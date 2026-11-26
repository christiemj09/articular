"""Extract every link in the article text, with where it sits, one worker per dump file.

    uv run python -m articular.pipeline.extract_links data/enwiki/*.bz2 --out data/enwiki_links --workers 8

For each input file this writes <out>/pages/<name>.parquet and <out>/links/<name>.parquet:

    pages: page_id, title, redirect_target, is_disambig, n_chars, n_sections, first_sentence
    links: page_id, pos, target, sec_idx, section, offset, word, word_abs, in_first_sentence, in_paren, italic, in_list

`pos` counts links through the whole article; `sec_idx` 0 is the lead and `section` its heading
("" for the lead); `offset` is the character position within the section after templates,
tables and references are removed; `word` counts the whitespace-separated words before the link in
its section and `word_abs` in the whole article. Links made by templates (navigation boxes, infoboxes) are
not in the article text and so are left out. Parsing rules match articular/pipeline/extract_leads.py, and
the lead links of the two agree. Finished files are skipped on a rerun.
"""

import argparse
import re
from bisect import bisect_left
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from articular.pipeline.extract_leads import COMMENT, DISAMBIG, REF, SKIP_PREFIX, TOKEN, iter_pages, norm_title, strip_blocks

HEADING_LINE = re.compile(r"^(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*$", re.M)
BLOCK_TAG = re.compile(r"<(gallery|timeline|imagemap|math|syntaxhighlight|source|score)\b.*?</\1>", re.S | re.I)
NOT_PROSE = ("[[file", "[[image", "{", "|", "*", "#", ":", ";", "<", "=", "__")
ABBREVIATION = re.compile(r"(?:\b[A-Z]|\b(?:St|Dr|Mr|Mrs|Ms|Jr|Sr|Prof|c|ca|b|d|e\.g|i\.e|vs|No|Mt|Ft|Inc|Ltd|Co|Corp|Bros|Gen|Col|Lt|Sgt|Capt|Rev|Gov|Sen|Rep|Hon|Ave|Blvd|Rd))$")
WORD = re.compile(r"\S+")
MAX_LINKS = 3000
BATCH_PAGES = 20_000

PAGES_SCHEMA = pa.schema([
    ("page_id", pa.int64()), ("title", pa.string()), ("redirect_target", pa.string()),
    ("is_disambig", pa.bool_()), ("n_chars", pa.int64()), ("n_sections", pa.int32()), ("first_sentence", pa.string()),
])
LINKS_SCHEMA = pa.schema([
    ("page_id", pa.int64()), ("pos", pa.int32()), ("target", pa.string()), ("sec_idx", pa.int32()),
    ("section", pa.dictionary(pa.int32(), pa.string())), ("offset", pa.int32()), ("word", pa.int32()), ("word_abs", pa.int32()),
    ("in_first_sentence", pa.bool_()), ("in_paren", pa.bool_()), ("italic", pa.bool_()), ("in_list", pa.bool_()),
])


def first_sentence_span(lead):
    """(start, end) of the first sentence of the first prose paragraph of the lead, or (0, 0)."""
    start = 0
    for line in lead.split("\n"):
        if len(line) > 30 and not line.lstrip().lower().startswith(NOT_PROSE):
            break
        start += len(line) + 1
    else:
        return 0, 0
    stop = lead.find("\n", start)
    stop = len(lead) if stop < 0 else stop
    link = paren = 0
    for i in range(start, stop):
        ch = lead[i]
        if ch == "[":
            link += 1
        elif ch == "]":
            link = max(0, link - 1)
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "." and not link and not paren and (i + 1 == stop or lead[i + 1] in " \n") \
                and not ABBREVIATION.search(lead[max(start, i - 6):i]):
            return start, i + 1
    return start, stop


def scan_links(text, sentence=(0, 0)):
    """Yield (target, offset, in_first_sentence, in_paren, italic, in_list) for links in one section."""
    pos, paren, italic = 0, 0, False
    in_list = text[:1] in ("*", "#", ":", ";")
    while m := TOKEN.search(text, pos):
        tok, pos = m.group(), m.end()
        if tok == "[[":
            depth, end = 1, pos
            while depth and (n := TOKEN.search(text, end)):
                end = n.end()
                depth += (n.group() == "[[") - (n.group() == "]]")
            if depth:
                break
            inner, pos = text[m.end(): end - 2], end
            target = inner.split("|")[0]
            if target.strip(" :").lower().startswith(SKIP_PREFIX) or "[[" in inner:
                continue
            if title := norm_title(target):
                yield title, m.start(), sentence[0] <= m.start() < sentence[1], paren > 0, italic, in_list
        elif tok == "(":
            paren += 1
        elif tok == ")":
            paren = max(0, paren - 1)
        elif tok == "\n":
            paren, italic, in_list = 0, False, text[pos:pos + 1] in ("*", "#", ":", ";")
        elif tok[0] == "'" and len(tok) != 3 and len(tok) != 4:
            italic = not italic


def article_links(wikitext):
    """(links, number of sections, first sentence). Each link: (target, sec_idx, section, offset, word, word_abs, flags…)."""
    text = strip_blocks(BLOCK_TAG.sub("", REF.sub("", COMMENT.sub("", wikitext))))
    cuts = [(m.start(), m.end(), m.group(2)) for m in HEADING_LINE.finditer(text)]
    bounds = [(0, cuts[0][0] if cuts else len(text), "")]
    bounds += [(end, cuts[i + 1][0] if i + 1 < len(cuts) else len(text), name) for i, (_, end, name) in enumerate(cuts)]
    lead = text[: bounds[0][1]]
    sentence = first_sentence_span(lead)
    links, words_before = [], 0
    for sec_idx, (start, stop, name) in enumerate(bounds):
        body = text[start:stop]
        word_starts = [m.start() for m in WORD.finditer(body)]
        for target, offset, *flags in scan_links(body, sentence if sec_idx == 0 else (0, 0)):
            word = bisect_left(word_starts, offset)
            links.append((target, sec_idx, name, offset, word, words_before + word, *flags))
            if len(links) >= MAX_LINKS:
                return links, len(bounds), lead[sentence[0]:sentence[1]][:600]
        words_before += len(word_starts)
    return links, len(bounds), lead[sentence[0]:sentence[1]][:600]


def extract_file(job):
    path, out = job
    name = path.name.removesuffix(".bz2").removesuffix(".xml")
    finals = [out / kind / f"{name}.parquet" for kind in ("pages", "links")]
    if all(p.exists() for p in finals):
        return name, None, None
    temps = [p.with_suffix(".tmp") for p in finals]
    n_pages = n_links = 0
    with pq.ParquetWriter(temps[0], PAGES_SCHEMA) as pages_out, pq.ParquetWriter(temps[1], LINKS_SCHEMA) as links_out:
        pages = {name: [] for name in PAGES_SCHEMA.names}
        links = {name: [] for name in LINKS_SCHEMA.names}

        def flush():
            pages_out.write_table(pa.table(pages, schema=PAGES_SCHEMA))
            links_out.write_table(pa.table(links, schema=LINKS_SCHEMA))
            for column in (*pages.values(), *links.values()):
                column.clear()

        for page_id, title, redirect, text in iter_pages(path):
            found, n_sections, sentence = ([], 0, None) if redirect else article_links(text)
            pages["page_id"].append(page_id)
            pages["title"].append(title)
            pages["redirect_target"].append(norm_title(redirect) if redirect else None)
            pages["is_disambig"].append(bool(DISAMBIG.search(text)))
            pages["n_chars"].append(len(text))
            pages["n_sections"].append(n_sections)
            pages["first_sentence"].append(sentence)
            n_pages += 1
            for pos, (target, sec_idx, section, offset, word, word_abs, in_first, in_paren, italic, in_list) in enumerate(found):
                links["page_id"].append(page_id)
                links["pos"].append(pos)
                links["target"].append(target)
                links["sec_idx"].append(sec_idx)
                links["section"].append(section)
                links["offset"].append(offset)
                links["word"].append(word)
                links["word_abs"].append(word_abs)
                links["in_first_sentence"].append(in_first)
                links["in_paren"].append(in_paren)
                links["italic"].append(italic)
                links["in_list"].append(in_list)
            n_links += len(found)
            if len(pages["page_id"]) >= BATCH_PAGES:
                flush()
        flush()
    for temp, final in zip(temps, finals):
        temp.rename(final)  # only complete files get their final name
    return name, n_pages, n_links


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="pages-articles .xml.bz2 files")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    args = parser.parse_args()

    for kind in ("pages", "links"):
        (args.out / kind).mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.out) for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)]
    total_pages = total_links = 0
    with Pool(args.workers) as pool:
        for name, n_pages, n_links in tqdm(pool.imap_unordered(extract_file, jobs), total=len(jobs), unit="file"):
            if n_pages is None:
                tqdm.write(f"{name}: already done, skipped")
            else:
                tqdm.write(f"{name}: {n_pages:,} pages, {n_links:,} links")
                total_pages += n_pages
                total_links += n_links
    print(f"done: {total_pages:,} pages and {total_links:,} links from {len(jobs)} files")


if __name__ == "__main__":
    main()

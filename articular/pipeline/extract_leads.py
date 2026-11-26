"""Extract lead links from pages-articles dump files, one worker per file.

    uv run python -m articular.pipeline.extract_leads data/enwiki/*.bz2 --out data/enwiki_extract --workers 8

For each input file this writes <out>/pages/<name>.parquet and <out>/links/<name>.parquet:

    pages: page_id, title, redirect_target, is_disambig, n_chars     (namespace 0 only)
    links: page_id, pos, target, in_paren, italic                    (links in the lead, in order)

Finished files are skipped on a rerun, so an interrupted run can be resumed.
The parsing rules match notebooks/01_simplewiki_eda.ipynb.
"""

import argparse
import bz2
import html
import re
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

COMMENT = re.compile(r"<!--.*?-->", re.S)
REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
HEADING = re.compile(r"^==", re.M)
BRACE = re.compile(r"\{\{|\}\}|\{\||\|\}")
TOKEN = re.compile(r"\[\[|\]\]|\(|\)|'{2,}|\n")
SKIP_PREFIX = ("file:", "image:", "category:", "media:")
DISAMBIG = re.compile(r"\{\{\s*(disambig|disamb|dab|hndis|geodis)", re.I)
MAX_LINKS = 30
BATCH_PAGES = 50_000

PAGES_SCHEMA = pa.schema([
    ("page_id", pa.int64()), ("title", pa.string()), ("redirect_target", pa.string()),
    ("is_disambig", pa.bool_()), ("n_chars", pa.int64()),
])
LINKS_SCHEMA = pa.schema([
    ("page_id", pa.int64()), ("pos", pa.int64()), ("target", pa.string()),
    ("in_paren", pa.bool_()), ("italic", pa.bool_()),
])


def strip_blocks(text):
    """Drop {{templates}} and {| tables |}, including nested ones."""
    out, stack, pos, keep_from = [], [], 0, 0
    while m := BRACE.search(text, pos):
        tok, pos = m.group(), m.end()
        if tok in ("{{", "{|"):
            if not stack:
                out.append(text[keep_from:m.start()])
            stack.append(tok)
        elif stack and (tok == "}}") == (stack[-1] == "{{"):
            stack.pop()
            if not stack:
                keep_from = pos
        elif tok == "|}":
            pos = m.start() + 1  # e.g. "|}}" closing a template: rescan from the brace
    if not stack:
        out.append(text[keep_from:])
    return "".join(out)


def norm_title(target):
    t = html.unescape(target).split("#")[0].replace("_", " ").strip(" :")
    t = " ".join(t.split())
    return t[:1].upper() + t[1:]


def lead_links(wikitext):
    """Ordered (target, in_paren, italic) for links in the lead."""
    m = HEADING.search(wikitext)
    text = wikitext[: m.start()] if m else wikitext
    text = strip_blocks(REF.sub("", COMMENT.sub("", text)))
    links, pos, paren, italic = [], 0, 0, False
    while (m := TOKEN.search(text, pos)) and len(links) < MAX_LINKS:
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
                links.append((title, paren > 0, italic))
        elif tok == "(":
            paren += 1
        elif tok == ")":
            paren = max(0, paren - 1)
        elif tok == "\n":
            paren, italic = 0, False
        elif tok[0] == "'" and len(tok) != 3 and len(tok) != 4:
            italic = not italic
    return links


def iter_pages(path):
    """Yield (page_id, title, redirect_title_or_None, wikitext) for namespace-0 pages."""
    page, root = {}, None
    with bz2.open(path, "rb") as f:
        for event, el in ET.iterparse(f, events=("start", "end")):
            if event == "start":
                if root is None:
                    root = el
                continue
            tag = el.tag.rsplit("}", 1)[-1]
            if tag in ("title", "ns", "text"):
                page[tag] = el.text or ""
            elif tag == "id":
                page.setdefault("id", int(el.text))  # first <id> in a page is the page id
            elif tag == "redirect":
                page["redirect"] = el.get("title")
            elif tag == "page":
                if page.get("ns") == "0":
                    yield page["id"], page["title"], page.get("redirect"), page.get("text", "")
                page = {}
                root.clear()  # drop finished pages so memory stays flat on large files


def extract_file(job):
    """Process one dump file; returns (name, pages written, links written)."""
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
            pages["page_id"].append(page_id)
            pages["title"].append(title)
            pages["redirect_target"].append(norm_title(redirect) if redirect else None)
            pages["is_disambig"].append(bool(DISAMBIG.search(text)))
            pages["n_chars"].append(len(text))
            n_pages += 1
            if not redirect:
                for pos, (target, in_paren, italic) in enumerate(lead_links(text)):
                    links["page_id"].append(page_id)
                    links["pos"].append(pos)
                    links["target"].append(target)
                    links["in_paren"].append(in_paren)
                    links["italic"].append(italic)
                    n_links += 1
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
    # largest files first so the long ones do not all end up at the tail of the run
    jobs = [(p, args.out) for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)]
    total_pages = total_links = 0
    with Pool(args.workers) as pool:
        for name, n_pages, n_links in tqdm(pool.imap_unordered(extract_file, jobs), total=len(jobs), unit="file"):
            if n_pages is None:
                tqdm.write(f"{name}: already done, skipped")
            else:
                tqdm.write(f"{name}: {n_pages:,} pages, {n_links:,} lead links")
                total_pages += n_pages
                total_links += n_links
    print(f"done: {total_pages:,} pages and {total_links:,} lead links from {len(jobs)} files")


if __name__ == "__main__":
    main()

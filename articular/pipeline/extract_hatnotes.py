"""Extract the notes at the head of sections that point to another article, one worker per dump file.

    uv run python -m articular.pipeline.extract_hatnotes data/enwiki/*.bz2 --out data/enwiki_hatnotes --workers 8

For each input file this writes <out>/<name>.parquet, one row for each article a note names:

    page_id, template, target, sec_idx, section, level

`template` is one of

    main      {{Main}}, {{Main article}}: "Main article: X". The section summarises X.
    further   {{Further}}, {{Further information}}: "Further information: X"
    see also  {{See also}}: "See also: X"
    broader   {{Broader}}: "For broader coverage of this topic, see X"

`sec_idx` 0 is the lead, `section` the heading the note sits under ("" for the lead) and `level`
the number of = signs of that heading (0 for the lead). Finished files are skipped on a rerun.
"""

import argparse
import re
from bisect import bisect_right
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from articular.pipeline.extract_leads import COMMENT, iter_pages, norm_title
from articular.pipeline.extract_links import HEADING_LINE

NOTE = re.compile(r"\{\{\s*(main(?:[ _]articles?)?|further(?:[ _]information)?|see[ _]also|broader)\s*\|([^{}]*)\}\}", re.I)
NAMED = re.compile(r"^\s*[\w ]+=")   # l1=, label 2=, selfref= and the like: not an article
BATCH_ROWS = 200_000

SCHEMA = pa.schema([
    ("page_id", pa.int64()), ("template", pa.dictionary(pa.int32(), pa.string())), ("target", pa.string()),
    ("sec_idx", pa.int32()), ("section", pa.string()), ("level", pa.int32()),
])


def template_name(raw):
    name = raw.lower().replace("_", " ")
    return "main" if name.startswith("main") else "further" if name.startswith("further") else name


def article_notes(wikitext):
    """(template, target, sec_idx, section, level) for each article named by a note."""
    text = COMMENT.sub("", wikitext)
    if "{{" not in text:
        return []
    heads = [(m.start(), m.group(2), len(m.group(1))) for m in HEADING_LINE.finditer(text)]
    starts = [h[0] for h in heads]
    notes = []
    for m in NOTE.finditer(text):
        sec_idx = bisect_right(starts, m.start())
        _, section, level = heads[sec_idx - 1] if sec_idx else (0, "", 0)
        for arg in m.group(2).split("|"):
            if NAMED.match(arg):
                continue
            if title := norm_title(arg.replace("[[", "").replace("]]", "")):
                notes.append((template_name(m.group(1)), title, sec_idx, section, level))
    return notes


def extract_file(job):
    path, out = job
    name = path.name.removesuffix(".bz2").removesuffix(".xml")
    final = out / f"{name}.parquet"
    if final.exists():
        return name, None
    temp, n_notes = final.with_suffix(".tmp"), 0
    with pq.ParquetWriter(temp, SCHEMA) as writer:
        rows = {name: [] for name in SCHEMA.names}

        def flush():
            writer.write_table(pa.table(rows, schema=SCHEMA))
            for column in rows.values():
                column.clear()

        for page_id, _, redirect, text in iter_pages(path):
            if redirect:
                continue
            for note in article_notes(text):
                for column, value in zip(SCHEMA.names, (page_id, *note)):
                    rows[column].append(value)
                n_notes += 1
            if len(rows["page_id"]) >= BATCH_ROWS:
                flush()
        flush()
    temp.rename(final)  # only complete files get their final name
    return name, n_notes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="pages-articles .xml.bz2 files")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.out) for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)]
    total = 0
    with Pool(args.workers) as pool:
        for name, n_notes in tqdm(pool.imap_unordered(extract_file, jobs), total=len(jobs), unit="file"):
            if n_notes is None:
                tqdm.write(f"{name}: already done, skipped")
            else:
                tqdm.write(f"{name}: {n_notes:,} notes")
                total += n_notes
    print(f"done: {total:,} notes from {len(jobs)} files")


if __name__ == "__main__":
    main()

"""Measure the title of every article: its length, its form, and how many other titles are built on it.

    uv run python -m articular.pipeline.title_features --wiki enwiki

Reads data/<wiki>_text_features.parquet (articular/pipeline/text_features.py) for the page ids and titles.
Writes data/<wiki>_title_features.parquet, one row per article:

  norm                the title in lower case, words only, without a closing "(...)", with a space at
                      each end, so that one title can be looked for inside another as a run of words
  title_words         words in the title, without a closing "(...)"
  title_chars         characters in the title, without a closing "(...)"
  title_caps          words after the first that start with a capital letter
  title_digit         1 if the title has a digit in it
  title_paren         1 if the title ends in "(...)"
  title_comma         1 if the title has a comma in it
  titles_containing   other articles whose title contains this one as a run of words and is longer:
                      "Prime number" and "Number theory" both count for "Number". Null for a title
                      that ends in "(...)" or is spelled like an earlier one, as for name_df.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

WORD = re.compile(r"[^\W_]+")
CLOSING_PAREN = re.compile(r"\s*\([^()]*\)$")
MAX_RUN = 5   # the longest title, in words, that is looked for inside other titles


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data, wiki = args.data, args.wiki

    titles = duckdb.sql(f"select page_id, title from '{data}/{wiki}_text_features.parquet' order by page_id").fetchall()
    rows = {name: [] for name in ("page_id", "norm", "title_words", "title_chars", "title_caps", "title_digit", "title_paren", "title_comma")}
    parts_of, owner = [], {}
    for page_id, title in titles:
        base = CLOSING_PAREN.sub("", title)
        paren = base != title
        parts = WORD.findall(base.lower())
        parts_of.append(parts)
        if parts and not paren:
            owner.setdefault(" ".join(parts), page_id)
        rows["page_id"].append(page_id)
        rows["norm"].append(" " + " ".join(parts) + " ")
        rows["title_words"].append(len(parts))
        rows["title_chars"].append(len(base))
        rows["title_caps"].append(sum(w[0].isupper() for w in WORD.findall(base)[1:]))
        rows["title_digit"].append(int(any(c.isdigit() for c in title)))
        rows["title_paren"].append(int(paren))
        rows["title_comma"].append(int("," in title))

    # for each title, every shorter run of its words that is itself a title
    containing = Counter()
    for (page_id, _), parts in zip(titles, parts_of):
        found = set()
        for i in range(len(parts)):
            for j in range(i + 1, min(len(parts), i + MAX_RUN) + 1):
                if j - i < len(parts):
                    hit = owner.get(" ".join(parts[i:j]))
                    if hit is not None and hit != page_id:
                        found.add(hit)
        containing.update(found)
    rows["titles_containing"] = [containing[page_id] if owner.get(norm.strip()) == page_id and not paren else None
                                 for page_id, norm, paren in zip(rows["page_id"], rows["norm"], rows["title_paren"])]

    out = data / f"{wiki}_title_features.parquet"
    pq.write_table(pa.table(rows), out, compression="zstd")
    print(f"{out}: {len(titles):,} articles, {len(owner):,} titles looked for inside other titles")
    print(duckdb.sql(f"""
        select f.title, h.title_words, h.title_caps, h.titles_containing
        from '{out}' h join '{data}/{wiki}_text_features.parquet' f using (page_id)
        where f.title in ('Animal', 'Dog', 'Poodle', 'Science', 'Physics', 'Quantum mechanics', 'France', 'Paris', 'Number',
                          'Prime number', 'Battle', 'Battle of Stalingrad', 'Music', 'Jazz', 'Mountain', 'Mount Everest')
        order by h.titles_containing desc
    """).df().to_string(index=False))


if __name__ == "__main__":
    main()

"""Print what the extracts hold for the lead of some articles: the relations read from the opening
sentence, and the first links in order.

    uv run python -m articular.tools.show_lead --wiki enwiki --pages 47962,1127541

Reads data/<wiki>_relations/ (articular/pipeline/extract_relations.py) and data/<wiki>_links/
(articular/pipeline/extract_links.py). The sentence itself is not kept by either.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--pages", required=True, help="comma-separated page ids")
    parser.add_argument("--links", type=int, default=8, help="links to show for each article")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    con = duckdb.connect()
    for page in args.pages.split(","):
        print(f"\n======== page {page} ========\n")
        print(con.sql(f"select relation, prep, head, phrase, target, pos from read_parquet('{data}/{wiki}_relations/*.parquet') where page_id = {page} order by pos").df().to_string(index=False))
        print()
        print(con.sql(f"""select pos, target, word_abs, in_first_sentence, in_paren, italic from read_parquet('{data}/{wiki}_links/links/*.parquet')
                          where page_id = {page} and sec_idx = 0 order by pos limit {args.links}""").df().to_string(index=False))


if __name__ == "__main__":
    main()

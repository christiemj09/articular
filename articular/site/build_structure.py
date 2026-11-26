"""Gather the tables behind the site's Structure page into one small file.

    uv run python -m articular.site.build_structure --wiki enwiki --out data/enwiki_site

Reads data/<wiki>_flow_stats.json (articular/experiments/compare_score_flow.py) and data/<wiki>_link_position.csv
(articular/experiments/link_specificity_by_position.py). Writes <out>/structure.json, gzip-compressed unless
--no-gzip is given:

  maps      for each map ("first", "kind"): pages by number of steps to the end of their path with
            the score's quartiles at each distance, flow by band of the score, and the share of
            steps that go to a more general page
  places    share of links pointing to a more general article, by place in the article and by how
            general the linking article is; the same by section number and by tenth of the links
  headings  the section headings whose links point up least and most often
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-gzip", action="store_true", help="write plain JSON (for a local file server)")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki

    links = pd.read_csv(data / f"{wiki}_link_position.csv")
    name = lambda v: v.split(" ", 1)[1]          # "1 first sentence" -> "first sentence"

    def table(title, key, convert=name):
        """One table of the file as {sources, rows: [{key, up: [...], target: [...], links: [...]}]}, sources in order."""
        part = links[links["table"].str.startswith(title)]
        sources = sorted(part["source"].unique())
        rows = []
        for value, group in part.groupby(key, sort=True):
            group = group.set_index("source").reindex(sources)
            known = lambda column: [None if pd.isna(v) else v for v in group[column]]      # JSON has no NaN
            rows.append({key: convert(value), "up": known("up_pct"), "target": known("target_median"),
                         "links": group["links"].fillna(0).astype(int).tolist()})
        return {"sources": [name(s) for s in sources], "rows": rows}

    def headings(title):
        part = links[links["table"] == title]
        return [{"heading": r.heading, "links": int(r.links), "up": r.up_pct, "in_lists": r.in_lists_pct} for r in part.itertuples()]

    least = next(t for t in links["table"].unique() if t.startswith("headings whose links point up least often"))
    out = {
        "wiki": wiki,
        "links": int(links[links["table"].str.startswith("by place")]["links"].sum()),
        "maps": json.loads((data / f"{wiki}_flow_stats.json").read_text()),
        "places": table("by place", "place"),
        "sections": table("by section number", "section_number", int),
        "tenths": table("by tenth", "tenth", int),
        "headings": {"down": headings(least), "up": headings("headings whose links point up most often"),
                     "common": headings("the commonest headings"),
                     "general_down": headings("linking article in the most general 10%: headings whose links point up least often"),
                     "general_up": headings("linking article in the most general 10%: headings whose links point up most often")},
    }
    text = json.dumps(out, separators=(",", ":")).encode()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "structure.json").write_bytes(text if args.no_gzip else gzip.compress(text, 6))
    print(f"structure.json: {len(text):,} bytes; {out['links']:,} links, maps: {', '.join(out['maps'])}")


if __name__ == "__main__":
    main()

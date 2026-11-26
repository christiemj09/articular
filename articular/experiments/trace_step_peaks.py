"""Which links put one kind of article at one number of steps from the end of its path?

    uv run python -m articular.experiments.trace_step_peaks --wiki enwiki

Reads, for each map, the table of the link each article follows: data/<wiki>_scores.parquet (first
link) and data/<wiki>_scores_kind.parquet (kind link); and data/<wiki>_kind_families.parquet
(articular/pipeline/build_kind_families.py).

articular/experiments/compare_score_flow.py counts the articles at each number of steps and finds peaks that
belong to single kinds: people at 17 steps on the first-link map, for one. A peak is given here as
map:steps:kind. For each peak this prints:
  - the links the kind's articles follow, commonest first, with the number of steps each link's
    target is from the end (an article is one step further than the page it links to), so the
    peak can be set beside the kind's other links;
  - the same for the articles in the peak only;
  - the trunk: at each number of steps between the peak and the end, the pages that most of the
    peak's paths pass through.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAPS = {"first": "{wiki}_scores.parquet", "kind": "{wiki}_scores_kind.parquet"}
PEAKS = "first:17:People,first:24:Settlements and regions,kind:5:People,kind:16:Creative works"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--peaks", default=PEAKS, help="comma-separated map:steps:kind")
    parser.add_argument("--top", type=int, default=20, help="links to list for each peak")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    kinds = data / f"{wiki}_kind_families.parquet"
    kind_names = json.loads(kinds.with_suffix(".json").read_text())
    families = pd.read_parquet(kinds).set_index("page_id")["family"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)

    peaks = [p.split(":", 2) for p in args.peaks.split(",")]
    for label, file in MAPS.items():
        mine = [(int(k), name) for m, k, name in peaks if m == label]
        if not mine:
            continue
        pages = pd.read_parquet(data / file.format(wiki=wiki), columns=["page_id", "title", "first_link_page_id"])
        n = len(pages)
        row_of = pd.Series(np.arange(n), index=pages["page_id"].to_numpy())
        has_link = (pages["first_link_page_id"] >= 0).to_numpy()
        nxt = np.full(n, -1)
        nxt[has_link] = row_of[pages["first_link_page_id"].to_numpy()[has_link]].to_numpy()
        # springs peeled inward, as articular/experiments/compare_score_flow.py does; then steps to the end, downstream pages first
        remaining = np.bincount(nxt[nxt >= 0], minlength=n)
        frontier, layers = np.flatnonzero(remaining == 0), []
        while len(frontier):
            layers.append(frontier)
            f = frontier[nxt[frontier] >= 0]
            np.subtract.at(remaining, nxt[f], 1)
            hit = np.unique(nxt[f])
            frontier = hit[remaining[hit] == 0]
        steps = np.zeros(n, dtype=np.int64)
        for layer in reversed(layers):
            f = layer[nxt[layer] >= 0]
            steps[f] = steps[nxt[f]] + 1
        title = pages["title"].to_numpy()
        family = families.reindex(pages["page_id"]).fillna(len(kind_names) - 1).to_numpy().astype(np.int64)

        def followed(chosen, top):          # the links these articles follow, commonest first
            target = nxt[chosen & (nxt >= 0)]
            count = np.bincount(target, minlength=n)
            order = np.argsort(-count, kind="stable")[:top]
            order = order[count[order] > 0]
            return pd.DataFrame({"link followed": title[order], "articles": count[order], "% of them": 100 * count[order] / chosen.sum(),
                                 "kind of the target": [kind_names[k] for k in family[order]], "steps of the article": steps[order] + 1})

        for k, name in mine:
            of_kind = family == kind_names.index(name)
            peak = of_kind & (steps == k)
            print(f"\n\n======== {label} map, {name}, {k} steps: {int(peak.sum()):,} of {int(of_kind.sum()):,} articles ({100 * peak.sum() / of_kind.sum():.1f}%) ========")
            print(f"\nthe links that {name} articles follow, at any number of steps\n")
            print(followed(of_kind, args.top).round(1).to_string(index=False))
            print(f"\nthe links that the articles at {k} steps follow\n")
            print(followed(peak, args.top).round(1).to_string(index=False))
            distinct = len(np.unique(nxt[peak]))
            print(f"\n  the {int(peak.sum()):,} articles follow links to {distinct:,} different pages")

            # the peak's paths, counted on every page they pass through
            through = peak.astype(np.int64)
            for layer in layers:
                f = layer[(nxt[layer] >= 0) & (through[layer] > 0)]
                np.add.at(through, nxt[f], through[f])
            print(f"\nthe trunk: pages the peak's paths pass through, by steps from the end (% of the peak)\n")
            rows = []
            for d in range(k - 1, -1, -1):
                at = np.flatnonzero((steps == d) & (through > 0))
                best = at[np.argsort(-through[at], kind="stable")[:3]]
                rows.append({"steps": d, "pages passed through": len(at),
                             "the three most used": "; ".join(f"{title[b]} {100 * through[b] / peak.sum():.1f}%" for b in best)})
            print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

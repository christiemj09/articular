"""Steps to the end of a path on kind links alone, with the first-link fallback steps removed.

    uv run python -m articular.experiments.kind_only_steps --wiki enwiki

Reads data/<wiki>_scores_kind.parquet (articular/pipeline/build_link_choice.py --rule kind),
data/<wiki>_kind_families.parquet (articular/pipeline/build_kind_families.py) and, for comparison,
data/<wiki>_flow_stats.json (articular/experiments/compare_score_flow.py).

On the kind-link map an article with no kind link follows its first link, and
articular/experiments/count_fallback_paths.py finds that 43% of the steps on that map are such fallbacks. Here
they are left out, as articular/pipeline/kind_graph.py does: a chain stops at an article with no kind link, or in
a loop. This prints, for all articles and for each kind of thing:
  - the articles at each number of steps;
  - the mean, the variance over the mean (1 for a Poisson distribution) and the share of articles
    that would have to move to make the counts a Poisson distribution with the same mean;
  - the same figures for the two maps, from the flow statistics;
  - where the chains end.
"""

import argparse
import json
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd


def shape(counts):
    """mean, variance / mean, share to move to reach a Poisson distribution, share at 0, commonest."""
    counts = np.asarray(counts, dtype=float)
    k, n = np.arange(len(counts)), counts.sum()
    mean = (k * counts).sum() / n
    var = ((k - mean) ** 2 * counts).sum() / n
    poisson = np.exp(k * np.log(mean) - mean - np.array([lgamma(i + 1) for i in k])) if mean > 0 else (k == 0).astype(float)
    move = 0.5 * (np.abs(counts / n - poisson).sum() + 1 - poisson.sum())
    return {"articles": int(n), "mean steps": mean, "variance / mean": var / mean if mean else np.nan, "to move %": 100 * move,
            "at 0 %": 100 * counts[0] / n, "commonest": int(counts.argmax()), "longest": int(np.flatnonzero(counts).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--top", type=int, default=25, help="chain ends to list")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    kinds = data / f"{wiki}_kind_families.parquet"
    kind_names = json.loads(kinds.with_suffix(".json").read_text())
    families = pd.read_parquet(kinds).set_index("page_id")["family"]
    pd.set_option("display.width", 250)

    pages = pd.read_parquet(data / f"{wiki}_scores_kind.parquet", columns=["page_id", "title", "is_disambig", "first_link_page_id", "rule"])
    n = len(pages)
    row_of = pd.Series(np.arange(n), index=pages["page_id"].to_numpy())
    has = (pages["rule"].str.startswith("kind") & (pages["first_link_page_id"] >= 0)).to_numpy()          # not the first-link fallback
    nxt = np.full(n, -1)
    nxt[has] = row_of[pages["first_link_page_id"].to_numpy()[has]].to_numpy()
    title = pages["title"].to_numpy()
    family = families.reindex(pages["page_id"]).fillna(len(kind_names) - 1).to_numpy().astype(np.int64)
    # springs peeled inward, as articular/experiments/compare_score_flow.py does; what is left is on a loop
    remaining = np.bincount(nxt[nxt >= 0], minlength=n)
    frontier, layers = np.flatnonzero(remaining == 0), []
    while len(frontier):
        layers.append(frontier)
        f = frontier[nxt[frontier] >= 0]
        np.subtract.at(remaining, nxt[f], 1)
        hit = np.unique(nxt[f])
        frontier = hit[remaining[hit] == 0]
    in_loop = remaining > 0
    # steps to the end of the chain and the page it ends at, downstream pages first
    depth, end = np.zeros(n, dtype=np.int64), np.arange(n)
    for layer in reversed(layers):
        f = layer[nxt[layer] >= 0]
        depth[f] = depth[nxt[f]] + 1
        end[f] = end[nxt[f]]
    longest = depth.max() + 1
    print(f"{n:,} articles; {int(has.sum()):,} ({100 * has.mean():.1f}%) have a kind link")
    print(f"at 0 steps: {int((depth == 0).sum()):,}: in a loop {int(in_loop.sum()):,}, disambiguation pages {int(pages['is_disambig'].sum()):,}, "
          f"other articles with no kind link {int((~has & ~pages['is_disambig'].to_numpy()).sum()):,}")

    print("\narticles by the number of kind-link steps to the end of the chain\n")
    by_kind = pd.DataFrame({name: np.bincount(depth[family == k], minlength=longest) for k, name in enumerate(kind_names)})
    print(pd.DataFrame({"articles": np.bincount(depth, minlength=longest)}).rename_axis("steps").T.to_string())
    print("\nthe same count for each kind of thing\n")
    print(by_kind.to_string())

    left_out = ["Lists and timelines", "No kind found"]
    plain = ~np.isin(family, [kind_names.index(name) for name in left_out if name in kind_names])
    rows = {"kind links only: all articles": shape(np.bincount(depth)),
            "kind links only: articles with a kind link": shape(np.bincount(depth[has])),
            f"kind links only: without {' and '.join(left_out)}": shape(np.bincount(depth[plain]))}
    stats = data / f"{wiki}_flow_stats.json"
    if stats.exists():
        for label, kept in json.loads(stats.read_text()).items():
            rows[f"{kept['name']} map"] = shape([r["pages"] for r in sorted(kept["steps"], key=lambda r: r["steps"])])
    print("\n\nthe shape of the counts (variance / mean is 1 for a Poisson distribution)\n")
    print(pd.DataFrame(rows).T.round(2).to_string())

    print("\nby kind of thing, kind links only\n")
    table = pd.DataFrame({name: shape(by_kind[name]) for name in kind_names}).T
    if stats.exists():
        for label, kept in json.loads(stats.read_text()).items():
            if "kinds" in kept:
                table[f"{label} map: mean"] = [shape(k["pages"])["mean steps"] for k in kept["kinds"]]
                table[f"{label} map: variance / mean"] = [shape(k["pages"])["variance / mean"] for k in kept["kinds"]]
    print(table.round(2).to_string())

    print(f"\n\nwhere the chains end: the {args.top} commonest (articles with a kind link)\n")
    ends = pd.Series(end[has]).value_counts().head(args.top)
    print(pd.DataFrame({"ends at": title[ends.index], "articles": ends.to_numpy(),
                        "% of those with a kind link": (100 * ends.to_numpy() / has.sum()).round(1),
                        "rule": pages["rule"].to_numpy()[ends.index], "loop": in_loop[ends.index]}).to_string(index=False))
    print(f"\n  chains that end in a loop: {100 * in_loop[end[has]].mean():.1f}%; at an article with no kind link: {100 * (~in_loop[end[has]]).mean():.1f}%")
    print(f"  different ends: {len(np.unique(end[has])):,}")


if __name__ == "__main__":
    main()

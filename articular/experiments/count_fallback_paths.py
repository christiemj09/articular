"""How much of the kind-link map is first-link behaviour? Count the paths that pass through a fallback.

    uv run python -m articular.experiments.count_fallback_paths --wiki enwiki

Reads data/<wiki>_scores_kind.parquet (articular/pipeline/build_link_choice.py --rule kind) and
data/<wiki>_kind_families.parquet (articular/pipeline/build_kind_families.py).

On the kind-link map an article follows the link that says what kind of thing it is. Where the
opening sentence has no such link, the article falls back to its first link; the `rule` column
records that as "first". This prints:
  - the articles by rule;
  - for each article, the number of fallback steps on its path to the end, counting its own;
  - the same split by the article's own rule, by kind of thing and by the length of the path;
  - at each number of steps from the end, the share of pages that are fallbacks, and the share of
    the paths passing through that distance that are on a fallback page there;
  - the pages on the closing cycles, and how many are fallbacks;
  - the fallback pages that the most paths pass through.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--top", type=int, default=40, help="fallback pages to list")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    kinds = data / f"{wiki}_kind_families.parquet"
    kind_names = json.loads(kinds.with_suffix(".json").read_text())
    families = pd.read_parquet(kinds).set_index("page_id")["family"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)

    pages = pd.read_parquet(data / f"{wiki}_scores_kind.parquet", columns=["page_id", "title", "first_link_page_id", "rule"])
    n = len(pages)
    print(f"{n:,} articles\n\nby the rule that supplied the link\n")
    print(pages["rule"].value_counts().to_frame("articles").assign(**{"%": lambda t: (100 * t["articles"] / n).round(1)}).to_string())
    row_of = pd.Series(np.arange(n), index=pages["page_id"].to_numpy())
    has_link = (pages["first_link_page_id"] >= 0).to_numpy()
    nxt = np.full(n, -1)
    nxt[has_link] = row_of[pages["first_link_page_id"].to_numpy()[has_link]].to_numpy()
    fallback = (pages["rule"] == "first").to_numpy() & has_link
    title = pages["title"].to_numpy()
    family = families.reindex(pages["page_id"]).fillna(len(kind_names) - 1).to_numpy().astype(np.int64)

    # flow and layers, by peeling springs inward as articular/experiments/compare_score_flow.py does
    flow, remaining = np.ones(n, dtype=np.int64), np.bincount(nxt[nxt >= 0], minlength=n)
    frontier, layers = np.flatnonzero(remaining == 0), []
    while len(frontier):
        layers.append(frontier)
        f = frontier[nxt[frontier] >= 0]
        np.add.at(flow, nxt[f], flow[f])
        np.subtract.at(remaining, nxt[f], 1)
        hit = np.unique(nxt[f])
        frontier = hit[remaining[hit] == 0]
    on_cycle = remaining > 0
    # steps to the end, and fallback steps on the way (the article's own included), downstream pages first
    steps, falls = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for layer in reversed(layers):
        f = layer[nxt[layer] >= 0]
        steps[f] = steps[nxt[f]] + 1
        falls[f] = falls[nxt[f]] + fallback[f]
    moving = steps > 0          # articles with a path: not on a cycle and with a link to follow

    def split(keep):
        f = falls[keep]
        return {"articles": int(keep.sum()), "none %": 100 * (f == 0).mean(), "one %": 100 * (f == 1).mean(), "two %": 100 * (f == 2).mean(),
                "three %": 100 * (f == 3).mean(), "four or more %": 100 * (f >= 4).mean(), "mean fallbacks": f.mean(),
                "mean steps": steps[keep].mean(), "fallback steps as % of steps": 100 * f.sum() / steps[keep].sum()}

    print(f"\n\nfallback steps on the path of each article that has a path ({int(moving.sum()):,} articles)\n")
    rows = {"all": split(moving), "the article follows a kind link": split(moving & ~fallback), "the article itself falls back": split(moving & fallback)}
    print(pd.DataFrame(rows).T.round(2).to_string())
    later = falls - fallback          # fallbacks after the article's own step
    kind_start = moving & ~fallback
    print(f"\n  of the {int(kind_start.sum()):,} that start on a kind link, {100 * (later[kind_start] > 0).mean():.1f}% pass through a fallback later")
    print(f"  paths made of kind links only, start to end: {int((moving & (falls == 0)).sum()):,} ({100 * (falls[moving] == 0).mean():.1f}%)")

    print("\n\nby kind of thing\n")
    print(pd.DataFrame({name: split(moving & (family == k)) for k, name in enumerate(kind_names)}).T.round(2).to_string())

    print("\n\nby the length of the path\n")
    rows = {}
    for d in range(1, steps.max() + 1):
        keep = steps == d
        if keep.sum() >= 1000:
            f = falls[keep]
            rows[d] = {"articles": int(keep.sum()), "none %": 100 * (f == 0).mean(), "mean fallbacks": f.mean(), "fallback steps as % of steps": 100 * f.mean() / d}
    print(pd.DataFrame(rows).T.round(2).rename_axis("steps").to_string())
    by_falls = pd.DataFrame({"articles": np.bincount(falls[moving]), "mean steps": pd.Series(steps[moving]).groupby(falls[moving]).mean()})
    print("\npath length by the number of fallback steps on it\n")
    print(by_falls.round(2).rename_axis("fallback steps").to_string())

    print("\n\nby steps from the end: pages there that are fallbacks, and paths passing through that are on one\n")
    rows = {}
    for d in range(0, steps.max() + 1):
        at = (steps == d) & (nxt >= 0)
        if at.sum() >= 100:
            rows[d] = {"pages": int(at.sum()), "fallback pages %": 100 * fallback[at].mean(), "paths through": int(flow[at].sum()),
                       "paths on a fallback page %": 100 * flow[at & fallback].sum() / flow[at].sum()}
    print(pd.DataFrame(rows).T.round(1).rename_axis("steps").to_string())

    cycle = on_cycle & (nxt >= 0)
    print(f"\n\npages on a closing cycle: {int(cycle.sum()):,}, of which fallbacks {int((cycle & fallback).sum()):,} ({100 * fallback[cycle].mean():.1f}%)")
    big = np.flatnonzero(cycle)
    big = big[np.argsort(-flow[big], kind="stable")[:30]]
    print("\nthe cycle pages most paths arrive at\n")
    print(pd.DataFrame({"title": title[big], "arriving": flow[big], "rule": pages["rule"].to_numpy()[big], "next": title[nxt[big]]}).to_string(index=False))

    print("\n\nthe fallback pages that the most paths pass through\n")
    top = np.flatnonzero(fallback)
    top = top[np.argsort(-flow[top], kind="stable")[:args.top]]
    print(pd.DataFrame({"title": title[top], "paths through": flow[top], "% of all articles": 100 * flow[top] / n, "first link followed": title[nxt[top]],
                        "steps from the end": steps[top], "on a cycle": on_cycle[top]}).round(1).to_string(index=False))


if __name__ == "__main__":
    main()

"""Compare the specificity score with the maps: are the general articles the ones paths flow through?

    uv run python -m articular.experiments.compare_score_flow --wiki enwiki

Reads data/<wiki>_specificity_score.parquet (articular/pipeline/build_specificity_score.py) and, for each map,
the table of the link each article follows: data/<wiki>_scores.parquet (first link) and
data/<wiki>_scores_kind.parquet (kind link).

The score does not use the maps. For each map this prints:
  - the rank correlation between the score and the number of articles whose path runs through a
    page, with lead in-links beside it for comparison;
  - for each band of the score, how much flows through its pages; and the other way round, which
    bands the pages with the most flow are in;
  - the pages where the two disagree most, each way;
  - whether a step along a path goes to a more general page, by the band of the page it leaves;
  - how many pages lie n steps from where their path ends, the ratio of each count to the one
    before (the fan-out), and the score of the pages at each distance.

If data/<wiki>_kind_families.parquet exists (articular/pipeline/build_kind_families.py), the pages at each number
of steps are also counted for each kind of thing.

Writes data/<wiki>_flow_stats.json: the tables above (not the lists of pages), for the charts that
articular/site/build_structure.py puts on the site.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from articular.site.build_tiles import BANDS

COLUMN = "no first-200-word vocabulary, others clipped"
MAPS = {"first link": "{wiki}_scores.parquet", "kind link": "{wiki}_scores_kind.parquet"}


def ranks(v):
    out = np.empty(len(v))
    out[np.argsort(v, kind="stable")] = np.arange(len(v))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    score = pd.read_parquet(data / f"{wiki}_specificity_score.parquet", columns=["page_id", "in_deg", COLUMN]).set_index("page_id")
    names = [name for _, name in BANDS]
    kinds = data / f"{wiki}_kind_families.parquet"
    if kinds.exists():
        kind_names = json.loads(kinds.with_suffix(".json").read_text())
        families = pd.read_parquet(kinds).set_index("page_id")["family"]
    pd.set_option("display.width", 250)

    stats = {}
    for label, file in MAPS.items():
        path = data / file.format(wiki=wiki)
        if not path.exists():
            continue
        kept = stats[label.split()[0]] = {"name": label}
        pages = pd.read_parquet(path, columns=["page_id", "title", "first_link_page_id"])
        n = len(pages)
        row_of = pd.Series(np.arange(n), index=pages["page_id"].to_numpy())
        has_link = (pages["first_link_page_id"] >= 0).to_numpy()
        nxt = np.full(n, -1)
        nxt[has_link] = row_of[pages["first_link_page_id"].to_numpy()[has_link]].to_numpy()
        # flow, by peeling springs inward as articular/site/build_tiles.py does; depth = steps from the furthest spring
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
        # steps from each page to where its path ends (a cycle or a page with no link): downstream pages first
        steps = np.zeros(n, dtype=np.int64)
        for layer in reversed(layers):
            f = layer[nxt[layer] >= 0]
            steps[f] = steps[nxt[f]] + 1

        rank = score[COLUMN].reindex(pages["page_id"]).to_numpy()          # 0 = most general
        in_deg = score["in_deg"].reindex(pages["page_id"]).to_numpy()
        ok = ~np.isnan(rank)
        band = np.searchsorted([below for below, _ in BANDS], np.nan_to_num(rank), side="right")
        title = pages["title"].to_numpy()
        print(f"\n\n======== {label} map: {n:,} articles, {int(ok.sum()):,} with a score ========")

        def spearman(a, b, keep):
            return np.corrcoef(ranks(a[keep]), ranks(b[keep]))[0, 1]
        inner = ok & (flow > 1)
        print("\nrank correlation with the number of articles flowing through a page (1 = same order)\n")
        print(f"  generality by the score      all pages {spearman(-rank, flow, ok):.3f}   pages something flows into {spearman(-rank, flow, inner):.3f}")
        print(f"  lead in-links                all pages {spearman(in_deg, flow, ok):.3f}   pages something flows into {spearman(in_deg, flow, inner):.3f}")
        print(f"  pages that something flows into: {int(inner.sum()):,} ({100 * inner.sum() / ok.sum():.1f}%)")
        kept.update(articles=n, scored=int(ok.sum()), flowed_into=int(inner.sum()), on_cycle=int(on_cycle.sum()),
                    score_flow={"all": spearman(-rank, flow, ok), "flowed_into": spearman(-rank, flow, inner)},
                    in_links_flow={"all": spearman(in_deg, flow, ok), "flowed_into": spearman(in_deg, flow, inner)})

        print("\nby band of the score: how much flows through its pages\n")
        rows = {}
        for b, name in enumerate(names):
            mine = ok & (band == b)
            f = flow[mine]
            rows[name] = {"pages": mine.sum(), "springs %": 100 * (f == 1).mean(), "flow 10+ %": 100 * (f >= 10).mean(),
                          "flow 100+ %": 100 * (f >= 100).mean(), "flow 1,000+ %": 100 * (f >= 1000).mean(),
                          "flow 10,000+ %": 100 * (f >= 10_000).mean(), "median flow": np.median(f), "on a cycle": int(on_cycle[mine].sum())}
        print(pd.DataFrame(rows).T.round(1).to_string())
        kept["bands"] = [{"name": name, **row} for name, row in rows.items()]

        print("\nthe other way round: the pages with the most flow, by band of the score (%)\n")
        order = np.argsort(-flow, kind="stable")
        order = order[ok[order]]
        rows = {}
        for k in (100, 691, 6910, 69_100):
            top = order[:k]
            rows[k] = {"least flow": flow[top[-1]], **{name: 100 * (band[top] == b).mean() for b, name in enumerate(names)}}
        print(pd.DataFrame(rows).T.round(1).rename(index=lambda k: f"top {k:,} by flow").to_string())
        kept["most_flow"] = [{"pages": k, **row} for k, row in rows.items()]

        table = pd.DataFrame({"title": title, "flow": flow, "in_links": in_deg, "more specific than %": rank})
        print("\ngeneral by the score, little flow: the most general 0.01% with the least flowing through\n")
        top = np.flatnonzero(ok & (band == 0))
        print(f"  of {len(top):,}: {int((flow[top] == 1).sum())} are springs, {int((flow[top] < 10).sum())} have flow under 10, {int((flow[top] < 100).sum())} under 100\n")
        print(table.iloc[top[np.argsort(flow[top], kind="stable")[:40]]].round(3).to_string(index=False))
        print("\nmuch flow, specific by the score: pages in the top 6,910 by flow that the score ranks lowest\n")
        top = order[:6910]
        print(table.iloc[top[np.argsort(-rank[top], kind="stable")[:40]]].round(1).to_string(index=False))

        print("\na step along a path: does it go to a page the score calls more general?\n")
        step = ok & (nxt >= 0)
        step[step] &= ok[nxt[step]]
        up = rank[nxt] < rank
        rows = {}
        for b, name in enumerate(names):
            mine = step & (band == b)
            if mine.any():
                rows[name] = {"steps": mine.sum(), "to a more general page %": 100 * up[mine].mean(),
                              "median ranks gained": np.median(rank[mine] - rank[nxt[mine]]),
                              "lands in the same or a more general band %": 100 * (band[nxt[mine]] <= b).mean()}
        rows["all"] = {"steps": step.sum(), "to a more general page %": 100 * up[step].mean(),
                       "median ranks gained": np.median(rank[step] - rank[nxt[step]]),
                       "lands in the same or a more general band %": 100 * (band[nxt[step]] <= band[step]).mean()}
        print(pd.DataFrame(rows).T.round(1).to_string())
        off = step & ~on_cycle
        print(f"\n  steps that are not on a cycle: {100 * up[off].mean():.1f}% go to a more general page ({int(off.sum()):,} steps)")
        kept["step_direction"] = [{"name": name, **row} for name, row in rows.items()]
        kept["off_cycle_up"] = 100 * up[off].mean()

        print("\npages by the number of steps to where their path ends (0 = on a cycle or at a dead end)\n")
        counts = np.bincount(steps)
        typical = pd.Series(rank[ok]).groupby(steps[ok]).quantile([0.25, 0.5, 0.75]).unstack()
        rows = [{"steps": d, "pages": int(c), "fan-out (pages / pages one step nearer)": c / counts[d - 1] if d else np.nan,
                 "score: quarter": typical[0.25].get(d, np.nan), "score: median": typical[0.5].get(d, np.nan),
                 "score: three quarters": typical[0.75].get(d, np.nan)} for d, c in enumerate(counts) if c]
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        print(f"\n  steps: mean {steps.mean():.2f}, median {np.median(steps):.0f}, commonest {counts.argmax()}, longest {steps.max()}")
        print(f"  rank correlation between steps and specificity by the score: {spearman(rank, steps.astype(float), ok):.3f}")
        kept["steps"] = [{"steps": r["steps"], "pages": r["pages"], "quarter": r["score: quarter"], "median": r["score: median"],
                          "three_quarters": r["score: three quarters"]} for r in rows]
        if kinds.exists():           # as on the map: a page with no family counts under the last one
            family = families.reindex(pages["page_id"]).fillna(len(kind_names) - 1).to_numpy().astype(np.int64)
            by_kind = pd.DataFrame({name: np.bincount(steps[family == k], minlength=len(counts)) for k, name in enumerate(kind_names)})
            print("\nthe same count for each kind of thing\n")
            print(by_kind.to_string())
            print("\n" + pd.DataFrame({"pages": by_kind.sum(), "mean steps": (by_kind.mul(by_kind.index, axis=0).sum() / by_kind.sum()).round(2),
                                       "commonest": by_kind.idxmax(), "at 0 %": (100 * by_kind.iloc[0] / by_kind.sum()).round(1)}).to_string())
            kept["kinds"] = [{"name": name, "pages": by_kind[name].tolist()} for name in kind_names]
        kept.update(steps_mean=steps.mean(), steps_median=np.median(steps), steps_longest=steps.max(),
                    steps_score=spearman(rank, steps.astype(float), ok))

    def plain(v):          # numpy numbers, and no NaN, which JSON has no word for
        v = v.item() if hasattr(v, "item") else v
        return None if isinstance(v, float) and np.isnan(v) else round(v, 3) if isinstance(v, float) else v

    def tidy(v):
        return {k: tidy(x) for k, x in v.items()} if isinstance(v, dict) else [tidy(x) for x in v] if isinstance(v, list) else plain(v)
    (data / f"{wiki}_flow_stats.json").write_text(json.dumps(tidy(stats), indent=1))


if __name__ == "__main__":
    main()

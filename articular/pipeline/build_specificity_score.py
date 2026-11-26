"""Give every article one specificity score, from the measures that belong to a single article.

    uv run python -m articular.pipeline.build_specificity_score --wiki enwiki

Reads what articular/experiments/score_hatnotes.py reads.

The combinations of Tests 7 to 13 judge a pair of articles. Some of their measures only exist for a
pair (does one title contain the other, which article links to the other, which carries a "Main
article" note naming the other, the pair of nouns). The rest belong to one article, and a weighted
sum of those can be worked out for every article on its own: the fit has no constant and compares
parent minus child, so the article with the larger sum is the one the fit calls more general.

Two fits, because Test 15 found the weights for places differ from the rest:
  things   subclass-of, instance-of and part-of pairs together, the same number from each
  places   located-in pairs
The score written for every article is the things sum, as a rank over all articles from 0 (most
general) to 100 (most specific). The places sum is kept beside it but the two are not on one
scale, so they are not mixed. A measure an article lacks counts as the middle value over all
articles.

Several versions are fitted and compared (see the comment above them in the code): they differ in
whether the two measures of the first 200 words' vocabulary are used and whether measures other
than counts are held within the 1st and 99th hundredth of their values. For each the script prints
the share of held-out pairs put the right way round, how many articles with hardly any in-links
reach the top of the ranking, and which measures put them there.

Writes data/<wiki>_specificity_score.parquet (a rank for each version; `specificity` is the one
named in CHOSEN) and data/<wiki>_specificity_weights.json.
With --weights-from, the weights of another run are applied and nothing is fitted (used to preview
the map on a small wiki with the English weights).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from articular.experiments.score_hatnotes import COUNTS as NOTE_COUNTS
from articular.experiments.score_hatnotes import note_tables
from articular.experiments.score_link_direction import resolve_tables
from articular.experiments.score_text_features import COUNTS, PROPS, TEXT, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles

OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
NAMES = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"] + NOTE_COUNTS
LOGGED = set(COUNTS) | set(NOTE_COUNTS)
SOURCE = {"in_deg": "o", "kinds_below": "s", **{n: "m" for n in NOTE_COUNTS}}   # the rest are in t
GROUPS = {"things": [279, 31, 361], "places": [131]}
CHOSEN = "no first-200-word vocabulary"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--statements", type=Path, default=None)
    parser.add_argument("--page-props", type=Path, default=None)
    parser.add_argument("--weights-from", type=Path, default=None)
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    statements = args.statements or data / "wd" / "statements.tsv.gz"
    page_props = args.page_props or data / "wd" / f"{wiki}-page_props.sql.gz"

    con = open_tables(data, wiki, statements, page_props)
    add_titles(con, data, wiki)
    resolve_tables(con, data, wiki)
    note_tables(con, data, wiki)

    columns = ", ".join(f"{SOURCE.get(n, 't')}.{n}" for n in NAMES)
    articles = con.sql(f"""
        select s.page_id, s.title, {columns}
        from s join t using (page_id) join o using (page_id) join m using (page_id) order by s.page_id
    """).df()
    values = articles[NAMES].to_numpy(dtype=float)
    logged = [i for i, n in enumerate(NAMES) if n in LOGGED]
    values[:, logged] = np.log1p(values[:, logged])
    row_of = pd.Series(np.arange(len(articles)), index=articles["page_id"].to_numpy())
    pairs = con.sql("select prop, parent, child from pair").df()
    pairs["p"], pairs["c"] = pairs["parent"].map(row_of), pairs["child"].map(row_of)
    pairs = pairs.dropna(subset=["p", "c"]).astype({"p": int, "c": int})
    held = (pairs["parent"].to_numpy() * 2654435761 % 5 == 0)
    turned = (pairs["child"].to_numpy() % 2 == 1)
    prop, p, c = pairs["prop"].to_numpy(), pairs["p"].to_numpy(), pairs["c"].to_numpy()
    in_links = values[:, NAMES.index("in_deg")]
    stored = json.loads(args.weights_from.read_text()) if args.weights_from else None

    def scores(label, values, used):
        """The things and places sums of every article (bigger = more general) from these measures."""
        cols = [NAMES.index(n) for n in used]
        values = values[:, cols]
        # as in the pair fits: a measure missing for either article says nothing about the pair
        diff = np.nan_to_num(values[p] - values[c])

        def fitted(props):
            """Spread and weights from the fitting pairs of these relations, the same number from each."""
            chosen = [np.flatnonzero((prop == number) & ~held) for number in props]
            chosen = [rows for rows in chosen if len(rows) >= 1000]
            least = min(len(rows) for rows in chosen)
            chosen = [rows[np.argsort(pairs["child"].to_numpy()[rows] * 2654435761 % 2**32, kind="stable")[:least]] for rows in chosen]
            spread = np.sqrt(np.mean([diff[rows].var(axis=0) for rows in chosen], axis=0)) + 1e-9
            rows = np.concatenate(chosen)
            x = diff[rows] / spread
            x[turned[rows]] *= -1
            return spread, fit(x, (~turned[rows]).astype(float))

        if stored:
            fits = {group: (np.array(f["spread"]), np.array(f["weights"])) for group, f in stored[label]["fits"].items()}
        else:
            fits = {group: fitted(props) for group, props in GROUPS.items()}
        middle = np.nanmedian(values, axis=0)
        centred = np.where(np.isnan(values), middle, values) - middle
        parts = {group: centred / spread * w for group, (spread, w) in fits.items()}   # what each measure adds
        return used, fits, parts, {group: part.sum(axis=1) for group, part in parts.items()}

    # Versions. The first run put articles with no in-links at the top through kl_200, whose weight is
    # large and nearly cancelled by mean_idf_200's. Holding every measure within its 1st and 99th
    # hundredth made that worse, because it also cut off the largest in-link counts. So the counts are
    # left alone here; only the other measures are clipped, or the two first-200-word measures dropped.
    low, high = np.nanpercentile(values, [1, 99], axis=0)
    clipped = values.copy()
    plain = [i for i, n in enumerate(NAMES) if n not in LOGGED]
    clipped[:, plain] = np.clip(values[:, plain], low[plain], high[plain])
    no_kl = [n for n in NAMES if n != "kl_200"]
    fewer = [n for n in no_kl if n != "mean_idf_200"]
    made = {"as first run": (values, NAMES), "measures other than counts clipped": (clipped, NAMES),
            "no kl_200": (values, no_kl), "no first-200-word vocabulary": (values, fewer),
            "no first-200-word vocabulary, others clipped": (clipped, fewer)}
    versions = {label: scores(label, v, used) for label, (v, used) in made.items()}
    if stored:
        print(f"weights taken from {args.weights_from}; nothing fitted here")
    else:
        (data / f"{wiki}_specificity_weights.json").write_text(json.dumps({
            label: {"names": used, "fits": {g: {"spread": s.tolist(), "weights": w.tolist()} for g, (s, w) in fits.items()}}
            for label, (used, fits, _, _) in versions.items()}, indent=1))

    print("\n== % of held-out pairs put the right way round (ties count half) ==\n")
    rows = {}
    for number, relation in PROPS.items():
        mine = held & (prop == number)
        if mine.sum() == 0:
            continue
        def right(v):
            return 100 * ((v[p[mine]] > v[c[mine]]).mean() + (v[p[mine]] == v[c[mine]]).mean() / 2)
        rows[relation] = {"lead in-links alone": right(in_links)}
        for label, (_, _, _, general) in versions.items():
            rows[relation][f"{label}: things score"] = right(general["things"])
            rows[relation][f"{label}: places score"] = right(general["places"])
        rows[relation]["held-out pairs"] = mine.sum()
    print(pd.DataFrame(rows).round(1).to_string())

    out = pd.DataFrame({"page_id": articles["page_id"], "title": articles["title"], "in_deg": articles["in_deg"]})
    few = in_links <= np.log1p(2)
    print("\n== articles with two lead in-links or fewer among the most general, by the things score ==\n")
    for label, (used, fits, parts, general) in versions.items():
        order = np.argsort(-general["things"], kind="stable")
        rank = np.empty(len(order))
        rank[order] = 100 * np.arange(len(order)) / max(len(order) - 1, 1)
        out[label] = rank          # 0 = most general, 100 = most specific
        out[label + ", places"] = general["places"]
        print(f"{label}: " + ", ".join(f"{int(few[order[:k]].sum()):,} of the top {k:,}" for k in (100, 1000, 10_000, 100_000)))
    out["specificity"] = out[CHOSEN]
    out.to_parquet(data / f"{wiki}_specificity_score.parquet", index=False)
    print(f"\n{len(out):,} articles scored; `specificity` is the rank of the version \"{CHOSEN}\"")

    for label, (used, fits, parts, general) in versions.items():
        print(f"\n==== {label} ====\n\nweights (in standard deviations of parent minus child; bigger = more general)\n")
        table = pd.DataFrame({group: w for group, (_, w) in fits.items()}, index=used)
        print(table.reindex(table.abs().max(axis=1).sort_values(ascending=False).index).round(2).to_string())
        order = np.argsort(-general["things"], kind="stable")
        print("\nthe most general\n")
        print(out.iloc[order[:40]][["title", "in_deg"]].to_string(index=False))
        print("\nthe most general with two lead in-links or fewer, and the measures that add most to their sum\n")
        for i in order[few[order]][:12]:
            top = np.argsort(-np.abs(parts["things"][i]))[:4]
            print(f"  {articles['title'][i][:60]:<60} rank {out[label][i]:.4f}; sum {general['things'][i]:.1f}: "
                  + ", ".join(f"{used[j]} {parts['things'][i, j]:+.1f} (value {values[i, NAMES.index(used[j])]:.2f})" for j in top))
        print("\nat every tenth of the ranking\n")
        print(out.iloc[order[np.linspace(0, len(order) - 1, 11).astype(int)]][["title", "in_deg", label]].round(1).to_string(index=False))
        print("\nthe most general by the places score\n")
        print(out.iloc[np.argsort(-general["places"], kind="stable")[:25]][["title", "in_deg"]].to_string(index=False))


if __name__ == "__main__":
    main()

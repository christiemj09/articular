"""Check the vocabulary measures against Wikidata: is the parent ranked as more general than the child?

    uv run python -m articular.experiments.score_text_features --wiki enwiki

Reads data/<wiki>_text_features.parquet (articular/pipeline/text_features.py), data/<wiki>_specificity.parquet,
data/<wiki>_scores.parquet, data/wd/statements.tsv.gz and data/wd/<wiki>-page_props.sql.gz.

Pairs and scoring are those of articular/experiments/score_specificity.py. Each measure is written with the
direction we expect, so that bigger = more general; a score well under 50 means the measure works
the other way round. A pair where either article lacks the measure is left out for that measure.

Then, for each relation, a weighted sum of the measures is fitted on four fifths of the parents and
scored on the pairs of the remaining fifth: with lead in-links alone, with the vocabulary measures
alone, and with both. The weights of the last fit show which measures carry it.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from articular.experiments.score_rules import read_items

PROPS = {279: "subclass of", 31: "instance of", 361: "part of", 131: "located in"}
# every measure is turned so that bigger = more general, as we expect it to run
MEASURES = {
    "name used in more articles": "t.name_df",
    "longer article": "t.n_tokens",
    "more distinct words in first 200": "t.types_200",
    "commoner words, first 200": "-t.mean_idf_200",
    "commoner words, whole article": "-t.mean_idf",
    "closer to average English, first 200": "-t.kl_200",
    "fewer rare words": "-t.share_rare_types",
    "weight spread over many words": "-t.top10_share",
    "commoner title words": "-t.title_idf_sum",
    "title scores lower on own text (BM25)": "-t.title_bm25",
    "lead in-links": "o.in_deg",
    "kinds_below": "s.kinds_below",
}
COUNTS = ("name_df", "n_tokens", "in_deg", "kinds_below", "titles_containing")   # compared on a log scale in the fits
TEXT = ["name_df", "n_tokens", "types_200", "mean_idf_200", "mean_idf", "kl_200", "share_rare_types", "top10_share",
        "title_idf_sum", "title_bm25"]
FITS = {"lead in-links": ["in_deg"], "vocabulary": TEXT, "both": ["in_deg"] + TEXT, "both and kinds_below": ["in_deg", "kinds_below"] + TEXT}


def fit(x, y):
    """Weights of a logistic regression without a constant, lightly shrunk."""
    def loss(w):
        z = x @ w
        return np.logaddexp(0, z).sum() - y @ z + 0.5 * w @ w, x.T @ (1 / (1 + np.exp(-z)) - y) + w
    return minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B").x


JOINS = """pair x join s sc on sc.page_id = x.child join s sp on sp.page_id = x.parent
           join t tc on tc.page_id = x.child join t tp on tp.page_id = x.parent
           join o oc on oc.page_id = x.child join o op on op.page_id = x.parent"""
SOURCE = {"in_deg": "o", "kinds_below": "s"}   # the table each fitted measure comes from; the rest are in t


def open_tables(data, wiki, statements, page_props):
    """A connection holding s, t and o (one row per article) and pair (child, parent, prop)."""
    con = duckdb.connect()
    con.sql(f"set temp_directory = '{data}/duckdb_tmp'; set preserve_insertion_order = false;")
    con.register("items_raw", read_items(page_props))
    con.sql(f"""
    create table s as select * from '{data}/{wiki}_specificity.parquet';
    create table t as select * from '{data}/{wiki}_text_features.parquet';
    create table o as select page_id, in_deg from '{data}/{wiki}_scores.parquet';
    create table item as select i.page_id, i.q from items_raw i join s using (page_id);
    create table pair as
    select distinct c.page_id as child, p.page_id as parent, t.prop
    from (
        select cast(substr(column0, 2) as bigint) as item, cast(substr(column1, 2) as int) as prop,
               cast(substr(column2, 2) as bigint) as value
        from read_csv('{statements}', delim = '\t', header = false, quote = '', escape = '', auto_detect = false,
                      columns = {{'column0': 'varchar', 'column1': 'varchar', 'column2': 'varchar'}})
    ) t join item c on c.q = t.item join item p on p.q = t.value
    where t.prop in {tuple(PROPS)} and c.page_id <> p.page_id;
    """)
    return con


def differences(con, names):
    """One row per pair: parent minus child for each measure (counts on a log scale); 0 where either is missing."""
    diffs = ", ".join(
        (f"ln(1 + {SOURCE.get(n, 't')}p.{n}) - ln(1 + {SOURCE.get(n, 't')}c.{n})" if n in COUNTS
         else f"{SOURCE.get(n, 't')}p.{n} - {SOURCE.get(n, 't')}c.{n}") + f" as {n}" for n in names)
    pairs = con.sql(f"select x.prop, x.parent, x.child, {diffs} from {JOINS}").df()
    pairs[names] = pairs[names].fillna(0)   # a measure that is missing for either article says nothing about the pair
    return pairs


def design(part, names):
    """x, y and the held-out mask for the pairs of one relation.

    Each difference is scaled to unit spread; half the pairs are turned round so both answers occur;
    a fifth of the parents are held out.
    """
    x = part[names].to_numpy(dtype=float)
    x = x / (x.std(axis=0) + 1e-9)
    turned = (part["child"].to_numpy() % 2 == 1)
    x[turned] *= -1
    y = (~turned).astype(float)
    held = (part["parent"].to_numpy() * 2654435761 % 5 == 0)
    return x, y, held


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--statements", type=Path, default=None)
    parser.add_argument("--page-props", type=Path, default=None)
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    statements = args.statements or data / "wd" / "statements.tsv.gz"
    page_props = args.page_props or data / "wd" / f"{wiki}-page_props.sql.gz"

    con = open_tables(data, wiki, statements, page_props)
    rows = []
    for name, expr in MEASURES.items():
        child, parent = [expr.replace("s.", f"s{side}.").replace("o.", f"o{side}.").replace("t.", f"t{side}.") for side in "cp"]
        rows.append(con.sql(f"""
            select '{name}' as measure, x.prop, (op.in_deg < oc.in_deg) as inlinks_wrong, count(*) as pairs,
                   count(*) filter ({parent} > {child}) as correct, count(*) filter ({parent} = {child}) as tied,
                   count(*) filter ({parent} < {child}) as wrong
            from {JOINS} where {parent} is not null and {child} is not null group by all
        """).df())
    table = pd.concat(rows)
    table["relation"] = table["prop"].map(PROPS)
    columns = [c for c in PROPS.values() if c in set(table["relation"])]
    for label, part in [("all pairs", table), ("pairs that lead in-links gets wrong", table[table["inlinks_wrong"]])]:
        if part.empty:
            continue
        total = part.groupby(["relation", "measure"], sort=False)[["pairs", "correct", "tied", "wrong"]].sum()
        # with ties split evenly, 50 is what a coin would score
        total["score"] = (100 * (total["correct"] + total["tied"] / 2) / total["pairs"]).round(1)
        wide = total.reset_index().pivot(index="measure", columns="relation")
        print(f"\n== {label}: score (correct % + half of tied %; a coin scores 50) ==\n")
        print(wide["score"].reindex(list(MEASURES))[columns].to_string())
        print(f"\n{label}: number of pairs the measure could judge\n")
        print(wide["pairs"].reindex(list(MEASURES))[columns].to_string())
    table.to_csv(data / f"{wiki}_text_features_check.csv", index=False)

    # the fits: parent minus child for each measure, scaled; half the pairs are turned round so both answers occur
    names = FITS["both and kinds_below"]
    pairs = differences(con, names)
    print("\n== fitted combinations: % of held-out pairs put the right way round ==")
    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop]
        if len(part) < 1000:
            continue
        x, y, held = design(part, names)
        print(f"\n{relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out")
        for label, used in FITS.items():
            cols = [names.index(n) for n in used]
            w = fit(x[~held][:, cols], y[~held])
            right = ((x[held][:, cols] @ w > 0) == (y[held] > 0)).mean()
            print(f"  {label:<24} {100 * right:5.1f}")
            if label == "both":
                order = np.argsort(-np.abs(w))
                print("    weights (parent minus child, in standard deviations): " + ", ".join(f"{used[i]} {w[i]:+.2f}" for i in order))


if __name__ == "__main__":
    main()

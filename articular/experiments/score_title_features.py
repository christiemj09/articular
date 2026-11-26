"""Check the title measures against Wikidata: is the parent ranked as more general than the child?

    uv run python -m articular.experiments.score_title_features --wiki enwiki

Reads data/<wiki>_title_features.parquet (articular/pipeline/title_features.py) and what
articular/experiments/score_text_features.py reads. Pairs, scoring and fitting are the same as there.

Three parts:
  - each title measure on its own, over all pairs and over the pairs lead in-links gets wrong;
  - the rule "the child's title contains the parent's" (Prime number, Number): how often it applies
    and how often it points the right way;
  - fitted combinations on held-out pairs: lead in-links with each title measure, with all of them,
    and with the vocabulary measures of articular/pipeline/text_features.py as well.
"""

import argparse
from pathlib import Path

import pandas as pd

from articular.experiments.score_text_features import JOINS, PROPS, TEXT, design, differences, fit, open_tables

# every measure is turned so that bigger = more general, as we expect it to run
MEASURES = {
    "fewer words in title": "-t.title_words",
    "shorter title": "-t.title_chars",
    "fewer capitalised words after the first": "-t.title_caps",
    "no digit in title": "-t.title_digit",
    "no (...) in title": "-t.title_paren",
    "no comma in title": "-t.title_comma",
    "title found inside more titles": "t.titles_containing",
    "commoner title words": "-t.title_idf_sum",
    "lead in-links": "o.in_deg",
}
TITLE = ["title_words", "title_chars", "title_caps", "title_digit", "title_paren", "title_comma", "titles_containing",
         "title_idf_sum", "contains"]
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
FITS = {
    "lead in-links": ["in_deg"],
    "title measures": TITLE,
    "lead in-links + title measures": ["in_deg"] + TITLE,
    "lead in-links + vocabulary": ["in_deg"] + TEXT,
    "lead in-links + vocabulary + title": ["in_deg"] + OTHER_TEXT + TITLE,
    "all of those + kinds_below": ["in_deg", "kinds_below"] + OTHER_TEXT + TITLE,
}


def add_titles(con, data, wiki):
    """Put the title measures into t, so the shared joins and differences apply to them unchanged."""
    con.sql(f"create or replace table t as select t.*, h.* exclude (page_id) from t join '{data}/{wiki}_title_features.parquet' h using (page_id)")


def containment(con):
    """One row per pair. `contains`: +1 when the child's title contains the parent's, -1 the other way round."""
    return con.sql(f"""
        select x.prop, x.parent, x.child, (op.in_deg < oc.in_deg) as inlinks_wrong,
               cast(length(tp.norm) > 2 and tc.norm <> tp.norm and contains(tc.norm, tp.norm) as int)
             - cast(length(tc.norm) > 2 and tc.norm <> tp.norm and contains(tp.norm, tc.norm) as int) as contains
        from {JOINS}
    """).df()


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
    add_titles(con, data, wiki)

    rows = []
    for name, expr in MEASURES.items():
        child, parent = [expr.replace("o.", f"o{side}.").replace("t.", f"t{side}.") for side in "cp"]
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
        total["tied %"] = (100 * total["tied"] / total["pairs"]).round(1)
        wide = total.reset_index().pivot(index="measure", columns="relation")
        print(f"\n== {label}: score (correct % + half of tied %; a coin scores 50) ==\n")
        print(wide["score"].reindex(list(MEASURES))[columns].to_string())
        print(f"\n{label}: % of pairs the measure leaves tied\n")
        print(wide["tied %"].reindex(list(MEASURES))[columns].to_string())
    table.to_csv(data / f"{wiki}_title_features_check.csv", index=False)

    inside = containment(con)
    print("\n== one title inside the other ==\n")
    for label, part in [("all pairs", inside), ("pairs that lead in-links gets wrong", inside[inside["inlinks_wrong"]])]:
        print(f"{label}: % of pairs where the child's title contains the parent's, % the other way round")
        for prop, relation in PROPS.items():
            got = part[part["prop"] == prop]["contains"]
            if len(got):
                print(f"  {relation:<12} {100 * (got == 1).mean():5.1f}  {100 * (got == -1).mean():5.1f}   of {len(got):,} pairs")

    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    pairs = differences(con, names).merge(inside.drop(columns="inlinks_wrong"), on=["prop", "parent", "child"])
    names = names + ["contains"]
    print("\n== fitted combinations: % of held-out pairs put the right way round ==")
    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop]
        if len(part) < 1000:
            continue
        x, y, held = design(part, names)

        def score(used):
            cols = [names.index(n) for n in used]
            w = fit(x[~held][:, cols], y[~held])
            return 100 * ((x[held][:, cols] @ w > 0) == (y[held] > 0)).mean()

        print(f"\n{relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out")
        alone = None
        for label, used in FITS.items():
            got = score(used)
            alone = got if alone is None else alone
            print(f"  {label:<38} {got:5.1f}   {got - alone:+5.1f}")
        print("  lead in-links plus one title measure")
        for n in TITLE:
            got = score(["in_deg", n])
            print(f"    + {n:<34} {got:5.1f}   {got - alone:+5.1f}")


if __name__ == "__main__":
    main()

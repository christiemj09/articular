"""Check the specificity measures against Wikidata: is the parent ranked as more general than the child?

    uv run python -m articular.experiments.score_specificity --wiki enwiki

Reads data/<wiki>_specificity.parquet (articular/pipeline/specificity.py), data/<wiki>_kind_graph.parquet,
data/<wiki>_scores.parquet (notebook 02, for the older measures), data/wd/statements.tsv.gz and
data/wd/<wiki>-page_props.sql.gz (scripts/fetch_wikidata.sh).

A pair is a Wikidata statement between two articles: child -> parent by subclass of, instance of,
part of or located in. For each measure the pair is correct if the parent comes out more general,
tied if the two are equal, wrong otherwise. Pairs where our own kind link is exactly that statement
are also counted apart, because kinds_below gets those right by construction. So are the pairs
that the plain count of lead in-links gets wrong (the child is linked to more than the parent):
a measure only adds something if it does better than a coin there.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from articular.experiments.score_rules import read_items

PROPS = {279: "subclass of", 31: "instance of", 361: "part of", 131: "located in"}
# every measure is turned so that bigger = more general
MEASURES = {
    "kinds_below": "s.kinds_below",
    "named_in": "s.named_in",
    "kinds_below, ties by named_in": "s.kinds_below * 1e9 + s.named_in",
    "lead in-links": "o.in_deg",
    "first-link traffic": "o.pages_through",
    "earlier level (lower)": "-o.level_all",
}


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

    con = duckdb.connect()
    con.sql(f"set temp_directory = '{data}/duckdb_tmp'; set preserve_insertion_order = false;")
    con.register("items_raw", read_items(page_props))
    con.sql(f"""
    create table s as select * from '{data}/{wiki}_specificity.parquet';
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
    create table ours as select page_id as child, kind_page_id as parent from '{data}/{wiki}_kind_graph.parquet' where kind_page_id >= 0;
    """)
    old = f"'{data}/{wiki}_scores.parquet'"
    rows = []
    for name, expr in MEASURES.items():
        child, parent = expr.replace("s.", "sc.").replace("o.", "oc."), expr.replace("s.", "sp.").replace("o.", "op.")
        rows.append(con.sql(f"""
            select '{name}' as measure, x.prop, (u.child is not null) as our_kind_link,
                   (op.in_deg < oc.in_deg) as inlinks_wrong, count(*) as pairs,
                   count(*) filter ({parent} > {child}) as correct, count(*) filter ({parent} = {child}) as tied,
                   count(*) filter ({parent} < {child}) as wrong
            from pair x join s sc on sc.page_id = x.child join s sp on sp.page_id = x.parent
            join {old} oc on oc.page_id = x.child join {old} op on op.page_id = x.parent
            left join ours u on u.child = x.child and u.parent = x.parent
            group by all
        """).df())
    table = pd.concat(rows)
    table["relation"] = table["prop"].map(PROPS)
    for label, part in [("all pairs", table), ("pairs that are not our own kind link", table[~table["our_kind_link"]]),
                        ("not our own kind link, and lead in-links gets it wrong", table[~table["our_kind_link"] & table["inlinks_wrong"]])]:
        total = part.groupby(["relation", "measure"], sort=False)[["pairs", "correct", "tied", "wrong"]].sum()
        for column in ("correct", "tied", "wrong"):
            total[column + " %"] = (100 * total[column] / total["pairs"]).round(1)
        # with ties split evenly, 50 is what a coin would score
        total["score"] = (total["correct %"] + total["tied %"] / 2).round(1)
        wide = total.reset_index().pivot(index="measure", columns="relation")
        order, columns = list(MEASURES), list(PROPS.values())
        print(f"\n== {label}: score (correct % + half of tied %; a coin scores 50) ==\n")
        columns = [c for c in columns if c in wide["score"].columns]
        print(wide["score"].reindex(order)[columns].to_string())
        print(f"\n{label}: tied %\n")
        print(wide["tied %"].reindex(order)[columns].to_string())
        print(f"\n{label}: number of pairs: " + ", ".join(f"{c} {int(wide['pairs'][c].max()):,}" for c in columns))
    table.to_csv(data / f"{wiki}_specificity_check.csv", index=False)


if __name__ == "__main__":
    main()

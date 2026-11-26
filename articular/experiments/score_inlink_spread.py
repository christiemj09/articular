"""Check how widely an article's in-links are spread against Wikidata: is a general article linked from many areas?

    uv run python -m articular.experiments.score_inlink_spread --wiki enwiki

Reads data/<wiki>_kind_families.parquet and what articular/experiments/score_inlink_position.py reads. Pairs,
held-out fifth and fitting are the same as in articular/experiments/score_text_features.py.

The articles that link to an article are sorted three ways, and the spread of each sorting is its
entropy (0 when all in-linking articles fall in one group):

  family   the fifteen named families of the map; articles in "Other kinds" or with no kind are left out
  basin    where the article's chain of first links ends; articles whose chain ends nowhere are left out
  kind     the article's own kind (the moths of one family all have the kind Moth)

  fam_h, fam_h_lead   spread over families, counting all in-links and only those from leads
  fam_top             share of the in-linking articles in the commonest family
  fam_other           share in a family other than the article's own
  basin_h, kind_h     spread over basins and over kinds
  kind_n              number of different kinds among the in-linking articles

A measure is missing for an article with no in-link that can be sorted that way.

Prints each measure on its own, fitted combinations on held-out pairs with and without the measures
of the earlier tests, and examples. Writes data/<wiki>_inlink_spread.parquet (one row per article)
and data/<wiki>_inlink_spread_check.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from articular.experiments.score_inlink_position import EARLIER, EXAMPLES, link_edges
from articular.experiments.score_kind_pairs import KINDS, LEARNED, kind_table, learn
from articular.experiments.score_link_direction import DIRECTION, direction_table, link_positions
from articular.experiments.score_text_features import JOINS, PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

NAMED = 15   # families 0 to 14 are named; 15 is "Other kinds" and 16 "No kind found"
# every measure is turned so that bigger = more general, as we expect it to run
MEASURES = {
    "in-links spread over more families": "h.fam_h",
    "the same, in-links from leads only": "h.fam_h_lead",
    "smaller share from the commonest family": "-h.fam_top",
    "larger share from outside its own family": "h.fam_other",
    "in-links spread over more basins": "h.basin_h",
    "in-links spread over more kinds": "h.kind_h",
    "in-linking articles of more kinds": "h.kind_n",
    "lead in-links": "o.in_deg",
}
SPREAD = ["fam_h", "fam_h_lead", "fam_top", "fam_other", "basin_h", "kind_h", "kind_n"]
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
FITS = {
    "lead in-links": ["in_deg"],
    "spread measures": SPREAD,
    "lead in-links + spread measures": ["in_deg"] + SPREAD,
    "every measure of Tests 7 to 10": EARLIER,
    "every measure of Tests 7 to 10 + spread": EARLIER + SPREAD,
}
WITH_H = f"{JOINS} join h hp on hp.page_id = x.parent join h hc on hc.page_id = x.child"
ENTROPY = "ln(sum(c)) - sum(c * ln(c)) / sum(c)"


def spread_table(con, data, wiki):
    """Table h: for each article, how its in-linking articles are spread. Needs edge and kind."""
    con.sql(f"""
    create table family as select page_id, family from '{data}/{wiki}_kind_families.parquet' where family < {NAMED};
    create table basin as select page_id, ending from '{data}/{wiki}_scores.parquet' where ending not like '(%';

    create table by_family as
    select e.dst, f.family, count(*) as c, count(*) filter (e.lead) as c_lead
    from edge e join family f on f.page_id = e.src group by 1, 2;
    create table fam as
    select b.dst, sum(c) as fam_n, {ENTROPY} as fam_h, max(c) / sum(c) as fam_top,
           ln(nullif(sum(c_lead), 0)) - sum(c_lead * ln(nullif(c_lead, 0))) / nullif(sum(c_lead), 0) as fam_h_lead,
           (sum(c) filter (b.family <> own.family)) / sum(c) as fam_other
    from by_family b left join family own on own.page_id = b.dst group by b.dst, own.family;
    update fam set fam_other = coalesce(fam_other, 0) where dst in (select page_id from family);

    create table bas as
    select dst, {ENTROPY} as basin_h
    from (select e.dst, b.ending, count(*) as c from edge e join basin b on b.page_id = e.src group by 1, 2) group by 1;
    create table kin as
    select dst, {ENTROPY} as kind_h, count(*) as kind_n
    from (select e.dst, k.kind_page_id, count(*) as c from edge e join kind k on k.page_id = e.src
          where k.kind_page_id is not null group by 1, 2) group by 1;

    create table h as
    select s.page_id, s.title, fam.fam_n, fam.fam_h, fam.fam_top, fam.fam_h_lead, fam.fam_other, bas.basin_h, kin.kind_h, kin.kind_n
    from s left join fam on fam.dst = s.page_id left join bas on bas.dst = s.page_id left join kin on kin.dst = s.page_id;
    """)


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
    link_positions(con, data, wiki)
    direction = direction_table(con)
    kinds = kind_table(con, data, wiki)
    link_edges(con, data, wiki)
    spread_table(con, data, wiki)
    con.sql(f"copy h to '{data}/{wiki}_inlink_spread.parquet' (format parquet)")
    print(con.sql("select count(*) as articles, count(fam_h) as with_family_spread, count(basin_h) as with_basin_spread, count(kind_h) as with_kind_spread from h").df().to_string(index=False))

    # each measure on its own
    rows = []
    for name, expr in MEASURES.items():
        child, parent = [expr.replace("h.", f"h{side}.").replace("o.", f"o{side}.") for side in "cp"]
        rows.append(con.sql(f"""
            select '{name}' as measure, x.prop, (op.in_deg < oc.in_deg) as inlinks_wrong, count(*) as pairs,
                   count(*) filter ({parent} > {child}) as correct, count(*) filter ({parent} = {child}) as tied,
                   count(*) filter ({parent} < {child}) as wrong
            from {WITH_H} where {parent} is not null and {child} is not null group by all
        """).df())
    table = pd.concat(rows)
    table["relation"] = table["prop"].map(PROPS)
    columns = [c for c in PROPS.values() if c in set(table["relation"])]
    for label, part in [("all pairs", table), ("pairs that lead in-links gets wrong", table[table["inlinks_wrong"]])]:
        if part.empty:
            continue
        total = part.groupby(["relation", "measure"], sort=False)[["pairs", "correct", "tied", "wrong"]].sum()
        total["score"] = (100 * (total["correct"] + total["tied"] / 2) / total["pairs"]).round(1)
        total["tied %"] = (100 * total["tied"] / total["pairs"]).round(1)
        wide = total.reset_index().pivot(index="measure", columns="relation")
        print(f"\n== {label}: score (correct % + half of tied %; a coin scores 50) ==\n")
        print(wide["score"].reindex(list(MEASURES))[columns].to_string())
        print(f"\n{label}: % of pairs the measure leaves tied\n")
        print(wide["tied %"].reindex(list(MEASURES))[columns].to_string())
        print(f"\n{label}: number of pairs the measure could judge\n")
        print(wide["pairs"].reindex(list(MEASURES))[columns].to_string())
    table.to_csv(data / f"{wiki}_inlink_spread_check.csv", index=False)

    # the fits
    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    spread = con.sql(f"""
        select x.prop, x.parent, x.child, ln(1 + hp.kind_n) - ln(1 + hc.kind_n) as kind_n,
               {", ".join(f"hp.{n} - hc.{n} as {n}" for n in SPREAD if n != "kind_n")}
        from {WITH_H}
    """).df()
    spread[SPREAD] = spread[SPREAD].fillna(0)   # a measure that is missing for either article says nothing about the pair
    pairs = (differences(con, names).merge(containment(con), on=keys)
             .merge(direction[keys + DIRECTION], on=keys).merge(kinds, on=keys).merge(spread, on=keys))
    names = names + ["contains"] + DIRECTION + LEARNED + SPREAD
    print("\n== fitted combinations: % of held-out pairs put the right way round ==")
    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop].reset_index(drop=True)
        if len(part) < 1000:
            continue
        held = (part["parent"].to_numpy() * 2654435761 % 5 == 0)   # the same fifth as design() holds out
        for form in KINDS:
            part[f"pair_{form}"], part[f"rank_{form}"] = learn(part, held, form)
        x, y, held = design(part, names)

        def score(used, only=None):
            cols = [names.index(n) for n in used]
            w = fit(x[~held][:, cols], y[~held])
            right = (x[held][:, cols] @ w > 0) == (y[held] > 0)
            return 100 * (right if only is None else right[only]).mean()

        wrong = part["inlinks_wrong"].to_numpy()[held]
        print(f"\n{relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out, {int(wrong.sum()):,} of them wrong by lead in-links")
        print(f"  {'':<44} {'all':>5}   {'change':>6}   on the pairs lead in-links gets wrong")
        alone = earlier = None
        for label, used in FITS.items():
            got = score(used)
            alone = got if alone is None else alone
            step = f"   {got - earlier:+5.1f} from the line above" if label.endswith("+ spread") else ""
            print(f"  {label:<44} {got:5.1f}   {got - alone:+6.1f}   {score(used, wrong):5.1f}{step}")
            earlier = got
        before = score(EARLIER)
        print("  one spread measure added: to lead in-links, and to every measure of Tests 7 to 10")
        for n in SPREAD:
            got, full = score(["in_deg", n]), score(EARLIER + [n])
            print(f"    + {n:<40} {got:5.1f}   {got - alone:+6.1f}   {full:5.1f}   {full - before:+5.1f}")

    # what the spread looks like: exp(entropy) is the number of equal groups that would give the same spread
    families = json.loads((data / f"{wiki}_kind_families.json").read_text())
    show = """h.title, cast(fam_n as bigint) as sorted_in_links, round(exp(fam_h), 2) as families, round(100 * fam_top, 1) as top_family_pct,
              round(100 * fam_other, 1) as other_family_pct, round(exp(basin_h), 2) as basins, round(exp(kind_h), 1) as kinds, kind_n"""
    print("\n== examples (families, basins, kinds: the number of equal groups that would give the same spread) ==\n")
    print(con.sql(f"select {show} from h where title in {EXAMPLES} order by fam_h desc").df().to_string(index=False))
    for floor in (1000, 20000):
        for column, name in [("fam_h", "families"), ("kind_h", "kinds")]:
            for way, word in [("desc", "widest"), ("asc", "narrowest")]:
                print(f"\n{word} spread over {name}, among articles with {floor:,} or more sorted in-linking articles\n")
                print(con.sql(f"select {show} from h where fam_n >= {floor} order by {column} {way}, fam_n desc limit 20").df().to_string(index=False))
    print("\nspread over families by the article's own family: articles with 100 or more sorted in-links, median number of equal families, median share from other families\n")
    by = con.sql("""
        select f.family, count(*) as articles, round(median(exp(fam_h)), 2) as families, round(100 * median(fam_other), 1) as other_family_pct
        from h join family f using (page_id) where fam_n >= 100 group by 1 order by 1
    """).df()
    by["family"] = by["family"].map(lambda i: families[i])
    print(by.to_string(index=False))


if __name__ == "__main__":
    main()

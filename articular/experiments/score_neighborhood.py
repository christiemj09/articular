"""Check how tightly an article's neighbours are linked against Wikidata: do general articles connect groups?

    uv run python -m articular.experiments.score_neighborhood --wiki enwiki

Reads what articular/experiments/score_hatnotes.py reads. Pairs, held-out fifth and fitting are the same as in
articular/experiments/score_text_features.py.

The graph is that of lead links: one edge for each article whose lead links to another, through
redirects. An article's neighbours are the articles it links to and the articles that link to it.

  cc        clustering: the share of pairs of its neighbours that are themselves linked, in either
            direction. Counted over all pairs among up to SAMPLE of its neighbours, so it is exact
            for an article with no more than SAMPLE neighbours and an estimate for the rest.
            Missing for an article with fewer than two neighbours.
  cc_adj    cc minus the average cc of articles with about as many neighbours (the same power of two)
  nb_in     average over the articles that link to it of the log of their own in-links
  nb_out    the same average over the articles it links to
  reach     log of the number of two-step paths that end at it, minus the log of its in-links: how
            much the second step widens the first. Paths are counted, not distinct articles.

Prints each measure on its own, fitted combinations on held-out pairs with and without the
measures of the earlier tests, and examples. Writes data/<wiki>_neighborhood.parquet (one row per
article) and data/<wiki>_neighborhood_check.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from articular.experiments.score_hatnotes import HATNOTE, note_rules, note_tables
from articular.experiments.score_inlink_position import EARLIER, EXAMPLES
from articular.experiments.score_kind_pairs import KINDS, LEARNED, kind_table, learn
from articular.experiments.score_link_direction import DIRECTION, direction_table, link_positions
from articular.experiments.score_text_features import JOINS, PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

SAMPLE = 40
# every measure is turned so that bigger = more general, as we expect it to run
MEASURES = {
    "neighbours less linked to each other": "-g.cc",
    "the same, against articles with as many neighbours": "-g.cc_adj",
    "linked from articles with more in-links": "g.nb_in",
    "links to articles with more in-links": "g.nb_out",
    "second step widens the first more": "g.reach",
    "lead in-links": "o.in_deg",
}
NEIGHBOURS = ["cc", "cc_adj", "nb_in", "nb_out", "reach"]
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
SO_FAR = EARLIER + HATNOTE
FITS = {
    "lead in-links": ["in_deg"],
    "neighbourhood measures": NEIGHBOURS,
    "lead in-links + neighbourhood measures": ["in_deg"] + NEIGHBOURS,
    "every measure of Tests 7 to 10 and 13": SO_FAR,
    "every measure of Tests 7 to 10 and 13 + neighbourhood": SO_FAR + NEIGHBOURS,
}
WITH_G = f"{JOINS} join g gp on gp.page_id = x.parent join g gc on gc.page_id = x.child"


def neighbourhood_table(con, data, wiki):
    """Table g: for each article, how its neighbours in the graph of lead links are linked. Needs resolve."""
    con.sql(f"""
    create table le as
    select distinct l.page_id as src, r.page_id as dst
    from read_parquet('{data}/{wiki}_links/links/*.parquet') l join resolve r on r.title = l.target
    join s on s.page_id = l.page_id join s d on d.page_id = r.page_id
    where l.sec_idx = 0 and l.page_id <> r.page_id;

    create table deg as
    select s.page_id, coalesce(i.n, 0) as d_in, coalesce(o.n, 0) as d_out
    from s left join (select dst, count(*) as n from le group by 1) i on i.dst = s.page_id
    left join (select src, count(*) as n from le group by 1) o on o.src = s.page_id;

    create table und as select src as a, dst as b from le union select dst, src from le;
    create table samp as
    select a, b, row_number() over (partition by a order by hash(a, b)) as rn from und qualify rn <= {SAMPLE};
    create table cc as
    select x.a as page_id, count(*) as wedges, count(u.a) as closed
    from samp x join samp y on y.a = x.a and y.rn > x.rn
    left join und u on u.a = x.b and u.b = y.b
    group by 1;

    create table g as
    select s.page_id, s.title, d.d_in, d.d_out, n.neighbours, c.wedges, c.closed / c.wedges::double as cc,
           i.nb_in, o.nb_out, ln(1 + i.two_in) - ln(1 + d.d_in) as reach
    from s join deg d using (page_id)
    left join (select a, count(*) as neighbours from und group by 1) n on n.a = s.page_id
    left join cc c on c.page_id = s.page_id
    left join (select e.dst, avg(ln(1 + d.d_in)) as nb_in, sum(d.d_in) as two_in
               from le e join deg d on d.page_id = e.src group by 1) i on i.dst = s.page_id
    left join (select e.src, avg(ln(1 + d.d_in)) as nb_out
               from le e join deg d on d.page_id = e.dst group by 1) o on o.src = s.page_id;
    alter table g add column cc_adj double;
    update g set cc_adj = g.cc - b.mean
    from (select floor(log2(neighbours)) as bin, avg(cc) as mean from g where cc is not null group by 1) b
    where g.cc is not null and floor(log2(g.neighbours)) = b.bin;
    drop table samp; drop table und;
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
    note_tables(con, data, wiki)
    notes = note_rules(con)
    neighbourhood_table(con, data, wiki)
    con.sql(f"copy g to '{data}/{wiki}_neighborhood.parquet' (format parquet)")
    print(con.sql("""
        select count(*) as articles, (select count(*) from le) as lead_links, count(cc) as with_clustering,
               round(avg(cc), 3) as mean_clustering, count(nb_in) as with_in_links
        from g
    """).df().to_string(index=False))

    # each measure on its own
    rows = []
    for name, expr in MEASURES.items():
        child, parent = [expr.replace("g.", f"g{side}.").replace("o.", f"o{side}.") for side in "cp"]
        rows.append(con.sql(f"""
            select '{name}' as measure, x.prop, (op.in_deg < oc.in_deg) as inlinks_wrong, count(*) as pairs,
                   count(*) filter ({parent} > {child}) as correct, count(*) filter ({parent} = {child}) as tied,
                   count(*) filter ({parent} < {child}) as wrong
            from {WITH_G} where {parent} is not null and {child} is not null group by all
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
    table.to_csv(data / f"{wiki}_neighborhood_check.csv", index=False)

    # the fits
    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    near = con.sql(f"select x.prop, x.parent, x.child, {', '.join(f'gp.{n} - gc.{n} as {n}' for n in NEIGHBOURS)} from {WITH_G}").df()
    near[NEIGHBOURS] = near[NEIGHBOURS].fillna(0)   # a measure that is missing for either article says nothing about the pair
    pairs = (differences(con, names).merge(containment(con), on=keys).merge(direction[keys + DIRECTION], on=keys)
             .merge(kinds, on=keys).merge(notes[keys + HATNOTE], on=keys).merge(near, on=keys))
    names = names + ["contains"] + DIRECTION + LEARNED + HATNOTE + NEIGHBOURS
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
            right = right if only is None else right[only]
            return (100 * right.mean() if len(right) else float("nan")), w

        wrong = part["inlinks_wrong"].to_numpy()[held]
        print(f"\n{relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out, {int(wrong.sum()):,} of them wrong by lead in-links")
        print(f"  {'':<54} {'all':>5}   {'change':>6}   on the pairs lead in-links gets wrong")
        alone = earlier = None
        for label, used in FITS.items():
            got = score(used)[0]
            alone = got if alone is None else alone
            step = f"   {got - earlier:+5.1f} from the line above" if label.endswith("+ neighbourhood") else ""
            print(f"  {label:<54} {got:5.1f}   {got - alone:+6.1f}   {score(used, wrong)[0]:5.1f}{step}")
            earlier = got
        w = score(["in_deg"] + NEIGHBOURS)[1]
        print("  weights of lead in-links + neighbourhood measures: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(["in_deg"] + NEIGHBOURS, w)))
        before = score(SO_FAR)[0]
        print("  one neighbourhood measure added: to lead in-links, and to every measure of Tests 7 to 10 and 13")
        for n in NEIGHBOURS:
            got, full = score(["in_deg", n])[0], score(SO_FAR + [n])[0]
            print(f"    + {n:<50} {got:5.1f}   {got - alone:+6.1f}   {full:5.1f}   {full - before:+5.1f}")

    # what the measures look like
    show = "title, d_in, d_out, round(cc, 3) as cc, round(cc_adj, 3) as cc_adj, round(nb_in, 2) as nb_in, round(nb_out, 2) as nb_out, round(reach, 2) as reach"
    print("\n== examples ==\n")
    print(con.sql(f"select {show} from g where title in {EXAMPLES} order by cc").df().to_string(index=False))
    for way, word in [("desc", "most"), ("asc", "least")]:
        print(f"\nneighbours {word} linked to each other, against articles with as many neighbours, among articles with 1,000 or more lead in-links\n")
        print(con.sql(f"select {show} from g where d_in >= 1000 and cc is not null order by cc_adj {way} limit 20").df().to_string(index=False))
    print("\nclustering by number of neighbours: articles, mean, median\n")
    print(con.sql("""
        select cast(pow(2, floor(log2(neighbours))) as bigint) as neighbours_from, count(*) as articles,
               round(avg(cc), 3) as mean_cc, round(median(cc), 3) as median_cc
        from g where cc is not null group by 1 order by 1
    """).df().to_string(index=False))


if __name__ == "__main__":
    main()

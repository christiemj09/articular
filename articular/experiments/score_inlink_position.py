"""Check where an article's in-links sit against Wikidata: is a general article linked from first sentences?

    uv run python -m articular.experiments.score_inlink_position --wiki enwiki

Reads what articular/experiments/score_kind_pairs.py reads. Pairs, held-out fifth and fitting are the same as in
articular/experiments/score_text_features.py.

For each article it counts the articles that link to it (through redirects, itself left out):

  in_all     anywhere in their text
  in_lead    in their lead
  in_first   in their first sentence
  kind_in    as their kind (the link the first sentence names as what the article is)

and from those the share of the in-linking articles that link from the lead and from the first
sentence. The kind of an article is often a noun with no link on it, so kind_in can exceed in_all;
its share is kind_in / (in_all + kind_in). A share is not defined for an article nothing links to. The fits use
shrunk shares: SHRINK imagined in-links at the rate of all articles together are added to each
article, so that a share from two or three links counts for little.

Three parts:
  - each count and share on its own, over all pairs and over the pairs lead in-links gets wrong;
  - fitted combinations on held-out pairs, with and without the measures of the earlier tests;
  - the shares of some well-known articles, and the articles with the largest and smallest shares.

Writes data/<wiki>_inlink_position.parquet (one row per article) and data/<wiki>_inlink_position_check.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from articular.experiments.score_kind_pairs import ALL, KINDS, LEARNED, kind_table, learn
from articular.experiments.score_link_direction import DIRECTION, direction_table, link_positions
from articular.experiments.score_text_features import JOINS, PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

SHRINK = 5
# every measure is turned so that bigger = more general, as we expect it to run
MEASURES = {
    "linked from more articles": "n.in_all",
    "linked from more leads": "n.in_lead",
    "linked from more first sentences": "n.in_first",
    "the kind of more articles": "n.kind_in",
    "larger share of in-links from leads": "n.in_lead / nullif(n.in_all, 0)::double",
    "larger share from first sentences": "n.in_first / nullif(n.in_all, 0)::double",
    "larger share as the kind": "n.kind_in / nullif(n.in_all + n.kind_in, 0)::double",
    "larger share from first sentences, shrunk": "n.first_share",
    "larger share as the kind, shrunk": "n.kind_share",
    "lead in-links": "o.in_deg",
}
POSITION_COUNTS = ["in_all", "in_first", "kind_in"]            # compared on a log scale in the fits
SHARES = ["lead_share", "first_share", "kind_share"]
POSITION = POSITION_COUNTS + SHARES
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
EARLIER = ALL + LEARNED
FITS = {
    "lead in-links": ["in_deg"],
    "in-link position measures": POSITION,
    "lead in-links + shares only": ["in_deg"] + SHARES,
    "lead in-links + in-link position measures": ["in_deg"] + POSITION,
    "every earlier measure": EARLIER,
    "every earlier measure + in-link position": EARLIER + POSITION,
}
WITH_N = f"{JOINS} join n np on np.page_id = x.parent join n nc on nc.page_id = x.child"
EXAMPLES = ("Animal", "Dog", "Poodle", "Disease", "Village", "City", "Music", "Jazz", "Science", "Physics", "Quantum mechanics",
            "France", "Paris", "United States", "World War II", "Battle", "Battle of Stalingrad", "Association football",
            "Football player", "Politician", "Barack Obama", "Species", "Album", "The Beatles")


def link_edges(con, data, wiki):
    """Table edge: one row for each article (src) that links to another (dst), through redirects. Needs resolve."""
    con.sql(f"""
    create table edge as
    select l.page_id as src, r.page_id as dst, bool_or(l.sec_idx = 0) as lead, bool_or(l.in_first_sentence) as first_sentence
    from read_parquet('{data}/{wiki}_links/links/*.parquet') l join resolve r on r.title = l.target
    join s on s.page_id = l.page_id
    where l.page_id <> r.page_id
    group by 1, 2;
    """)


def inlink_counts(con):
    """Table n: for each article, how many articles link to it, by where the link sits. Needs edge and kind."""
    con.sql(f"""
    create table counted as
    select s.page_id, s.title, coalesce(e.in_all, 0) as in_all, coalesce(e.in_lead, 0) as in_lead,
           coalesce(e.in_first, 0) as in_first, coalesce(k.kind_in, 0) as kind_in
    from s
    left join (select dst, count(*) as in_all, count(*) filter (lead) as in_lead, count(*) filter (first_sentence) as in_first
               from edge group by 1) e on e.dst = s.page_id
    left join (select kind_page_id, count(*) as kind_in from kind where kind_page_id <> page_id group by 1) k
           on k.kind_page_id = s.page_id;
    """)
    total, lead, first, kind = con.sql("select sum(in_all), sum(in_lead), sum(in_first), sum(kind_in) from counted").fetchall()[0]
    print(f"{total:,} pairs of articles where one links to the other: {100 * lead / total:.1f}% from the lead, "
          f"{100 * first / total:.1f}% from the first sentence; {kind:,} articles have another article as their kind")
    con.sql(f"""
    create table n as
    select *, (in_lead + {SHRINK} * {lead / total}) / (in_all + {SHRINK}) as lead_share,
           (in_first + {SHRINK} * {first / total}) / (in_all + {SHRINK}) as first_share,
           (kind_in + {SHRINK} * {kind / (total + kind)}) / (in_all + kind_in + {SHRINK}) as kind_share
    from counted
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
    inlink_counts(con)
    con.sql("drop table edge")
    con.sql(f"copy n to '{data}/{wiki}_inlink_position.parquet' (format parquet)")

    # each measure on its own
    rows = []
    for name, expr in MEASURES.items():
        child, parent = [expr.replace("n.", f"n{side}.").replace("o.", f"o{side}.") for side in "cp"]
        rows.append(con.sql(f"""
            select '{name}' as measure, x.prop, (op.in_deg < oc.in_deg) as inlinks_wrong, count(*) as pairs,
                   count(*) filter ({parent} > {child}) as correct, count(*) filter ({parent} = {child}) as tied,
                   count(*) filter ({parent} < {child}) as wrong
            from {WITH_N} where {parent} is not null and {child} is not null group by all
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
    table.to_csv(data / f"{wiki}_inlink_position_check.csv", index=False)

    # the fits
    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    position = con.sql(f"""
        select x.prop, x.parent, x.child,
               {", ".join(f"ln(1 + np.{n}) - ln(1 + nc.{n}) as {n}" for n in POSITION_COUNTS)},
               {", ".join(f"np.{n} - nc.{n} as {n}" for n in SHARES)}
        from {WITH_N}
    """).df()
    pairs = (differences(con, names).merge(containment(con), on=keys)
             .merge(direction[keys + DIRECTION], on=keys).merge(kinds, on=keys).merge(position, on=keys))
    names = names + ["contains"] + DIRECTION + LEARNED + POSITION
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
            step = f"   {got - earlier:+5.1f} from the line above" if label.endswith("+ in-link position") else ""
            print(f"  {label:<44} {got:5.1f}   {got - alone:+6.1f}   {score(used, wrong):5.1f}{step}")
            earlier = got
        before = score(EARLIER)
        print("  one in-link position measure added: to lead in-links, and to every earlier measure")
        for n in POSITION:
            got, full = score(["in_deg", n]), score(EARLIER + [n])
            print(f"    + {n:<40} {got:5.1f}   {got - alone:+6.1f}   {full:5.1f}   {full - before:+5.1f}")

    # what the shares look like
    show = "title, in_all, in_lead, in_first, kind_in, round(100.0 * in_first / nullif(in_all, 0), 1) as first_pct, round(100.0 * kind_in / nullif(in_all + kind_in, 0), 1) as kind_pct"
    print("\n== examples ==\n")
    print(con.sql(f"select {show} from n where title in {EXAMPLES} order by first_pct desc").df().to_string(index=False))
    for floor in (1000, 20000):
        for way, word in [("desc", "largest"), ("asc", "smallest")]:
            print(f"\n{word} share of in-links from first sentences, among articles with {floor:,} or more in-linking articles\n")
            print(con.sql(f"select {show} from n where in_all >= {floor} order by first_pct {way}, in_all desc limit 25").df().to_string(index=False))
    print("\nshare of in-links from first sentences by number of in-linking articles: articles, median %, a quarter below, a quarter above\n")
    print(con.sql("""
        select cast(pow(10, floor(log10(in_all))) as bigint) as in_all_from, count(*) as articles,
               round(100 * median(in_first / in_all::double), 1) as median_pct,
               round(100 * quantile_cont(in_first / in_all::double, 0.25), 1) as q1, round(100 * quantile_cont(in_first / in_all::double, 0.75), 1) as q3
        from n where in_all > 0 group by 1 order by 1
    """).df().to_string(index=False))


if __name__ == "__main__":
    main()

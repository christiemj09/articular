"""Check the notes at the head of sections against Wikidata: does "Main article: X" point from general to specific?

    uv run python -m articular.experiments.score_hatnotes --wiki enwiki

Reads data/<wiki>_hatnotes/ (articular/pipeline/extract_hatnotes.py) and what articular/experiments/score_kind_pairs.py reads.
Pairs, held-out fifth and fitting are the same as in articular/experiments/score_text_features.py.

Within a pair, each rule says +1 (the parent is the more general, as Wikidata has it), -1 (the child
is) or 0 (no opinion):

  main       one article has a "Main article" note naming the other and is not named back: the one
             with the note is the general one
  further    the same for "Further information" notes
  see_also   the same for "See also" notes
  broader    one article has a "For broader coverage" note naming the other: the one named is the general one

Two variants of the main rule look at the heading of the section the note sits under:

  main_nobg  the main rule, leaving out notes under a heading that contains "background"
  main_head  the main rule weighted by how often notes under that heading were right among the pairs
             used for fitting (log odds with one added to each count, so a heading that is usually
             wrong turns the rule round). A pair used for fitting does not count itself.

For each article it also counts, through redirects:

  main_out, main_in     articles its "Main article" notes name, and articles whose notes name it
  other_out, other_in   the same for "Further information" and "See also" notes

Prints each rule and count on its own, fitted combinations on held-out pairs with and without the
measures of the earlier tests, and examples. Writes data/<wiki>_hatnotes.parquet (one row per
article) and data/<wiki>_hatnotes_check.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from articular.experiments.score_inlink_position import EARLIER, EXAMPLES
from articular.experiments.score_kind_pairs import KINDS, LEARNED, kind_table, learn
from articular.experiments.score_link_direction import DIRECTION, direction_table, link_positions
from articular.experiments.score_text_features import JOINS, PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

RULES = ["main", "further", "see_also", "broader"]
COUNTS = ["main_out", "main_in", "other_out", "other_in"]   # compared on a log scale in the fits
HATNOTE = RULES + COUNTS
NO_BACKGROUND = ["main_nobg"] + [n for n in HATNOTE if n != "main"]
# every count is turned so that bigger = more general, as we expect it to run
MEASURES = {
    "names more main articles": "m.main_out",
    "named as a main article by fewer": "-m.main_in",
    "names more further and see-also articles": "m.other_out",
    "named as further or see-also by fewer": "-m.other_in",
    "lead in-links": "o.in_deg",
}
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
FITS = {
    "lead in-links": ["in_deg"],
    "note measures": HATNOTE,
    "lead in-links + note measures": ["in_deg"] + HATNOTE,
    "every measure of Tests 7 to 10": EARLIER,
    "every measure of Tests 7 to 10 + notes": EARLIER + HATNOTE,
    "the same, Background sections left out": EARLIER + NO_BACKGROUND,
    "the same, main rule weighted by heading": EARLIER + HATNOTE + ["main_head"],
    "the same, both": EARLIER + NO_BACKGROUND + ["main_head"],
}
WITH_M = f"{JOINS} join m mp on mp.page_id = x.parent join m mc on mc.page_id = x.child"


def note_tables(con, data, wiki):
    """Tables note (src, dst, template), m (counts per article) and pn (notes within a pair). Needs resolve and wanted."""
    con.sql(f"""
    create table note as
    select distinct h.page_id as src, r.page_id as dst, h.template, lower(h.section) as section
    from read_parquet('{data}/{wiki}_hatnotes/*.parquet') h join resolve r on r.title = h.target
    join s on s.page_id = h.page_id
    where h.page_id <> r.page_id;

    create table m as
    select s.page_id, s.title, coalesce(a.main_out, 0) as main_out, coalesce(b.main_in, 0) as main_in,
           coalesce(a.other_out, 0) as other_out, coalesce(b.other_in, 0) as other_in
    from s
    left join (select src, count(distinct dst) filter (template = 'main') as main_out,
                      count(distinct dst) filter (template in ('further', 'see also')) as other_out from note group by 1) a
           on a.src = s.page_id
    left join (select dst, count(distinct src) filter (template = 'main') as main_in,
                      count(distinct src) filter (template in ('further', 'see also')) as other_in from note group by 1) b
           on b.dst = s.page_id;

    create table pn as
    select n.src, n.dst, bool_or(template = 'main') as main, bool_or(template = 'further') as further,
           bool_or(template = 'see also') as see_also, bool_or(template = 'broader') as broader,
           bool_or(template = 'main' and section not like '%background%') as main_nobg,
           min(section) filter (template = 'main') as main_sec
    from note n join wanted w on w.src = n.src and w.dst = n.dst group by 1, 2;
    """)


def note_rules(con):
    """One row per pair: what each rule says, and parent minus child for each count."""
    # down: the parent's note naming the child; up: the child's note naming the parent
    def rule(name, a="down", b="up"):
        return f"cast(coalesce({a}.{name}, false) as int) - cast(coalesce({b}.{name}, false) as int) as {name}"
    return con.sql(f"""
        select x.prop, x.parent, x.child, (op.in_deg < oc.in_deg) as inlinks_wrong,
               {rule("main")}, {rule("further")}, {rule("see_also")}, {rule("broader", "up", "down")}, {rule("main_nobg")},
               coalesce(down.main_sec, up.main_sec) as main_sec,
               {", ".join(f"ln(1 + mp.{n}) - ln(1 + mc.{n}) as {n}" for n in COUNTS)}
        from {WITH_M}
        left join pn down on down.src = x.parent and down.dst = x.child
        left join pn up on up.src = x.child and up.dst = x.parent
    """).df()


def learn_heading(part, held):
    """main_head for every pair of one relation, counted over the pairs that are not held out."""
    main, heading = part["main"].to_numpy(), part["main_sec"]
    train = part[~held & (main != 0)]
    right = heading.map(train[train["main"] == 1]["main_sec"].value_counts()).fillna(0).to_numpy() - (~held & (main == 1))
    wrong = heading.map(train[train["main"] == -1]["main_sec"].value_counts()).fillna(0).to_numpy() - (~held & (main == -1))
    return np.where(main != 0, main * np.log((right + 1) / (wrong + 1)), 0.0)


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
    con.sql(f"copy m to '{data}/{wiki}_hatnotes.parquet' (format parquet)")
    print(con.sql("""
        select template, count(distinct hash(src, dst)) as notes, count(distinct src) as articles_with_one, count(distinct dst) as articles_named
        from note group by 1 order by 2 desc
    """).df().to_string(index=False))
    print(con.sql("""
        select count(*) as articles, count(*) filter (main_out > 0) as name_a_main_article, count(*) filter (main_in > 0) as named_as_one,
               count(*) filter (main_out > 0 and main_in > 0) as both from m
    """).df().to_string(index=False))

    # each rule on its own
    notes = note_rules(con)
    rows = []
    for label, part in [("all pairs", notes), ("pairs that lead in-links gets wrong", notes[notes["inlinks_wrong"]])]:
        print(f"\n== {label} ==")
        print("\neach rule: % of pairs it gets right, % it gets wrong, % it has no opinion on, and its score")
        print("(right % + half of no opinion %; a coin scores 50); then the pairs it has an opinion on and % right of those\n")
        for prop, relation in PROPS.items():
            got = part[part["prop"] == prop]
            if got.empty:
                continue
            print(f"  {relation} ({len(got):,} pairs)")
            for name in RULES + ["main_nobg"]:
                n_right, n_wrong = int((got[name] == 1).sum()), int((got[name] == -1).sum())
                right, wrong = 100 * n_right / len(got), 100 * n_wrong / len(got)
                silent = 100 - right - wrong
                of_called = 100 * n_right / (n_right + n_wrong) if n_right + n_wrong else float("nan")
                print(f"    {name:<10} {right:6.2f} {wrong:6.2f} {silent:6.2f}   score {right + silent / 2:5.1f}"
                      f"   speaks on {n_right + n_wrong:>8,}   right {of_called:5.1f}")
                rows.append({"pairs": label, "relation": relation, "measure": name, "n": len(got), "right": n_right, "wrong": n_wrong})

    print("\n== the main rule by the heading its note sits under: pairs with an answer, % right (the 20 commonest headings) ==")
    for prop, relation in PROPS.items():
        got = notes[(notes["prop"] == prop) & (notes["main"] != 0)]
        if got.empty:
            continue
        by = got.assign(right=got["main"] == 1).groupby("main_sec")["right"].agg(["size", "mean"]).sort_values("size", ascending=False).head(20)
        print(f"\n  {relation}")
        for heading, row in by.iterrows():
            print(f"    {heading[:50] or '(lead)':<50} {int(row['size']):>7,}   {100 * row['mean']:5.1f}")

    # each count on its own
    counted = []
    for name, expr in MEASURES.items():
        child, parent = [expr.replace("m.", f"m{side}.").replace("o.", f"o{side}.") for side in "cp"]
        counted.append(con.sql(f"""
            select '{name}' as measure, x.prop, (op.in_deg < oc.in_deg) as inlinks_wrong, count(*) as pairs,
                   count(*) filter ({parent} > {child}) as correct, count(*) filter ({parent} = {child}) as tied,
                   count(*) filter ({parent} < {child}) as wrong
            from {WITH_M} group by all
        """).df())
    table = pd.concat(counted)
    table["relation"] = table["prop"].map(PROPS)
    columns = [c for c in PROPS.values() if c in set(table["relation"])]
    for label, part in [("all pairs", table), ("pairs that lead in-links gets wrong", table[table["inlinks_wrong"]])]:
        if part.empty:
            continue
        total = part.groupby(["relation", "measure"], sort=False)[["pairs", "correct", "tied", "wrong"]].sum()
        total["score"] = (100 * (total["correct"] + total["tied"] / 2) / total["pairs"]).round(1)
        total["tied %"] = (100 * total["tied"] / total["pairs"]).round(1)
        total["of untied"] = (100 * total["correct"] / (total["correct"] + total["wrong"])).round(1)
        wide = total.reset_index().pivot(index="measure", columns="relation")
        print(f"\n== {label}: counts, score (correct % + half of tied %; a coin scores 50) ==\n")
        print(wide["score"].reindex(list(MEASURES))[columns].to_string())
        print(f"\n{label}: % of pairs the count leaves tied\n")
        print(wide["tied %"].reindex(list(MEASURES))[columns].to_string())
        print(f"\n{label}: % correct of the pairs not tied\n")
        print(wide["of untied"].reindex(list(MEASURES))[columns].to_string())
    pd.concat([pd.DataFrame(rows), table.drop(columns="prop")]).to_csv(data / f"{wiki}_hatnotes_check.csv", index=False)

    # the fits
    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    pairs = (differences(con, names).merge(containment(con), on=keys)
             .merge(direction[keys + DIRECTION], on=keys).merge(kinds, on=keys).merge(notes[keys + HATNOTE + ["main_nobg", "main_sec"]], on=keys))
    names = names + ["contains"] + DIRECTION + LEARNED + HATNOTE + ["main_nobg", "main_head"]
    print("\n== fitted combinations: % of held-out pairs put the right way round ==")
    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop].reset_index(drop=True)
        if len(part) < 1000:
            continue
        held = (part["parent"].to_numpy() * 2654435761 % 5 == 0)   # the same fifth as design() holds out
        for form in KINDS:
            part[f"pair_{form}"], part[f"rank_{form}"] = learn(part, held, form)
        part["main_head"] = learn_heading(part, held)
        x, y, held = design(part, names)

        def score(used, only=None):
            cols = [names.index(n) for n in used]
            w = fit(x[~held][:, cols], y[~held])
            right = (x[held][:, cols] @ w > 0) == (y[held] > 0)
            right = right if only is None else right[only]
            return (100 * right.mean() if len(right) else float("nan")), w

        wrong = part["inlinks_wrong"].to_numpy()[held]
        spoke = (part[RULES].to_numpy()[held] != 0).any(axis=1)
        print(f"\n{relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out, {int(wrong.sum()):,} of them wrong by lead in-links,"
              f" {int(spoke.sum()):,} with a note between the two")
        print(f"  {'':<44} {'all':>5}   {'change':>6}   on the pairs lead in-links gets wrong, and on the pairs with a note")
        alone = earlier = None
        for label, used in FITS.items():
            got = score(used)[0]
            alone = got if alone is None else alone
            step = f"   {got - earlier:+5.1f} from the line above" if label.endswith("+ notes") else ""
            print(f"  {label:<44} {got:5.1f}   {got - alone:+6.1f}   {score(used, wrong)[0]:5.1f}   {score(used, spoke)[0]:5.1f}{step}")
            earlier = got
        w = score(["in_deg"] + HATNOTE)[1]
        print("  weights of lead in-links + note measures: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(["in_deg"] + HATNOTE, w)))
        before = score(EARLIER)[0]
        print("  one note measure added: to lead in-links, and to every measure of Tests 7 to 10")
        for n in HATNOTE:
            got, full = score(["in_deg", n])[0], score(EARLIER + [n])[0]
            print(f"    + {n:<40} {got:5.1f}   {got - alone:+6.1f}   {full:5.1f}   {full - before:+5.1f}")

    # what the notes look like
    show = "title, main_out, main_in, other_out, other_in"
    print("\n== examples ==\n")
    print(con.sql(f"select {show} from m where title in {EXAMPLES} order by main_out desc").df().to_string(index=False))
    for column, what in [("main_out", "name the most main articles"), ("main_in", "are named as a main article by the most articles")]:
        print(f"\narticles that {what}\n")
        print(con.sql(f"select {show} from m order by {column} desc limit 25").df().to_string(index=False))
    print("\npairs where the main rule is wrong (the child has a Main article note naming the parent), 12 per relation\n")
    print(con.sql(f"""
        select prop, child, parent from (
            select x.prop, sc.title as child, sp.title as parent,
                   row_number() over (partition by x.prop order by hash(x.child, x.parent)) as i
            from {JOINS} join pn up on up.src = x.child and up.dst = x.parent and up.main
            where not exists (select 1 from pn d where d.src = x.parent and d.dst = x.child and d.main)
        ) where i <= 12 order by prop, i
    """).df().to_string(index=False))


if __name__ == "__main__":
    main()

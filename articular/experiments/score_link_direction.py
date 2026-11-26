"""Check link direction within a pair against Wikidata: does the child link to the parent, and earlier?

    uv run python -m articular.experiments.score_link_direction --wiki enwiki

Reads data/<wiki>_links/ (articular/pipeline/extract_links.py) and what articular/experiments/score_title_features.py reads.
Pairs, scoring and fitting are the same as in articular/experiments/score_text_features.py.

For each pair it finds the first link from the child's article to the parent's and the first link
back, with the number of words before each. Each rule below then says +1 (the parent is the more
general, as Wikidata has it), -1 (the child is) or 0 (no opinion):

  links           one article links to the other and is not linked back: the one linking is the specific one
  in the lead     the same, counting only links in the lead
  first sentence  the same, counting only links in the first sentence
  earlier         the article that links to the other after fewer words is the specific one; an article
                  that does not link at all counts as linking last

Three parts:
  - how the pairs divide by which way the links run;
  - each rule on its own, over all pairs and over the pairs lead in-links gets wrong;
  - fitted combinations on held-out pairs, with and without the measures of the earlier tests.

Writes data/<wiki>_link_direction.parquet (one row per pair) and data/<wiki>_link_direction_check.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from articular.experiments.score_text_features import JOINS, PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

RULES = {
    "links": "links",
    "in the lead": "lead",
    "first sentence": "first_sentence",
    "earlier": "earlier",
}
# `early` is graded: 1 for a link at the first word, falling with the log of the words before it, 0 for no link
DIRECTION = ["links", "lead", "first_sentence", "early"]
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
BEFORE = ["in_deg"] + OTHER_TEXT + TITLE
FITS = {
    "lead in-links": ["in_deg"],
    "direction measures": DIRECTION,
    "lead in-links + direction": ["in_deg"] + DIRECTION,
    "lead in-links + vocabulary + title": BEFORE,
    "those + direction": BEFORE + DIRECTION,
    "lead in-links + vocabulary + title + kinds_below": BEFORE + ["kinds_below"],
    "those + direction ": BEFORE + ["kinds_below"] + DIRECTION,
}


def resolve_tables(con, data, wiki):
    """Tables resolve (title or redirect -> article) and wanted (each Wikidata pair, both ways round)."""
    links = f"{data}/{wiki}_links"
    con.sql(f"""
    create table pages as select page_id, title, redirect_target from read_parquet('{links}/pages/*.parquet');
    create table resolve as
    select p.title, coalesce(a0.page_id, a1.page_id, a2.page_id) as page_id
    from pages p
    left join pages a0 on a0.title = p.title and a0.redirect_target is null
    left join pages a1 on a1.title = p.redirect_target and a1.redirect_target is null
    left join pages r on r.title = p.redirect_target
    left join pages a2 on a2.title = r.redirect_target and a2.redirect_target is null
    where coalesce(a0.page_id, a1.page_id, a2.page_id) is not null;

    create table wanted as
    select distinct child as src, parent as dst from pair union select distinct parent, child from pair;
    """)


def link_positions(con, data, wiki):
    """Table hop: for each ordered pair of articles that Wikidata pairs, the first link from src to dst."""
    resolve_tables(con, data, wiki)
    links = f"{data}/{wiki}_links"
    con.sql(f"""
    create table hop as
    select l.page_id as src, r.page_id as dst, min(l.word_abs) as word,
           bool_or(l.sec_idx = 0) as lead, bool_or(l.in_first_sentence) as first_sentence
    from read_parquet('{links}/links/*.parquet') l join resolve r on r.title = l.target
    join wanted w on w.src = l.page_id and w.dst = r.page_id
    group by 1, 2;
    """)


def direction_table(con):
    """One row per pair: where each article first links to the other, and what each rule says."""
    # up: the child's link to the parent; down: the parent's link to the child
    return con.sql(f"""
        select x.prop, x.parent, x.child, (op.in_deg < oc.in_deg) as inlinks_wrong,
               up.word as up_word, down.word as down_word,
               cast(up.src is not null as int) - cast(down.src is not null as int) as links,
               cast(coalesce(up.lead, false) as int) - cast(coalesce(down.lead, false) as int) as lead,
               cast(coalesce(up.first_sentence, false) as int) - cast(coalesce(down.first_sentence, false) as int) as first_sentence,
               sign(coalesce(down.word, 1e9) - coalesce(up.word, 1e9))::int as earlier,
               coalesce(1 / (1 + ln(1 + up.word)), 0) - coalesce(1 / (1 + ln(1 + down.word)), 0) as early
        from {JOINS}
        left join hop up on up.src = x.child and up.dst = x.parent
        left join hop down on down.src = x.parent and down.dst = x.child
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
    link_positions(con, data, wiki)

    direction = direction_table(con)
    direction.to_parquet(data / f"{wiki}_link_direction.parquet", index=False)

    rows = []
    for label, part in [("all pairs", direction), ("pairs that lead in-links gets wrong", direction[direction["inlinks_wrong"]])]:
        print(f"\n== {label} ==")
        print("\nwhich way the links run, % of pairs (and the median number of words before the link, where both link)\n")
        print(f"  {'relation':<12} {'pairs':>10} {'neither':>8} {'child only':>11} {'parent only':>12} {'both':>6}   {'child':>6} {'parent':>7}")
        for prop, relation in PROPS.items():
            got = part[part["prop"] == prop]
            if got.empty:
                continue
            up, down = got["up_word"].notna(), got["down_word"].notna()
            both = got[up & down]
            print(f"  {relation:<12} {len(got):>10,} {100 * (~up & ~down).mean():>8.1f} {100 * (up & ~down).mean():>11.1f}"
                  f" {100 * (~up & down).mean():>12.1f} {100 * (up & down).mean():>6.1f}"
                  f"   {both['up_word'].median():>6.0f} {both['down_word'].median():>7.0f}")
        print("\neach rule: % of pairs it gets right, % it gets wrong, % it has no opinion on, and its score")
        print("(right % + half of no opinion %; a coin scores 50); last, % right of the pairs it has an opinion on\n")
        for prop, relation in PROPS.items():
            got = part[part["prop"] == prop]
            if got.empty:
                continue
            print(f"  {relation}")
            for name, column in RULES.items():
                right, wrong, silent = [100 * (got[column] == v).mean() for v in (1, -1, 0)]
                of_called = 100 * right / (right + wrong) if right + wrong else float("nan")
                print(f"    {name:<16} {right:5.1f} {wrong:5.1f} {silent:5.1f}   score {right + silent / 2:5.1f}   when it speaks {of_called:5.1f}")
                rows.append({"pairs": label, "relation": relation, "rule": name, "n": len(got), "right": right, "wrong": wrong,
                             "no_opinion": silent})
    pd.DataFrame(rows).to_csv(data / f"{wiki}_link_direction_check.csv", index=False)

    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    pairs = (differences(con, names).merge(containment(con).drop(columns="inlinks_wrong"), on=keys)
             .merge(direction[keys + DIRECTION], on=keys))
    names = names + ["contains"] + DIRECTION
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
        alone = previous = None
        for label, used in FITS.items():
            got = score(used)
            alone = got if alone is None else alone
            step = f"   {got - previous:+5.1f} from the line above" if label.startswith("those") else ""
            print(f"  {label:<50} {got:5.1f}   {got - alone:+5.1f}{step}")
            previous = got
        print("  lead in-links plus one direction measure")
        for n in DIRECTION:
            got = score(["in_deg", n])
            print(f"    + {n:<46} {got:5.1f}   {got - alone:+5.1f}")


if __name__ == "__main__":
    main()

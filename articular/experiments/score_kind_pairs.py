"""Check whether the kinds of two articles decide which is the more general, against Wikidata.

    uv run python -m articular.experiments.score_kind_pairs --wiki enwiki

Reads data/<wiki>_kind_graph.parquet, data/<wiki>_relations/ and what articular/experiments/score_link_direction.py
reads. Pairs, held-out fifth and fitting are the same as in articular/experiments/score_text_features.py.

Each article has a kind in two forms: the article its first sentence names as its kind (Village),
and the noun of that sentence alone ("village"), which more articles have. Unlike every earlier
measure, these are learned from Wikidata: the pairs of the four fifths of parents used for fitting
are counted, and the held-out pairs are judged from those counts.

  pair   how often, among pairs with these two kinds, the article of the parent's kind was the
         parent, against the other way round (a village lies in a district; never the reverse)
  rank   how often an article of each kind is a parent rather than a child, parent's kind minus child's

Both are log odds with one added to each count; 0 when the two kinds are the same or either is
missing. A pair used for fitting does not count itself. The counts are kept apart by relation.

Prints each measure on its own over the held-out pairs, and fitted combinations with and without
the measures of the earlier tests. Writes data/<wiki>_kind_pairs_check.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from articular.experiments.score_link_direction import BEFORE, DIRECTION, direction_table, link_positions
from articular.experiments.score_text_features import PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

KINDS = {"link": "kind_page_id", "noun": "head"}   # the two forms of an article's kind
LEARNED = [f"{how}_{form}" for form in KINDS for how in ("pair", "rank")]
OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
ALL = BEFORE + ["kinds_below"] + DIRECTION
FITS = {
    "lead in-links": ["in_deg"],
    "kind measures": LEARNED,
    "lead in-links + kind measures": ["in_deg"] + LEARNED,
    "every earlier measure": ALL,
    "every earlier measure + kind measures": ALL + LEARNED,
}


def learn(part, held, form):
    """pair and rank for every pair of one relation, counted over the pairs that are not held out."""
    kp, kc = part[f"{form}_p"], part[f"{form}_c"]
    known = kp.notna() & kc.notna() & (kp != kc)
    own = (~held & known).astype(int)                      # a fitting pair must not count itself
    train = pd.DataFrame({"a": kp[~held & known], "b": kc[~held & known]})

    both = train.groupby(["a", "b"]).size().rename("n").reset_index()
    frame = pd.DataFrame({"a": kp.to_numpy(), "b": kc.to_numpy()})
    forward = frame.merge(both, on=["a", "b"], how="left")["n"].fillna(0).to_numpy() - own
    back = frame.merge(both.rename(columns={"a": "b", "b": "a"}), on=["a", "b"], how="left")["n"].fillna(0).to_numpy()
    pair = np.where(known, np.log((forward + 1) / (back + 1)), 0.0)

    as_parent, as_child = train["a"].value_counts(), train["b"].value_counts()
    def count(series, kind):
        return kind.map(series).fillna(0).to_numpy()
    rank = (np.log((count(as_parent, kp) - own + 1) / (count(as_child, kp) + 1))
            - np.log((count(as_parent, kc) + 1) / (count(as_child, kc) - own + 1)))
    return pair, np.where(known, rank, 0.0)


def kind_table(con, data, wiki):
    """Table kind (one row per article, both forms of its kind); returns both articles' kinds for every pair."""
    con.sql(f"""
    create table kind as
    select g.page_id, nullif(g.kind_page_id, -1) as kind_page_id, h.head
    from '{data}/{wiki}_kind_graph.parquet' g
    left join (select page_id, any_value(head) as head from read_parquet('{data}/{wiki}_relations/*.parquet')
               where relation in ('kind', 'kind_text') and head is not null group by 1) h using (page_id)
    """)
    return con.sql("""
        select x.prop, x.parent, x.child, kp.kind_page_id as link_p, kc.kind_page_id as link_c, kp.head as noun_p, kc.head as noun_c
        from pair x left join kind kp on kp.page_id = x.parent left join kind kc on kc.page_id = x.child
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
    kinds = kind_table(con, data, wiki)

    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    pairs = (differences(con, names).merge(containment(con), on=keys)
             .merge(direction[keys + DIRECTION], on=keys).merge(kinds, on=keys))
    names = names + ["contains"] + DIRECTION + LEARNED

    rows = []
    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop].reset_index(drop=True)
        if len(part) < 1000:
            continue
        held = (part["parent"].to_numpy() * 2654435761 % 5 == 0)   # the same fifth as design() holds out
        for form in KINDS:
            part[f"pair_{form}"], part[f"rank_{form}"] = learn(part, held, form)
        x, y, held = design(part, names)

        def score(used):
            cols = [names.index(n) for n in used]
            w = fit(x[~held][:, cols], y[~held])
            return 100 * ((x[held][:, cols] @ w > 0) == (y[held] > 0)).mean()

        print(f"\n== {relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out ==")
        out = part[held]
        for form in KINDS:
            has = out[f"{form}_p"].notna() & out[f"{form}_c"].notna()
            same = has & (out[f"{form}_p"] == out[f"{form}_c"])
            print(f"  kind as {form}: both articles have one in {100 * has.mean():.1f}% of held-out pairs; the same one in {100 * same.mean():.1f}%")
        print("\n  each measure alone on held-out pairs: right %, wrong %, no opinion %, score; then the same on the pairs lead in-links gets wrong")
        for n in LEARNED:
            line = f"    {n:<12}"
            for label, got in [("all", out), ("in-links wrong", out[out["inlinks_wrong"]])]:
                right, wrong, silent = [100 * v.mean() for v in (got[n] > 0, got[n] < 0, got[n] == 0)]
                line += f"   {right:5.1f} {wrong:5.1f} {silent:5.1f}  score {right + silent / 2:5.1f}"
                rows.append({"relation": relation, "measure": n, "pairs": label, "n": len(got), "right": right, "wrong": wrong, "no_opinion": silent})
            print(line)
        print("\n  fitted combinations: % of held-out pairs put the right way round")
        alone = earlier = None
        for label, used in FITS.items():
            got = score(used)
            alone = got if alone is None else alone
            step = f"   {got - earlier:+5.1f} from the line above" if label == "every earlier measure + kind measures" else ""
            print(f"    {label:<40} {got:5.1f}   {got - alone:+5.1f}{step}")
            earlier = got
        print("    every earlier measure plus one kind measure")
        before = score(ALL)
        for n in LEARNED:
            got = score(ALL + [n])
            print(f"      + {n:<34} {got:5.1f}   {got - before:+5.1f}")

        # what was learned: the commonest pairs of kinds among the pairs used for fitting
        train = part[~held & part["noun_p"].notna() & part["noun_c"].notna() & (part["noun_p"] != part["noun_c"])]
        top = train.groupby(["noun_c", "noun_p"]).size().sort_values(ascending=False).head(15)
        both = train.groupby(["noun_c", "noun_p"]).size()
        print("\n  commonest (child noun, parent noun) among fitting pairs: pairs this way, pairs the other way")
        print("    " + "; ".join(f"{c} under {p} {n:,}/{int(both.get((p, c), 0)):,}" for (c, p), n in top.items()))
    pd.DataFrame(rows).to_csv(data / f"{wiki}_kind_pairs_check.csv", index=False)


if __name__ == "__main__":
    main()

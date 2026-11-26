"""List the pairs that the full fitted combination still puts the wrong way round, for reading.

    uv run python -m articular.experiments.sample_errors --wiki enwiki

Reads what articular/experiments/score_hatnotes.py reads. For each relation it fits the measures of Tests 7 to 10
and 13 on four fifths of the parents (as in that script) and takes the held-out pairs it gets wrong.

Prints counts that describe those pairs, and writes
  data/<wiki>_errors.parquet       every held-out pair the combination gets wrong
  data/<wiki>_errors_sample.csv    a random --sample of them per relation, with both first sentences
"""

import argparse
from pathlib import Path

import pandas as pd

from articular.experiments.score_hatnotes import HATNOTE, note_rules, note_tables
from articular.experiments.score_inlink_position import EARLIER
from articular.experiments.score_kind_pairs import KINDS, LEARNED, kind_table, learn
from articular.experiments.score_text_features import PROPS, TEXT, design, differences, fit, open_tables
from articular.experiments.score_link_direction import DIRECTION, direction_table, link_positions
from articular.experiments.score_title_features import TITLE, add_titles, containment

OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]
USED = EARLIER + HATNOTE


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--statements", type=Path, default=None)
    parser.add_argument("--page-props", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=150)
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

    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    pairs = (differences(con, names).merge(containment(con), on=keys)
             .merge(direction[keys + DIRECTION + ["up_word", "down_word"]], on=keys)
             .merge(kinds, on=keys).merge(notes[keys + HATNOTE], on=keys))
    names = names + ["contains"] + DIRECTION + LEARNED + HATNOTE
    wrong = []
    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop].reset_index(drop=True)
        if len(part) < 1000:
            continue
        held = (part["parent"].to_numpy() * 2654435761 % 5 == 0)   # the same fifth as design() holds out
        for form in KINDS:
            part[f"pair_{form}"], part[f"rank_{form}"] = learn(part, held, form)
        x, y, held = design(part, names)
        cols = [names.index(n) for n in USED]
        w = fit(x[~held][:, cols], y[~held])
        z = x[held][:, cols] @ w
        # margin: how far on the wrong side the pair falls, in the fit's own units
        got = part[held][(z > 0) != (y[held] > 0)].assign(relation=relation, margin=abs(z[(z > 0) != (y[held] > 0)]))
        print(f"{relation}: {int(held.sum()):,} held out, {len(got):,} wrong ({100 * len(got) / held.sum():.1f}%)")
        wrong.append(got)
    wrong = pd.concat(wrong)
    con.register("wrong", wrong[["relation", "prop", "parent", "child", "margin", "inlinks_wrong", "contains",
                                 "links", "first_sentence", "up_word", "down_word", "main", "noun_c", "noun_p"]])
    links = f"{data}/{wiki}_links"
    full = con.sql(f"""
        with sentence as (select page_id, first_sentence from read_parquet('{links}/pages/*.parquet') where redirect_target is null),
             -- the same two articles paired the other way round, or under another relation
             other as (select child, parent, string_agg(prop::varchar, ' ' order by prop) as props from pair group by all)
        select w.relation, tc.title as specific, tp.title as general, w.margin, w.inlinks_wrong,
               oc.in_deg as specific_in_links, op.in_deg as general_in_links,
               sc.kinds_below as specific_kinds_below, sp.kinds_below as general_kinds_below,
               tc.n_tokens as specific_words, tp.n_tokens as general_words,
               w.contains, w.links, w.first_sentence, w.up_word, w.down_word, w.main as main_note,
               w.noun_c as specific_noun, w.noun_p as general_noun, 'Q' || ic.q as specific_item, 'Q' || ip.q as general_item,
               same.props as same_way_props, back.props as reversed_props,
               left(zc.first_sentence, 220) as specific_sentence, left(zp.first_sentence, 220) as general_sentence,
               w.child, w.parent
        from wrong w
        join t tc on tc.page_id = w.child join t tp on tp.page_id = w.parent
        join o oc on oc.page_id = w.child join o op on op.page_id = w.parent
        join s sc on sc.page_id = w.child join s sp on sp.page_id = w.parent
        left join item ic on ic.page_id = w.child left join item ip on ip.page_id = w.parent
        left join sentence zc on zc.page_id = w.child left join sentence zp on zp.page_id = w.parent
        left join other same on same.child = w.child and same.parent = w.parent
        left join other back on back.child = w.parent and back.parent = w.child
    """).df()
    full.to_parquet(data / f"{wiki}_errors.parquet", index=False)

    print("\n== the pairs still wrong, % of them ==\n")
    print(f"  {'relation':<12} {'pairs':>8} {'in-links also wrong':>20} {'in-links tied':>14} {'Wikidata has it both ways':>26}"
          f" {'no link either way':>19} {'only general links':>19} {'general 1 kind below':>21} {'specific < 100 words':>21}")
    for relation in PROPS.values():
        got = full[full["relation"] == relation]
        if got.empty:
            continue
        shares = [got["inlinks_wrong"], got["specific_in_links"] == got["general_in_links"], got["reversed_props"].notna(),
                  got["up_word"].isna() & got["down_word"].isna(), got["up_word"].isna() & got["down_word"].notna(),
                  got["general_kinds_below"] <= 1, got["specific_words"] < 100]
        print(f"  {relation:<12} {len(got):>8,} " + " ".join(f"{100 * v.mean():>{n}.1f}" for v, n in zip(shares, (20, 14, 26, 19, 19, 21, 21))))
    print("\ncommonest general articles among the pairs still wrong (pairs)")
    for relation in PROPS.values():
        top = full[full["relation"] == relation]["general"].value_counts().head(25)
        print(f"\n  {relation}: " + "; ".join(f"{title} {n:,}" for title, n in top.items()))

    sample = pd.concat([g.sample(min(args.sample, len(g)), random_state=0) for _, g in full.groupby("relation", sort=False)])
    sample.drop(columns=["child", "parent"]).to_csv(data / f"{wiki}_errors_sample.csv", index=False)


if __name__ == "__main__":
    main()

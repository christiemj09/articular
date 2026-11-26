"""Find which vocabulary measures carry the gain over lead in-links, by fitting with measures left out or added.

    uv run python -m articular.experiments.refit_text_features --wiki enwiki

Reads what articular/experiments/score_text_features.py reads, and fits the same way: for each relation a weighted
sum of parent-minus-child differences, fitted on four fifths of the parents and scored on the pairs
of the remaining fifth.

For each relation it prints the held-out score of
  - lead in-links alone, and lead in-links with all ten vocabulary measures (the two ends);
  - lead in-links with one measure, or one group of like measures, added;
  - everything with one measure, or one group, left out;
  - a short list built one measure at a time, each step adding the measure that helps most. The
    choice at each step is made on a second fifth of the parents, so the held-out fifth plays no
    part in choosing.
Every score is also given for the held-out pairs where both articles have the first-200-word
measures, since those measures count as "no difference" elsewhere.
"""

import argparse
from pathlib import Path

import numpy as np

from articular.experiments.score_text_features import PROPS, TEXT, design, differences, fit, open_tables

# measures that are close to the same quantity are also added and left out together
GROUPS = {
    "first 200 words": ["mean_idf_200", "kl_200", "types_200"],
    "title": ["title_idf_sum", "title_bm25"],
    "mentions and size": ["name_df", "n_tokens", "top10_share"],
    "whole-article rarity": ["mean_idf", "share_rare_types"],
}
STEPS = 4   # length of the short list, after lead in-links


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

    names = ["in_deg"] + TEXT
    con = open_tables(data, wiki, statements, page_props)
    pairs = differences(con, names)
    # whether both articles of a pair have the first-200-word measures
    full = con.sql("""select x.prop, x.parent, x.child, (tc.kl_200 is not null and tp.kl_200 is not null) as has_200
                      from pair x join t tc on tc.page_id = x.child join t tp on tp.page_id = x.parent""").df()
    pairs = pairs.merge(full, on=["prop", "parent", "child"])

    for prop, relation in PROPS.items():
        part = pairs[pairs["prop"] == prop]
        if len(part) < 1000:
            continue
        x, y, held = design(part, names)
        has_200 = part["has_200"].to_numpy()
        chooser = (part["parent"].to_numpy() * 2654435761 % 5 == 1)   # the fifth used to choose the short list

        def score(used, train=~held, test=held):
            cols = [names.index(n) for n in used]
            w = fit(x[train][:, cols], y[train])
            right = (x[:, cols] @ w > 0) == (y > 0)
            return 100 * right[test].mean(), 100 * right[test & has_200].mean()

        def line(label, used, base=None):
            got = score(used)
            change = "" if base is None else f"   {got[0] - base[0]:+5.1f}  {got[1] - base[1]:+5.1f}"
            print(f"  {label:<34} {got[0]:5.1f}  {got[1]:5.1f}{change}")
            return got

        print(f"\n== {relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out, "
              f"{int((held & has_200).sum()):,} of them with first-200-word measures ==")
        print(f"  {'':<34} {'all':>5}  {'w/200':>5}   change (all, w/200)")
        alone = line("lead in-links alone", ["in_deg"])
        everything = line("lead in-links + all vocabulary", names, alone)
        line("vocabulary alone", TEXT, alone)

        print("  lead in-links plus one measure (change from lead in-links alone)")
        for n in TEXT:
            line(f"  + {n}", ["in_deg", n], alone)
        print("  lead in-links plus one group")
        for group, members in GROUPS.items():
            line(f"  + {group}", ["in_deg"] + members, alone)

        print("  everything less one measure (change from everything)")
        for n in TEXT:
            line(f"  - {n}", [m for m in names if m != n], everything)
        print("  everything less one group")
        for group, members in GROUPS.items():
            line(f"  - {group}", [m for m in names if m not in members], everything)

        print("  short list, one measure added at a time (change from lead in-links alone)")
        chosen = ["in_deg"]
        for _ in range(STEPS):
            rest = [n for n in TEXT if n not in chosen]
            best = max(rest, key=lambda n: score(chosen + [n], train=~held & ~chooser, test=chooser)[0])
            chosen.append(best)
            line(f"  then + {best}", chosen, alone)
        print("  short list in full: " + ", ".join(chosen))


if __name__ == "__main__":
    main()

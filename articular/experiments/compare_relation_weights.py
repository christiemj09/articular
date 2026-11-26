"""Fit the measures on each relation separately and compare the weights: are kind and scope separate axes?

    uv run python -m articular.experiments.compare_relation_weights --wiki enwiki

Reads what articular/experiments/score_hatnotes.py reads. Pairs, held-out fifth and fitting are the same as in
articular/experiments/score_text_features.py, with one change: each difference is scaled by one spread shared by
the four relations (the root of the average of their variances), so that a weight means the same
thing in every fit.

The measures are those of Tests 7 to 9 and 13. The kind measures of Test 10 are left out because
they are counted from the pairs of one relation and have no meaning in another.

Three parts:
  - the weights of the fit on each relation, side by side, and how alike the four sets are;
  - every fit scored on the held-out pairs of every relation;
  - one fit on the four relations together (the same number of pairs from each), scored on each.

Writes data/<wiki>_relation_weights.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from articular.experiments.score_hatnotes import HATNOTE, note_rules, note_tables
from articular.experiments.score_link_direction import DIRECTION, direction_table, link_positions
from articular.experiments.score_text_features import PROPS, TEXT, differences, fit, open_tables
from articular.experiments.score_title_features import TITLE, add_titles, containment

OTHER_TEXT = [n for n in TEXT if n != "title_idf_sum"]


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
    note_tables(con, data, wiki)
    notes = note_rules(con)

    names = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    keys = ["prop", "parent", "child"]
    pairs = (differences(con, names).merge(containment(con), on=keys)
             .merge(direction[keys + DIRECTION], on=keys).merge(notes[keys + HATNOTE], on=keys))
    names = names + ["contains"] + DIRECTION + HATNOTE

    parts = {relation: pairs[pairs["prop"] == prop] for prop, relation in PROPS.items()}
    parts = {relation: part for relation, part in parts.items() if len(part) >= 1000}
    spread = np.sqrt(np.mean([part[names].to_numpy(dtype=float).var(axis=0) for part in parts.values()], axis=0)) + 1e-9

    def design(part):
        """x, y and the held-out mask, as in score_text_features.design but with the shared spread."""
        x = part[names].to_numpy(dtype=float) / spread
        turned = (part["child"].to_numpy() % 2 == 1)
        x[turned] *= -1
        return x, (~turned).astype(float), (part["parent"].to_numpy() * 2654435761 % 5 == 0)

    made = {relation: design(part) for relation, part in parts.items()}
    weights = {relation: fit(x[~held], y[~held]) for relation, (x, y, held) in made.items()}

    # the same number of fitting pairs from each relation, chosen by a hash of the child
    least = min(int((~held).sum()) for _, _, held in made.values())
    xs, ys = [], []
    for relation, (x, y, held) in made.items():
        order = np.argsort(parts[relation]["child"].to_numpy()[~held] * 2654435761 % 2**32, kind="stable")[:least]
        xs.append(x[~held][order])
        ys.append(y[~held][order])
    weights["all four together"] = fit(np.concatenate(xs), np.concatenate(ys))

    table = pd.DataFrame(weights, index=names)
    relations = list(parts)
    print("== weights of the fit on each relation (parent minus child, in shared standard deviations) ==\n")
    print(table.reindex(table[relations].abs().max(axis=1).sort_values(ascending=False).index).round(2).to_string())
    table.to_csv(data / f"{wiki}_relation_weights.csv")

    print("\n== how alike the sets of weights are: correlation, and below it the cosine of the angle between them ==\n")
    print(table[relations].corr().round(2).to_string())
    unit = table[relations] / np.sqrt((table[relations] ** 2).sum())
    print()
    print((unit.T @ unit).round(2).to_string())
    print("\nmeasures whose weight has a different sign in two fits, where both weights are 0.1 or more in size\n")
    for i, a in enumerate(relations):
        for b in relations[i + 1:]:
            flipped = table[(table[a] * table[b] < 0) & (table[[a, b]].abs().min(axis=1) >= 0.1)]
            print(f"  {a} / {b}: " + (", ".join(f"{n} ({r[a]:+.2f}, {r[b]:+.2f})" for n, r in flipped.iterrows()) or "none"))

    print("\n== % of held-out pairs put the right way round: fitted on the row, scored on the column ==\n")
    scores = pd.DataFrame({
        scored: {fitted: 100 * ((x[held] @ w > 0) == (y[held] > 0)).mean() for fitted, w in weights.items()}
        for scored, (x, y, held) in made.items()
    })
    column = names.index("in_deg")
    scores.loc["lead in-links alone"] = {scored: 100 * ((x[held][:, column] > 0) == (y[held] > 0)).mean() for scored, (x, y, held) in made.items()}
    print(scores.round(1).to_string())
    print("\nloss from using another relation's weights (own fit minus the row's fit)\n")
    print((np.diag(scores.loc[relations, relations]) - scores.loc[list(weights)]).round(1).to_string())


if __name__ == "__main__":
    main()

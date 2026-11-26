"""Compare the tf-idf vector of a parent article with the sum of its children's vectors.

    uv run python -m articular.experiments.family_vectors data/enwiki/*.bz2 --wiki enwiki --workers 8

Reads what articular/experiments/pair_vectors.py reads, and builds the vectors the same way: one weight per word,
(1 + ln count) x idf, each article's vector scaled to length 1.

A family is a parent and its children under one Wikidata relation. Families with fewer than --least
children are left out; of a larger family --most children are kept, chosen by a hash; families are
taken in hash order until a relation has --cap pairs. The children's unit vectors are added up.

Likeness, one row per family: the cosine between the parent and the sum of its children, beside the
cosine between the parent and one child, and between the parent and the sum of another parent's
children (same relation, drawn at random).

Direction, one row per pair, scored as in Test 7. The sum of a child's siblings (the family's sum
less the child itself) is compared with the child and with the parent:

  siblings        cosine between the parent and the sum of the child's siblings, minus the cosine
                  between the child and that sum. Positive when the parent is the closer.

Writes data/<wiki>_family_vectors.parquet (one row per pair) and data/<wiki>_families.parquet (one
row per family). The vectors are kept in data/<wiki>_text/families/vectors/; finished files are
skipped on a rerun with the same pairs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, diags

from articular.experiments import pair_vectors
from articular.experiments.pair_vectors import load_vectors, score, vector_file
from articular.experiments.score_text_features import JOINS, PROPS, design, differences, fit, open_tables
from articular.experiments.score_title_features import OTHER_TEXT, TITLE, add_titles, containment
from articular.pipeline.text_features import run

CHUNK = 2000        # pairs compared at a time; a family's sum is repeated for each of its pairs
SIZES = [3, 5, 10, 20, np.inf]


def rowdot(a, b):
    """Dot product of each row of a with the same row of b."""
    return np.concatenate([a[i:i + CHUNK].multiply(b[i:i + CHUNK]).sum(axis=1).A1 for i in range(0, a.shape[0], CHUNK)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="pages-articles .xml.bz2 files")
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--statements", type=Path, default=None)
    parser.add_argument("--page-props", type=Path, default=None)
    parser.add_argument("--cap", type=int, default=150_000, help="pairs kept for each relation")
    parser.add_argument("--least", type=int, default=3, help="children a family must have")
    parser.add_argument("--most", type=int, default=30, help="children kept of a larger family")
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    statements = args.statements or data / "wd" / "statements.tsv.gz"
    page_props = args.page_props or data / "wd" / f"{wiki}-page_props.sql.gz"
    out = data / f"{wiki}_text" / "families"
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    con = open_tables(data, wiki, statements, page_props)
    add_titles(con, data, wiki)
    con.sql(f"""
        create or replace table pair as
        with kept as (
            select x.child, x.parent, x.prop, count(*) over (partition by x.prop, x.parent) as children from {JOINS}
            qualify row_number() over (partition by x.prop, x.parent order by hash(x.parent, x.child), x.child) <= {args.most}
        ), family as (
            select prop, parent, sum(count(*)) over (partition by prop order by hash(parent), parent) as running
            from kept where children >= {args.least} group by prop, parent
        )
        select k.child, k.parent, k.prop from kept k join family f using (prop, parent) where f.running <= {args.cap}
    """)
    pairs = con.sql("select prop, parent, child from pair order by prop, parent, child").df()
    needed = set(pairs["parent"].tolist()) | set(pairs["child"].tolist())
    print(f"{len(pairs):,} pairs in {len(pairs.drop_duplicates(['prop', 'parent'])):,} families, {len(needed):,} articles in them")
    # vectors kept from a run with other pairs would lack some of these articles
    marker, chosen = out / "vectors" / "pairs.json", {"cap": args.cap, "pairs": len(pairs), "articles": len(needed)}
    if marker.exists() and json.loads(marker.read_text()) != chosen:
        for p in (out / "vectors").glob("*.*"):
            p.unlink()
    marker.write_text(json.dumps(chosen))

    # the workers of articular/experiments/pair_vectors.py read these from their own module
    vocab = pq.read_table(data / f"{wiki}_text" / "vocab.parquet", columns=["token", "df"]).to_pandas()
    n_docs = con.sql(f"select count(*) from '{data}/{wiki}_text_features.parquet'").fetchall()[0][0]
    pair_vectors.VOCAB.update(zip(vocab["token"], range(len(vocab))))
    pair_vectors.NEEDED.update(needed)
    pair_vectors.IDF = np.log(n_docs / vocab["df"].to_numpy()).clip(min=0)
    n_terms = len(vocab)
    del vocab
    run(vector_file, [(p, out) for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)], args.workers, "building vectors")

    ids, weights, _ = load_vectors(out, n_terms)
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    unit = (diags(1 / np.sqrt(weights.multiply(weights).sum(axis=1).A1)) @ weights).tocsr()
    del weights

    # an article with no word in the vocabulary has no vector; a family left with too few children is dropped
    pairs = pairs[pairs["parent"].isin(row_of.index) & pairs["child"].isin(row_of.index)]
    pairs = pairs[pairs.groupby(["prop", "parent"])["child"].transform("size") >= args.least].reset_index(drop=True)
    pairs["family"] = pairs.groupby(["prop", "parent"], sort=False).ngroup()
    families = pairs.drop_duplicates("family").set_index("family")[["prop", "parent"]].sort_index()
    families["children"] = pairs.groupby("family").size()
    parent, child, family = row_of[pairs["parent"]].to_numpy(), row_of[pairs["child"]].to_numpy(), pairs["family"].to_numpy()
    family_parent = row_of[families["parent"]].to_numpy()

    member = csr_matrix((np.ones(len(pairs)), (family, child)), shape=(len(families), len(ids)))
    total = (member @ unit).tocsr()                     # one row per family: the sum of its children
    total_sq = total.multiply(total).sum(axis=1).A1
    parent_total = rowdot(unit[family_parent], total)
    parent_child = rowdot(unit[parent], unit[child])
    child_total = rowdot(unit[child], total[family])
    # the sum of the child's siblings is the family's sum less the child; its length follows from the three products
    rest = np.sqrt((total_sq[family] - 2 * child_total + 1).clip(min=1e-12))
    pairs["cosine"] = parent_child
    pairs["parent_siblings"] = (parent_total[family] - parent_child) / rest
    pairs["child_siblings"] = (child_total - 1) / rest
    pairs["siblings"] = pairs["parent_siblings"] - pairs["child_siblings"]

    families["cos_sum"] = parent_total / np.sqrt(total_sq)
    families["cos_child"] = pairs.groupby("family")["cosine"].median()
    families["cos_best_child"] = pairs.groupby("family")["cosine"].max()
    rng = np.random.default_rng(0)
    families["cos_random"] = np.nan
    for prop in PROPS:          # the same parents, each with the children of another parent of the relation
        mine = np.flatnonzero(families["prop"] == prop)
        if len(mine):
            other = rng.permutation(mine)
            families.loc[mine, "cos_random"] = rowdot(unit[family_parent[mine]], total[other]) / np.sqrt(total_sq[other])
    families.to_parquet(data / f"{wiki}_families.parquet")

    side = con.sql(f"""
        select x.prop, x.parent, x.child, op.in_deg as parent_in, oc.in_deg as child_in, tp.n_tokens as parent_words, tc.n_tokens as child_words
        from {JOINS}
    """).df()
    pairs = pairs.merge(side, on=["prop", "parent", "child"])
    pairs.to_parquet(data / f"{wiki}_family_vectors.parquet", index=False)
    pairs["relation"], families["relation"] = pairs["prop"].map(PROPS), families["prop"].map(PROPS)
    by, by_family = pairs.groupby("relation", sort=False), families.groupby("relation", sort=False)

    print("\n== the families ==\n")
    print(pd.DataFrame({"families": by_family.size(), "pairs": by.size(), "children, median": by_family["children"].median(),
                        f"families with {args.most} children kept": by_family["children"].apply(lambda s: (s == args.most).sum())}).to_string())

    print("\n== likeness: cosine between the parent and the sum of its children (medians over families) ==\n")
    print(pd.DataFrame({
        "one child": by_family["cos_child"].median(), "sum of its children": by_family["cos_sum"].median(),
        "sum of another parent's children": by_family["cos_random"].median(),
        "own sum the more alike %": by_family.apply(lambda g: score((g["cos_sum"] - g["cos_random"]).to_numpy()), include_groups=False),
        "sum beats its middle child %": by_family.apply(lambda g: score((g["cos_sum"] - g["cos_child"]).to_numpy()), include_groups=False),
        "sum beats every child %": by_family.apply(lambda g: score((g["cos_sum"] - g["cos_best_child"]).to_numpy()), include_groups=False),
        "child and sum of its siblings": by["child_siblings"].median(),
    }).round(3).to_string())

    print("\n== the same cosine by the number of children summed (median; families in brackets) ==\n")
    labels = [f"{int(a)} to {int(b) - 1}" if np.isfinite(b) else f"{int(a)} or more" for a, b in zip(SIZES, SIZES[1:])]
    cell = families.groupby(["relation", pd.cut(families["children"], SIZES, right=False, labels=labels)], observed=True, sort=False)["cos_sum"].agg(["median", "size"])
    print(cell.apply(lambda r: f"{r['median']:.3f} ({int(r['size']):,})", axis=1).unstack().reindex(columns=labels).to_string())

    print("\n== direction: score (correct % + half of tied %; a coin scores 50) ==\n")
    measures = {"lead in-links": pairs["parent_in"] - pairs["child_in"], "longer article": pairs["parent_words"] - pairs["child_words"],
                "closer to the child's siblings": pairs["siblings"]}
    wrong = (pairs["parent_in"] < pairs["child_in"]).to_numpy()
    shorter = (pairs["parent_words"] < pairs["child_words"]).to_numpy()
    for label, mask in [("all pairs", None), ("pairs that lead in-links gets wrong", wrong), ("pairs where the parent is the shorter article", shorter)]:
        table = {name: {relation: score(values.to_numpy(dtype=float)[index], None if mask is None else mask[index])
                        for relation, index in by.indices.items()} for name, values in measures.items()}
        print(f"{label}\n")
        print(pd.DataFrame(table).T.round(1).to_string() + "\n")
    print("pairs in each group\n")
    print(pd.DataFrame({"all pairs": by.size(), "lead in-links wrong": pairs.groupby("relation", sort=False)["parent"].apply(lambda s: wrong[s.index].sum()),
                        "parent shorter": pairs.groupby("relation", sort=False)["parent"].apply(lambda s: shorter[s.index].sum())}).T.to_string() + "\n")

    # the fits of Test 7 and after, with the new measure added
    old = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    fitted = differences(con, old).merge(containment(con).drop(columns="inlinks_wrong"), on=["prop", "parent", "child"])
    fitted = fitted.merge(pairs[["prop", "parent", "child", "siblings"]], on=["prop", "parent", "child"])
    names = old + ["contains", "siblings"]
    fits = {"lead in-links": ["in_deg"], "lead in-links + siblings": ["in_deg", "siblings"], "lead in-links + length": ["in_deg", "n_tokens"],
            "lead in-links + length + siblings": ["in_deg", "n_tokens", "siblings"], "everything before": old + ["contains"],
            "everything before + siblings": names}
    print("== fitted combinations: % of held-out pairs put the right way round, and the gain over lead in-links ==")
    for prop, relation in PROPS.items():
        part = fitted[fitted["prop"] == prop]
        if len(part) < 1000:
            continue
        x, y, held = design(part, names)
        print(f"\n{relation}: {int((~held).sum()):,} pairs to fit, {int(held.sum()):,} held out")
        alone = None
        for label, used in fits.items():
            cols = [names.index(n) for n in used]
            w = fit(x[~held][:, cols], y[~held])
            got = 100 * ((x[held][:, cols] @ w > 0) == (y[held] > 0)).mean()
            alone = got if alone is None else alone
            print(f"  {label:<36} {got:5.1f}   {got - alone:+5.1f}")
            if label == "lead in-links + length + siblings":
                print("    weights (parent minus child, in standard deviations): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(used, w)))


if __name__ == "__main__":
    main()

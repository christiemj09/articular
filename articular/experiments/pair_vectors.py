"""Compare the tf-idf vectors of a parent article and its child: are they alike, and does the text say which is which?

    uv run python -m articular.experiments.pair_vectors data/enwiki/*.bz2 --wiki enwiki --workers 8

Reads the dump files once, data/<wiki>_text/vocab.parquet (articular/pipeline/text_features.py),
data/<wiki>_title_features.parquet and what articular/experiments/score_text_features.py reads.

Test 7 summed each article's vocabulary up in one number (average idf and the like). Here each
article is a vector with one weight per word, (1 + ln count) x idf, over the words found in three or
more articles, and two articles are compared by the cosine of the angle between their vectors: 1 for
the same mix of words, 0 for no word in common.

Pairs are those of articular/experiments/score_text_features.py (child, parent, relation, from Wikidata), at most
--cap for each relation, chosen by a hash. The cosine is the same whichever way round a pair is
taken, so it cannot say which article is the more general. It is reported as a measure of likeness,
beside the cosine of the same children with parents drawn at random from the same relation. Three
measures that do have a direction are then scored as in Test 7:

  covers          share of the child's weight that lies on words the parent also uses, minus the
                  share of the parent's weight on words the child uses
  top_covers      the same with each article's twenty heaviest words only: the share of the child's
                  twenty found anywhere in the parent, minus the share of the parent's twenty found
                  anywhere in the child
  centroid_cos    cosine between the article and the sum of all articles' vectors (one number for
                  each article: how close its mix of words is to that of the whole encyclopedia)

Writes data/<wiki>_pair_vectors.parquet (one row per pair) and data/<wiki>_centroid_cos.parquet (the
articles in the pairs). The vectors are kept in data/<wiki>_text/vectors/; finished files are skipped
on a rerun with the same pairs.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from articular.pipeline.extract_leads import iter_pages
from articular.experiments.score_text_features import COUNTS, JOINS, PROPS, design, differences, fit, open_tables
from articular.experiments.score_title_features import OTHER_TEXT, TITLE, add_titles, containment
from articular.pipeline.text_features import article_words, run

TOP = 20
CHUNK = 5000        # pairs compared at a time; a parent in many pairs has its vector repeated in each
VECTOR_SCHEMA = pa.schema([("page_id", pa.int64()), ("terms", pa.large_list(pa.int32())), ("weights", pa.large_list(pa.float32()))])
VOCAB, NEEDED = {}, set()   # word -> row in IDF; the articles whose vectors are kept. Inherited by the workers
IDF = None
NEW = ["covers", "top_covers", "centroid_cos"]


def vector_file(job):
    """One dump file: the vectors of the articles in the pairs, and the sum of every article's unit vector."""
    path, out = job
    name = path.name.removesuffix(".bz2").removesuffix(".xml")
    final, total = out / "vectors" / f"{name}.parquet", out / "vectors" / f"{name}.npy"
    if total.exists():
        return name
    rows, summed = {column: [] for column in VECTOR_SCHEMA.names}, np.zeros(len(IDF))
    for page_id, _, redirect, text in iter_pages(path):
        if redirect:
            continue
        lead, rest = article_words(text)
        counts = Counter(lead)
        counts.update(rest)
        terms = np.fromiter((VOCAB.get(w, -1) for w in counts), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        terms, tf = terms[terms >= 0], tf[terms >= 0]
        weight = (1 + np.log(tf)) * IDF[terms]
        if not weight.any():
            continue
        summed[terms] += weight / math.sqrt(weight @ weight)
        if page_id in NEEDED:
            order = np.argsort(-weight, kind="stable")          # heaviest words first: the first TOP are the top words
            rows["page_id"].append(page_id)
            rows["terms"].append(terms[order].astype(np.int32))
            rows["weights"].append(weight[order].astype(np.float32))
    pq.write_table(pa.table(rows, schema=VECTOR_SCHEMA), final)
    np.save(total, summed)          # written last: marks the file as done
    return name


def load_vectors(out, n_terms):
    """(page ids, matrix of weights, matrix marking each article's TOP heaviest words), one row per article."""
    table = pa.concat_tables([pq.read_table(p) for p in sorted((out / "vectors").glob("*.parquet"))]).combine_chunks()
    ids = table["page_id"].to_numpy()
    terms, weights = table["terms"].chunk(0), table["weights"].chunk(0)
    indptr = terms.offsets.to_numpy()
    indices, data = terms.values.to_numpy(), weights.values.to_numpy().astype(np.float64)
    place = np.arange(len(indices)) - np.repeat(indptr[:-1], np.diff(indptr))          # 0 for an article's heaviest word
    top = place < TOP
    row = np.repeat(np.arange(len(ids)), np.diff(indptr))
    shape = (len(ids), n_terms)
    return ids, csr_matrix((data, indices, indptr), shape=shape), csr_matrix((np.ones(top.sum()), (row[top], indices[top])), shape=shape)


def compare(weights, tops, parent, child):
    """cosine, covers and top_covers for pairs given as rows of the matrices."""
    out = {"cosine": [], "covers": [], "top_covers": []}
    for start in range(0, len(parent), CHUNK):
        p, c = weights[parent[start:start + CHUNK]], weights[child[start:start + CHUNK]]
        tp, tc = tops[parent[start:start + CHUNK]], tops[child[start:start + CHUNK]]
        length = np.sqrt(p.multiply(p).sum(axis=1).A1 * c.multiply(c).sum(axis=1).A1)
        out["cosine"].append(p.multiply(c).sum(axis=1).A1 / length)
        in_p, in_c = p.sign(), c.sign()
        out["covers"].append(c.multiply(in_p).sum(axis=1).A1 / c.sum(axis=1).A1 - p.multiply(in_c).sum(axis=1).A1 / p.sum(axis=1).A1)
        out["top_covers"].append(tc.multiply(in_p).sum(axis=1).A1 / tc.sum(axis=1).A1 - tp.multiply(in_c).sum(axis=1).A1 / tp.sum(axis=1).A1)
    return {k: np.concatenate(v) for k, v in out.items()}


def score(values, wrong=None):
    """% of pairs where the value is positive, ties counted as half; a coin scores 50."""
    values = values[~np.isnan(values)] if wrong is None else values[wrong & ~np.isnan(values)]
    return 100 * ((values > 0).mean() + (values == 0).mean() / 2) if len(values) else np.nan


def main():
    global IDF
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="pages-articles .xml.bz2 files")
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--statements", type=Path, default=None)
    parser.add_argument("--page-props", type=Path, default=None)
    parser.add_argument("--cap", type=int, default=150_000, help="pairs kept for each relation")
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    statements = args.statements or data / "wd" / "statements.tsv.gz"
    page_props = args.page_props or data / "wd" / f"{wiki}-page_props.sql.gz"
    out = data / f"{wiki}_text"
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    con = open_tables(data, wiki, statements, page_props)
    add_titles(con, data, wiki)
    con.sql(f"""
        create or replace table pair as
        select x.child, x.parent, x.prop from {JOINS}
        qualify row_number() over (partition by x.prop order by hash(x.parent, x.child), x.parent, x.child) <= {args.cap}
    """)
    pairs = con.sql("select prop, parent, child from pair order by prop, parent, child").df()
    NEEDED.update(pairs["parent"].tolist(), pairs["child"].tolist())
    print(f"{len(pairs):,} pairs, {len(NEEDED):,} articles in them")
    print(pairs["prop"].map(PROPS).value_counts().to_string())
    # vectors kept from a run with other pairs would lack some of these articles
    marker, chosen = out / "vectors" / "pairs.json", {"cap": args.cap, "pairs": len(pairs), "articles": len(NEEDED)}
    if marker.exists() and json.loads(marker.read_text()) != chosen:
        for p in (out / "vectors").glob("*.*"):
            p.unlink()
    marker.write_text(json.dumps(chosen))

    vocab = pq.read_table(out / "vocab.parquet", columns=["token", "df"]).to_pandas()
    n_docs = con.sql(f"select count(*) from '{data}/{wiki}_text_features.parquet'").fetchall()[0][0]
    VOCAB.update(zip(vocab["token"], range(len(vocab))))
    IDF = np.log(n_docs / vocab["df"].to_numpy()).clip(min=0)
    del vocab
    print(f"{len(IDF):,} words in the vocabulary")
    run(vector_file, [(p, out) for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)], args.workers, "building vectors")

    centroid = sum(np.load(p) for p in (out / "vectors").glob("*.npy"))
    ids, weights, tops = load_vectors(out, len(IDF))
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    print(f"{len(ids):,} vectors, {weights.nnz / len(ids):.0f} words each on average")
    lengths = np.sqrt(weights.multiply(weights).sum(axis=1).A1)
    centroid_cos = (weights @ centroid) / (lengths * math.sqrt(centroid @ centroid))
    pd.DataFrame({"page_id": ids, "centroid_cos": centroid_cos}).to_parquet(data / f"{wiki}_centroid_cos.parquet", index=False)
    con.sql(f"create or replace table t as select t.*, v.centroid_cos from t left join '{data}/{wiki}_centroid_cos.parquet' v using (page_id)")

    # an article with no word in the vocabulary has no vector; its pairs are left out
    pairs = pairs[pairs["parent"].isin(row_of.index) & pairs["child"].isin(row_of.index)].reset_index(drop=True)
    parent, child = row_of[pairs["parent"]].to_numpy(), row_of[pairs["child"]].to_numpy()
    for name, values in compare(weights, tops, parent, child).items():
        pairs[name] = values
    rng = np.random.default_rng(0)
    pairs["cosine_random"] = np.nan
    for prop in PROPS:          # the same children, each with a parent drawn from another pair of the relation
        mine = np.flatnonzero(pairs["prop"] == prop)
        if len(mine):
            pairs.loc[mine, "cosine_random"] = compare(weights, tops, parent[rng.permutation(mine)], child[mine])["cosine"]
    side = con.sql(f"""
        select x.prop, x.parent, x.child, op.in_deg as parent_in, oc.in_deg as child_in, tp.n_tokens as parent_words, tc.n_tokens as child_words,
               tp.centroid_cos - tc.centroid_cos as centroid_cos
        from {JOINS}
    """).df()
    pairs = pairs.merge(side, on=["prop", "parent", "child"])
    pairs.to_parquet(data / f"{wiki}_pair_vectors.parquet", index=False)
    pairs["relation"] = pairs["prop"].map(PROPS)
    wrong = (pairs["parent_in"] < pairs["child_in"]).to_numpy()
    by = pairs.groupby("relation", sort=False)

    print("\n== likeness: cosine between parent and child, and between the child and a parent drawn at random ==\n")
    print(pd.DataFrame({
        "pairs": by.size(), "median": by["cosine"].median(), "quarter": by["cosine"].quantile(0.25), "three quarters": by["cosine"].quantile(0.75),
        "median, random parent": by["cosine_random"].median(),
        "true pair the more alike %": by.apply(lambda g: score((g["cosine"] - g["cosine_random"]).to_numpy()), include_groups=False),
    }).round(3).to_string())

    print("\n== does likeness say when lead in-links can be trusted? pairs by quarter of the cosine ==\n")
    rows = []
    for relation, part in by:
        quarter = pd.qcut(part["cosine"], 4, labels=["least alike", "second", "third", "most alike"])
        right = 100 * ((part["parent_in"] > part["child_in"]) + (part["parent_in"] == part["child_in"]) / 2)
        rows.append(right.groupby(quarter, observed=True).mean().rename(relation))
    print(pd.DataFrame(rows).round(1).to_string())

    print("\n== direction: score (correct % + half of tied %; a coin scores 50) ==\n")
    measures = {"lead in-links": pairs["parent_in"] - pairs["child_in"], "longer article": pairs["parent_words"] - pairs["child_words"],
                "covers": pairs["covers"], "top_covers": pairs["top_covers"], "closer to the centroid": pairs["centroid_cos"]}
    for label, mask in [("all pairs", None), ("pairs that lead in-links gets wrong", wrong)]:
        table = {name: {relation: score(values.to_numpy(dtype=float)[index], None if mask is None else mask[index])
                        for relation, index in by.indices.items()} for name, values in measures.items()}
        print(f"{label}\n")
        print(pd.DataFrame(table).T.round(1).to_string() + "\n")

    # the fits of Test 7 and after, with the new measures added
    old = ["in_deg", "kinds_below"] + OTHER_TEXT + [n for n in TITLE if n != "contains"]
    fitted = differences(con, old).merge(containment(con).drop(columns="inlinks_wrong"), on=["prop", "parent", "child"])
    fitted = fitted.merge(pairs[["prop", "parent", "child", "covers", "top_covers", "centroid_cos"]].fillna(0), on=["prop", "parent", "child"])
    names = old + ["contains"] + NEW
    fits = {"lead in-links": ["in_deg"], **{f"lead in-links + {n}": ["in_deg", n] for n in NEW},
            "lead in-links + length": ["in_deg", "n_tokens"], "lead in-links + length + covers": ["in_deg", "n_tokens", "covers"],
            "lead in-links + the three": ["in_deg"] + NEW, "everything before": old + ["contains"], "everything before + the three": names}
    assert set(COUNTS) >= {"in_deg", "n_tokens"}
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
            if label == "lead in-links + the three":
                print("    weights (parent minus child, in standard deviations): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(used, w)))


if __name__ == "__main__":
    main()

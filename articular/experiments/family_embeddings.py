"""Pilot: do the tests of articular/experiments/pair_vectors.py and articular/experiments/family_vectors.py on dense embeddings.

    uv run --group embed python -m articular.experiments.family_embeddings data/enwiki/*.bz2 --wiki enwiki --workers 8

Reads data/<wiki>_family_vectors.parquet (articular/experiments/family_vectors.py): its pairs, their tf-idf
measures, lead in-links and article lengths. The pairs of the first --families families of each
relation, in hash order, are used.

One pass over the dump files writes the opening of every article in that table, as plain text with
its capitals and punctuation, cut at --words words, to data/<wiki>_text/openings/ (finished files
are skipped on a rerun). Each model then embeds two inputs for the articles of the chosen families:
the title alone and the opening. Embeddings are kept as the model gives them, not scaled to length
1, in data/<wiki>_text/embeddings/, and reused on a rerun with the same families.

For each model and input:

  likeness      cosine of parent and child beside a random parent; cosine of the parent and the sum
                of its children beside one child and another parent's children; rank correlation
                of the cosine with the tf-idf cosine of the same pair
  direction     scored as before, a coin scoring 50: the longer vector (norm), closeness to the
                centroid of the embedded articles, and "siblings" as in articular/experiments/family_vectors.py
  probe         a weighted sum of the coordinates of (parent - child), fitted on four fifths of the
                parents of a relation and scored on the pairs of the rest, of the same relation and
                of each other relation; then fitted together with lead in-links, length and the
                tf-idf siblings measure
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from articular.pipeline.extract_leads import COMMENT, REF, iter_pages, strip_blocks
from articular.pipeline.extract_links import BLOCK_TAG, HEADING_LINE
from articular.experiments.pair_vectors import score
from articular.experiments.score_text_features import PROPS, fit
from articular.pipeline.text_features import EXTERNAL, LINK, MEDIA_LINE, NOT_TEXT_LINK, TAG, URL, run

import re

QUOTES = re.compile(r"'{2,}")
EMPTY_BRACKETS = re.compile(r"\(\s*[;,]?\s*\)")
OPENING_SCHEMA = pa.schema([("page_id", pa.int64()), ("title", pa.string()), ("opening", pa.string())])
NEEDED, WORDS = set(), [200]   # inherited by the workers
FOLDS = 4


def opening_text(wikitext, words):
    """The article's text before its first heading, without markup, cut at a number of words."""
    text = strip_blocks(BLOCK_TAG.sub("", REF.sub("", COMMENT.sub("", wikitext))))
    if m := HEADING_LINE.search(text):
        text = text[: m.start()]
    text = NOT_TEXT_LINK.sub(" ", MEDIA_LINE.sub(" ", text))
    text = TAG.sub(" ", URL.sub(" ", EXTERNAL.sub(r"\1", LINK.sub(r"\1", text))))
    text = EMPTY_BRACKETS.sub("", QUOTES.sub("", text).replace("&nbsp;", " "))
    return " ".join(text.split()[:words])


def opening_file(job):
    path, out = job
    final = out / f"{path.name.removesuffix('.bz2').removesuffix('.xml')}.parquet"
    if final.exists():
        return final.name
    rows = [(page_id, title, opening_text(text, WORDS[0])) for page_id, title, redirect, text in iter_pages(path)
            if not redirect and page_id in NEEDED]
    part = final.with_suffix(".part")
    pq.write_table(pa.table(list(zip(*rows)) if rows else [[], [], []], schema=OPENING_SCHEMA), part)
    part.rename(final)
    return final.name


def embed(model_name, texts, cache):
    """Embeddings as the model gives them, one row per text; (matrix, texts per second or None if read back)."""
    if cache.exists():
        return np.load(cache), None
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    start = time.time()
    matrix = model.encode(texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
    np.save(cache, matrix)
    return matrix, len(texts) / (time.time() - start)


def flipped(part, x):
    """As design() of articular/experiments/score_text_features.py: scaled, half the pairs turned round, a fifth of the parents held out."""
    scale = x.std(axis=0) + 1e-9
    turned = part["child"].to_numpy() % 2 == 1
    sign = np.where(turned, -1.0, 1.0)
    held = part["parent"].to_numpy() * 2654435761 % 5 == 0
    return x / scale * sign[:, None], (~turned).astype(float), held, sign


def right(x, w, y):
    return 100 * ((x @ w > 0) == (y > 0)).mean() if len(y) else np.nan


def analyse(pairs, families, vectors, label):
    norm = np.linalg.norm(vectors, axis=1)
    unit = vectors / norm[:, None]
    parent, child, family = pairs["p"].to_numpy(), pairs["c"].to_numpy(), pairs["family"].to_numpy()
    total = np.zeros((len(families), unit.shape[1]))
    np.add.at(total, family, unit[child])
    total_norm = np.linalg.norm(total, axis=1)
    family_parent = families["p"].to_numpy()
    parent_total = (unit[family_parent] * total).sum(axis=1)
    parent_child = (unit[parent] * unit[child]).sum(axis=1)
    child_total = (unit[child] * total[family]).sum(axis=1)
    rest = np.sqrt((total_norm[family] ** 2 - 2 * child_total + 1).clip(min=1e-12))
    centre = unit.mean(axis=0)
    centre_cos = unit @ centre / np.linalg.norm(centre)
    rng = np.random.default_rng(0)
    pairs = pairs.assign(cos=parent_child, cos_random=np.nan, siblings_e=(parent_total[family] - parent_child - child_total + 1) / rest,
                         norm=norm[parent] - norm[child], centre=centre_cos[parent] - centre_cos[child])
    families = families.assign(cos_sum=parent_total / total_norm, cos_child=pairs.groupby("family")["cos"].median(),
                               cos_best=pairs.groupby("family")["cos"].max(), cos_other=np.nan)
    for prop in PROPS:
        mine = np.flatnonzero(pairs["prop"] == prop)
        pairs.loc[mine, "cos_random"] = (unit[parent[rng.permutation(mine)]] * unit[child[mine]]).sum(axis=1)
        mine = np.flatnonzero(families["prop"] == prop)
        other = rng.permutation(mine)
        families.loc[mine, "cos_other"] = (unit[family_parent[mine]] * total[other]).sum(axis=1) / total_norm[other]
    by, by_family = pairs.groupby("relation", sort=False), families.groupby("relation", sort=False)

    print(f"\n\n######## {label} ########")
    print("\n== likeness (medians) ==\n")
    print(pd.DataFrame({
        "parent and child": by["cos"].median(), "random parent": by["cos_random"].median(),
        "true pair the more alike %": by.apply(lambda g: score((g["cos"] - g["cos_random"]).to_numpy()), include_groups=False),
        "parent and sum of children": by_family["cos_sum"].median(), "sum of another's children": by_family["cos_other"].median(),
        "sum beats every child %": by_family.apply(lambda g: score((g["cos_sum"] - g["cos_best"]).to_numpy()), include_groups=False),
        "rank correlation with tf-idf cosine": by.apply(lambda g: spearmanr(g["cos"], g["cosine"]).statistic, include_groups=False),
    }).round(3).to_string())

    wrong = (pairs["parent_in"] < pairs["child_in"]).to_numpy()
    shorter = (pairs["parent_words"] < pairs["child_words"]).to_numpy()
    measures = {"lead in-links": pairs["parent_in"] - pairs["child_in"], "longer article": pairs["parent_words"] - pairs["child_words"],
                "siblings, tf-idf": pairs["siblings"], "siblings, embedding": pairs["siblings_e"],
                "longer vector": pairs["norm"], "closer to the centroid": pairs["centre"]}
    print("\n== direction: score (a coin scores 50) ==")
    for name, mask in [("all pairs", None), ("pairs that lead in-links gets wrong", wrong), ("pairs where the parent is the shorter article", shorter)]:
        table = {m: {relation: score(values.to_numpy(dtype=float)[index], None if mask is None else mask[index])
                     for relation, index in by.indices.items()} for m, values in measures.items()}
        print(f"\n{name}\n")
        print(pd.DataFrame(table).T.round(1).to_string())

    # the probe: weights on the coordinates of (parent - child)
    difference = unit[parent] - unit[child]
    sides = ["in_deg", "n_tokens", "siblings"]
    side = np.column_stack([np.log1p(pairs["parent_in"]) - np.log1p(pairs["child_in"]),
                            np.log1p(pairs["parent_words"]) - np.log1p(pairs["child_words"]), pairs["siblings"]])
    fitted, transfer, combined = {}, {}, {}
    for relation, index in by.indices.items():
        part = pairs.iloc[index]
        x, y, held, sign = flipped(part, difference[index])
        w = fit(x[~held], y[~held])
        fitted[relation] = (w, x[held], y[held], wrong[index][held])
        # the probe's score for a pair it was not fitted on: held-out pairs from w, the others from fits that leave their fold out
        fold = part["parent"].to_numpy() * 40503 % FOLDS
        probe = x @ w
        for k in range(FOLDS):
            inside = ~held & (fold == k)
            probe[inside] = x[inside] @ fit(x[~held & (fold != k)], y[~held & (fold != k)])
        z = np.column_stack([side[index] * sign[:, None], probe])
        z = z / (z[~held].std(axis=0) + 1e-9)
        names = sides + ["probe"]
        rows = {}
        for name, used in {"lead in-links": ["in_deg"], "+ length": ["in_deg", "n_tokens"], "+ probe": ["in_deg", "probe"],
                           "+ length + probe": ["in_deg", "n_tokens", "probe"], "+ length + tf-idf siblings": sides,
                           "+ length + tf-idf siblings + probe": names}.items():
            cols = [names.index(n) for n in used]
            rows[name] = right(z[held][:, cols], fit(z[~held][:, cols], y[~held]), y[held])
        combined[relation] = rows
    for trained, (w, *_) in fitted.items():
        transfer[trained] = {tested: right(x, w, y) for tested, (_, x, y, _) in fitted.items()}
    print("\n== probe: % of held-out pairs put the right way round. Rows: relation fitted on; columns: relation scored on ==\n")
    print(pd.DataFrame(transfer).T.round(1).to_string())
    print("\nthe same probe, own relation, on held-out pairs that lead in-links gets wrong (pairs in brackets)\n")
    print("  " + "; ".join(f"{relation} {right(x[bad], w, y[bad]):.1f} ({int(bad.sum())})" for relation, (w, x, y, bad) in fitted.items()))
    print("\n== fitted with the other measures: % of held-out pairs put the right way round ==\n")
    print(pd.DataFrame(combined).round(1).to_string())
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="pages-articles .xml.bz2 files")
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--families", type=int, default=1000, help="families used for each relation")
    parser.add_argument("--words", type=int, default=200, help="words of the opening that are embedded")
    parser.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-mpnet-base-v2"])
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    out = data / f"{wiki}_text"
    (out / "openings").mkdir(parents=True, exist_ok=True)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    pairs = pd.read_parquet(data / f"{wiki}_family_vectors.parquet")
    NEEDED.update(pairs["parent"].tolist(), pairs["child"].tolist())
    WORDS[0] = args.words
    run(opening_file, [(p, out / "openings") for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)], args.workers, "reading openings")
    openings = pd.concat([pd.read_parquet(p) for p in sorted((out / "openings").glob("*.parquet"))], ignore_index=True)
    lengths = openings["opening"].str.split().str.len()
    print(f"{len(openings):,} openings of {len(NEEDED):,} articles; median {lengths.median():.0f} words, {(lengths >= args.words).mean():.0%} cut at {args.words}, "
          f"{(lengths < 10).sum():,} under 10 words (left out)")
    openings = openings[lengths >= 10]

    # the first families of each relation in hash order, then only pairs whose two articles have an opening
    families = pairs.drop_duplicates(["prop", "parent"])[["prop", "parent"]]
    families = families[families["parent"].isin(openings["page_id"])]
    families = families.assign(order=families["parent"] * 2654435761 % 2**32).sort_values(["prop", "order"]).groupby("prop").head(args.families)
    pairs = pairs.drop(columns="family").merge(families[["prop", "parent"]], on=["prop", "parent"])
    pairs = pairs[pairs["child"].isin(openings["page_id"])]
    pairs = pairs[pairs.groupby(["prop", "parent"])["child"].transform("size") >= 3].reset_index(drop=True)
    pairs["family"] = pairs.groupby(["prop", "parent"], sort=False).ngroup()
    pairs["relation"] = pairs["prop"].map(PROPS)
    articles = openings[openings["page_id"].isin(set(pairs["parent"]) | set(pairs["child"]))].sort_values("page_id").reset_index(drop=True)
    row_of = pd.Series(np.arange(len(articles)), index=articles["page_id"])
    pairs["p"], pairs["c"] = row_of[pairs["parent"]].to_numpy(), row_of[pairs["child"]].to_numpy()
    families = pairs.drop_duplicates("family").set_index("family")[["prop", "relation", "parent", "p"]].sort_index()
    print(f"{len(pairs):,} pairs in {len(families):,} families, {len(articles):,} articles")
    print(pd.DataFrame({"families": families.groupby("relation", sort=False).size(), "pairs": pairs.groupby("relation", sort=False).size()}).to_string())

    tag = f"{args.families}x{args.words}_{len(articles)}"
    for model_name in args.models:
        for kind, texts in [("title", articles["title"].tolist()), ("opening", articles["opening"].tolist())]:
            cache = out / "embeddings" / f"{model_name.split('/')[-1]}_{kind}_{tag}.npy"
            vectors, rate = embed(model_name, texts, cache)
            speed = "read back" if rate is None else f"{rate:.0f} texts a second, {len(NEEDED) / rate / 3600:.1f} hours for all {len(NEEDED):,} articles at that rate"
            result = analyse(pairs, families, vectors.astype(np.float64), f"{model_name}, {kind} ({vectors.shape[1]} dimensions; {speed})")
            result.drop(columns=["p", "c"]).to_parquet(data / f"{wiki}_family_embeddings_{model_name.split('/')[-1]}_{kind}.parquet", index=False)


if __name__ == "__main__":
    main()

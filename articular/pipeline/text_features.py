"""Measure the vocabulary of every article: how common its words are, and how widely its name is used.

    uv run python -m articular.pipeline.text_features data/enwiki/*.bz2 --wiki enwiki --workers 8

Reads the dump files twice and data/<wiki>_links/pages/ (articular/pipeline/extract_links.py) for the titles.

The first pass counts, for every word, the number of articles it appears in (df) and how often it
appears in all (cf), and writes data/<wiki>_text/vocab.parquet. The second pass writes
data/<wiki>_text_features.parquet, one row per article:

  n_tokens, n_types   words and distinct words in the prose (templates, tables, references and the
                      closing sections such as "See also" are left out)
  lead_tokens         words before the first heading
  mean_idf            average over the article's words of ln(articles / df): low when the words are common
  mean_idf_types      the same average over distinct words
  share_rare_types    share of distinct words found in fewer than 1 article in 10,000
  top10_share         share of the article's tf-idf weight carried by its ten heaviest words
  types_200, mean_idf_200, kl_200
                      the first 200 words only, so that long and short articles can be compared:
                      distinct words, average idf, and the Kullback-Leibler divergence of their
                      word frequencies from those of the whole encyclopedia. Null under 200 words.
  title_idf_sum       sum of idf over the words of the title
  title_bm25          BM25 score of the article's own text for its title as a query
  name_df             articles, other than itself, whose prose contains the title as a run of words.
                      Titles with a bracketed qualifier are not counted (null), and an article whose
                      title is spelled like another's shares the count with it.

Finished files are skipped on a rerun.
"""

import argparse
import json
import math
import re
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from articular.pipeline.extract_leads import COMMENT, REF, iter_pages, strip_blocks
from articular.pipeline.extract_links import BLOCK_TAG, HEADING_LINE

CLOSING = re.compile(r"^==\s*(see also|references|external links|further reading|notes|sources|bibliography|citations|footnotes)\s*==\s*$", re.M | re.I)
MEDIA_LINE = re.compile(r"^\[\[(?:file|image):.*$", re.M | re.I)
NOT_TEXT_LINK = re.compile(r"\[\[(?:category|file|image|media):[^\]]*\]\]", re.I)
LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")
EXTERNAL = re.compile(r"\[https?://\S+\s*([^\]]*)\]")
URL = re.compile(r"https?://\S+")
TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[^\W_]+")
MIN_DF = 3          # rarer words are dropped from the vocabulary and treated as appearing in one article
WINDOW = 200
MAX_NAME_WORDS = 5
K1, B = 1.2, 0.75   # the usual BM25 settings

FEATURES_SCHEMA = pa.schema([
    ("page_id", pa.int64()), ("title", pa.string()), ("n_tokens", pa.int32()), ("n_types", pa.int32()),
    ("lead_tokens", pa.int32()), ("mean_idf", pa.float32()), ("mean_idf_types", pa.float32()),
    ("share_rare_types", pa.float32()), ("top10_share", pa.float32()), ("types_200", pa.int32()),
    ("mean_idf_200", pa.float32()), ("kl_200", pa.float32()), ("title_idf_sum", pa.float32()), ("title_bm25", pa.float32()),
])
VOCAB = {}   # word -> row in IDF and LOG_P; filled before the second pass and inherited by the workers
IDF = LOG_P = None
NAMES, PREFIXES = {}, set()
CORPUS = {}


def words(text):
    """Lower-case words of a piece of wikitext, with link targets, addresses and tags removed."""
    text = NOT_TEXT_LINK.sub(" ", MEDIA_LINE.sub(" ", text))
    text = TAG.sub(" ", URL.sub(" ", EXTERNAL.sub(r"\1", LINK.sub(r"\1", text))))
    return WORD.findall(text.replace("&nbsp;", " ").lower())


def article_words(wikitext):
    """(words of the lead, words of the rest), stopping at the closing sections."""
    text = strip_blocks(BLOCK_TAG.sub("", REF.sub("", COMMENT.sub("", wikitext))))
    if m := CLOSING.search(text):
        text = text[: m.start()]
    m = HEADING_LINE.search(text)
    return (words(text[: m.start()]), words(text[m.start():])) if m else (words(text), [])


def count_file(job):
    """First pass over one dump file: df and cf of every word."""
    path, out = job
    name = path.name.removesuffix(".bz2").removesuffix(".xml")
    final, totals = out / "df" / f"{name}.parquet", out / "df" / f"{name}.json"
    if totals.exists():
        return name
    df, cf, n_docs, n_tokens = Counter(), Counter(), 0, 0
    for _, _, redirect, text in iter_pages(path):
        if redirect:
            continue
        lead, rest = article_words(text)
        seen = Counter(lead)
        seen.update(rest)
        cf.update(seen)
        df.update(seen.keys())
        n_docs += 1
        n_tokens += len(lead) + len(rest)
    tokens = list(df)
    pq.write_table(pa.table({"token": tokens, "df": [df[t] for t in tokens], "cf": [cf[t] for t in tokens]}), final)
    totals.write_text(json.dumps({"n_docs": n_docs, "n_tokens": n_tokens}))  # written last: marks the file as done
    return name


def idf_of(counts):
    """(idf, log corpus probability, count) arrays for a Counter of words."""
    rows = np.fromiter((VOCAB.get(w, -1) for w in counts), dtype=np.int64, count=len(counts))
    return IDF[rows], LOG_P[rows], np.fromiter(counts.values(), dtype=np.float64, count=len(counts))


def names_in(tokens, own):
    """Titles of other articles that appear in the text as a run of words."""
    found, n = set(), len(tokens)
    for i, run in enumerate(tokens):
        j = i + 1
        while True:
            if run in NAMES:
                found.add(NAMES[run])
            if run not in PREFIXES or j >= n or j - i >= MAX_NAME_WORDS:
                break
            run = f"{run} {tokens[j]}"
            j += 1
    found.discard(own)
    return found


def measure_file(job):
    """Second pass over one dump file: the features of each article, and which names each one uses."""
    path, out = job
    name = path.name.removesuffix(".bz2").removesuffix(".xml")
    finals = [out / kind / f"{name}.parquet" for kind in ("features", "names")]
    if all(p.exists() for p in finals):
        return name
    n_docs, avg_len = CORPUS["n_docs"], CORPUS["n_tokens"] / CORPUS["n_docs"]
    rare = math.log(10_000)
    rows, used = {column: [] for column in FEATURES_SCHEMA.names}, Counter()
    for page_id, title, redirect, text in iter_pages(path):
        if redirect:
            continue
        lead, rest = article_words(text)
        tokens = lead + rest
        counts = Counter(tokens)
        idf, _, tf = idf_of(counts)
        n = len(tokens)
        weight = np.sort(tf * idf)
        first = Counter(tokens[:WINDOW]) if n >= WINDOW else None
        if first:
            idf_w, log_p, tf_w = idf_of(first)
            p = tf_w / WINDOW
        title_words = set(WORD.findall(title.lower()))
        title_idf = {w: IDF[VOCAB.get(w, -1)] for w in title_words}
        bm25 = sum(math.log((n_docs - df + 0.5) / (df + 0.5) + 1) * counts[w] * (K1 + 1) / (counts[w] + K1 * (1 - B + B * n / avg_len))
                   for w, df in ((w, n_docs / math.exp(i)) for w, i in title_idf.items()) if counts[w])
        rows["page_id"].append(page_id)
        rows["title"].append(title)
        rows["n_tokens"].append(n)
        rows["n_types"].append(len(counts))
        rows["lead_tokens"].append(len(lead))
        rows["mean_idf"].append(float(tf @ idf / n) if n else None)
        rows["mean_idf_types"].append(float(idf.mean()) if n else None)
        rows["share_rare_types"].append(float((idf > rare).mean()) if n else None)
        rows["top10_share"].append(float(weight[-10:].sum() / weight.sum()) if n and weight.sum() else None)
        rows["types_200"].append(len(first) if first else None)
        rows["mean_idf_200"].append(float(tf_w @ idf_w / WINDOW) if first else None)
        rows["kl_200"].append(float(p @ (np.log(p) - log_p)) if first else None)
        rows["title_idf_sum"].append(float(sum(title_idf.values())))
        rows["title_bm25"].append(float(bm25))
        used.update(names_in(tokens, NAMES.get(" ".join(WORD.findall(title.lower())), -1)))
    pq.write_table(pa.table({"name_id": list(used), "df": list(used.values())}), finals[1])
    pq.write_table(pa.table(rows, schema=FEATURES_SCHEMA), finals[0])
    return name


def run(function, jobs, workers, label):
    # forked, so that the workers of the second pass inherit the vocabulary and the names
    with get_context("fork").Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(function, jobs), total=len(jobs), unit="file", desc=label):
            pass


def main():
    global IDF, LOG_P
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="pages-articles .xml.bz2 files")
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    out = data / f"{wiki}_text"
    for kind in ("df", "features", "names"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    jobs = [(p, out) for p in sorted(args.inputs, key=lambda p: -p.stat().st_size)]

    run(count_file, jobs, args.workers, "counting words")
    for key in ("n_docs", "n_tokens"):
        CORPUS[key] = sum(json.loads(p.read_text())[key] for p in (out / "df").glob("*.json"))
    con = duckdb.connect()
    con.sql(f"""
        copy (select token, sum(df)::bigint as df, sum(cf)::bigint as cf from read_parquet('{out}/df/*.parquet')
              group by 1 having sum(df) >= {MIN_DF})
        to '{out}/vocab.parquet' (format parquet)
    """)
    vocab = pq.read_table(out / "vocab.parquet").to_pandas()
    print(f"{CORPUS['n_docs']:,} articles, {CORPUS['n_tokens']:,} words, {len(vocab):,} distinct words in {MIN_DF} or more articles")
    VOCAB.update(zip(vocab["token"], range(len(vocab))))
    # the last row stands for every word left out of the vocabulary
    IDF = np.append(np.log(CORPUS["n_docs"] / vocab["df"].to_numpy()), math.log(CORPUS["n_docs"]))
    LOG_P = np.append(np.log(vocab["cf"].to_numpy() / CORPUS["n_tokens"]), -math.log(CORPUS["n_tokens"]))
    del vocab

    titles = con.sql(f"""
        select page_id, title from read_parquet('{data}/{wiki}_links/pages/*.parquet')
        where redirect_target is null and not is_disambig and title not like '%(%' order by page_id
    """).fetchall()
    for page_id, title in titles:
        parts = WORD.findall(title.lower())
        if 0 < len(parts) <= MAX_NAME_WORDS and NAMES.setdefault(" ".join(parts), page_id) == page_id:
            PREFIXES.update(" ".join(parts[:k]) for k in range(1, len(parts)))
    print(f"{len(NAMES):,} names to look for")

    run(measure_file, jobs, args.workers, "measuring articles")
    final = data / f"{wiki}_text_features.parquet"
    con.execute("create table name as select * from (select unnest($1) as name, unnest($2) as name_id)", [list(NAMES), list(NAMES.values())])
    con.sql(f"""
        create table used as select name_id, sum(df)::bigint as name_df from read_parquet('{out}/names/*.parquet') group by 1;
        create table spelled as
        select f.page_id, array_to_string(regexp_extract_all(lower(f.title), '[^\\W_]+'), ' ') as name
        from read_parquet('{out}/features/*.parquet') f where f.title not like '%(%';
        copy (
            select f.*, case when s.page_id is null or n.name_id is null then null else coalesce(u.name_df, 0) end as name_df
            from read_parquet('{out}/features/*.parquet') f
            left join spelled s using (page_id) left join name n using (name) left join used u using (name_id)
        ) to '{final}' (format parquet)
    """)
    print(f"{final}: {con.sql(f"select count(*) from '{final}'").fetchall()[0][0]:,} articles")
    print(con.sql(f"""
        select title, n_tokens, n_types, types_200, round(mean_idf, 2) as mean_idf, round(mean_idf_200, 2) as mean_idf_200,
               round(kl_200, 2) as kl_200, round(top10_share, 2) as top10, round(title_bm25, 1) as title_bm25, name_df
        from '{final}' where title in ('Animal', 'Dog', 'Poodle', 'Science', 'Physics', 'Quantum mechanics', 'France', 'Paris',
                                       'Eiffel Tower', 'Music', 'Jazz', 'The Beatles', 'Mathematics', 'Triangle', 'Albert Einstein')
        order by name_df desc
    """).df().to_string(index=False))


if __name__ == "__main__":
    main()

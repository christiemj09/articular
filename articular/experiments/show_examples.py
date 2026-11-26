"""Print worked examples for the write-up: what each rule picks for some articles, and the measures for some pairs.

    uv run python -m articular.experiments.show_examples --wiki enwiki

Reads data/<wiki>_links/pages/, data/<wiki>_scores_first.parquet, data/<wiki>_scores_kind.parquet,
data/<wiki>_specificity.parquet, data/<wiki>_scores.parquet and, when they exist,
data/<wiki>_text_features.parquet, data/wd/statements.tsv.gz and data/wd/<wiki>-page_props.sql.gz.

For each article in ARTICLES: its opening sentence, its first link and its kind link.
For each (general, specific) pair in PAIRS: the Wikidata statements that join the two, and each
measure for both articles.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from articular.experiments.score_rules import read_items

ARTICLES = ["Albert Einstein", "Taylor Swift", "Marie Curie", "Ludwig van Beethoven", "Mona Lisa", "Eiffel Tower",
            "Mount Everest", "Amazon River", "Tokyo", "Red-tailed hawk", "Poodle", "Toyota", "The Beatles", "Titanic",
            "Hamlet", "Photosynthesis", "Violin", "Chess", "Malaria", "Bicycle", "Harvard University", "Moby-Dick",
            "Quantum mechanics", "Battle of Stalingrad", "Gold", "Jazz", "Stoke-on-Trent", "Abraham Lincoln"]
PAIRS = [("Musical instrument", "Violin"), ("Vehicle", "Bicycle"), ("Disease", "Malaria"), ("Bird", "Penguin"),
         ("Novel", "Moby-Dick"), ("Mountain", "Mount Everest"), ("Philosopher", "Immanuel Kant"), ("Chemical element", "Gold"),
         ("Sport", "Tennis"), ("Emotion", "Fear"), ("Number", "Prime number"), ("Building", "Skyscraper"), ("Planet", "Mars"),
         ("Europe", "France"), ("Solar System", "Mars"), ("Mathematics", "Number theory"), ("Biology", "Genetics"),
         ("World War II", "Battle of Stalingrad"), ("Human body", "Heart"), ("United States", "Texas"),
         ("Computer science", "Algorithm"), ("Buddhism", "Mahayana"), ("Software engineering", "Software architecture"),
         ("Economics", "Inflation"), ("Roman Empire", "Julius Caesar"), ("Jazz", "Miles Davis"), ("Chess", "Checkmate"),
         ("Atom", "Electron"), ("Physics", "Quantum mechanics"), ("France", "Paris")]
PROPS = {279: "subclass of", 31: "instance of", 361: "part of", 131: "located in", 17: "country", 106: "occupation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    pd.set_option("display.width", 250, "display.max_colwidth", 120)

    con = duckdb.connect()
    con.execute("create table wanted as select unnest($1) as title", [sorted({*ARTICLES, *(t for pair in PAIRS for t in pair)})])
    con.sql(f"""
        create table page as
        select p.page_id, p.title, p.first_sentence from read_parquet('{data}/{wiki}_links/pages/*.parquet') p
        join wanted using (title) where p.redirect_target is null;
    """)
    picks = con.sql(f"""
        select p.title, left(p.first_sentence, 220) as sentence, f2.title as first_link, k2.title as kind_link, k.rule as kind_rule
        from page p
        left join '{data}/{wiki}_scores_first.parquet' f on f.page_id = p.page_id
        left join '{data}/{wiki}_scores_first.parquet' f2 on f2.page_id = f.first_link_page_id
        left join '{data}/{wiki}_scores_kind.parquet' k on k.page_id = p.page_id
        left join '{data}/{wiki}_scores_kind.parquet' k2 on k2.page_id = k.first_link_page_id
    """).df().set_index("title")
    print("== what each rule picks ==\n")
    for title in ARTICLES:
        if title in picks.index:
            row = picks.loc[title]
            print(f"{title}\n  sentence:   {row['sentence']}\n  first link: {row['first_link']}\n  kind link:  {row['kind_link']} ({row['kind_rule']})\n")

    text = data / f"{wiki}_text_features.parquet"
    extra = ", t.n_tokens, t.name_df, round(t.mean_idf_200, 2) as mean_idf_200, round(t.title_bm25, 1) as title_bm25" if text.exists() else ""
    values = con.sql(f"""
        select p.page_id, p.title, s.kinds_below, s.named_in, o.in_deg as lead_in_links, o.pages_through as first_link_traffic {extra}
        from page p left join '{data}/{wiki}_specificity.parquet' s using (page_id)
        left join '{data}/{wiki}_scores.parquet' o using (page_id)
        {f"left join '{text}' t using (page_id)" if text.exists() else ""}
    """).df().set_index("title")

    joined = {}
    statements, page_props = data / "wd" / "statements.tsv.gz", data / "wd" / f"{wiki}-page_props.sql.gz"
    if statements.exists() and page_props.exists():
        con.register("items_raw", read_items(page_props))
        found = con.sql(f"""
            with item as (select p.title, i.q from page p join items_raw i using (page_id))
            select c.title as child, p.title as parent, cast(substr(t.column1, 2) as int) as prop
            from read_csv('{statements}', delim = '\t', header = false, quote = '', escape = '', auto_detect = false,
                          columns = {{'column0': 'varchar', 'column1': 'varchar', 'column2': 'varchar'}}) t
            join item c on c.q = cast(substr(t.column0, 2) as bigint) join item p on p.q = cast(substr(t.column2, 2) as bigint)
        """).fetchall()
        for child, parent, prop in found:
            joined.setdefault((parent, child), []).append(PROPS.get(prop, f"P{prop}"))
    print("== pairs: general, then specific ==\n")
    for general, specific in PAIRS:
        if general in values.index and specific in values.index:
            stated = ", ".join(joined.get((general, specific), [])) or "no direct statement"
            print(f"{general} / {specific}   Wikidata: {specific} is [{stated}] {general}")
            print(values.loc[[general, specific]].drop(columns="page_id").to_string() + "\n")


if __name__ == "__main__":
    main()

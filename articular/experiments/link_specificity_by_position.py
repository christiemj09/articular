"""Where in an article do links point to more general articles, and where to more specific ones?

    uv run python -m articular.experiments.link_specificity_by_position --wiki enwiki

Reads data/<wiki>_links/ (articular/pipeline/extract_links.py) and data/<wiki>_specificity_score.parquet
(articular/pipeline/build_specificity_score.py).

Every link in the text of an article is taken, with the place it stands: first sentence, rest of
the lead, or a later section (its number and heading), in running text or in a list. Both ends get
the specificity rank (0 = most general, 100 = most specific). A link "points up" when its target
is more general than the article it stands in.

A specific article points up almost wherever it links, because nearly everything is more general
than it. So each table is split by how general the linking article is, and gives the target's
rank beside the share pointing up: the comparison that matters is between places in one column.

Prints: by place; by section number; by tenth of the article's links in order; by heading.
Writes data/<wiki>_link_position.csv (the tables, stacked).
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from articular.experiments.score_link_direction import resolve_tables

COLUMN = "no first-200-word vocabulary, others clipped"
# the linking article, by its own rank
SOURCES = "case when s.r < 1 then '1 most general 1%' when s.r < 10 then '2 next 9%' when s.r < 50 then '3 rest of top half' else '4 most specific half' end"
STATS = """count(*) as links, round(100 * avg(cast(d.r < s.r as int)), 1) as up_pct,
           round(approx_quantile(d.r, 0.5), 2) as target_median, round(approx_quantile(s.r - d.r, 0.5), 2) as gain_median"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)

    con = duckdb.connect()
    con.sql(f"set temp_directory = '{data}/duckdb_tmp'; set preserve_insertion_order = false;")
    con.sql("create table pair (child bigint, parent bigint, prop int)")      # resolve_tables wants it; no pairs are used here
    resolve_tables(con, data, wiki)
    con.sql(f"""
    create table score as select page_id, "{COLUMN}" as r from '{data}/{wiki}_specificity_score.parquet';
    create table l as
    select l.page_id as src, r.page_id as dst, l.sec_idx, lower(l.section) as section, l.pos, l.in_first_sentence, l.in_list
    from read_parquet('{data}/{wiki}_links/links/*.parquet') l join resolve r on r.title = l.target
    where l.page_id <> r.page_id;
    create table last as select src, max(pos) as last from l group by 1;
    """)
    joined = "l join score s on s.page_id = l.src join score d on d.page_id = l.dst"
    total = con.sql(f"select count(*) from {joined}").fetchall()[0][0]
    print(f"{total:,} links between scored articles\n")
    tables = []

    def show(title, key, sql, wide=True):
        table = con.sql(sql).df()
        tables.append(table.assign(table=title))
        print(f"\n== {title} ==\n")
        if not wide:
            print(table.to_string(index=False))
            return
        for value, label in (("up_pct", "% of links pointing up"), ("target_median", "median rank of the target (0 = most general)"), ("links", "links")):
            print(f"{label}\n")
            print(table.pivot(index=key, columns="source", values=value).to_string())
            print()

    place = """case when l.in_first_sentence then '1 first sentence' when l.sec_idx = 0 then '2 rest of the lead'
                    when not l.in_list then '3 later sections, running text' else '4 later sections, lists' end"""
    show("by place in the article, and by how general the linking article is", "place",
         f"select {place} as place, {SOURCES} as source, {STATS} from {joined} group by all order by 1, 2")
    show("by section number (0 = the lead)", "section_number",
         f"select least(l.sec_idx, 12) as section_number, {SOURCES} as source, {STATS} from {joined} group by all order by 1, 2")
    show("by tenth of the article's links, in order (articles with 20 links or more)", "tenth",
         f"""select least(cast(floor(10.0 * l.pos / (e.last + 1)) as int), 9) + 1 as tenth, {SOURCES} as source, {STATS}
             from {joined} join last e on e.src = l.src where e.last >= 19 group by all order by 1, 2""")

    floor = max(2000, total // 5000)
    headings = f"""select l.section as heading, {STATS}, round(100 * avg(cast(l.in_list as int)), 0) as in_lists_pct,
                          round(approx_quantile(s.r, 0.5), 1) as source_median
                   from {joined} where l.sec_idx > 0 group by all having count(*) >= {floor}"""
    show(f"headings whose links point up least often (headings with {floor:,} links or more)", "heading",
         f"select * from ({headings}) order by up_pct limit 40", wide=False)
    show("headings whose links point up most often", "heading", f"select * from ({headings}) order by up_pct desc limit 25", wide=False)
    show("the commonest headings", "heading", f"select * from ({headings}) order by links desc limit 40", wide=False)

    # the same headings for general articles only, where pointing up is not the default
    general = f"""select l.section as heading, {STATS}, round(100 * avg(cast(l.in_list as int)), 0) as in_lists_pct
                  from {joined} where l.sec_idx > 0 and s.r < 10 group by all having count(*) >= {max(500, floor // 10)}"""
    show("linking article in the most general 10%: headings whose links point up least often", "heading",
         f"select * from ({general}) order by up_pct limit 40", wide=False)
    show("linking article in the most general 10%: headings whose links point up most often", "heading",
         f"select * from ({general}) order by up_pct desc limit 25", wide=False)
    pd.concat(tables).to_csv(data / f"{wiki}_link_position.csv", index=False)


if __name__ == "__main__":
    main()

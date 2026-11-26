"""Score link rules against Wikidata: does the link a rule picks say what kind of thing the article is?

    uv run python -m articular.experiments.score_rules --wiki enwiki

Reads
  data/<wiki>_links/pages/            articles and redirects (articular/pipeline/extract_links.py)
  data/<wiki>_scores_first.parquet    the first-link rule      (articular/pipeline/build_link_choice.py --rule first)
  data/<wiki>_scores_defining.parquet the old pattern rule     (articular/pipeline/build_link_choice.py --rule defining)
  data/<wiki>_scores_kind.parquet     the kind rule            (articular/pipeline/build_link_choice.py --rule kind)
  data/<wiki>_relations/              parser-labeled links    (articular/pipeline/extract_relations.py)
  data/wd/statements.tsv.gz           Wikidata statements      (scripts/fetch_wikidata.sh)
  data/wd/<wiki>-page_props.sql.gz    article -> Wikidata item (scripts/fetch_wikidata.sh)

Every link "article -> target" gets one outcome, the first that applies:
  kind, 1 step      Wikidata says the article is an instance or subclass of the target
  kind, 2-3 steps   the target is a superclass of one of those, that many steps up in total
  kind, 4+ steps    the same, further up (classes this far up are often very broad)
  occupation        the target is the article's occupation (people are "instance of human" in Wikidata)
  part of, located in, country
  reversed          the target is a kind of the article
  unrelated         none of the above
Only links where both ends have a Wikidata item and the article has an instance-of or subclass-of
statement are scored.

Writes data/<wiki>_rule_scores.parquet (rule, page_id, target_page_id, outcome) and
data/<wiki>_rule_scores_summary.csv, and prints the summary with examples.
"""

import argparse
import gzip
import re
from pathlib import Path

import duckdb
import pyarrow as pa

ITEM = re.compile(rb"\((\d+),'wikibase_item','Q(\d+)'")
OUTCOMES = ["kind, 1 step", "kind, 2-3 steps", "kind, 4+ steps", "occupation", "part of", "located in",
            "country", "reversed", "unrelated"]
PARSER_RELATIONS = ["kind", "kind_also", "kind_part", "qualifier", "prep", "clause", "field", "aside", "other"]


def read_items(path):
    """page_id and Wikidata item number for every page that has one, from the page_props SQL dump."""
    pages, items = [], []
    with gzip.open(path, "rb") as dump:
        for line in dump:
            for m in ITEM.finditer(line):
                pages.append(int(m.group(1)))
                items.append(int(m.group(2)))
    return pa.table({"page_id": pa.array(pages, pa.int64()), "q": pa.array(items, pa.int64())})


def steps_up(name, src, dst):
    """SQL for a table of scored links whose `dst` item is a class of, or above a class of, the `src` item."""
    return f"""
    create table {name} as
    select p.rule, p.page_id, p.dst, min(u.depth) + 1 as steps
    from linked p
    join st s on s.item = p.{src} and s.prop in (31, 279)
    join up u on u.start = s.value and u.anc = p.{dst}
    group by all;
    """


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--statements", type=Path, default=None)
    parser.add_argument("--page-props", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=12, help="how far up the subclass tree to follow")
    parser.add_argument("--examples", type=int, default=25)
    args = parser.parse_args()
    data, wiki = args.data, args.wiki
    statements = args.statements or data / "wd" / "statements.tsv.gz"
    page_props = args.page_props or data / "wd" / f"{wiki}-page_props.sql.gz"

    con = duckdb.connect()
    con.sql(f"set temp_directory = '{data}/duckdb_tmp'; set preserve_insertion_order = false;")

    con.sql(f"""
    create table pages as
    select page_id, title, redirect_target, is_disambig from read_parquet('{data}/{wiki}_links/pages/*.parquet');
    create table resolve as
    select p.title, coalesce(a0.page_id, a1.page_id, a2.page_id) as page_id
    from pages p
    left join pages a0 on a0.title = p.title and a0.redirect_target is null
    left join pages a1 on a1.title = p.redirect_target and a1.redirect_target is null
    left join pages r on r.title = p.redirect_target
    left join pages a2 on a2.title = r.redirect_target and a2.redirect_target is null
    where coalesce(a0.page_id, a1.page_id, a2.page_id) is not null;
    create table article as select page_id, title from pages where redirect_target is null and not is_disambig;
    """)

    con.register("items_raw", read_items(page_props))
    con.sql("create table item as select i.page_id, i.q from items_raw i join article a using (page_id)")

    # keep the statements about articles, and every subclass statement (needed to walk up the tree)
    con.sql(f"""
    create table st as
    select item, prop, value from (
        select cast(substr(column0, 2) as bigint) as item, cast(substr(column1, 2) as int) as prop,
               cast(substr(column2, 2) as bigint) as value
        from read_csv('{statements}', delim = '\t', header = false, quote = '', escape = '', auto_detect = false,
                      columns = {{'column0': 'varchar', 'column1': 'varchar', 'column2': 'varchar'}})
    )
    where prop = 279 or item in (select q from item);
    create table kinded as select distinct item from st where prop in (31, 279) and item in (select q from item);
    """)

    # up(start, anc, depth): every class an article belongs to, with each of its superclasses
    con.sql("""
    create table up as
    select distinct value as start, value as anc, 0 as depth
    from st where prop in (31, 279) and item in (select q from item)
    """)
    for depth in range(1, args.max_steps):
        before = con.sql("select count(*) from up").fetchall()[0][0]
        con.sql(f"""
        insert into up
        select distinct u.start, e.value, {depth}
        from up u join st e on e.item = u.anc and e.prop = 279
        where u.depth = {depth - 1}
          and not exists (select 1 from up x where x.start = u.start and x.anc = e.value)
        """)
        after = con.sql("select count(*) from up").fetchall()[0][0]
        print(f"superclasses {depth} step(s) up: {after - before:,} new, {after:,} in all", flush=True)
        if after == before:
            break

    con.sql(f"""
    create table pairs as
    select distinct rule, page_id, dst from (
        select 'first link' as rule, page_id, first_link_page_id as dst
        from '{data}/{wiki}_scores_first.parquet' where first_link_page_id >= 0
        union all
        select 'pattern rule', page_id, first_link_page_id
        from '{data}/{wiki}_scores_defining.parquet' where rule = 'defining'
        union all
        select 'kind rule', page_id, first_link_page_id
        from '{data}/{wiki}_scores_kind.parquet' where rule in ('kind', 'kind_noun')
        union all
        select 'kind rule: ' || (case when rule = 'kind' then 'the link' else 'the noun' end), page_id, first_link_page_id
        from '{data}/{wiki}_scores_kind.parquet' where rule in ('kind', 'kind_noun')
        union all
        select 'parser: ' || l.relation, l.page_id, r.page_id
        from read_parquet('{data}/{wiki}_relations/*.parquet') l join resolve r on r.title = l.target
        where r.page_id <> l.page_id
    );
    create table linked as
    select p.rule, p.page_id, p.dst, a.q as qa, b.q as qb
    from pairs p join item a on a.page_id = p.page_id join item b on b.page_id = p.dst
    where a.q in (select item from kinded);
    """)
    con.sql(steps_up("fwd", "qa", "qb"))
    con.sql(steps_up("rev", "qb", "qa"))
    con.sql("""
    create table oth as
    select p.rule, p.page_id, p.dst,
           bool_or(s.prop = 106) as occupation, bool_or(s.prop = 361) as part_of,
           bool_or(s.prop = 131) as located_in, bool_or(s.prop = 17) as country
    from linked p join st s on s.item = p.qa and s.value = p.qb and s.prop in (106, 361, 131, 17)
    group by all;

    create table scored as
    select p.rule, p.page_id, p.dst as target_page_id,
           case when f.steps = 1 then 'kind, 1 step'
                when f.steps <= 3 then 'kind, 2-3 steps'
                when f.steps > 3 then 'kind, 4+ steps'
                when o.occupation then 'occupation'
                when o.part_of then 'part of'
                when o.located_in then 'located in'
                when o.country then 'country'
                when r.steps is not null then 'reversed'
                else 'unrelated' end as outcome
    from linked p
    left join fwd f using (rule, page_id, dst)
    left join rev r using (rule, page_id, dst)
    left join oth o using (rule, page_id, dst);
    """)
    con.sql(f"copy scored to '{data}/{wiki}_rule_scores.parquet' (format parquet)")

    n_articles, n_item, n_kinded = con.sql("""
        select (select count(*) from article), (select count(*) from item), (select count(*) from kinded)
    """).fetchall()[0]
    print(f"\n{n_articles:,} articles; {n_item:,} have a Wikidata item; "
          f"{n_kinded:,} have an instance-of or subclass-of statement")

    counts = con.sql("""
        select p.rule, count(*) as links, count(distinct p.page_id) as articles
        from pairs p group by 1
    """).df().set_index("rule")
    table = con.sql("select rule, outcome, count(*) as n from scored group by all").df()
    table = table.pivot(index="rule", columns="outcome", values="n").reindex(columns=OUTCOMES).fillna(0)
    scored = table.sum(axis=1)
    share = table.div(scored, axis=0).mul(100).round(1)
    share.insert(0, "kind, any", share[OUTCOMES[:3]].sum(axis=1).round(1))
    summary = counts.join(scored.rename("scored")).fillna({"scored": 0}).astype({"scored": int}).join(share)
    order = ["first link", "pattern rule", "kind rule", "kind rule: the link", "kind rule: the noun"] + [f"parser: {r}" for r in PARSER_RELATIONS]
    summary = summary.reindex([r for r in order if r in summary.index])
    summary.to_csv(data / f"{wiki}_rule_scores_summary.csv")
    print("\nlinks = links the rule supplies; scored = those that can be checked; the rest are % of scored\n")
    print(summary.to_string())

    # what share of all articles does each one-link rule get right?
    reach = con.sql(f"""
        select rule, round(100.0 * count(distinct page_id) filter (outcome like 'kind%') / {n_articles}, 1) as kind_any,
               round(100.0 * count(distinct page_id) filter (outcome = 'kind, 1 step') / {n_articles}, 1) as kind_1_step
        from scored where rule in ('first link', 'pattern rule', 'kind rule', 'parser: kind') group by 1 order by 1
    """).df()
    print("\n% of ALL articles for which the rule's link is confirmed as a kind\n")
    print(reach.to_string(index=False))

    # Wikidata files people as "human", species as "taxon", and a film's genre under another property, so it
    # cannot confirm "footballer", "moth" or "drama film". Score those groups apart from everything else.
    con.sql("""
    create table grp as
    select i.page_id,
           case when bool_or(s.value = 5) then 'human'
                when bool_or(s.value in (16521, 23038290)) then 'taxon'
                when bool_or(w.start is not null) then 'creative work'
                else 'everything else' end as grp
    from item i join st s on s.item = i.q and s.prop in (31, 279)
    left join (select distinct start from up where anc in (17537576, 386724)) w on w.start = s.value
    group by 1
    """)
    groups = con.sql("""
        select g.grp as article_is, c.rule, count(*) as scored,
               round(100.0 * count(*) filter (c.outcome like 'kind%') / count(*), 1) as kind_any,
               round(100.0 * count(*) filter (c.outcome = 'kind, 1 step') / count(*), 1) as kind_1_step,
               round(100.0 * count(*) filter (c.outcome = 'occupation') / count(*), 1) as occupation,
               round(100.0 * count(*) filter (c.outcome in ('part of', 'located in', 'country')) / count(*), 1) as place_or_part,
               round(100.0 * count(*) filter (c.outcome = 'unrelated') / count(*), 1) as unrelated
        from scored c join grp g using (page_id)
        where c.rule in ('first link', 'pattern rule', 'kind rule', 'kind rule: the link', 'kind rule: the noun', 'parser: kind')
        group by all order by 1, 2
    """).df()
    groups.to_csv(data / f"{wiki}_rule_scores_groups.csv", index=False)
    print("\n== the three one-link rules, by what Wikidata says the article is (% of scored) ==\n")
    print(con.sql("select grp as article_is, count(*) as articles from grp group by 1 order by 2 desc").df().to_string(index=False))
    print()
    print(groups.to_string(index=False))

    # where does each rule fail? outcomes split by the article's own Wikidata class (an article can have several)
    for rule in ["parser: kind", "first link"]:
        by_class = con.sql(f"""
            select coalesce(any_value(k.title), 'Q' || s.value) as article_is_a, count(*) as links,
                   round(100.0 * count(*) filter (c.outcome like 'kind%') / count(*), 1) as kind_any,
                   round(100.0 * count(*) filter (c.outcome = 'occupation') / count(*), 1) as occupation,
                   round(100.0 * count(*) filter (c.outcome = 'unrelated') / count(*), 1) as unrelated,
                   mode(t.title) filter (c.outcome = 'unrelated') as commonest_unrelated_link
            from scored c join item i on i.page_id = c.page_id
            join st s on s.item = i.q and s.prop = 31
            join article t on t.page_id = c.target_page_id
            left join item ki on ki.q = s.value left join article k on k.page_id = ki.page_id
            where c.rule = '{rule}' group by s.value order by links desc limit 40
        """).df()
        print(f"\n== {rule}: outcomes by the article's Wikidata class, 40 largest ==\n")
        print(by_class.to_string(index=False))

    for rule, outcome in [("parser: kind", "unrelated"), ("parser: kind", "reversed"), ("first link", "unrelated")]:
        examples = con.sql(f"""
            with sample as (
                select * from (select page_id, target_page_id from scored where rule = '{rule}' and outcome = '{outcome}')
                using sample reservoir({args.examples} rows) repeatable (1)     -- sampling runs before `where`, hence the subquery
            )
            select a.title as article, t.title as link,
                   (select string_agg(distinct k.title, '; ')
                    from item i join st s on s.item = i.q and s.prop in (31, 279)
                    join item ki on ki.q = s.value join article k on k.page_id = ki.page_id
                    where i.page_id = sample.page_id) as wikidata_says
            from sample join article a on a.page_id = sample.page_id join article t on t.page_id = sample.target_page_id
        """).df()
        print(f"\n== {rule}, {outcome}: examples (wikidata_says lists only classes that have an article) ==\n")
        print(examples.to_string(index=False))


if __name__ == "__main__":
    main()

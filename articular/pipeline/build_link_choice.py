"""Pick one outgoing link per article by a rule, as input for the map (articular/site/build_tiles.py).

    uv run python -m articular.pipeline.build_link_choice --wiki enwiki --rule defining

Reads data/<wiki>_links/ (from articular/pipeline/extract_links.py) and writes data/<wiki>_scores_<rule>.parquet
with page_id, title, is_disambig, first_link_page_id (-1 for none) and the rule that supplied it.

Rules:
  first      the first link in the lead that is not in parentheses or italics (the Philosophy game)
  defining   the link that says what kind of thing the article is: in "X is an American [[singer]]
             of [[pop music]]" it is Singer. Found by a text pattern, which misreads some sentences.
  kind       the same idea, read from the grammatical parse (articular/pipeline/extract_relations.py): the link on
             the kind noun, or the article titled with the noun and the words joined to it, longest
             first ("a professional ice hockey player" -> Ice hockey player; "a village in …" ->
             Village). A link whose target's title does not contain the noun gives way to the noun's
             own article: "[[Association football|footballer]]" -> Football player. The `rule`
             column says which was used: kind (the link) or kind_noun (the noun's article).

Where a rule finds nothing, the article falls back to the `first` rule, so the maps cover the same
articles; the `rule` column records that too.

Disambiguation pages get no link under either rule.
"""

import argparse
import re
from pathlib import Path

import duckdb

LINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
VERB_OR_PAREN = re.compile(r"[()]|\b(?:is|are|was|were)\b")
# kind nouns too vague to look up by title when nothing more specific was found
VAGUE = ("type", "kind", "form", "sort", "variety", "species", "specie", "subspecies", "subspecie", "genus", "breed",
         "style", "example", "member", "one", "part", "name", "term", "word")
STOP = re.compile(r"\b(of|in|on|at|from|by|for|with|that|which|who|whose|where|when|and|or|to|as|near|between)\b|[,;:(]")


def norm_title(target):
    t = target.split("#")[0].replace("_", " ").strip(" :")
    t = " ".join(t.split())
    return t[:1].upper() + t[1:]


def defining_link(sentence):
    """Title of the last link in the noun phrase after the first is/are/was/were, or None."""
    if not sentence:
        return None
    plain = LINK.sub(lambda m: " " * len(m.group()), sentence)     # look for the verb outside link text
    depth, verb = 0, None
    for m in VERB_OR_PAREN.finditer(plain):
        if m.group() == "(":
            depth += 1
        elif m.group() == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            verb = m
            break
    if verb is None:
        return None
    last, cursor = None, verb.end()
    for m in LINK.finditer(sentence, verb.end()):
        if STOP.search(plain, cursor, m.start()):
            break
        last, cursor = m, m.end()
    return norm_title(last.group(1)) if last else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--rule", choices=["first", "defining", "kind"], default="defining")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    links = args.data / f"{args.wiki}_links"

    con = duckdb.connect()
    con.sql(f"""
    create table pages as select * from read_parquet('{links}/pages/*.parquet');
    create table resolve as
    select p.title, coalesce(a0.page_id, a1.page_id, a2.page_id) as page_id
    from pages p
    left join pages a0 on a0.title = p.title and a0.redirect_target is null
    left join pages a1 on a1.title = p.redirect_target and a1.redirect_target is null
    left join pages r on r.title = p.redirect_target
    left join pages a2 on a2.title = r.redirect_target and a2.redirect_target is null
    where coalesce(a0.page_id, a1.page_id, a2.page_id) is not null;

    create table first_link as
    select l.page_id as src, arg_min(r.page_id, l.pos) as dst
    from read_parquet('{links}/links/*.parquet') l join resolve r on r.title = l.target
    where l.sec_idx = 0 and not l.in_paren and not l.italic and r.page_id <> l.page_id
    group by 1;
    """)
    if args.rule == "defining":
        sentences = con.sql("select page_id, first_sentence from pages where redirect_target is null and not is_disambig").df()
        sentences["target"] = [defining_link(s) for s in sentences["first_sentence"]]
        con.register("found", sentences[["page_id", "target"]].dropna())
        con.sql("""
        create table defining as
        select f.page_id as src, r.page_id as dst, 'defining' as how
        from found f join resolve r on r.title = f.target
        where r.page_id <> f.page_id
        """)
    elif args.rule == "kind":
        relations = f"read_parquet('{args.data}/{args.wiki}_relations/*.parquet')"
        con.sql(f"""
        create table stated as                       -- one row per article that states a kind
        select page_id as src, any_value(head) as head, any_value(phrase) as phrase,
               any_value(target) filter (relation = 'kind') as target
        from {relations} where relation in ('kind', 'kind_text') group by 1;

        create table by_link as
        select s.src, r.page_id as dst, contains(lower(t.title), s.head) as names_it
        from stated s join resolve r on r.title = s.target join pages t on t.page_id = r.page_id
        where r.page_id <> s.src and not t.is_disambig;

        create table by_noun as                      -- the longest ending of the phrase that is an article title
        select s.src, arg_max(r.page_id, w.n) as dst
        from stated s, lateral (
            select n, array_to_string(string_split(s.phrase, ' ')[-n:], ' ') as words from range(1, 5) t(n)
            where n <= len(string_split(s.phrase, ' '))
        ) w
        join resolve r on r.title = upper(w.words[1]) || w.words[2:]
        join pages t on t.page_id = r.page_id
        where r.page_id <> s.src and not t.is_disambig and not (w.n = 1 and s.head in {VAGUE})
          -- several words can hit a redirect to something else ("American politician" -> Politics of the
          -- United States, "British actor" -> List of British actors): the title must still name the noun
          and (w.n = 1 or (contains(lower(t.title), s.head) and t.title not like 'List%'))
        group by 1;

        create table defining as
        select coalesce(l.src, n.src) as src,
               case when l.dst is null or (not l.names_it and n.dst is not null) then n.dst else l.dst end as dst,
               case when l.dst is null or (not l.names_it and n.dst is not null) then 'kind_noun' else 'kind' end as how
        from by_link l full join by_noun n on n.src = l.src
        """)
    else:
        con.sql("create table defining as select src, dst, 'defining' as how from first_link where false")

    out = args.data / f"{args.wiki}_scores_{args.rule}.parquet"
    con.sql(f"""
    copy (
        select p.page_id, p.title, p.is_disambig,
               case when p.is_disambig then -1 else coalesce(d.dst, f.dst, -1) end as first_link_page_id,
               case when p.is_disambig then 'none' when d.dst is not null then d.how
                    when f.dst is not null then 'first' else 'none' end as rule
        from pages p left join defining d on d.src = p.page_id left join first_link f on f.src = p.page_id
        where p.redirect_target is null order by p.page_id
    ) to '{out}' (format parquet)
    """)
    print(con.sql(f"select rule, count(*) as articles from '{out}' group by 1 order by 2 desc").df().to_string(index=False))
    changed = con.sql(f"""
        select count(*) from '{out}' s join first_link f on f.src = s.page_id
        where s.rule <> 'first' and s.first_link_page_id <> f.dst
    """).fetchall()[0][0]
    print(f"{out}: {changed:,} articles point somewhere different from their first link")


if __name__ == "__main__":
    main()

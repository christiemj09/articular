# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Does "defining" decay with link position?
#
# Runs on the output of `articular/pipeline/extract_links.py` (every link in the article text, with its
# position and section). Set the `WIKI` environment variable; default `simplewiki`.
#
# The original premise: an article's opening anchors the topic in more general terms, so early
# links should point "up". Two questions:
#
# 1. **Is the first link the defining link?** The defining link is taken from the wording of the
#    first sentence: in "X is a [[Y]] of [[Z]]", it is Y.
# 2. **How do links change with position?** Without an outside source of truth, three proxies:
#    does the target have more incoming links than the source, does the target link back, and
#    how many incoming links does the target have.
#
# These are proxies, not ground truth. Wikidata would give that, and is not used here.
# No network access is needed.

# %%
import os
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WIKI = os.environ.get("WIKI", "simplewiki")
DATA = Path("../data") if Path.cwd().name == "notebooks" else Path("data")
LINKS = DATA / f"{WIKI}_links"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 11, "lines.linewidth": 2,
})
pd.set_option("display.max_rows", 60, "display.width", 180, "display.max_colwidth", 110)

# %% [markdown]
# ## 1. Resolve links
#
# Targets are matched to articles through redirects. Self-links and repeat links to the same
# target are dropped (the first occurrence is kept), and `ordinal` renumbers what remains.
# Disambiguation pages are left out as sources.

# %%
con = duckdb.connect()
con.sql(f"""
create table pages as select * from read_parquet('{LINKS}/pages/*.parquet');
create table articles as select page_id, title, first_sentence from pages where redirect_target is null and not is_disambig;
create table resolve as
select p.title, coalesce(a0.page_id, a1.page_id, a2.page_id) as page_id
from pages p
left join pages a0 on a0.title = p.title and a0.redirect_target is null
left join pages a1 on a1.title = p.redirect_target and a1.redirect_target is null
left join pages r on r.title = p.redirect_target
left join pages a2 on a2.title = r.redirect_target and a2.redirect_target is null
where coalesce(a0.page_id, a1.page_id, a2.page_id) is not null;

create table links as
with raw as (
    select l.page_id as src, r.page_id as dst, l.pos, l.sec_idx, l.section, l.word, l.word_abs,
           l.in_first_sentence, l.in_paren, l.italic, l.in_list, l.target
    from read_parquet('{LINKS}/links/*.parquet') l
    join articles a on a.page_id = l.page_id
    join resolve r on r.title = l.target
    where r.page_id <> l.page_id
)
select *, row_number() over (partition by src order by pos) as ordinal
from (select * from raw qualify row_number() over (partition by src, dst order by pos) = 1);

create table indeg as select dst as page_id, count(*) as n from links group by 1;
create table pairs as select distinct src, dst from links;
""")
con.sql("""
select (select count(*) from articles) as articles, (select count(*) from links) as links,
       (select count(*) from links where sec_idx = 0) as in_lead,
       (select count(*) from links where in_first_sentence) as in_first_sentence
""").df().T.rename(columns={0: "count"})

# %% [markdown]
# ## 2. The defining link, from the wording of the first sentence
#
# After the first "is / are / was / were", links are read until the first word that ends the
# noun phrase (of, in, from, that, and, a comma…). The last link before that point is the
# defining link: in "is an American [[singer]] and [[songwriter]]" it is *singer*; in "is the
# [[capital city]] of [[France]]" it is *capital city*.

# %%
COPULA = re.compile(r"\b(is|are|was|were)\b")
LINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
STOP = re.compile(r"\b(of|in|on|at|from|by|for|with|that|which|who|whose|where|when|and|or|to|as|near|between)\b|[,;:(]")


def norm_title(target):
    t = target.split("#")[0].replace("_", " ").strip(" :")
    t = " ".join(t.split())
    return t[:1].upper() + t[1:]


def defining_link(sentence):
    """(target title, how it was found) for the link that says what kind of thing this is."""
    if not sentence:
        return None, "no sentence"
    plain = LINK.sub(lambda m: " " * len(m.group()), sentence)     # look for the verb outside link text
    depth, verb = 0, None
    for m in re.finditer(r"[()]|\b(?:is|are|was|were)\b", plain):
        if m.group() == "(":
            depth += 1
        elif m.group() == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            verb = m
            break
    if verb is None:
        return None, "no verb"
    last, cursor = None, verb.end()
    for m in LINK.finditer(sentence, verb.end()):
        if STOP.search(plain, cursor, m.start()):
            break
        last, cursor = m, m.end()
    if last is None:
        return None, "no link in the noun phrase"
    return norm_title(last.group(1)), "found"


for example in ["'''Kleenex''' is a [[brand]] [[name]] for many paper products, such as [[facial tissue]].",
                "'''Paris''' ([[nickname]]d the \"City of light\") is the [[capital (city)|capital city]] of [[France]].",
                "'''Dogs''' are [[mammal]]s, usually to be kept as [[pet]]s.",
                "'''Jane Doe''' (born 1950 in [[Ohio]]) is an [[United States|American]] [[singer]] and [[songwriter]].",
                "'''Quantum mechanics''' explains how the [[universe]] works."]:
    print(defining_link(example), "<-", example[:70])

# %%
sentences = con.sql("select page_id, first_sentence from articles").df()
found = [defining_link(s) for s in sentences["first_sentence"]]
sentences["target"], sentences["how"] = [f[0] for f in found], [f[1] for f in found]
con.register("defining_raw", sentences[["page_id", "target", "how"]])
con.sql("""
create table defining as
select d.page_id as src, d.how, r.page_id as dst
from defining_raw d left join resolve r on r.title = d.target
""")
how = con.sql("""
select case when how = 'found' and dst is null then 'found, but target is not an article' else how end as outcome,
       count(*) as articles from defining group by 1 order by 2 desc
""").df()
how["share"] = (how["articles"] / how["articles"].sum()).map("{:.1%}".format)
how

# %% [markdown]
# ### Is the first link the defining link?
#
# For articles with a defining link, where does it fall among the article's links? "Game
# order" skips links in parentheses and italics, as the Philosophy game does.

# %%
where = con.sql("""
with ranked as (
    select l.src, l.dst, l.ordinal,
           case when not l.in_paren and not l.italic
                then sum((not l.in_paren and not l.italic)::int) over (partition by l.src order by l.pos) end as game_ordinal
    from links l
)
select least(r.ordinal, 6) as ordinal, least(coalesce(r.game_ordinal, 99), 6) as game_ordinal, count(*) as n
from defining d join ranked r on r.src = d.src and r.dst = d.dst
where d.dst is not null group by 1, 2
""").df()
total = where["n"].sum()
table = pd.DataFrame({
    "by plain order": where.groupby("ordinal")["n"].sum() / total,
    "by game order": where.groupby("game_ordinal")["n"].sum() / total,
}).rename(index={6: "6th or later / skipped"})
table.index.name = "defining link is the…"
print(f"{total:,} articles with a defining link")
table.style.format("{:.1%}")

# %% [markdown]
# And the other way round: when the first link is *not* the defining link, what is it?

# %%
con.sql("""
select a.title, a.first_sentence, t1.title as first_link, t2.title as defining_link
from defining d
join links l on l.src = d.src and l.ordinal = 1 and l.dst <> d.dst
join articles a on a.page_id = d.src
join pages t1 on t1.page_id = l.dst join pages t2 on t2.page_id = d.dst
where d.dst is not null using sample 12 rows (reservoir, 7)
""").df()

# %% [markdown]
# ## 3. How links change with position
#
# Three proxies for "this link points to something more general", by the link's ordinal:
#
# - the target has more incoming links than the source,
# - the target does **not** link back (a one-way reference),
# - the target's incoming-link count.

# %%
by_ordinal = con.sql("""
select least(l.ordinal, 40) as ordinal, count(*) as links,
       avg((coalesce(ti.n, 0) > coalesce(si.n, 0))::int) as target_more_linked,
       avg((b.src is null)::int) as one_way,
       median(coalesce(ti.n, 0)) as median_target_in_links,
       avg((d.dst is not null)::int) as is_defining
from links l
left join indeg ti on ti.page_id = l.dst
left join indeg si on si.page_id = l.src
left join pairs b on b.src = l.dst and b.dst = l.src
left join defining d on d.src = l.src and d.dst = l.dst
group by 1 order by 1
""").df().set_index("ordinal")
by_ordinal.head(12).style.format({"links": "{:,}", "target_more_linked": "{:.1%}", "one_way": "{:.1%}",
                                  "median_target_in_links": "{:.0f}", "is_defining": "{:.1%}"})

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
shown = by_ordinal.loc[:30]
axes[0].plot(shown.index, shown["is_defining"], color=BLUE, marker="o", markersize=3)
axes[0].set(title="Link is the defining link", xlabel="link ordinal in the article", ylim=(0, None))
axes[1].plot(shown.index, shown["target_more_linked"], color=BLUE, marker="o", markersize=3)
axes[1].set(title="Target has more incoming links than source", xlabel="link ordinal in the article", ylim=(0, 1))
axes[2].plot(shown.index, shown["median_target_in_links"], color=BLUE, marker="o", markersize=3)
axes[2].set(title="Median incoming links of the target", xlabel="link ordinal in the article", yscale="log")
for ax in axes[:2]:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fig.tight_layout()

# %% [markdown]
# ### By where the link sits
#
# The same proxies by region of the article instead of ordinal.

# %%
by_place = con.sql("""
select case when l.in_first_sentence then '1 first sentence'
            when l.sec_idx = 0 then '2 rest of lead'
            when l.section in ('Related pages', 'See also', 'Other websites', 'References', 'Notes') then '5 related pages / see also'
            when l.in_list then '4 body, in a list'
            else '3 body, in prose' end as place,
       count(*) as links,
       avg((coalesce(ti.n, 0) > coalesce(si.n, 0))::int) as target_more_linked,
       avg((b.src is null)::int) as one_way,
       median(coalesce(ti.n, 0)) as median_target_in_links
from links l
left join indeg ti on ti.page_id = l.dst
left join indeg si on si.page_id = l.src
left join pairs b on b.src = l.dst and b.dst = l.src
group by 1 order by 1
""").df().set_index("place")
by_place.style.format({"links": "{:,}", "target_more_linked": "{:.1%}", "one_way": "{:.1%}", "median_target_in_links": "{:.0f}"})

# %% [markdown]
# ### By word position in the lead
#
# Position measured in words from the start of the lead, which does not depend on how many
# links came before.

# %%
by_word = con.sql("""
select case when l.word < 5 then '000–4' when l.word < 10 then '005–9' when l.word < 20 then '010–19'
            when l.word < 40 then '020–39' when l.word < 80 then '040–79' when l.word < 160 then '080–159' else '160+' end as words_into_lead,
       count(*) as links,
       avg((coalesce(ti.n, 0) > coalesce(si.n, 0))::int) as target_more_linked,
       avg((b.src is null)::int) as one_way,
       median(coalesce(ti.n, 0)) as median_target_in_links,
       avg((d.dst is not null)::int) as is_defining
from links l
left join indeg ti on ti.page_id = l.dst
left join indeg si on si.page_id = l.src
left join pairs b on b.src = l.dst and b.dst = l.src
left join defining d on d.src = l.src and d.dst = l.dst
where l.sec_idx = 0 group by 1 order by 1
""").df().set_index("words_into_lead")
by_word.style.format({"links": "{:,}", "target_more_linked": "{:.1%}", "one_way": "{:.1%}",
                      "median_target_in_links": "{:.0f}", "is_defining": "{:.1%}"})

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
# # Page specificity on Simple English Wikipedia
#
# Following the first link of an article, repeatedly, usually ends at *Philosophy*. The number
# of steps is a rough measure of how specific a topic is. This notebook builds the lead-link
# graph of Simple English Wikipedia and tests the premises behind a de-noised version:
#
#     spec(p) = 1 + agg(spec(l) for l in first k lead links of p),   spec(Philosophy) = 0
#
# 1. Extract lead links from the dump
# 2. Build the graph
# 3. The first-link game: reach, shape, funnels
# 4. How noisy is a single link?
# 5. Iterating specificity: convergence and variants
# 6. Face validity
#
# Wikipedia text is CC BY-SA 4.0; the tables derived here carry the same license.

# %%
import bz2
import html
import re
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from tqdm import tqdm

WIKI = "simplewiki"
ROOT = "Philosophy"
DATA = Path("../data") if Path.cwd().name == "notebooks" else Path("data")
DATA.mkdir(exist_ok=True)
DUMP = DATA / f"{WIKI}-latest-pages-articles.xml.bz2"
DUMP_URL = f"https://dumps.wikimedia.org/{WIKI}/latest/{DUMP.name}"
PAGES = DATA / f"{WIKI}_pages.parquet"
LINKS = DATA / f"{WIKI}_lead_links.parquet"

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
pd.set_option("display.max_rows", 60, "display.width", 160)

# %% [markdown]
# ## 1. Extract lead links
#
# The lead is everything before the first section heading. Templates, tables, references and
# file links are removed. Each remaining link is kept in order, flagged if it sits inside
# parentheses or italics, so the Philosophy Game rules stay a choice made later.

# %%
if not DUMP.exists():
    # Wikimedia rejects requests without a descriptive User-Agent
    req = urllib.request.Request(DUMP_URL, headers={"User-Agent": "articular/0.1 (research notebook)"})
    part = DUMP.with_suffix(".part")
    with urllib.request.urlopen(req) as resp, open(part, "wb") as f, \
            tqdm(total=int(resp.headers["Content-Length"]), unit="B", unit_scale=True, desc=DUMP.name) as bar:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    part.rename(DUMP)
print(f"{DUMP.name}: {DUMP.stat().st_size / 1e6:.0f} MB")

# %%
COMMENT = re.compile(r"<!--.*?-->", re.S)
REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
HEADING = re.compile(r"^==", re.M)
BRACE = re.compile(r"\{\{|\}\}|\{\||\|\}")
TOKEN = re.compile(r"\[\[|\]\]|\(|\)|'{2,}|\n")
SKIP_PREFIX = ("file:", "image:", "category:", "media:")
DISAMBIG = re.compile(r"\{\{\s*(disambig|disamb|dab|hndis|geodis)", re.I)
MAX_LINKS = 30


def strip_blocks(text):
    """Drop {{templates}} and {| tables |}, including nested ones."""
    out, stack, pos, keep_from = [], [], 0, 0
    while m := BRACE.search(text, pos):
        tok, pos = m.group(), m.end()
        if tok in ("{{", "{|"):
            if not stack:
                out.append(text[keep_from:m.start()])
            stack.append(tok)
        elif stack and (tok == "}}") == (stack[-1] == "{{"):
            stack.pop()
            if not stack:
                keep_from = pos
        elif tok == "|}":
            pos = m.start() + 1  # e.g. "|}}" closing a template: rescan from the brace
    if not stack:
        out.append(text[keep_from:])
    return "".join(out)


def norm_title(target):
    t = html.unescape(target).split("#")[0].replace("_", " ").strip(" :")
    t = " ".join(t.split())
    return t[:1].upper() + t[1:]


def lead_links(wikitext):
    """Ordered (target, in_paren, italic) for links in the lead."""
    m = HEADING.search(wikitext)
    text = wikitext[: m.start()] if m else wikitext
    text = strip_blocks(REF.sub("", COMMENT.sub("", text)))
    links, pos, paren, italic = [], 0, 0, False
    while (m := TOKEN.search(text, pos)) and len(links) < MAX_LINKS:
        tok, pos = m.group(), m.end()
        if tok == "[[":
            depth, end = 1, pos
            while depth and (n := TOKEN.search(text, end)):
                end = n.end()
                depth += (n.group() == "[[") - (n.group() == "]]")
            if depth:
                break
            inner, pos = text[m.end(): end - 2], end
            target = inner.split("|")[0]
            if target.strip(" :").lower().startswith(SKIP_PREFIX) or "[[" in inner:
                continue
            if title := norm_title(target):
                links.append((title, paren > 0, italic))
        elif tok == "(":
            paren += 1
        elif tok == ")":
            paren = max(0, paren - 1)
        elif tok == "\n":
            paren, italic = 0, False
        elif tok[0] == "'" and len(tok) != 3 and len(tok) != 4:
            italic = not italic
    return links


# A quick check of the rules on a made-up lead
lead_links(
    "{{Infobox|a=[[Skip me]]}}\n[[File:x.png|thumb|A [[caption]] link]]\n"
    "'''Kleenex''' (from [[Greek language|Greek]]) is a ''[[brand]]'' of [[facial tissue]].<ref>[[x]]</ref>\n"
    "== History ==\n[[Later]]"
)

# %%
if not (PAGES.exists() and LINKS.exists()):
    pages = {"page_id": [], "title": [], "redirect_target": [], "is_disambig": [], "n_chars": []}
    links = {"page_id": [], "pos": [], "target": [], "in_paren": [], "italic": []}
    page = {}
    with bz2.open(DUMP, "rb") as f:
        for _, el in tqdm(ET.iterparse(f), desc="elements", unit_scale=True):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag in ("title", "ns", "text"):
                page[tag] = el.text or ""
            elif tag == "id":
                page.setdefault("id", int(el.text))  # first <id> in a page is the page id
            elif tag == "redirect":
                page["redirect"] = el.get("title")
            elif tag == "page":
                if page.get("ns") == "0":
                    text, redirect = page.get("text", ""), page.get("redirect")
                    pages["page_id"].append(page["id"])
                    pages["title"].append(page["title"])
                    pages["redirect_target"].append(norm_title(redirect) if redirect else None)
                    pages["is_disambig"].append(bool(DISAMBIG.search(text)))
                    pages["n_chars"].append(len(text))
                    if not redirect:
                        for pos, (target, in_paren, italic) in enumerate(lead_links(text)):
                            links["page_id"].append(page["id"])
                            links["pos"].append(pos)
                            links["target"].append(target)
                            links["in_paren"].append(in_paren)
                            links["italic"].append(italic)
                page = {}
                el.clear()
    pq.write_table(pa.table(pages), PAGES)
    pq.write_table(pa.table(links), LINKS)

# %% [markdown]
# ## 2. Build the graph
#
# Link targets are matched to article titles, redirects are followed (up to two hops), and
# links to missing pages are dropped. `edges` applies the game rules: no parenthesised or
# italic links, no self-links, each target counted once. `rank` 1 is the first valid link.

# %%
con = duckdb.connect()
con.sql(f"""
create table pages as select * from '{PAGES}';
create table links as select * from '{LINKS}';

create table articles as
select page_id, title, is_disambig, n_chars, row_number() over (order by page_id) - 1 as idx
from pages where redirect_target is null;

-- every title (article or redirect) -> the article it ends at
create table resolve as
select p.title, coalesce(a0.page_id, a1.page_id, a2.page_id) as page_id
from pages p
left join articles a0 on a0.title = p.title
left join articles a1 on a1.title = p.redirect_target
left join pages r on r.title = p.redirect_target
left join articles a2 on a2.title = r.redirect_target
where coalesce(a0.page_id, a1.page_id, a2.page_id) is not null;

create table resolved as
select l.page_id as src, l.pos, r.page_id as dst, l.in_paren, l.italic
from links l join resolve r on r.title = l.target;

create table edges as
select s.idx as src, d.idx as dst, row_number() over (partition by e.src order by e.pos) as rank
from (
    select src, dst, min(pos) as pos from resolved
    where not in_paren and not italic and src <> dst group by src, dst
) e
join articles s on s.page_id = e.src
join articles d on d.page_id = e.dst;
""")
con.sql("""
select
  (select count(*) from pages) as ns0_pages,
  (select count(*) from articles) as articles,
  (select count(*) from pages where redirect_target is not null) as redirects,
  (select count(*) from articles where is_disambig) as disambig,
  (select count(*) from links) as raw_lead_links,
  (select count(*) from resolved) as resolved_links,
  (select count(*) from edges) as game_edges
""").df().T.rename(columns={0: "count"})

# %% [markdown]
# How many valid lead links does an article have, and how often do the game rules matter?

# %%
con.sql("""
select
  round(avg(in_paren::int) * 100, 1) as pct_links_in_parens,
  round(avg(italic::int) * 100, 1) as pct_links_italic,
  round(100 * (select count(*) from resolved r1 where pos = (select min(pos) from resolved r2 where r2.src = r1.src)
               and (in_paren or italic)) / count(distinct src), 1) as pct_pages_first_raw_link_rejected
from resolved
""").df()

# %%
articles = con.sql("select * from articles order by idx").df()
edges = con.sql("select src, dst, rank from edges order by src, rank").df()
N = len(articles)
title = articles["title"].to_numpy()
idx_of = pd.Series(articles.index.to_numpy(), index=articles["title"])
ROOT_IDX = int(idx_of[ROOT])

out_deg = np.bincount(edges["src"], minlength=N)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(*np.unique(out_deg, return_counts=True), color=BLUE, width=0.8)
ax.set(title=f"Valid lead links per article (capped at {MAX_LINKS})", xlabel="links in lead", ylabel="articles");
print(f"{N:,} articles; {np.mean(out_deg == 0):.1%} have no valid lead link; "
      f"{np.mean(out_deg >= 3):.1%} have at least 3")

# %% [markdown]
# ## 3. The first-link game
#
# Each article points to at most one other article, so every path ends in a cycle or a dead
# end. `d1` is the number of first-link steps to Philosophy.


# %%
def graph(k):
    """Adjacency matrix of the graph using each page's first k valid links."""
    e = edges[edges["rank"] <= k]
    return sp.csr_matrix((np.ones(len(e)), (e["src"], e["dst"])), shape=(N, N))


def dist_to_root(G):
    """Fewest steps from every page to the root (inf if it cannot be reached)."""
    return shortest_path(G.T.tocsr(), method="D", unweighted=True, indices=ROOT_IDX)


nxt = np.full(N, -1)
first = edges[edges["rank"] == 1]
nxt[first["src"]] = first["dst"]
d1 = dist_to_root(graph(1))
reached = np.isfinite(d1)
print(f"{reached.mean():.1%} of articles reach {ROOT} by first links "
      f"({reached.sum():,} of {N:,}); median distance {np.median(d1[reached]):.0f}, max {d1[reached].max():.0f}")

# %% [markdown]
# Where does Philosophy itself lead?

# %%
path, seen = [ROOT_IDX], {ROOT_IDX}
while (n := nxt[path[-1]]) >= 0:
    path.append(n)
    if n in seen:
        break
    seen.add(n)
print(" → ".join(title[path]))

# %% [markdown]
# ### Where do the other pages end up?
#
# Every page is followed to its end: a cycle (named by its members) or a dead end.

# %%
# Pointer doubling: after 2**20 steps every page sits on its cycle or dead end
jump = np.where(nxt >= 0, nxt, np.arange(N))
for _ in range(20):
    jump = jump[jump]


def cycle_name(i):
    members, j = [i], nxt[i]
    while j >= 0 and j != i:
        members.append(j)
        j = nxt[j]
    if j < 0:
        return "(no valid lead link)" if out_deg[i] == 0 else "(dead end)"
    names = sorted(title[members])
    return " ⟲ ".join(names[:4]) + (f" … ({len(names)} pages)" if len(names) > 4 else "")


end_name = {i: cycle_name(i) for i in np.unique(jump)}
ending = pd.Series(jump).map(end_name)
ending[reached] = f"→ {ROOT}"
ending[(out_deg == 0)] = "(no valid lead link)"
ending[(out_deg > 0) & (nxt[jump] < 0) & ~reached] = "(path ends at a page with no valid lead link)"
basins = ending.value_counts().rename("articles").to_frame()
basins["share"] = (basins["articles"] / N).map("{:.2%}".format)
basins.head(15)

# %% [markdown]
# ### Shape: how far are pages from Philosophy?

# %%
fig, ax = plt.subplots(figsize=(7, 3))
vals, counts = np.unique(d1[reached], return_counts=True)
ax.bar(vals, counts, color=BLUE, width=0.8)
ax.set(title=f"First-link distance to {ROOT}", xlabel="steps (d1)", ylabel="articles");

# %% [markdown]
# ### Funnels
#
# For each page, the number of pages whose first-link path passes through it. A few pages
# carry almost all the traffic, which is why one editor changing one first link can move
# thousands of distances.

# %%
through = np.ones(N)
through[~reached] = 0
for d in range(int(d1[reached].max()), 0, -1):
    level = np.flatnonzero(d1 == d)
    np.add.at(through, nxt[level], through[level])
funnels = pd.DataFrame({"title": title, "d1": d1, "pages_through": through.astype(int)})
funnels["share_of_reaching"] = (funnels["pages_through"] / reached.sum()).map("{:.1%}".format)
funnels[reached].nlargest(25, "pages_through").reset_index(drop=True)

# %% [markdown]
# ## 4. How noisy is a single link?
#
# If lead links all anchor a topic in more general terms, then the 2nd and 3rd links should
# sit about as far from Philosophy as the 1st. Compare, for pages with at least three valid
# links, the first-link distance of each of those three targets.

# %%
wide = edges[edges["rank"] <= 3].pivot(index="src", columns="rank", values="dst").dropna().astype(int)
tgt = pd.DataFrame({f"link{r}": d1[wide[r]] for r in (1, 2, 3)}, index=wide.index)
tgt = tgt[np.isfinite(tgt).all(axis=1)]
print(f"{len(tgt):,} pages with three valid links that all reach {ROOT}")
print("Spearman correlation between the distances of a page's first three link targets:")
print(tgt.corr(method="spearman").round(2))

rng = np.random.default_rng(0)
shuffled = np.abs(tgt["link1"].to_numpy() - rng.permutation(tgt["link2"].to_numpy()))
for r in (2, 3):
    diff = tgt[f"link{r}"] - tgt["link1"]
    print(f"link{r} vs link1: same distance {np.mean(diff == 0):.0%}, within ±1 {np.mean(diff.abs() <= 1):.0%}, "
          f"within ±2 {np.mean(diff.abs() <= 2):.0%}, mean {diff.mean():+.2f}")
print(f"random pairing (baseline):  within ±1 {np.mean(shuffled <= 1):.0%}, within ±2 {np.mean(shuffled <= 2):.0%}")

fig, ax = plt.subplots(figsize=(7, 3))
diff = (tgt["link2"] - tgt["link1"]).clip(-10, 10)
ax.bar(*np.unique(diff, return_counts=True), color=BLUE, width=0.8)
ax.set(title="Distance of 2nd link target minus distance of 1st link target",
       xlabel="difference in steps (clipped to ±10)", ylabel="articles");

# %% [markdown]
# The spread of a page's distance across its own first three links, next to the spread of
# distance across all pages, says how much of `d1` is signal.

# %%
within = tgt.std(axis=1).mean()
print(f"mean within-page std of link-target distance: {within:.2f} steps")
print(f"std of d1 across all reaching pages:          {d1[reached].std():.2f} steps")

# %% [markdown]
# ## 5. Iterating specificity
#
# `spec(p) = 1 + agg(spec of p's first k links)`, Philosophy pinned at 0, seeded with `d1`.
#
# Links to pages that cannot reach Philosophy (in the k-link graph) are ignored; otherwise
# their infinite value would spread to everything upstream. With `agg = mean` the fixed
# point is the expected number of steps of a random walk to Philosophy. With `agg = min` it
# is the shortest path. Median has no such guarantee, so it is simply iterated.


# %%
def k_graph(k):
    """First-k graph restricted to pages that can reach the root; the root has no out-links."""
    G = graph(k)
    dmin = dist_to_root(G)
    ok = np.isfinite(dmin)
    e = edges[(edges["rank"] <= k) & ok[edges["src"]] & ok[edges["dst"]] & (edges["src"] != ROOT_IDX)]
    return e, ok, dmin


def iterate(k, agg="mean", tol=1e-6, max_rounds=5000):
    e, ok, dmin = k_graph(k)
    seed = np.where(reached, d1, dmin)  # d1 where defined, else shortest path in the k-graph
    s = np.where(ok, seed, np.nan)
    if agg == "mean":
        A = sp.csr_matrix((np.ones(len(e)), (e["src"], e["dst"])), shape=(N, N))
        deg = np.asarray(A.sum(axis=1)).ravel()
        P = sp.diags(1 / np.maximum(deg, 1)) @ A
        step = lambda s: 1 + P @ np.nan_to_num(s)
    else:
        slots = np.full((N, k), -1)
        slot = e.groupby("src").cumcount().to_numpy()
        slots[e["src"], slot] = e["dst"]
        step = lambda s: 1 + np.nanmedian(np.where(slots >= 0, s[slots], np.nan), axis=1)
    has_links = np.bincount(e["src"], minlength=N) > 0
    history = []
    for _ in range(max_rounds):
        new = s.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows for pages without links
            new[has_links] = step(s)[has_links]
        history.append(np.nanmax(np.abs(new - s)))
        s = new
        if history[-1] < tol:
            break
    return s, np.array(history)


runs = {}
for k in (1, 2, 3, 5, 10):
    runs[k] = iterate(k)
    s, h = runs[k]
    print(f"mean, k={k:>2}: {np.isfinite(s).mean():.1%} of pages covered, "
          f"{len(h):>4} rounds, last change {h[-1]:.1e}, median spec {np.nanmedian(s):.1f}, max {np.nanmax(s):.0f}")

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
for (k, (_, h)), color in zip(list(runs.items())[1:], (BLUE, ORANGE, AQUA, "#6250d6")):
    ax.plot(np.arange(1, len(h) + 1), h, color=color, label=f"k = {k}")
ax.set(yscale="log", xscale="log", title="Convergence of the mean update: largest change in any page per round",
       xlabel="round", ylabel="max |change|");
ax.legend(frameon=False);

# %% [markdown]
# ### Do the variants agree?
#
# Rank correlation between variants, over pages covered by all of them.

# %%
med3, med3_hist = iterate(3, agg="median", max_rounds=500)
print(f"median, k=3: {len(med3_hist)} rounds, largest change in the last round {med3_hist[-1]:.1e}")
variants = pd.DataFrame({
    "d1 (first link)": np.where(reached, d1, np.nan),
    "mean k=2": runs[2][0], "mean k=3": runs[3][0], "mean k=5": runs[5][0], "mean k=10": runs[10][0],
    "median k=3": med3,
    "min k=3": np.where(np.isfinite(k_graph(3)[2]), k_graph(3)[2], np.nan),
}, index=title)
common = variants.dropna()
print(f"{len(common):,} pages covered by every variant")
common.corr(method="spearman").round(2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
sample = common.sample(min(20000, len(common)), random_state=0)
jitter = rng.uniform(-0.35, 0.35, len(sample))
axes[0].scatter(sample["d1 (first link)"] + jitter, sample["mean k=3"], s=3, alpha=0.15, color=BLUE, linewidths=0)
axes[0].set(title=f"First-link distance vs mean k=3 (pages reaching {ROOT})", xlabel="d1 (jittered)", ylabel="spec, mean k=3");
axes[1].hist(variants["mean k=3"].dropna(), bins=80, color=BLUE)
axes[1].set(title="Distribution of spec, mean k=3, all covered pages", xlabel="spec", ylabel="articles");
fig.tight_layout()

# %% [markdown]
# ### Does specificity relate to how often a page is linked?
#
# General topics should be linked from many leads.

# %%
in_deg = np.bincount(edges["dst"], minlength=N)
rows = {name: spearmanr(col[col.notna().to_numpy()], in_deg[col.notna().to_numpy()]).statistic
        for name, col in variants.items()}
pd.Series(rows, name="Spearman with lead in-degree").round(2).to_frame()

# %% [markdown]
# ## 6. Face validity

# %%
spot = ["Philosophy", "Science", "Knowledge", "Mathematics", "Physics", "Quantum mechanics", "Electron",
        "Schrödinger equation", "Animal", "Dog", "Poodle", "Brand", "Kleenex", "Cotton swab",
        "Anime", "Dragon Ball Z", "Country", "France", "Paris", "Eiffel Tower", "Music", "The Beatles"]
variants.reindex([t for t in spot if t in variants.index]).round(1)

# %%
main = variants["mean k=3"].dropna()
linked = main[in_deg[idx_of[main.index]] >= 20]  # well-linked pages only, to keep the lists recognizable
pd.DataFrame({
    "most general": linked.nsmallest(25).index, "spec": linked.nsmallest(25).round(1).to_numpy(),
    "most specific (≥20 lead in-links)": linked.nlargest(25).index, "spec ": linked.nlargest(25).round(1).to_numpy(),
})

# %% [markdown]
# ### Classifying a page's links
#
# For one page, compare each lead link's specificity to the page's own: lower means the link
# generalizes, similar means lateral, higher means a narrower topic. The ±1 threshold is
# arbitrary.


# %%
def classify_links(page, measure="mean k=3", threshold=1.0):
    own = variants.at[page, measure]
    e = edges[edges["src"] == idx_of[page]]
    out = pd.DataFrame({"rank": e["rank"].to_numpy(), "link": title[e["dst"]]})
    out["spec"] = variants[measure].to_numpy()[e["dst"]].round(1)
    out["delta"] = (out["spec"] - own).round(1)
    out["kind"] = np.select([out["delta"] < -threshold, out["delta"] > threshold], ["generalizing", "narrower"], "lateral")
    print(f"{page}: {measure} = {own:.1f}")
    return out


classify_links("Quantum mechanics")

# %%
classify_links("Dog")

# %% [markdown]
# ## 7. Two alternative measures
#
# **Rootless levels.** Give every page a level `h` so that, over all lead links at once,
# `h[page] - h[link]` is as close to 1 as possible (least squares). This is the "trophic
# level" of network science. It keeps the "a page is one step more specific than what its
# lead links to" idea, but needs no root and tolerates cycles. Levels are only defined up to
# a constant, so the lowest page is set to 0. The *incoherence* is the mean squared miss:
# 0 for a perfect hierarchy, 1 for a graph with no direction at all.

# %%
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg


def levels(k=None):
    """Least-squares levels on the first-k graph (all lead links if k is None)."""
    e = edges if k is None else edges[edges["rank"] <= k]
    src, dst = e["src"].to_numpy(), e["dst"].to_numpy()
    A = sp.csr_matrix((np.ones(len(e)), (src, dst)), shape=(N, N))
    out_w, in_w = np.asarray(A.sum(axis=1)).ravel(), np.asarray(A.sum(axis=0)).ravel()
    L = (sp.diags(out_w + in_w) - A - A.T).tocsr()
    # levels are only comparable within one connected piece: keep the largest
    _, label = connected_components(A, directed=True, connection="weak")
    giant = label == np.bincount(label).argmax()
    x, info = cg(L[giant][:, giant], (out_w - in_w)[giant], rtol=1e-10, maxiter=20000)
    h = np.full(N, np.nan)
    h[giant] = x - x.min()
    miss = h[src] - h[dst] - 1
    return h, {"k": k or "all", "covered": giant.mean(), "cg_converged": info == 0,
               "incoherence": np.nanmean(miss ** 2), "links_pointing_down": np.nanmean(h[src] > h[dst])}


level_runs = {k: levels(k) for k in (3, 5, None)}
pd.DataFrame([info for _, info in level_runs.values()]).round(3)

# %% [markdown]
# **Discounted mean.** The original update with a discount `g` on the links' values:
# `spec(p) = 1 + g * mean(spec of first k links)`, root pinned at 0. Steps far from the page
# count for less, so the update converges quickly and values stay below `1 / (1 - g)`. Pages
# with no links are held at that ceiling. It still needs a root.


# %%
def discounted(k, g, root=ROOT, tol=1e-9, max_rounds=2000):
    e = edges[(edges["rank"] <= k) & (edges["src"] != idx_of[root])]
    A = sp.csr_matrix((np.ones(len(e)), (e["src"], e["dst"])), shape=(N, N))
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1 / np.maximum(deg, 1)) @ A
    s = np.full(N, 1 / (1 - g))
    s[idx_of[root]] = 0
    for rounds in range(1, max_rounds + 1):
        new = np.where(deg > 0, 1 + g * (P @ s), s)
        change, s = np.abs(new - s).max(), new
        if change < tol:
            break
    return s, rounds


disc_runs = {}
for g in (0.5, 0.85, 0.95):
    disc_runs[g], rounds = discounted(3, g)
    below = np.mean(disc_runs[g] < 0.99 / (1 - g))
    print(f"g={g}: converged in {rounds} rounds; ceiling {1 / (1 - g):.1f}; "
          f"{below:.1%} of pages are more than 1% below the ceiling")

# %% [markdown]
# ### Comparison
#
# Three checks: do the measures agree with each other, do they order pairs we know the
# answer to, and do the extremes look right?

# %%
measures = pd.DataFrame({
    "levels k=3": level_runs[3][0], "levels k=5": level_runs[5][0], "levels all": level_runs[None][0],
    "discounted g=0.5": disc_runs[0.5], "discounted g=0.85": disc_runs[0.85], "discounted g=0.95": disc_runs[0.95],
    "mean k=3": variants["mean k=3"].to_numpy(), "min k=3": variants["min k=3"].to_numpy(),
}, index=title)
measures.dropna().corr(method="spearman").round(2)

# %%
# (more general, more specific) pairs with an uncontroversial answer
PAIRS = [("Science", "Physics"), ("Physics", "Quantum mechanics"), ("Physics", "Electron"),
         ("Mathematics", "Algebra"), ("Mathematics", "Triangle"), ("Animal", "Mammal"), ("Mammal", "Dog"),
         ("Animal", "Dog"), ("Plant", "Tree"), ("Tree", "Oak"), ("Country", "France"), ("France", "Paris"),
         ("City", "Paris"), ("Paris", "Eiffel Tower"), ("Music", "Jazz"), ("Music", "The Beatles"),
         ("Brand", "Kleenex"), ("Anime", "Dragon Ball Z"), ("Sport", "Football"), ("Language", "English language"),
         ("Religion", "Christianity"), ("Christianity", "Pope"), ("Food", "Bread"), ("Chemistry", "Oxygen"),
         ("Art", "Painting"), ("Painting", "Mona Lisa"), ("Computer", "Laptop"), ("Planet", "Mars"),
         ("Disease", "Influenza"), ("Government", "Parliament")]
pairs = [(a, b) for a, b in PAIRS if a in measures.index and b in measures.index]
general, specific = measures.loc[[a for a, _ in pairs]].to_numpy(), measures.loc[[b for _, b in pairs]].to_numpy()
score = pd.DataFrame({
    "correct": (general < specific).sum(axis=0), "tied": (general == specific).sum(axis=0),
    "wrong": (general > specific).sum(axis=0), "missing": np.isnan(general - specific).sum(axis=0),
}, index=measures.columns)
print(f"{len(pairs)} test pairs")
score

# %%
measures.reindex([t for t in spot if t in measures.index]).round(2)

# %%
lv = measures["levels all"].dropna()
lv_linked = lv[in_deg[idx_of[lv.index]] >= 20]
pd.DataFrame({
    "lowest level (most general)": lv_linked.nsmallest(25).index, "level": lv_linked.nsmallest(25).round(2).to_numpy(),
    "highest level (≥20 lead in-links)": lv_linked.nlargest(25).index, "level ": lv_linked.nlargest(25).round(2).to_numpy(),
})

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].hist(lv, bins=100, color=BLUE)
axes[0].set(title="Distribution of levels (all lead links)", xlabel="level", ylabel="articles")
axes[1].hist(measures["discounted g=0.85"], bins=100, color=BLUE)
axes[1].set(title="Distribution of discounted mean, g=0.85, k=3", xlabel="spec", ylabel="articles", yscale="log")
fig.tight_layout()

# %%
pd.Series({name: spearmanr(col.dropna(), in_deg[idx_of[col.dropna().index]]).statistic
           for name, col in measures.items()}, name="Spearman with lead in-degree").round(2).to_frame()

# %%
variants["levels all"] = measures["levels all"]
classify_links("Quantum mechanics", measure="levels all", threshold=0.25)

# %%
classify_links("Dog", measure="levels all", threshold=0.25)

# %% [markdown]
# ## Findings so far
#
# - **Philosophy is not the root here.** Only 0.2% of Simple English articles reach it by
#   first links. Philosophy's own path runs into the cycle Existence ⟲ Fact ⟲ Reality, and
#   the first-link graph splits into several basins, the largest being Measurement ⟲ Quantity
#   (36%). The root has to be chosen from the data, or the measure must not need one.
# - **A single link is noisy.** Changing which lead link is followed changes the basin, not
#   just the distance. Too few pages reach Philosophy to measure the noise in steps.
# - **Mean hitting time does not measure specificity.** With k ≥ 2 almost every page gets
#   nearly the same value (about 800 ± 10 for k = 3). The random walk mixes through the whole
#   graph long before it finds Philosophy, so the value mostly reflects how rarely the walk
#   visits Philosophy, not where it started. The update also had not converged after 5,000
#   rounds, and the median update does not converge at all.
# - **Min (shortest path)** is well defined but squeezes everything into 4–9 steps.
#
# **Alternatives (section 7)**
#
# - **Rootless levels work best.** They order 28–29 of 30 hand-picked general/specific pairs
#   correctly; the discounted mean, the plain mean and shortest path are at or near chance.
# - **But the levels are shallow.** Almost all pages sit between 0.9 and 3, and levels
#   correlate strongly with how often a page is linked (Spearman -0.65 using all lead links).
#   The lead-link graph looks more like two layers (linked-to hubs, and the pages that link
#   to them) than a deep hierarchy. Margins between related topics are thin, e.g. Physics
#   1.35 vs Quantum mechanics 1.46.
# - **Discounting fixes convergence, not meaning.** The discounted mean converges in tens to
#   hundreds of rounds, but most pages sit at or near the ceiling and it barely agrees with
#   the levels (Spearman about 0). The problem is the root, not the lack of damping.
# - The 30 test pairs are a small hand-picked list; a larger independent check (for example
#   Wikidata "subclass of" pairs) is needed before trusting the ranking of measures.

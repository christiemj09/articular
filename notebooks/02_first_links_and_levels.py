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
# # First links and levels on a full wiki
#
# Runs on the output of `articular/pipeline/extract_leads.py` (set the `WIKI` environment variable;
# default `enwiki`). It answers the questions notebook 01 left open for English Wikipedia:
#
# 1. Build the lead-link graph
# 2. The first-link game: does Philosophy hold as the root? Where does everything else go?
# 3. How noisy is a single link?
# 4. Rootless levels (the measure that worked best on Simple English)
# 5. Save per-page results
#
# The mean and discounted-mean updates from notebook 01 are left out: they did not measure
# specificity there. Disambiguation pages are kept as link targets but their own links are
# dropped, since "X may refer to A, B, C" does not define X in more general terms.
#
# Wikipedia text is CC BY-SA 4.0; the tables derived here carry the same license.

# %%
import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import cg
from scipy.stats import spearmanr

WIKI = os.environ.get("WIKI", "enwiki")
ROOT = "Philosophy"
DATA = Path("../data") if Path.cwd().name == "notebooks" else Path("data")
EXTRACT = DATA / f"{WIKI}_extract"
SCORES = DATA / f"{WIKI}_scores.parquet"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 11, "lines.linewidth": 2,
})
pd.set_option("display.max_rows", 60, "display.width", 180, "display.max_colwidth", 80)
print(WIKI, EXTRACT)

# %% [markdown]
# ## 1. Build the graph
#
# Link targets are matched to article titles, redirects are followed (up to two hops), and
# links to missing pages are dropped. `edges` applies the game rules: no parenthesised or
# italic links, no self-links, each target counted once. `rank` 1 is the first valid link.

# %%
con = duckdb.connect()
con.sql(f"""
create table pages as select * from read_parquet('{EXTRACT}/pages/*.parquet');
create view links as select * from read_parquet('{EXTRACT}/links/*.parquet');

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
join articles s on s.page_id = e.src and not s.is_disambig
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

# %%
articles = con.sql("select page_id, title, is_disambig from articles order by idx").df()
E = con.sql("select src, dst, rank from edges").fetchnumpy()
E_src, E_dst, E_rank = E["src"], E["dst"], E["rank"]
N = len(articles)
title = articles["title"].to_numpy()
idx_of = pd.Series(np.arange(N), index=title)
ROOT_IDX = int(idx_of[ROOT])
out_deg = np.bincount(E_src, minlength=N)
in_deg = np.bincount(E_dst, minlength=N)

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(*np.unique(out_deg, return_counts=True), color=BLUE, width=0.8)
ax.set(title="Valid lead links per article (capped at 30)", xlabel="links in lead", ylabel="articles");
content = ~articles["is_disambig"].to_numpy()
print(f"{N:,} articles; {np.mean(out_deg[content] == 0):.1%} of non-disambiguation articles have no valid "
      f"lead link; {np.mean(out_deg[content] >= 3):.1%} have at least 3")

# %% [markdown]
# ## 2. The first-link game
#
# Each article points to at most one other article, so every path ends in a cycle or a dead
# end. `d1` is the number of first-link steps to Philosophy.


# %%
def graph(k=None):
    """Adjacency matrix using each page's first k valid links (all if k is None)."""
    keep = slice(None) if k is None else E_rank <= k
    return sp.csr_matrix((np.ones(len(E_src[keep]), dtype=np.float32), (E_src[keep], E_dst[keep])), shape=(N, N))


nxt = np.full(N, -1)
nxt[E_src[E_rank == 1]] = E_dst[E_rank == 1]
d1 = shortest_path(graph(1).T.tocsr(), method="D", unweighted=True, indices=ROOT_IDX)
reached = np.isfinite(d1)
print(f"{reached.mean():.1%} of all articles reach {ROOT} by first links ({reached.sum():,} of {N:,})")
print(f"{reached[content].mean():.1%} of non-disambiguation articles")
print(f"{reached[content & (out_deg > 0)].mean():.1%} of articles that have at least one valid lead link")
print(f"median distance {np.median(d1[reached]):.0f}, mean {d1[reached].mean():.1f}, max {d1[reached].max():.0f}")

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
root_on_cycle = path[-1] == ROOT_IDX
print(f"{ROOT} is {'on' if root_on_cycle else 'not on'} the cycle its path ends in")

# %% [markdown]
# ### Where does everything end up?
#
# Every page is followed to its end: a cycle (named by its members) or a dead end. A page can
# end in Philosophy's cycle without passing through Philosophy itself, so both are counted.

# %%
# Pointer doubling: after 2**25 steps every page sits on its cycle or at its dead end
jump = np.where(nxt >= 0, nxt, np.arange(N))
for _ in range(25):
    jump = jump[jump]
dead_end = nxt[jump] < 0

# Pages can land on different members of the same cycle: name each cycle once
canon, cycle_names = np.full(N, -1), {}
for i in np.unique(jump[~dead_end]):
    if canon[i] >= 0:
        continue
    members, j = [i], nxt[i]
    while j != i:
        members.append(j)
        j = nxt[j]
    canon[members] = min(members)
    names = sorted(title[members])
    cycle_names[min(members)] = " ⟲ ".join(names[:4]) + (f" … ({len(names)} pages)" if len(names) > 4 else "")

ending = pd.Series(canon[jump]).map(cycle_names).to_numpy(dtype=object)
ending[dead_end] = "(path ends at a page with no valid lead link)"
ending[out_deg == 0] = "(no valid lead link)"
ending[~content] = "(disambiguation page)"
basins = pd.Series(ending).value_counts().rename("articles").to_frame()
basins["share"] = (basins["articles"] / N).map("{:.2%}".format)
root_basin = ending == ending[ROOT_IDX]
print(f"{root_basin.mean():.1%} of articles end in the same place as {ROOT}; {reached.mean():.1%} pass through {ROOT}")
print(f"{len(cycle_names):,} distinct cycles")
basins.head(20)

# %% [markdown]
# ### Shape: how far are pages from Philosophy?

# %%
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(*np.unique(d1[reached], return_counts=True), color=BLUE, width=0.8)
ax.set(title=f"First-link distance to {ROOT}", xlabel="steps (d1)", ylabel="articles");

# %% [markdown]
# ### Funnels
#
# For each page, the number of pages whose first-link path to Philosophy passes through it.

# %%
through = reached.astype(np.int64)
for d in range(int(d1[reached].max()), 0, -1):
    level = np.flatnonzero(d1 == d)
    np.add.at(through, nxt[level], through[level])
funnels = pd.DataFrame({"title": title, "d1": d1, "pages_through": through})
funnels["share_of_reaching"] = (funnels["pages_through"] / reached.sum()).map("{:.1%}".format)
funnels[reached].nlargest(30, "pages_through").reset_index(drop=True)

# %% [markdown]
# ## 3. How noisy is a single link?
#
# For pages with at least three valid links, compare the first-link distance of each of the
# three targets. If lead links all anchor a topic in more general terms, these should agree.

# %%
slots = np.full((N, 3), -1)
first3 = E_rank <= 3
slots[E_src[first3], E_rank[first3] - 1] = E_dst[first3]
has3 = (slots >= 0).all(axis=1)
tgt = pd.DataFrame(d1[slots[has3]], columns=["link1", "link2", "link3"])
all_reach = np.isfinite(tgt).all(axis=1)
print(f"{has3.sum():,} pages with three valid links; for {all_reach.mean():.1%} all three targets reach {ROOT}")
for r in (1, 2, 3):
    print(f"  link{r} target reaches {ROOT}: {np.isfinite(tgt[f'link{r}']).mean():.1%}")
tgt = tgt[all_reach]
print("Spearman correlation between the distances of a page's first three link targets:")
print(tgt.corr(method="spearman").round(2))

rng = np.random.default_rng(0)
shuffled = np.abs(tgt["link1"].to_numpy() - rng.permutation(tgt["link2"].to_numpy()))
for r in (2, 3):
    diff = tgt[f"link{r}"] - tgt["link1"]
    print(f"link{r} vs link1: same distance {np.mean(diff == 0):.0%}, within ±1 {np.mean(diff.abs() <= 1):.0%}, "
          f"within ±2 {np.mean(diff.abs() <= 2):.0%}, mean {diff.mean():+.2f}")
print(f"random pairing (baseline): same {np.mean(shuffled == 0):.0%}, within ±1 {np.mean(shuffled <= 1):.0%}, "
      f"within ±2 {np.mean(shuffled <= 2):.0%}")
print(f"mean within-page std of target distance: {tgt.std(axis=1).mean():.2f}; "
      f"std of d1 across all reaching pages: {d1[reached].std():.2f}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(*np.unique((tgt["link2"] - tgt["link1"]).clip(-10, 10), return_counts=True), color=BLUE, width=0.8)
ax.set(title="Distance of 2nd link target minus distance of 1st link target",
       xlabel="difference in steps (clipped to ±10)", ylabel="articles");

# %% [markdown]
# ## 4. Rootless levels
#
# Give every page a level `h` so that, over all lead links at once, `h[page] - h[link]` is as
# close to 1 as possible (least squares; the "trophic level" of network science). No root is
# needed and cycles are tolerated. The lowest page is set to 0. *Incoherence* is the mean
# squared miss: 0 for a perfect hierarchy, 1 for a graph with no direction at all.


# %%
def levels(k=None):
    A = graph(k).astype(np.float64)
    out_w, in_w = np.asarray(A.sum(axis=1)).ravel(), np.asarray(A.sum(axis=0)).ravel()
    L = (sp.diags(out_w + in_w) - A - A.T).tocsr()
    # levels are only comparable within one connected piece: keep the largest
    _, label = connected_components(A, directed=True, connection="weak")
    giant = label == np.bincount(label).argmax()
    Lg, rounds = L[giant][:, giant], [0]
    x, info = cg(Lg, (out_w - in_w)[giant], rtol=1e-8, maxiter=20000, M=sp.diags(1 / Lg.diagonal()),
                 callback=lambda _: rounds.__setitem__(0, rounds[0] + 1))
    h = np.full(N, np.nan)
    h[giant] = x - x.min()
    keep = slice(None) if k is None else E_rank <= k
    miss = h[E_src[keep]] - h[E_dst[keep]] - 1
    return h, {"k": k or "all", "covered": giant.mean(), "cg_rounds": rounds[0], "cg_converged": info == 0,
               "incoherence": np.nanmean(miss ** 2), "links_pointing_down": np.nanmean(miss > -1)}


level_runs = {k: levels(k) for k in (3, None)}
measures = pd.DataFrame({"levels k=3": level_runs[3][0], "levels all": level_runs[None][0],
                         "d1 (first link)": np.where(reached, d1, np.nan)}, index=title)
pd.DataFrame([info for _, info in level_runs.values()]).round(3)

# %%
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(measures["levels all"].dropna(), bins=120, color=BLUE)
ax.set(title="Distribution of levels (all lead links)", xlabel="level", ylabel="articles");
both = measures.dropna()
print(f"Spearman, over {len(both):,} pages with every measure:")
print(both.corr(method="spearman").round(2))
pd.Series({name: spearmanr(col.dropna(), in_deg[col.notna().to_numpy()]).statistic
           for name, col in measures.items()}, name="Spearman with lead in-degree").round(2).to_frame()

# %% [markdown]
# ### Do the measures order pairs we know the answer to?

# %%
# (more general, more specific) pairs with an uncontroversial answer
PAIRS = [("Science", "Physics"), ("Physics", "Quantum mechanics"), ("Physics", "Electron"),
         ("Mathematics", "Algebra"), ("Mathematics", "Triangle"), ("Animal", "Mammal"), ("Mammal", "Dog"),
         ("Animal", "Dog"), ("Plant", "Tree"), ("Tree", "Oak"), ("Country", "France"), ("France", "Paris"),
         ("City", "Paris"), ("Paris", "Eiffel Tower"), ("Music", "Jazz"), ("Music", "The Beatles"),
         ("Brand", "Kleenex"), ("Anime", "Dragon Ball Z"), ("Sport", "Association football"),
         ("Language", "English language"), ("Religion", "Christianity"), ("Christianity", "Pope"),
         ("Food", "Bread"), ("Chemistry", "Oxygen"), ("Art", "Painting"), ("Painting", "Mona Lisa"),
         ("Computer", "Laptop"), ("Planet", "Mars"), ("Disease", "Influenza"), ("Government", "Parliament")]
pairs = [(a, b) for a, b in PAIRS if a in idx_of.index and b in idx_of.index]
M = measures.to_numpy()
general, specific = M[idx_of[[a for a, _ in pairs]]], M[idx_of[[b for _, b in pairs]]]
print(f"{len(pairs)} test pairs")
pd.DataFrame({
    "correct": (general < specific).sum(axis=0), "tied": (general == specific).sum(axis=0),
    "wrong": (general > specific).sum(axis=0), "missing": np.isnan(general - specific).sum(axis=0),
}, index=measures.columns)

# %%
spot = ["Philosophy", "Science", "Knowledge", "Mathematics", "Physics", "Quantum mechanics", "Electron",
        "Schrödinger equation", "Animal", "Dog", "Poodle", "Brand", "Kleenex", "Cotton swab",
        "Anime", "Dragon Ball Z", "Country", "France", "Paris", "Eiffel Tower", "Music", "The Beatles"]
measures.iloc[idx_of[[t for t in spot if t in idx_of.index]]].round(2)

# %%
MIN_IN = 20 if N < 1_000_000 else 200  # well-linked pages only, to keep the lists recognizable
lv = measures["levels all"][in_deg >= MIN_IN].dropna()
pd.DataFrame({
    "lowest level (most general)": lv.nsmallest(30).index, "level": lv.nsmallest(30).round(2).to_numpy(),
    f"highest level (≥{MIN_IN} lead in-links)": lv.nlargest(30).index, "level ": lv.nlargest(30).round(2).to_numpy(),
})


# %% [markdown]
# ### Classifying a page's links
#
# Lower level than the page: the link generalizes. Similar: lateral. Higher: narrower.


# %%
def classify_links(page, measure="levels all", threshold=0.25):
    i = idx_of[page]
    mine = np.flatnonzero(E_src == i)
    mine = mine[np.argsort(E_rank[mine])]
    out = pd.DataFrame({"rank": E_rank[mine], "link": title[E_dst[mine]]})
    out["level"] = measures[measure].to_numpy()[E_dst[mine]].round(2)
    out["delta"] = (out["level"] - measures[measure].iloc[i]).round(2)
    out["kind"] = np.select([out["delta"] < -threshold, out["delta"] > threshold], ["generalizing", "narrower"], "lateral")
    print(f"{page}: {measure} = {measures[measure].iloc[i]:.2f}")
    return out


classify_links("Quantum mechanics")

# %%
classify_links("Dog")

# %% [markdown]
# ## 5. Save per-page results

# %%
scores = pd.DataFrame({
    "page_id": articles["page_id"], "title": title, "is_disambig": ~content,
    "out_deg": out_deg, "in_deg": in_deg,
    "first_link_page_id": np.where(nxt >= 0, articles["page_id"].to_numpy()[nxt], -1),
    "d1": np.where(reached, d1, np.nan), "ending": ending, "pages_through": through,
    "level_k3": level_runs[3][0], "level_all": level_runs[None][0],
})
scores.to_parquet(SCORES, index=False)
print(f"{SCORES}: {len(scores):,} rows, {SCORES.stat().st_size / 1e6:.0f} MB")

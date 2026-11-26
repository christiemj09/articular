"""Score how general or specific each article is, from the kind graph and from who names it.

    uv run python -m articular.pipeline.specificity --wiki enwiki

Reads data/<wiki>_kind_graph.parquet (articular/pipeline/kind_graph.py), data/<wiki>_relations/ and
data/<wiki>_links/pages/ (articular/pipeline/extract_relations.py, articular/pipeline/extract_links.py).

Writes data/<wiki>_specificity.parquet, one row per article:
  kinds_below   articles that are a kind of this one, directly or through other kinds, plus itself.
                A general thing has many (Person); a particular thing has 1. Articles on a loop of
                kinds share the loop's total.
  specificity   log2(articles / kinds_below): 0 for something everything is a kind of, about 22.7
                for a particular thing in English Wikipedia.
  depth         steps up the kind chain to where it ends (-1 if the article has no kind)
  named_in      articles whose opening sentence links to this one, in any role (kind, place, field…)
  has_kind      whether the article's own kind was found

Prints how well each measure orders hand-picked (general, specific) pairs, and example values.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# (more general, more specific). KIND: the second is a kind or instance of the first.
KIND = [("Science", "Physics"), ("Animal", "Mammal"), ("Mammal", "Dog"), ("Animal", "Dog"), ("Plant", "Tree"),
        ("Tree", "Oak"), ("Country", "France"), ("City", "Paris"), ("Music", "Jazz"), ("Brand", "Kleenex"),
        ("Anime", "Dragon Ball Z"), ("Sport", "Association football"), ("Language", "English language"),
        ("Religion", "Christianity"), ("Food", "Bread"), ("Art", "Painting"), ("Painting", "Mona Lisa"),
        ("Computer", "Laptop"), ("Planet", "Mars"), ("Disease", "Influenza")]
# SCOPE: the second lies within the first (a part, a place in it, a topic of it) without being a kind of it.
SCOPE = [("Physics", "Quantum mechanics"), ("Physics", "Electron"), ("Mathematics", "Algebra"),
         ("Mathematics", "Triangle"), ("France", "Paris"), ("Paris", "Eiffel Tower"), ("Music", "The Beatles"),
         ("Christianity", "Pope"), ("Chemistry", "Oxygen"), ("Government", "Parliament")]
SPOT = ["Entity", "Person", "Organism", "Animal", "Mammal", "Dog", "Poodle", "Science", "Physics", "Quantum mechanics",
        "Electron", "Mathematics", "Triangle", "Country", "France", "Paris", "Eiffel Tower", "Music", "Jazz",
        "The Beatles", "Software architecture", "Mahayana", "Philosophy", "Albert Einstein", "Village"]


def kinds_below(nxt):
    """Size of each node's subtree in a graph where node i points to nxt[i]; loop members share their total."""
    n = len(nxt)
    below, waiting = np.ones(n, dtype=np.int64), np.bincount(nxt[nxt >= 0], minlength=n)
    frontier = np.flatnonzero(waiting == 0)
    while len(frontier):
        f = frontier[nxt[frontier] >= 0]
        np.add.at(below, nxt[f], below[f])
        np.subtract.at(waiting, nxt[f], 1)
        hit = np.unique(nxt[f])
        frontier = hit[waiting[hit] == 0]
    on_loop, done = waiting > 0, np.zeros(n, dtype=bool)
    for i in np.flatnonzero(on_loop):
        if not done[i]:
            members, j = [i], nxt[i]
            while j != i:
                members.append(j)
                j = nxt[j]
            below[members], done[members] = below[members].sum(), True
    return below


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data, wiki = args.data, args.wiki

    graph = pd.read_parquet(data / f"{wiki}_kind_graph.parquet")
    n = len(graph)
    index = pd.Series(np.arange(n), index=graph["page_id"].to_numpy())
    nxt = np.where(graph["kind_page_id"] >= 0, index.reindex(graph["kind_page_id"]).fillna(-1).to_numpy(), -1).astype(int)
    below = kinds_below(nxt)

    con = duckdb.connect()
    named = con.sql(f"""
        with pages as (select page_id, title, redirect_target from read_parquet('{data}/{wiki}_links/pages/*.parquet')),
        resolve as (
            select p.title, coalesce(a0.page_id, a1.page_id) as page_id
            from pages p
            left join pages a0 on a0.title = p.title and a0.redirect_target is null
            left join pages a1 on a1.title = p.redirect_target and a1.redirect_target is null
            where coalesce(a0.page_id, a1.page_id) is not null
        )
        select r.page_id, count(distinct l.page_id) as named_in
        from read_parquet('{data}/{wiki}_relations/*.parquet') l join resolve r on r.title = l.target
        where l.relation <> 'aside' and r.page_id <> l.page_id group by 1
    """).df().set_index("page_id")["named_in"]

    out = pd.DataFrame({
        "page_id": graph["page_id"], "title": graph["title"], "kinds_below": below,
        "specificity": np.log2(n / below), "depth": np.where(nxt >= 0, graph["depth"], -1),
        "named_in": named.reindex(graph["page_id"]).fillna(0).to_numpy().astype(np.int64), "has_kind": nxt >= 0,
    })
    out.to_parquet(data / f"{wiki}_specificity.parquet", index=False)

    print(f"{n:,} articles; {(below > 1).sum():,} have other articles as kinds of them; "
          f"{(below == 1).sum():,} have none (specificity {np.log2(n):.1f})")
    print("\nkinds_below, articles:")
    edges = [1, 2, 10, 100, 1_000, 10_000, 100_000, 10 ** 9]
    print(pd.Series(np.histogram(below, edges)[0], index=["1", "2-9", "10-99", "100-999", "1,000-9,999", "10,000-99,999", "100,000+"]).to_string())

    by = out.drop_duplicates("title").set_index("title")
    # bigger = more general for the counts; depth runs the other way and only means something with a kind
    measures = {"kinds_below": by["kinds_below"], "named_in": by["named_in"],
                "depth (shallower)": -by["depth"].where(by["has_kind"])}
    scores = data / f"{wiki}_scores.parquet"
    if scores.exists():
        old = pd.read_parquet(scores, columns=["title", "in_deg", "pages_through", "level_all"]).drop_duplicates("title").set_index("title")
        measures.update({"lead in-links": old["in_deg"], "first-link traffic": old["pages_through"],
                         "earlier level (lower)": -old["level_all"]})
    rows = []
    for name, pairs in [("kind", KIND), ("scope", SCOPE)]:
        pairs = [(a, b) for a, b in pairs if a in by.index and b in by.index]
        for measure, values in measures.items():
            g = values.reindex([a for a, _ in pairs]).to_numpy(dtype=float)
            s = values.reindex([b for _, b in pairs]).to_numpy(dtype=float)
            rows.append({"pairs": f"{name} ({len(pairs)})", "measure": measure, "correct": int((g > s).sum()),
                         "tied": int((g == s).sum()), "wrong": int((g < s).sum()), "missing": int(np.isnan(g - s).sum())})
    print("\n== hand-picked pairs: does the measure rank the general one as more general? ==\n")
    print(pd.DataFrame(rows).to_string(index=False))
    wrong = [(a, b) for a, b in KIND + SCOPE if a in by.index and b in by.index and by.at[a, "kinds_below"] <= by.at[b, "kinds_below"]]
    print("\nkinds_below does not separate:", "; ".join(f"{a} ({by.at[a, 'kinds_below']:,}) / {b} ({by.at[b, 'kinds_below']:,})" for a, b in wrong))

    print("\n== examples ==\n")
    print(by.reindex([t for t in SPOT if t in by.index])[["kinds_below", "specificity", "depth", "named_in"]].round(1).to_string())
    print("\n== the 40 most general articles by kinds_below ==\n")
    print(out.nlargest(40, "kinds_below")[["title", "kinds_below", "specificity", "named_in"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()

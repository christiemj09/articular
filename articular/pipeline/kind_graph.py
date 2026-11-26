"""Follow "is a kind of" links upward from every article and describe the graph they make.

    uv run python -m articular.pipeline.kind_graph --wiki enwiki

Reads data/<wiki>_scores_kind.parquet (articular/pipeline/build_link_choice.py --rule kind), using only the links
the parser supplied (rule kind, kind_noun or kind_of), not the first-link fallback. Each article's chain
ends at an article with no kind of its own, or in a loop.

Writes data/<wiki>_kind_graph.parquet: page_id, title, kind_page_id (-1 for none), rule, depth (steps to
the end of the chain), end_page_id (where the chain ends; the lowest page id of a loop), in_loop.
Prints coverage, depths, the commonest endings with their first sentences, and some example chains.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

EXAMPLES = ["Paris", "Dog", "Oak", "Guitar", "Albert Einstein", "Jazz", "Triangle", "Philosophy", "Village",
            "Politician", "Film", "Song", "Tiger", "Hydrogen", "Association football", "Poland", "Quantum mechanics",
            "Electron", "Mathematics", "Banana", "Democracy", "Empirical evidence", "Moth", "River", "Novel"]


def follow(nxt):
    """depth, end and in_loop for every node of a graph where node i points to nxt[i] (-1 for none)."""
    n = len(nxt)
    depth, end, loop = np.full(n, -1), np.full(n, -1), np.zeros(n, bool)
    for i in range(n):
        if depth[i] >= 0:
            continue
        path, seen, j = [], {}, i
        while j != -1 and depth[j] < 0 and j not in seen:
            seen[j] = len(path)
            path.append(j)
            j = nxt[j]
        if j == -1:                 # the chain stops at an article with no kind
            last = path.pop()
            depth[last], end[last] = 0, last
            base, e, in_loop = 0, last, False
        elif j in seen:             # the chain closed a new loop
            cycle, path = path[seen[j]:], path[:seen[j]]
            e = min(cycle)
            for c in cycle:
                depth[c], end[c], loop[c] = 0, e, True
            base, in_loop = 0, True
        else:                       # the chain joined one already walked
            base, e, in_loop = depth[j], end[j], loop[j]
        for k, v in enumerate(reversed(path)):
            depth[v], end[v], loop[v] = base + k + 1, e, in_loop
    return depth, end, loop


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()
    data, wiki = args.data, args.wiki

    pages = duckdb.sql(f"""
        select page_id, title, first_link_page_id as dst, rule
        from '{data}/{wiki}_scores_kind.parquet' where not is_disambig order by page_id
    """).df()
    index = pd.Series(np.arange(len(pages)), index=pages["page_id"].values)
    parsed = pages["rule"].str.startswith("kind").values
    nxt = np.where(parsed, index.reindex(pages["dst"].values).fillna(-1).values.astype(int), -1)
    depth, end, loop = follow(nxt)
    title = pages["title"].values

    out = pd.DataFrame({
        "page_id": pages["page_id"], "title": title,
        "kind_page_id": np.where(nxt >= 0, pages["page_id"].values[nxt], -1), "rule": pages["rule"],
        "depth": depth, "end_page_id": pages["page_id"].values[end], "in_loop": loop,
    })
    out.to_parquet(data / f"{wiki}_kind_graph.parquet", index=False)

    has = nxt >= 0
    print(f"{len(pages):,} articles; {has.sum():,} ({100 * has.mean():.1f}%) have a kind")
    print(pages["rule"].value_counts().to_string())
    print(f"\nchains ending in a loop: {(loop & has).sum():,} articles; {len(set(end[loop])):,} loops")
    print("\nsteps to the end of the chain (articles with a kind):")
    print(pd.Series(depth[has]).value_counts().sort_index().to_string())

    sentences = duckdb.sql(f"""
        select page_id, first_sentence from read_parquet('{data}/{wiki}_links/pages/*.parquet')
        where redirect_target is null
    """).df().set_index("page_id")["first_sentence"]
    top = pd.Series(end[has]).value_counts().head(args.top)
    endings = pd.DataFrame({
        "ends_at": title[top.index], "articles": top.values, "loop": loop[top.index],
        "first_sentence": [str(sentences.get(pages["page_id"].values[e], ""))[:110] for e in top.index],
    })
    print(f"\n== where chains end: the {args.top} commonest ==\n")
    print(endings.to_string(index=False))

    direct = pd.Series(nxt[has]).value_counts().head(args.top)
    print(f"\n== the {args.top} commonest kinds (articles pointing straight at them) ==\n")
    print(pd.DataFrame({"kind": title[direct.index], "articles": direct.values}).to_string(index=False))

    print("\n== example chains ==\n")
    where = {t: i for i, t in enumerate(title)}
    for start in EXAMPLES:
        if start not in where:
            continue
        i, names, seen = where[start], [start], set()
        while nxt[i] >= 0 and i not in seen and len(names) < 20:
            seen.add(i)
            i = nxt[i]
            names.append(title[i])
        print(" → ".join(names) + (" ⟲" if nxt[i] >= 0 else ""))


if __name__ == "__main__":
    main()

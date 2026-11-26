"""Lay out the whole first-link graph and cut it into map tiles for the full-page viewer.

    uv run python -m articular.site.build_tiles --wiki enwiki --dump 2026-11-01 --out data/enwiki_site

Reads data/<wiki>_scores.parquet. Writes under <out>/:

    meta.json            world size, zoom levels, basins, lakes, search manifest
    tiles/index.json     which tiles exist at each zoom level
    tiles/<z>/<x>_<y>    pages first shown at zoom z inside that tile (most-traveled pages come first)
    idx/<k>              pages k*SHARD.. in title order: position, traffic, next page, biggest tributaries

With --kinds, tiles and idx files also carry each page's kind family ("c"), and meta.json names the
families ("kinds"), so the viewer can color the map by kind of thing instead of by basin.
With --specificity, they carry each page's specificity band ("g"), idx files also its rank ("s":
0 = most general, 100 = most specific), and meta.json names the bands ("specificity").

Layout: every page sits just upstream of the page its first link points to. Its springs (pages
nothing flows into) form a disc around it; its tributaries with their own catchments are set on an
arc beyond that disc, each catchment turned to face away from the confluence. Loops become lakes
with their member pages around the shore. Files are gzip-compressed unless --no-gzip is given.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

S = 1.0            # spacing between springs, in map units
GAP_R = 1.0        # clear water between a page's spring disc and its tributaries
# Layout variants. sector: radians of arc that tributaries may occupy, centered on "upstream".
# arc_limit: how far an arc may be pushed out (in units of its largest tributary) before the
# tributaries are spiralled around the page instead; below 0 always spirals. fan: radians that a
# page's springs spread over (a full disc is 2*pi). bend: how much channels wander.
LAYOUTS = {
    "burst": {"name": "Fireworks", "sector": 3.6, "arc_limit": 0.6, "fan": 2 * np.pi, "bend": 0.9},
    "branch": {"name": "Branches", "sector": 2.6, "arc_limit": 3.0, "fan": 2.6, "bend": 1.3},
    "bloom": {"name": "Blooms", "sector": 3.6, "arc_limit": -1.0, "fan": 2 * np.pi, "bend": 0.0},
}
GOLDEN = np.pi * (3 - np.sqrt(5))
Q = 4              # stored coordinates are map units * Q, as integers
SHARD = 2000
# Specificity bands: ranks below each bound, of 100. Each band is about ten times the size of the one
# before, as each zoom level shows about that many more pages; equal bands would leave the whole-map
# view, which shows only the most-traveled pages, in one color.
BANDS = [(0.01, "Most general 0.01%"), (0.1, "Top 0.1%"), (1, "Top 1%"), (10, "Top 10%"), (50, "Top half"),
         (100.01, "Most specific half")]


def rnd(i, salt):
    x = np.sin(i * 12.9898 + salt * 78.233) * 43758.5453
    return x - np.floor(x)


def ragged(starts, lens):
    """Indices start..start+len for each segment, concatenated, plus each index's segment number."""
    first = np.cumsum(lens) - lens
    return np.repeat(starts, lens) + np.arange(lens.sum()) - np.repeat(first, lens), np.repeat(np.arange(len(lens)), lens), first


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-cap", type=int, default=1500, help="most pages stored in one tile")
    parser.add_argument("--layout", choices=list(LAYOUTS), default="burst")
    parser.add_argument("--scores", type=Path, default=None,
                        help="table with one chosen link per article (default data/<wiki>_scores.parquet)")
    parser.add_argument("--rule", default="first link", help="how the link was chosen, shown in the viewer")
    parser.add_argument("--kinds", type=Path, default=None,
                        help="page_id -> family table from articular/pipeline/build_kind_families.py; its .json sits beside it")
    parser.add_argument("--specificity", type=Path, default=None,
                        help="page_id -> rank table from articular/pipeline/build_specificity_score.py")
    parser.add_argument("--specificity-column", default="specificity", help="the rank to use from that table")
    parser.add_argument("--no-gzip", action="store_true", help="write plain JSON (for a local file server)")
    args = parser.parse_args()

    lay = LAYOUTS[args.layout]
    SECTOR, FAN = lay["sector"], lay["fan"]

    def write(rel, obj):
        path = args.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        path.write_bytes(data if args.no_gzip else gzip.compress(data, 6))

    scores = pd.read_parquet(args.scores or args.data / f"{args.wiki}_scores.parquet",
                             columns=["page_id", "title", "is_disambig", "first_link_page_id"])
    n = len(scores)
    title = scores["title"].to_numpy()
    idx_of_id = pd.Series(np.arange(n), index=scores["page_id"].to_numpy())
    has_link = (scores["first_link_page_id"] >= 0).to_numpy()
    nxt = np.full(n, -1)
    nxt[has_link] = idx_of_id[scores["first_link_page_id"].to_numpy()[has_link]].to_numpy()

    # ---- Traffic, by peeling springs inward. `finished[r]` are pages whose tributaries are all done.
    traffic, remaining = np.ones(n, dtype=np.int64), np.bincount(nxt[nxt >= 0], minlength=n)
    frontier, finished = np.flatnonzero(remaining == 0), []
    while len(frontier):
        finished.append(frontier)
        f = frontier[nxt[frontier] >= 0]
        np.add.at(traffic, nxt[f], traffic[f])
        np.subtract.at(remaining, nxt[f], 1)
        hit = np.unique(nxt[f])
        frontier = hit[remaining[hit] == 0]
    on_cycle = remaining > 0
    tail = ~on_cycle & (nxt >= 0)
    spring = tail & (traffic == 1)
    river = tail & (traffic > 1)
    dry_root = ~on_cycle & (nxt < 0) & (traffic > 1)     # a dead end that other pages flow into
    print(f"{n:,} pages: {on_cycle.sum():,} on loops, {river.sum():,} rivers, {spring.sum():,} springs, "
          f"{dry_root.sum():,} dead ends with inflow, {(~on_cycle & (nxt < 0) & (traffic == 1)).sum():,} isolated")

    # ---- Loops: members in flow order
    canon, loops = np.full(n, -1), []
    for i in np.flatnonzero(on_cycle):
        if canon[i] < 0:
            members, j = [i], nxt[i]
            while j != i:
                members.append(j)
                j = nxt[j]
            canon[members] = len(loops)
            loops.append(np.array(members))

    # ---- Tributaries of each page, largest in the middle of the arc
    rk = np.flatnonzero(river)
    o = np.lexsort((-traffic[rk], nxt[rk]))
    rk, par = rk[o], nxt[rk[o]]
    rdeg = np.bincount(par, minlength=n)
    rstart = np.concatenate([[0], np.cumsum(rdeg)])[:-1]
    rank = np.arange(len(rk)) - rstart[par]
    rkids = rk[np.lexsort((np.where(rank % 2 == 1, rank, -rank), par))]

    nspring = np.bincount(nxt[spring], minlength=n)
    L = np.where(nspring > 0, S * np.sqrt(nspring * 2 * np.pi / FAN) + S, 0.6)   # reach of the springs
    ecx, ecy, enc_r = np.zeros(n), np.zeros(n), L.copy()        # circle enclosing the whole catchment
    theta, qx, qy = np.zeros(n), np.zeros(n), np.zeros(n)       # bearing and position in the parent's frame

    def place(P):
        P = P[rdeg[P] > 0]
        if not len(P):
            return
        m = rdeg[P]
        K, g, first = ragged(rstart[P], m)
        K = rkids[K]
        r = enc_r[K]
        lo = L[P] + GAP_R
        hi = lo + (np.pi / 2) * np.add.reduceat(2 * r, first) / SECTOR + 1
        for _ in range(40):   # smallest arc radius at which the tributaries fit side by side
            mid = (lo + hi) / 2
            fits = np.add.reduceat(2 * np.arcsin(r / (mid[g] + r)), first) <= SECTOR
            hi, lo = np.where(fits, mid, hi), np.where(fits, lo, mid)
        w = 2 * np.arcsin(r / (hi[g] + r))
        sum_w = np.add.reduceat(w, first)
        span = np.maximum(np.minimum(SECTOR, sum_w * 1.6 + 0.6), sum_w)
        ww = w + ((span - sum_w) / m)[g]
        before = np.cumsum(ww) - ww
        before -= np.repeat(before[first], m)
        th = -span[g] / 2 + before + ww / 2 + ((rnd(P, 3) - 0.5) * lay["bend"])[g]   # the last term makes channels wander
        d = hi[g] + r
        # A page with many sizeable tributaries would need a huge, empty arc. Spiral those around it
        # instead, largest nearest, each further out by the ground the earlier ones have used.
        by_size = np.lexsort((-r, g))
        rs = r[by_size]
        used = np.cumsum(4 * rs ** 2)
        used -= np.repeat((used - 4 * rs ** 2)[first], m) + 2 * rs ** 2
        base = (L[P] + GAP_R)[g]
        spiral_d, spiral_th = np.empty_like(r), np.empty_like(r)
        spiral_d[by_size] = np.maximum(base + 1.1 * rs, np.sqrt(base ** 2 + 1.5 * used / np.pi))
        spiral_th[by_size] = (np.arange(len(r)) - np.repeat(first, m)) * GOLDEN + (2 * np.pi * rnd(P, 4))[g]
        arc = ((hi <= L[P] + GAP_R + lay["arc_limit"] * np.maximum.reduceat(r, first)) | (m == 1))[g]
        d, th = np.where(arc, d, spiral_d), np.where(arc, th, spiral_th)
        cx, cy, c, s = d * np.cos(th), d * np.sin(th), np.cos(th), np.sin(th)
        theta[K] = th
        qx[K] = cx - (c * ecx[K] - s * ecy[K])
        qy[K] = cy - (s * ecx[K] + c * ecy[K])
        # A catchment's circle has the combined area of its parts and sits at their center of mass.
        # Strictly enclosing circles would waste half the ground at every even fork, and the map of a
        # deep tree would be almost all empty; this way neighbors interleave a little at the edges.
        area = np.add.reduceat(r ** 2, first) + L[P] ** 2
        ecx[P] = np.add.reduceat(cx * r ** 2, first) / area
        ecy[P] = np.add.reduceat(cy * r ** 2, first) / area
        enc_r[P] = np.sqrt(area)

    for f in finished:
        place(f)
    place(np.flatnonzero(on_cycle))

    # ---- Lakes: loop members around the shore, in flow order
    basin_pages = np.array([traffic[m].sum() for m in loops])
    lakes = []
    for k, members in enumerate(loops):
        r, shore = enc_r[members], 2 + 0.02 * np.sqrt(basin_pages[k])
        lo, hi = shore, shore + r.sum() * 2 + 1
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if (2 * np.arcsin(r / (mid + r))).sum() <= 2 * np.pi * 0.98 else (mid, hi)
        w = 2 * np.arcsin(r / (hi + r))
        w = w * (2 * np.pi / w.sum())
        th = np.cumsum(w) - w / 2
        d = hi + r
        cx, cy, c, s = d * np.cos(th), d * np.sin(th), np.cos(th), np.sin(th)
        theta[members] = th
        qx[members] = cx - (c * ecx[members] - s * ecy[members])
        qy[members] = cy - (s * ecx[members] + c * ecy[members])
        mx, my = ((cx - r).min() + (cx + r).max()) / 2, ((cy - r).min() + (cy + r).max()) / 2
        lakes.append({"shore": hi, "cx": mx, "cy": my, "r": max((np.hypot(cx - mx, cy - my) + r).max(), np.hypot(mx, my) + hi)})

    # ---- Pack the basins: largest first, in rows
    dry = np.flatnonzero(dry_root)
    radii = np.concatenate([[lake["r"] for lake in lakes], enc_r[dry]])
    order = np.argsort(-radii)
    row_width = max(2.4 * radii[order[0]], 1.15 * np.sqrt(((2.3 * radii + 4) ** 2).sum()))
    center_x, center_y = np.zeros(len(radii)), np.zeros(len(radii))
    x = y = row_h = 0.0
    for i in order:
        size = 2.3 * radii[i] + 4   # catchments spill a little past their nominal circle
        if x > 0 and x + size > row_width:
            x, y, row_h = 0.0, y + row_h, 0.0
        center_x[i], center_y[i] = x + size / 2, y + size / 2
        x, row_h = x + size, max(row_h, size)
    world = max(row_width, y + row_h)

    # ---- Global positions, from each basin's root upstream
    X, Y, PHI = np.zeros(n), np.zeros(n), np.zeros(n)
    root = np.full(n, -1)            # basin number: loops first, then dry basins
    for k, members in enumerate(loops):
        lakes[k]["x"], lakes[k]["y"] = center_x[k] - lakes[k]["cx"], center_y[k] - lakes[k]["cy"]
        X[members], Y[members], PHI[members] = lakes[k]["x"] + qx[members], lakes[k]["y"] + qy[members], theta[members]
        root[members] = k
    X[dry], Y[dry] = center_x[len(loops):] - ecx[dry], center_y[len(loops):] - ecy[dry]
    root[dry] = len(loops) + np.arange(len(dry))
    depth = np.zeros(n, dtype=np.int64)
    for f in reversed(finished):
        f = f[nxt[f] >= 0]
        depth[f] = depth[nxt[f]] + 1
        root[f] = root[nxt[f]]
    included = root >= 0
    rv = np.flatnonzero(river & included)
    rv = rv[np.argsort(depth[rv], kind="stable")]
    for level in np.split(rv, np.flatnonzero(np.diff(depth[rv])) + 1):
        p = nxt[level]
        c, s = np.cos(PHI[p]), np.sin(PHI[p])
        X[level], Y[level], PHI[level] = X[p] + c * qx[level] - s * qy[level], Y[p] + s * qx[level] + c * qy[level], PHI[p] + theta[level]
    sp = np.flatnonzero(spring & included)
    sp = sp[np.argsort(nxt[sp], kind="stable")]
    p = nxt[sp]
    k = np.arange(len(sp)) - np.concatenate([[0], np.cumsum(nspring)])[:-1][p]   # 0, 1, 2… around each page
    reach, bearing = S * np.sqrt((k + 1) * 2 * np.pi / FAN), PHI[p] + (k * GOLDEN) % FAN - FAN / 2
    X[sp], Y[sp] = X[p] + reach * np.cos(bearing), Y[p] + reach * np.sin(bearing)

    keep = np.flatnonzero(included)
    shift_x, shift_y = 8 - X[keep].min(), 8 - Y[keep].min()
    X, Y = X + shift_x, Y + shift_y
    for lake in lakes:
        lake["x"], lake["y"] = lake["x"] + shift_x, lake["y"] + shift_y
    world = max(X[keep].max(), Y[keep].max()) + 8
    print(f"placed {len(keep):,} pages in {len(loops):,} loop basins and {len(dry):,} dry basins; world {world:,.0f} units across")

    # ---- Ids in title order (so search can binary-search a small manifest), colors, integer coordinates
    lower = pd.Series(title[keep]).str.lower().to_numpy()
    by_title = np.argsort(lower, kind="stable")
    ident = np.full(n, -1)
    ident[keep[by_title]] = np.arange(len(keep))
    big = np.argsort(-basin_pages)[:8]
    color_of_loop = np.full(len(loops), 8)
    color_of_loop[big] = np.arange(len(big))
    color = np.where(root < len(loops), color_of_loop[np.clip(root, 0, len(loops) - 1)], 9)
    xi, yi = np.round(X * Q).astype(np.int64), np.round(Y * Q).astype(np.int64)
    kind_names, extra = None, lambda part: {}
    if args.kinds:
        kind_names = json.loads(args.kinds.with_suffix(".json").read_text())
        families = pd.read_parquet(args.kinds).set_index("page_id")["family"]
        family = families.reindex(scores["page_id"]).fillna(len(kind_names) - 1).to_numpy().astype(np.int64)
        extra = lambda part: {"c": family[part].tolist()}
    band, band_names = None, [name for _, name in BANDS] + ["Not scored"]
    if args.specificity:
        ranks = pd.read_parquet(args.specificity).set_index("page_id")[args.specificity_column]
        rank = ranks.reindex(scores["page_id"]).to_numpy()
        band = np.where(np.isnan(rank), len(BANDS), np.searchsorted([below for below, _ in BANDS], rank, side="right"))
        band = np.minimum(band, len(BANDS) - 1 + np.isnan(rank))
        tenths = np.where(np.isnan(rank), -1, np.round(np.nan_to_num(rank) * 10)).astype(np.int64)
        kinds_extra = extra
        extra = lambda part, records=False: {**kinds_extra(part), "g": band[part].tolist(), **({"s": tenths[part].tolist()} if records else {})}
    parent = np.where(nxt >= 0, nxt, np.arange(n))
    pxi, pyi = xi[parent], yi[parent]

    # ---- Search shards
    kid = np.flatnonzero(tail & included)
    kid = kid[np.lexsort((-traffic[kid], nxt[kid]))]
    kid_deg = np.bincount(nxt[kid], minlength=n)
    kid_rank = np.arange(len(kid)) - np.concatenate([[0], np.cumsum(kid_deg)])[:-1][nxt[kid]]
    top = np.full((n, 5), -1)
    top[nxt[kid[kid_rank < 5]], kid_rank[kid_rank < 5]] = ident[kid[kid_rank < 5]]
    ordered, manifest = keep[by_title], []
    for start in range(0, len(ordered), SHARD):
        part = ordered[start:start + SHARD]
        manifest.append(lower[by_title[start]])
        write(f"idx/{start // SHARD}", {
            "t": title[part].tolist(), "x": xi[part].tolist(), "y": yi[part].tolist(), "n": traffic[part].tolist(),
            "p": np.where(nxt[part] >= 0, ident[nxt[part]], -1).tolist(), "b": color[part].tolist(),
            "k": [[v for v in row if v >= 0] for row in top[part].tolist()], **(extra(part, True) if band is not None else extra(part)),
        })

    # ---- Tiles: the most-traveled pages claim the shallowest tile with room
    size = int(np.ceil(world * Q))
    todo = keep[np.argsort(-traffic[keep], kind="stable")]
    level_of, tile_of, z = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), 0
    while len(todo):
        cell = size / 2 ** z
        t = (xi[todo] // cell).astype(np.int64) * 2 ** z + (yi[todo] // cell).astype(np.int64)
        take = (pd.Series(t).groupby(t).cumcount().to_numpy() < args.tile_cap) | (cell < 24 * Q)
        level_of[todo[take]], tile_of[todo[take]] = z, t[take]
        todo, z = todo[~take], z + 1
    levels = z
    o = keep[np.lexsort((-traffic[keep], tile_of[keep], level_of[keep]))]
    key = level_of[o] * 2 ** 40 + tile_of[o]
    index = {}
    for part in np.split(o, np.flatnonzero(np.diff(key)) + 1):
        z, t = int(level_of[part[0]]), int(tile_of[part[0]])
        index.setdefault(z, []).append(t)
        write(f"tiles/{z}/{t >> z}_{t & (2 ** z - 1)}", {
            "i": ident[part].tolist(), "x": xi[part].tolist(), "y": yi[part].tolist(),
            "px": pxi[part].tolist(), "py": pyi[part].tolist(), "n": traffic[part].tolist(),
            "b": color[part].tolist(), "t": title[part].tolist(), **extra(part),
        })
    write("tiles/index.json", index)

    def loop_name(members):
        names = title[members[np.argsort(-traffic[members])]]
        return " ⟲ ".join(names[:2]) + (" …" if len(names) > 2 else "")

    # ---- Longest journeys: the pages furthest from where their path ends. A page one step along
    # a listed journey would be nearly as far, so a journey only counts if its first steps are its own.
    used, journeys = np.zeros(n, dtype=bool), []
    for v in keep[np.argsort(-depth[keep], kind="stable")][:500_000]:
        path, j = [], v
        while not used[j]:
            path.append(j)
            if on_cycle[j] or nxt[j] < 0:
                break
            j = nxt[j]
        if len(path) < min(6, depth[v] + 1) and journeys:
            continue
        joined = used[j]
        used[path] = True
        k = root[v]
        journeys.append({"id": int(ident[v]), "t": title[v], "steps": int(depth[v]), "own": len(path) - (0 if joined else 1),
                         "joins": title[j] if joined else None,
                         "end": loop_name(loops[k]) if k < len(loops) else title[dry[k - len(loops)]], "loop": bool(k < len(loops))})
        if len(journeys) == 100:
            break
    write("journeys.json", journeys)

    labeled = np.argsort(-basin_pages)[:400]
    isolated = int((~included).sum())
    write("meta.json", {
        "wiki": args.wiki, "dump": args.dump, "layout": args.layout, "layout_name": lay["name"], "rule": args.rule, "articles": n, "placed": int(len(keep)), "left_out": isolated,
        "loops": len(loops), "dry_basins": int(len(dry)), "dry_pages": int((color[keep] == 9).sum()),
        "size": size, "q": Q, "levels": levels, "shard": SHARD, "manifest": manifest,
        "lakes": [{"x": int(round(lakes[k]["x"] * Q)), "y": int(round(lakes[k]["y"] * Q)), "r": int(round(lakes[k]["shore"] * Q)),
                   "R": int(round(lakes[k]["r"] * Q)), "pages": int(basin_pages[k]), "b": int(color_of_loop[k]),
                   "name": loop_name(loops[k])} for k in labeled],
        **({"kinds": [{"name": name, "pages": int(c)} for name, c in
                      zip(kind_names, np.bincount(family[keep], minlength=len(kind_names)))]} if kind_names else {}),
        **({"specificity": [{"name": name, "pages": int(c)} for name, c in
                            zip(band_names, np.bincount(band[keep], minlength=len(band_names)))]} if band is not None else {}),
    })
    files = [p for p in args.out.rglob("*") if p.is_file()]
    print(f"{levels} zoom levels, {sum(len(v) for v in index.values()):,} tiles, {len(manifest):,} search shards; "
          f"{len(files):,} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

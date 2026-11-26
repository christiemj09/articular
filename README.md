# articular 📖

Compute specificity over Wikipedia pages.

The starting point was the
[Wikipedia philosophy phenomenon](https://en.wikipedia.org/wiki/Wikipedia_philosophy_phenomenon):
the claim that following the first link of an article, again and again, leads to Philosophy,
which would make the number of steps a rough measure of how specific a topic is. On the
November 2026 dump of English Wikipedia that no longer holds. Fewer than half of all articles
(45.5%) reach Philosophy by first links, and every chain ends in a loop or a dead end, most of
them in one of two loops: Existence and Reality, or a seven-article loop around Knowledge.

So this project looks for better measures of which of two related articles is the more general
one, and checks each against Wikidata's subclass-of, instance-of, part-of and located-in
statements.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

Everything runs from the repository root and reads and writes under `data/`, which is not
committed. Each module's docstring says what it reads, what it writes and how to run it:

```sh
uv run python -m articular.pipeline.extract_links data/enwiki/*.bz2 --out data/enwiki_links --workers 8
uv run python -m articular.experiments.score_text_features --wiki enwiki
```

Simple English Wikipedia (a few hundred MB) is enough to try any step on a laptop. English
Wikipedia needs a large machine: the runs so far used 8 cores, 64 GB of memory and a 200 GB disk.

## Layout

- `articular/pipeline/`: reads the dumps and builds the tables everything else uses: leads,
  links, parsed first sentences, hatnotes, the "kind of" graph, vocabulary and title measures,
  and the combined specificity score.
- `articular/experiments/`: one module per test. Each scores a measure, or a combination of
  measures, against the Wikidata pairs and prints its tables.
- `articular/site/`: builds the tiles and tables for the map viewer in `viz/full/index.html`.
- `articular/tools/`: small helpers for looking at single articles.
- `scripts/`: shell scripts: dump and Wikidata downloads, the site deploy, and `vm` and
  `vm-run` for working with an analysis VM on AWS.
- `notebooks/`: the early explorations of first links and link position, as
  [jupytext](https://jupytext.readthedocs.io/) `.py` files.

The embedding experiment needs an extra dependency group:

```sh
uv run --group embed python -m articular.experiments.family_embeddings data/enwiki/*.bz2 --wiki enwiki
```

## Data license

Wikipedia text is licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Tables derived from the dumps carry the same license and must credit Wikipedia contributors.
Wikidata statements are [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

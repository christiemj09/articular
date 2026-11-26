# articular 📖

Compute specificity over Wikipedia pages.

Following the first link of a Wikipedia article, again and again, usually ends at
[Philosophy](https://en.wikipedia.org/wiki/Wikipedia:Getting_to_Philosophy). The number of
steps it takes is a rough measure of how specific a topic is. This project tests that idea,
then looks for better measures of which of two related articles is the more general one,
checked against Wikidata's subclass-of, instance-of, part-of and located-in statements.

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

#!/usr/bin/env bash
# Build the maps and upload the site. Run on the VM.
#
#   scripts/deploy_site.sh <site-bucket> [wiki] [dump-date]
#
# Two maps of the same articles are built, differing in which link each article follows:
#   first   the first link in the lead (needs data/<wiki>_scores.parquet, from notebook 02)
#   kind    the link that says what kind of thing the article is, read by the parser, else the first link
#           (needs data/<wiki>_links/ and data/<wiki>_relations/, from articular/pipeline/extract_links.py and
#           articular/pipeline/extract_relations.py)
# Both carry each article's kind family, so the viewer can color either map by kind of thing, and,
# if data/<wiki>_specificity_score.parquet exists (articular/pipeline/build_specificity_score.py), each
# article's specificity band. If data/<wiki>_flow_stats.json (articular/experiments/compare_score_flow.py) and
# data/<wiki>_link_position.csv (articular/experiments/link_specificity_by_position.py) exist, the tables for the
# Structure page are gathered into structure.json.
# Data files are gzip-compressed on disk and uploaded with a matching Content-Encoding.
# Afterwards, clear the CDN cache from a machine with CloudFront access:
#   aws cloudfront create-invalidation --distribution-id <id> --paths '/*'
set -euo pipefail

bucket="$1"
wiki="${2:-enwiki}"
dump="${3:-2026-11-01}"
out="data/${wiki}_site"

if [ ! -d "data/${wiki}_links/links" ]; then
  uv run python -m articular.pipeline.extract_links "data/$wiki"/*.bz2 --out "data/${wiki}_links" 2>&1 | tail -1
fi
uv run python -m articular.pipeline.build_link_choice --wiki "$wiki" --rule kind | tail -1
uv run python -m articular.pipeline.kind_graph --wiki "$wiki" > "data/${wiki}_kind_graph.log"
uv run python -m articular.pipeline.build_kind_families --wiki "$wiki" > "data/${wiki}_kind_families.log"
grep -v "no such article" "data/${wiki}_kind_families.log" | tail -120
kinds="data/${wiki}_kind_families.parquet"
spec=()
if [ -f "data/${wiki}_specificity_score.parquet" ]; then
  spec=(--specificity "data/${wiki}_specificity_score.parquet" --specificity-column "no first-200-word vocabulary, others clipped")
fi

rm -rf "$out"
uv run python -m articular.site.build_tiles --wiki "$wiki" --dump "$dump" --rule "first link" --kinds "$kinds" "${spec[@]}" --out "$out/first" | tail -2
uv run python -m articular.site.build_tiles --wiki "$wiki" --dump "$dump" --rule "kind link (the link that says what kind of thing it is), or its first link where the opening sentence has none" \
  --scores "data/${wiki}_scores_kind.parquet" --kinds "$kinds" "${spec[@]}" --out "$out/kind" | tail -2
printf '%s' '[{"id":"first","name":"First link","path":"first/","about":"Each article follows the first link in its lead"},{"id":"kind","name":"Kind link","path":"kind/","about":"Each article follows the link that says what kind of thing it is"}]' | gzip > "$out/variants.json"

if [ -f "data/${wiki}_flow_stats.json" ] && [ -f "data/${wiki}_link_position.csv" ]; then      # the Structure page's charts
  uv run python -m articular.site.build_structure --wiki "$wiki" --out "$out"
fi

aws s3 sync --only-show-errors --delete --content-encoding gzip --content-type application/json "$out" "s3://$bucket/data/"
aws s3 cp --only-show-errors --content-type text/html viz/full/index.html "s3://$bucket/index.html"
aws s3 ls --recursive --summarize "s3://$bucket/" | tail -2

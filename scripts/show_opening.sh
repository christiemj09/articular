#!/usr/bin/env bash
# Print the opening of an article straight from the dump on the VM: the first lines of its
# wikitext that carry the bold title.
#
#   scripts/show_opening.sh <page-id> "<title>" [wiki-dir]
#
# The dump file is picked by the page-id range in its name.
cd "$(dirname "$0")/.."
id="$1"; title="$2"; dir="${3:-data/enwiki}"
for f in "$dir"/*.bz2; do
  range="${f##*.xml-p}"; range="${range%.bz2}"
  lo="${range%%p*}"; hi="${range##*p}"
  if [ "$id" -ge "$lo" ] && [ "$id" -le "$hi" ]; then
    echo "== $title ($(basename "$f"))"
    bzcat "$f" | grep -m1 -A80 -F "<title>$title</title>" | grep -m2 -F "'''" | cut -c1-900
  fi
done

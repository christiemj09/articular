#!/usr/bin/env bash
# Fetch Wikidata's "truthy" dump and keep only the relations we score against. Run on the VM.
#
#   scripts/fetch_wikidata.sh [wiki] [dump-date] [dest]     defaults: enwiki 20261101 data/wd
#
# Writes <dest>/statements.tsv.gz with lines "item<TAB>property<TAB>value" for
#   P31 instance of, P279 subclass of, P361 part of, P131 located in, P17 country, P106 occupation
# and downloads <wiki>'s page_props table, which maps each article to its Wikidata item.
# The 72 GB dump is downloaded to disk (resumable), filtered in one pass, then deleted.
set -euo pipefail

wiki="${1:-enwiki}"
dump="${2:-20261101}"
dest="${3:-data/wd}"
ua="articular/0.1 (research)"
mkdir -p "$dest"

wget -c -nv -U "$ua" "https://dumps.wikimedia.org/$wiki/$dump/$wiki-$dump-page_props.sql.gz" -O "$dest/$wiki-page_props.sql.gz"

if [ ! -f "$dest/statements.tsv.gz" ]; then
  wget -c -nv -U "$ua" "https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz" -O "$dest/truthy.nt.gz"
  gzip -dc "$dest/truthy.nt.gz" \
    | LC_ALL=C grep -E '^<http://www.wikidata.org/entity/Q[0-9]+> <http://www.wikidata.org/prop/direct/(P31|P279|P361|P131|P17|P106)> <http://www.wikidata.org/entity/Q' \
    | sed -E 's#^<http://www.wikidata.org/entity/(Q[0-9]+)> <http://www.wikidata.org/prop/direct/(P[0-9]+)> <http://www.wikidata.org/entity/(Q[0-9]+)> \.$#\1\t\2\t\3#' \
    | gzip > "$dest/statements.tmp.gz"
  mv "$dest/statements.tmp.gz" "$dest/statements.tsv.gz"
  rm -f "$dest/truthy.nt.gz" "$dest"/sample.*
fi
ls -l "$dest"
echo "done"

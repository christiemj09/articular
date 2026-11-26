#!/usr/bin/env bash
# Download a wiki's current-revisions article dump (no edit history) and verify it.
#
#   scripts/download_dump.sh [wiki] [dest]     defaults: enwiki data
#
# Fetches the split pages-articles files (a few dozen parts), which can be extracted in
# parallel, one worker per file. The dump date is pinned by the saved checksum file, so an
# interrupted run resumes the same dump. Wikimedia allows two connections per client.
set -euo pipefail

wiki="${1:-enwiki}"
dest="${2:-data}"
ua="articular/0.1 (research)"

mkdir -p "$dest/$wiki"
cd "$dest/$wiki"

if [ ! -f md5sums.txt ]; then
  curl -fsSL -A "$ua" "https://dumps.wikimedia.org/$wiki/latest/$wiki-latest-md5sums.txt" -o md5sums.txt
fi
grep -E "  $wiki-[0-9]{8}-pages-articles[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2$" md5sums.txt > parts.md5
date="$(head -1 parts.md5 | cut -d- -f2)"
echo "dump date: $date, $(wc -l < parts.md5) files"

awk -v base="https://dumps.wikimedia.org/$wiki/$date/" '{print base $2}' parts.md5 \
  | xargs -P 2 -n 1 wget -c -nv -U "$ua"

md5sum -c --quiet parts.md5
echo "done: all files verified"

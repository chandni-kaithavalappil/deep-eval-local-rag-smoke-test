#!/usr/bin/env bash
# Fetch a small Anthropic-themed corpus for the LightRAG ingest.
# Uses publicly available pages. If a URL is gated or moved, the script
# logs a warning and continues — the experiment is robust to missing docs.

set -euo pipefail
cd "$(dirname "$0")"

declare -a URLS=(
  "https://www.anthropic.com/news/core-views-on-ai-safety"
  "https://www.anthropic.com/news/anthropics-responsible-scaling-policy"
  "https://www.anthropic.com/news/claudes-constitution"
  "https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback"
  "https://www.anthropic.com/news/decomposing-language-models-into-understandable-components"
  "https://www.anthropic.com/news/towards-monosemanticity-decomposing-language-models-with-dictionary-learning"
)

i=1
for url in "${URLS[@]}"; do
  fname=$(printf "doc_%02d.md" "$i")
  echo "[$i] $url -> $fname"
  if curl -sSL --fail "$url" -o "_raw_$i.html"; then
    # Strip HTML to plain text — keeps the experiment dependency-free
    python3 - <<PY
import re, sys, pathlib
html = pathlib.Path("_raw_$i.html").read_text(encoding="utf-8", errors="ignore")
# Drop scripts/styles, then strip tags
html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
text = re.sub(r"<[^>]+>", " ", html)
text = re.sub(r"\s+", " ", text).strip()
pathlib.Path("$fname").write_text(f"Source: $url\n\n{text}\n", encoding="utf-8")
PY
    rm -f "_raw_$i.html"
  else
    echo "  ! failed to fetch — skipping"
  fi
  i=$((i + 1))
done

echo
echo "Corpus ready in $(pwd):"
ls -lh doc_*.md 2>/dev/null || echo "  (no docs were fetched — check network)"

#!/usr/bin/env bash
# Run the full generalized fast color clustering analysis pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -d .venv ]; then
  echo "No .venv found — run scripts/setup.sh first."
  exit 1
fi
echo "Running analysis pipeline..."
.venv/bin/python analysis.py
echo "Done. Outputs written to output/"

#!/usr/bin/env bash
# Remove generated environment and cache files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf .venv __pycache__ *.pyc
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "Cleaned up .venv and __pycache__."

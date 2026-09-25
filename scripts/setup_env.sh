#!/usr/bin/env bash
# Set up a fresh machine to run everything in PLAN.md (CPU is enough).
#
#   scripts/setup_env.sh          # install dependencies
#   scripts/setup_env.sh --data   # ... and fetch the Phase 3b/3c data now
#
# The CPU wheel of PyTorch is installed first so pip does not pull the
# multi-GB CUDA build; drop --index-url on a GPU machine.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,real]"
python -m pytest -q
if [ "${1:-}" = "--data" ]; then
  python scripts/fetch_quickdraw.py
  python scripts/fetch_real_data.py
fi
echo "ready: run scripts/run_all.sh"

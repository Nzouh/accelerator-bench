#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.venvs/accelerator-bench/bin/activate"
exec accelerator-bench hub sweep --device 'Samsung Galaxy S24' --qairt-version 2.50 \
  --manifest artifacts/hub/manifest.json --repeats 3 --max-active 4 --compile-options-only \
  --output-dir results/hub-s24-sweep-2026-10-13 "$@"

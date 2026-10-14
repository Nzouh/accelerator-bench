#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.venvs/accelerator-bench/bin/activate"
accelerator-bench hub sweep --device 'Samsung Galaxy S24' --qairt-version 2.50 \
  --manifest artifacts/hub-nhwc/manifest.json --variants optimized --repeats 3 --max-active 2 \
  --output-dir results/hub-s24-nhwc-2026-10-13 "$@"

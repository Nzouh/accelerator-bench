#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.venvs/accelerator-bench/bin/activate"
accelerator-bench run --output results/rtx3060-exploratory-2026-10-12.json \
  --context "EXPLORATORY: concurrent desktop/GPU apps present, including Rainbow Six and BlueStacks. User chose to keep them running. Power includes unrelated activity; this is not an isolated hardware comparison. AC state not confirmed."

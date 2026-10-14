#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.venvs/accelerator-bench/bin/activate"
accelerator-bench run --output results/rtx3060-ac-2026-10-13.json \
  --context "AC-powered desktop run: Windows confirmed AC power and Balanced power plan before starting; idle GPU preflight recorded separately. Rainbow Six was closed. Resident desktop apps remain; background activity is not isolated. GPU clocks and power limits were not changed. One full session, not an independent-session repeatability study. See results/rtx3060-ac-2026-10-13-host.json for before/after context."

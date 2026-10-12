#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv "$HOME/.venvs/accelerator-bench"
source "$HOME/.venvs/accelerator-bench/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,hub]'
accelerator-bench doctor

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[Run 1/3] warmup with reset"
python main.py --stage warmup --reset

echo "[Run 2/3] warmup without reset"
python main.py --stage warmup

echo "[Run 3/3] warmup without reset"
python main.py --stage warmup

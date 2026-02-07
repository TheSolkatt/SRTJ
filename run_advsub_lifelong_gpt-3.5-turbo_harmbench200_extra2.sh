#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

target_model="${TARGET_MODEL:-gpt-3.5-turbo-1106}"
safe_model="$(printf '%s' "$target_model" | sed 's/[^a-zA-Z0-9_.-]/_/g')"
python_cmd="${PYTHON_CMD:-python}"

save_interval="${SAVE_INTERVAL:-0}"
save_per_goal="${SAVE_PER_GOAL:-1}"
save_flags="--save-interval ${save_interval}"
if [ "${save_per_goal}" != "0" ]; then
  save_flags="${save_flags} --save-per-goal"
fi

echo "[Env] Using ${python_cmd}: $(${python_cmd} -V 2>&1)"
${python_cmd} - <<'PY'
import importlib

mods = ["openai", "tenacity", "dotenv", "clingo", "sentence_transformers"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[Env] Missing packages (check your conda env): {missing}")
print("[Env] OK: core dependencies importable")
PY
warmup_dataset="adv_subset_50"
harmbench_path="data/harmbench_200.json"

# Naming convention matches the main script
library_lifelong_prefix="$repo_root/library_${safe_model}_${warmup_dataset}_lifelong"

# We assume Run 1 is already done and exists
start_lib="${library_lifelong_prefix}_run1"

if [ ! -d "$start_lib" ]; then
  echo "[Error] Start library not found: $start_lib"
  echo "Expected Run 1 to be completed."
  exit 1
fi

log_root="${LOG_ROOT:-$repo_root/logs/${safe_model}}"
log_dir="$log_root"
log_subdir="${LOG_SUBDIR:-}"
if [ -n "$log_subdir" ]; then
  log_dir="$log_dir/$log_subdir"
fi
mkdir -p "$log_dir"
# Use a distinctive log name so we know this was the extension run
log_json="$log_dir/advsub_harmbench200_resume_runs23_$(date +%Y%m%d_%H%M%S).json"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
script_start="$(date -Is)"
runs_file="$(mktemp)"
combined_stdout="$(mktemp)"
combined_stderr="$(mktemp)"

run_cmd() {
  local run_index="$1"
  local label="$2"
  local cmd="$3"
  local log_path="${4:-}"
  local lib_path="${5:-}"
  local start_ts end_ts exit_code
  local stdout_file stderr_file

  if [ -n "$log_path" ]; then
    # Auto-resume logic (same as original)
    if [ -f "$log_path" ]; then
      resume_from="$(
        LOG_PATH="$log_path" ${python_cmd} - <<'PY'
import csv, os, sys, tempfile
log_path = os.environ["LOG_PATH"]
if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
    print("")
    sys.exit(0)
with open(log_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("")
    sys.exit(0)
last = rows[-1]
goal_text = (last.get("goal") or "").strip()
success = str(last.get("success", "")).strip().lower() == "true"
if success or not goal_text:
    print(log_path)
    sys.exit(0)
tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="")
with open(log_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(tmp, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        g = (row.get("goal") or "").strip()
        if g == goal_text:
            continue
        writer.writerow(row)
tmp.close()
print(tmp.name)
PY
      )"
      if [ -n "$resume_from" ]; then
        cmd="$cmd --resume-from-log $resume_from"
      fi
    fi
    cmd="$cmd --log-path $log_path"
  fi

  echo "$label"
  start_ts="$(date -Is)"
  stdout_file="$(mktemp)"
  stderr_file="$(mktemp)"

  set +e
  bash -c "$cmd" > >(tee -a "$stdout_file" "$combined_stdout") \
    2> >(tee -a "$stderr_file" "$combined_stderr" >&2)
  exit_code=$?
  set -e

  end_ts="$(date -Is)"

  RUN_INDEX="$run_index" RUN_LABEL="$label" RUN_CMD="$cmd" \
  RUN_START="$start_ts" RUN_END="$end_ts" RUN_EXIT="$exit_code" \
  RUN_STDOUT="$stdout_file" RUN_STDERR="$stderr_file" \
  LIB_ROOT="$lib_path" RUN_LOG_PATH="$log_path" \
  ${python_cmd} - <<'PY' >> "$runs_file"
import json
import os
from pathlib import Path

data = {
    "run_index": int(os.environ["RUN_INDEX"]),
    "label": os.environ["RUN_LABEL"],
    "command": os.environ["RUN_CMD"],
    "log_path": os.environ.get("RUN_LOG_PATH") or None,
    "start_time": os.environ["RUN_START"],
    "end_time": os.environ["RUN_END"],
    "exit_code": int(os.environ["RUN_EXIT"]),
}
with open(os.environ["RUN_STDOUT"], encoding="utf-8") as f:
    data["stdout"] = f.read()
with open(os.environ["RUN_STDERR"], encoding="utf-8") as f:
    data["stderr"] = f.read()

lib_root = Path(os.environ["LIB_ROOT"])
def _count_rules(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return 0
        parsed = json.loads(content)
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0

l1 = _count_rules(lib_root / "layer1_candidates.json")
l2 = _count_rules(lib_root / "layer2_buffer.json")
l3 = _count_rules(lib_root / "layer3_long_term.json")
data["library_state"] = {
    "layer1": l1,
    "layer2": l2,
    "layer3": l3,
    "total": l1 + l2 + l3,
}
print(json.dumps(data, ensure_ascii=False))
PY

  rm -f "$stdout_file" "$stderr_file"
}

prev_lib="$start_lib"

# Continue with Run 2 and Run 3
for i in 2 3; do
  run_lib="${library_lifelong_prefix}_run${i}"
  
  if [ -d "$run_lib" ]; then
    backup_root="${run_lib}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[Init] Existing library found at $run_lib. Moving to $backup_root"
    mv "$run_lib" "$backup_root"
  fi

  echo "[Init] Copying prev run lib to $run_lib"
  cp -R "$prev_lib" "$run_lib"

  run_cmd "$i" "[Run ${i}/3] lifelong without reset (harmbench_200)" \
    "${python_cmd} main.py --stage lifelong --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${run_lib} ${save_flags}" \
    "$log_dir/lifelong_${safe_model}_${run_tag}_run${i}.csv" \
    "$run_lib"

  prev_lib="$run_lib"
done

# Frozen Eval (based on Run 3)
frozen_lib="${library_lifelong_prefix}_run3_frozen"
if [ -d "$frozen_lib" ]; then
    backup_root="${frozen_lib}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[Init] Existing frozen library found. Moving to $backup_root"
    mv "$frozen_lib" "$backup_root"
fi
cp -R "$prev_lib" "$frozen_lib"

run_cmd 5 "[Run 1/1] frozen eval (harmbench_200)" \
  "${python_cmd} main.py --stage lifelong --frozen --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${frozen_lib} ${save_flags}" \
  "$log_dir/lifelong_frozen_${safe_model}_${run_tag}.csv" \
  "$frozen_lib"

script_end="$(date -Is)"

LOG_JSON="$log_json" RUNS_FILE="$runs_file" \
COMBINED_STDOUT="$combined_stdout" COMBINED_STDERR="$combined_stderr" \
SCRIPT_START="$script_start" SCRIPT_END="$script_end" \
TARGET_MODEL="$target_model" HARMBENCH_PATH="$harmbench_path" \
SAVE_INTERVAL="$save_interval" SAVE_PER_GOAL="$save_per_goal" LOG_DIR="$log_dir" \
${python_cmd} - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

runs_path = Path(os.environ["RUNS_FILE"])
run_lines = runs_path.read_text(encoding="utf-8").splitlines() if runs_path.exists() else []
runs = [json.loads(line) for line in run_lines if line.strip()]

combined_stdout = Path(os.environ["COMBINED_STDOUT"]).read_text(encoding="utf-8")
combined_stderr = Path(os.environ["COMBINED_STDERR"]).read_text(encoding="utf-8")

def _cmd_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=Path.cwd(), stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

git_head = _cmd_output(["git", "rev-parse", "HEAD"])
git_status = _cmd_output(["git", "status", "--porcelain=v1"])

payload = {
    "script_start": os.environ["SCRIPT_START"],
    "script_end": os.environ["SCRIPT_END"],
    "meta": {
        "cwd": str(Path.cwd()),
        "log_dir": os.environ.get("LOG_DIR") or None,
        "target_model": os.environ.get("TARGET_MODEL") or None,
        "harmbench_path": os.environ.get("HARMBENCH_PATH") or None,
        "save_interval": (lambda v: int(v) if str(v).strip().isdigit() else 0)(os.environ.get("SAVE_INTERVAL") or 0),
        "save_per_goal": str(os.environ.get("SAVE_PER_GOAL") or "").strip() not in {"0", "false", "False"},
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "git": {
            "head": git_head or None,
            "dirty": bool(git_status.strip()),
            "status": git_status.splitlines()[:200] if git_status else [],
        },
    },
    "runs": runs,
    "combined_stdout": combined_stdout,
    "combined_stderr": combined_stderr,
}

Path(os.environ["LOG_JSON"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

rm -f "$runs_file" "$combined_stdout" "$combined_stderr"

echo "[Done] JSON log saved to $log_json"

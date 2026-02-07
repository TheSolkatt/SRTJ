#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

script_basename="$(basename "${BASH_SOURCE[0]}")"

# Directory/log label comes from the script name (keeps paths stable and readable).
model_alias="${script_basename#run_advsub_lifelong_}"
model_alias="${model_alias%.sh}"
if [ "$model_alias" = "gpt-3.5-turbo" ]; then
  model_alias="gpt-3.5-turbo-1106"
fi

# Actual target model id (can include provider prefixes like openrouter).
default_target_model="$model_alias"
case "$model_alias" in
  claude-3-5-sonnet-20240620)
    default_target_model="anthropic/claude-3.5-sonnet-20240620"
    ;;
  gpt-4o)
    default_target_model="openai/gpt-4o-2024-08-06"
    ;;
  llama-3-8B-Instruct)
    default_target_model="meta-llama/llama-3-8b-instruct"
    ;;
  llama-3-70B-Instruct)
    default_target_model="meta-llama/llama-3-70b-instruct"
    ;;
esac

target_model="${TARGET_MODEL:-$default_target_model}"
safe_model="$(printf '%s' "$model_alias" | sed 's/[^a-zA-Z0-9_.-]/_/g')"
python_cmd="${PYTHON_CMD:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Speed/robustness knobs (do NOT change algorithmic behavior; only persistence frequency).
# - SAVE_INTERVAL=0 means only save on explicit flush (we pass --save-per-goal to keep crash safety).
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
warmup_path="data/adv_subset_50.json"
harmbench_path="data/harmbench_200.json"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
frozen_only="${FROZEN_ONLY:-0}"

if [ "$frozen_only" != "0" ] && [ -z "${RUN_TAG:-}" ]; then
  echo "[Error] FROZEN_ONLY requires RUN_TAG to locate the existing run5 library."
  exit 1
fi

log_root="${LOG_ROOT:-$repo_root/logs/${safe_model}}"
log_dir="$log_root"
log_subdir="${LOG_SUBDIR:-}"
if [ -n "$log_subdir" ]; then
  log_dir="$log_dir/$log_subdir"
fi
mkdir -p "$log_dir"
log_json="$log_dir/advsub_harmbench200_${run_tag}.json"
script_start="$(date -Is)"
runs_file="$(mktemp)"
combined_stdout="$(mktemp)"
combined_stderr="$(mktemp)"

# Rule libraries: keep everything under libreay_2.5/<model>/ to avoid mixing runs.
# Each invocation uses a per-run tag in directory names so previous runs stay intact.
library_base="${LIBRARY_BASE:-$repo_root/libreay_2.5/$safe_model}"
mkdir -p "$library_base"
library_root="$library_base/warmup_${run_tag}"
library_lifelong_prefix="$library_base/lifelong_${run_tag}"
base_library="$repo_root/library"

warmup_log="$log_dir/warmup_${safe_model}_${run_tag}.csv"
if [ "$frozen_only" = "0" ]; then
  if [ -f "$warmup_log" ] && [ ! -d "$library_root" ]; then
    echo "[Error] Warmup log exists but library dir missing: $library_root"
    echo "Set RUN_TAG to start a fresh run, or restore the library dir to resume."
    exit 1
  fi
  if [ -d "$library_root" ] && [ ! -f "$warmup_log" ]; then
    backup_root="${library_root}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[Init] Existing warmup library found. Moving to $backup_root"
    mv "$library_root" "$backup_root"
  fi
  mkdir -p "$library_root"
  for filename in solver.lp asp_config.json ontology.json; do
    if [ -f "$base_library/$filename" ] && [ ! -f "$library_root/$filename" ]; then
      cp "$base_library/$filename" "$library_root/$filename"
    fi
  done
fi

run_cmd() {
  local run_index="$1"
  local label="$2"
  local cmd="$3"
  local log_path="${4:-}"
  local lib_path="${5:-$library_root}"
  local start_ts end_ts exit_code
  local stdout_file stderr_file

  if [ -n "$log_path" ]; then
    # Auto-resume: if last goal failed, rerun it with full attempts.
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
# Exclude the last (failed) goal so it will be rerun.
tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="")
writer = None
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

if [ "$frozen_only" = "0" ]; then
  warmup_reset="--reset"
  warmup_label="[Run 1/1] warmup (reset) (${warmup_dataset})"
  if [ -f "$warmup_log" ]; then
    # Resume warmup without wiping the existing library state.
    warmup_reset=""
    warmup_label="[Run 1/1] warmup (resume) (${warmup_dataset})"
  fi

  run_cmd 1 "$warmup_label" \
    "${python_cmd} main.py --stage warmup --warmup-path ${warmup_path} ${warmup_reset} --target-model ${target_model} --library-root ${library_root} ${save_flags}" \
    "$warmup_log" \
    "$library_root"

  prev_lib="$library_root"
  for i in 1 2 3 4 5; do
    run_lib="${library_lifelong_prefix}_run${i}"
    run_log="$log_dir/lifelong_${safe_model}_${run_tag}_run${i}.csv"

    if [ -f "$run_log" ] && [ ! -d "$run_lib" ]; then
      echo "[Error] Lifelong log exists but library dir missing: $run_lib"
      echo "Restore the library dir to resume, or set RUN_TAG to start a fresh run."
      exit 1
    fi

    if [ -d "$run_lib" ]; then
      if [ -f "$run_log" ]; then
        echo "[Init] Found existing run lib+log; resuming in-place: $run_lib"
      else
        backup_root="${run_lib}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[Init] Existing run lib found without log. Moving to $backup_root"
        mv "$run_lib" "$backup_root"
        echo "[Init] Copying prev run lib to $run_lib"
        cp -R "$prev_lib" "$run_lib"
      fi
    else
      echo "[Init] Copying prev run lib to $run_lib"
      cp -R "$prev_lib" "$run_lib"
    fi

    run_cmd $((i + 1)) "[Run ${i}/3] lifelong without reset (harmbench_200)" \
      "${python_cmd} main.py --stage lifelong --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${run_lib} ${save_flags}" \
      "$run_log" \
      "$run_lib"

    prev_lib="$run_lib"
  done
else
  prev_lib="${library_lifelong_prefix}_run5"
  if [ ! -d "$prev_lib" ]; then
    echo "[Error] Lifelong run5 library not found: $prev_lib"
    echo "Provide the correct RUN_TAG or complete run5 before frozen."
    exit 1
  fi
fi

frozen_lib="${library_lifelong_prefix}_run5_frozen"
frozen_log="$log_dir/lifelong_frozen_${safe_model}_${run_tag}_run5.csv"
if [ -f "$frozen_log" ] && [ ! -d "$frozen_lib" ]; then
  echo "[Error] Frozen log exists but library dir missing: $frozen_lib"
  echo "Restore the library dir to resume, or set RUN_TAG to start a fresh run."
  exit 1
fi
if [ -d "$frozen_lib" ]; then
  if [ -f "$frozen_log" ]; then
    echo "[Init] Found existing frozen lib+log; resuming in-place: $frozen_lib"
  else
    backup_root="${frozen_lib}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[Init] Existing frozen lib found without log. Moving to $backup_root"
    mv "$frozen_lib" "$backup_root"
    cp -R "$prev_lib" "$frozen_lib"
  fi
else
  cp -R "$prev_lib" "$frozen_lib"
fi

run_cmd 5 "[Run 1/1] frozen eval (harmbench_200)" \
  "${python_cmd} main.py --stage lifelong --frozen --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${frozen_lib} ${save_flags}" \
  "$frozen_log" \
  "$frozen_lib"

script_end="$(date -Is)"

LOG_JSON="$log_json" RUNS_FILE="$runs_file" \
COMBINED_STDOUT="$combined_stdout" COMBINED_STDERR="$combined_stderr" \
SCRIPT_START="$script_start" SCRIPT_END="$script_end" \
TARGET_MODEL="$target_model" WARMUP_PATH="$warmup_path" HARMBENCH_PATH="$harmbench_path" \
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
        "warmup_path": os.environ.get("WARMUP_PATH") or None,
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

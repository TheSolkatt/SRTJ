#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_cmd="${PYTHON_CMD:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

model_alias="${MODEL_ALIAS:-}"
run_tag="${RUN_TAG:-}"
run_index="${RUN_INDEX:-}"
if [ -z "$model_alias" ] || [ -z "$run_tag" ] || [ -z "$run_index" ]; then
  echo "[Usage] MODEL_ALIAS=<name> RUN_TAG=<YYYYMMDD_HHMMSS> RUN_INDEX=<lifelong_run_idx> bash scripts/run_frozen_runs12.sh"
  echo "Example: MODEL_ALIAS=gpt-4o RUN_TAG=20260205_112651 RUN_INDEX=4 bash scripts/run_frozen_runs12.sh"
  exit 1
fi

if [ "$model_alias" = "gpt-3.5-turbo" ]; then
  model_alias="gpt-3.5-turbo-1106"
fi

default_target_model="$model_alias"
case "$model_alias" in
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

save_interval="${SAVE_INTERVAL:-0}"
save_per_goal="${SAVE_PER_GOAL:-1}"
save_flags="--save-interval ${save_interval}"
if [ "${save_per_goal}" != "0" ]; then
  save_flags="${save_flags} --save-per-goal"
fi

log_root="${LOG_ROOT:-$repo_root/logs/${model_alias}}"
log_dir="$log_root"
log_subdir="${LOG_SUBDIR:-}"
if [ -n "$log_subdir" ]; then
  log_dir="$log_dir/$log_subdir"
fi
mkdir -p "$log_dir"

library_base="${LIBRARY_BASE:-$repo_root/libreay_2.5/$model_alias}"
lifelong_prefix="$library_base/lifelong_${run_tag}"
source_lib="${lifelong_prefix}_run${run_index}"
frozen_lib="${lifelong_prefix}_run${run_index}_frozen"
frozen_log="$log_dir/lifelong_frozen_${model_alias}_${run_tag}_run${run_index}.csv"

harmbench_path="${HARMBENCH_PATH:-data/harmbench_200.json}"

prepare_frozen_dir() {
  local source_lib="$1"
  local frozen_lib="$2"
  local frozen_log="$3"

  if [ ! -d "$source_lib" ]; then
    echo "[Error] Source library not found: $source_lib"
    exit 1
  fi
  if [ -f "$frozen_log" ] && [ ! -d "$frozen_lib" ]; then
    echo "[Error] Frozen log exists but library dir missing: $frozen_lib"
    exit 1
  fi
  if [ -d "$frozen_lib" ]; then
    if [ -f "$frozen_log" ]; then
      echo "[Init] Found frozen lib+log; resuming in-place: $frozen_lib"
      return
    fi
    backup_root="${frozen_lib}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[Init] Existing frozen lib found without log. Moving to $backup_root"
    mv "$frozen_lib" "$backup_root"
  fi
  echo "[Init] Copying $source_lib -> $frozen_lib"
  cp -R "$source_lib" "$frozen_lib"
}

run_cmd() {
  local label="$1"
  local cmd="$2"
  local log_path="$3"

  if [ -n "$log_path" ] && [ -f "$log_path" ]; then
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

  echo "$label"
  cmd="$cmd --log-path $log_path"
  bash -c "$cmd"
}

case "$run_index" in
  ''|*[!0-9]*)
    echo "[Error] RUN_INDEX must be a positive integer (e.g., 4)."
    exit 1
    ;;
esac
if [ "$run_index" -lt 1 ]; then
  echo "[Error] RUN_INDEX must be >= 1."
  exit 1
fi

prepare_frozen_dir "$source_lib" "$frozen_lib" "$frozen_log"
cmd_run="${python_cmd} main.py --stage lifelong --frozen --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${frozen_lib} ${save_flags}"
run_cmd "[Frozen Run${run_index}] ${model_alias} ${run_tag}" "$cmd_run" "$frozen_log"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

target_model="gpt-3.5-turbo-1106"
safe_model="$(printf '%s' "$target_model" | sed 's/[^a-zA-Z0-9_.-]/_/g')"
warmup_dataset="adv_subset_50"
warmup_path="data/adv_subset_50.json"
library_root="$repo_root/library_${safe_model}_${warmup_dataset}"
library_lifelong_prefix="$repo_root/library_${safe_model}_${warmup_dataset}_lifelong"
base_library="$repo_root/library"

shopt -s nullglob
for path in "${library_root}"* "${library_lifelong_prefix}"*; do
  if [ -d "$path" ]; then
    backup_root="${path}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[Init] Existing library found. Moving to $backup_root"
    mv "$path" "$backup_root"
  fi
done
shopt -u nullglob
mkdir -p "$library_root"
for filename in solver.lp asp_config.json ontology.json; do
  if [ -f "$base_library/$filename" ]; then
    cp "$base_library/$filename" "$library_root/$filename"
  fi
done

log_dir="$repo_root/logs/gpt-3.5-turbo-1106_harmbench200/advsub"
mkdir -p "$log_dir"
log_json="$log_dir/advsub_harmbench200_$(date +%Y%m%d_%H%M%S).json"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
script_start="$(date -Is)"
runs_file="$(mktemp)"
combined_stdout="$(mktemp)"
combined_stderr="$(mktemp)"
harmbench_path="data/harmbench_200.json"

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
      resume_from="$(python - <<'PY'
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
goal_text = (last.get("goal") or last.get("goal_id") or "").strip()
behavior_id = (last.get("behavior_id") or "").strip()
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
        g = (row.get("goal") or row.get("goal_id") or "").strip()
        b = (row.get("behavior_id") or "").strip()
        if g == goal_text and (not behavior_id or b == behavior_id):
            continue
        writer.writerow(row)
tmp.close()
print(tmp.name)
PY
LOG_PATH="$log_path")"
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
  python - <<'PY' >> "$runs_file"
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

run_cmd 1 "[Run 1/1] warmup with reset (${warmup_dataset})" \
  "python main.py --stage warmup --warmup-path ${warmup_path} --reset --target-model ${target_model} --library-root ${library_root}" \
  "$log_dir/warmup_${safe_model}_${run_tag}.csv" \
  "$library_root"

prev_lib="$library_root"
for i in 1 2 3; do
  run_lib="${library_lifelong_prefix}_run${i}"
  cp -R "$prev_lib" "$run_lib"

  run_cmd $((i + 1)) "[Run ${i}/3] lifelong without reset (harmbench_200)" \
    "python main.py --stage lifelong --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${run_lib}" \
    "$log_dir/lifelong_${safe_model}_${run_tag}_run${i}.csv" \
    "$run_lib"

  prev_lib="$run_lib"
done

frozen_lib="${library_lifelong_prefix}_run3_frozen"
cp -R "$prev_lib" "$frozen_lib"

run_cmd 5 "[Run 1/1] frozen eval (harmbench_200)" \
  "python main.py --stage lifelong --frozen --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${frozen_lib}" \
  "$log_dir/lifelong_frozen_${safe_model}_${run_tag}.csv" \
  "$frozen_lib"

script_end="$(date -Is)"

LOG_JSON="$log_json" RUNS_FILE="$runs_file" \
COMBINED_STDOUT="$combined_stdout" COMBINED_STDERR="$combined_stderr" \
SCRIPT_START="$script_start" SCRIPT_END="$script_end" \
python - <<'PY'
import json
import os
from pathlib import Path

runs_path = Path(os.environ["RUNS_FILE"])
run_lines = runs_path.read_text(encoding="utf-8").splitlines() if runs_path.exists() else []
runs = [json.loads(line) for line in run_lines if line.strip()]

combined_stdout = Path(os.environ["COMBINED_STDOUT"]).read_text(encoding="utf-8")
combined_stderr = Path(os.environ["COMBINED_STDERR"]).read_text(encoding="utf-8")

payload = {
    "script_start": os.environ["SCRIPT_START"],
    "script_end": os.environ["SCRIPT_END"],
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

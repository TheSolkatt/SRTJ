#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

target_model="gpt-3.5-turbo-1106"
safe_model="$(printf '%s' "$target_model" | sed 's/[^a-zA-Z0-9_.-]/_/g')_v1"
library_root="$repo_root/library_${safe_model}"
base_library="$repo_root/library"

mkdir -p "$library_root"
for filename in solver.lp asp_config.json ontology.json; do
  if [ -f "$base_library/$filename" ] && [ ! -f "$library_root/$filename" ]; then
    cp "$base_library/$filename" "$library_root/$filename"
  fi
done

log_dir="$repo_root/logs/$safe_model"
mkdir -p "$log_dir"
log_json="$log_dir/warmup_lifelong_cycles_$(date +%Y%m%d_%H%M%S).json"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
script_start="$(date -Is)"
runs_file="$(mktemp)"
combined_stdout="$(mktemp)"
combined_stderr="$(mktemp)"
harmbench_path="data/harmbench_behaviors_text_test.csv"

run_cmd() {
  local run_index="$1"
  local label="$2"
  local cmd="$3"
  local log_path="${4:-}"
  local allow_resume="${5:-1}"
  local start_ts end_ts exit_code
  local stdout_file stderr_file

  if [ -n "$log_path" ] && [ "$allow_resume" -eq 1 ]; then
    if [ -f "$log_path" ]; then
      cmd="$cmd --resume-from-log $log_path"
    fi
    cmd="$cmd --log-path $log_path"
  elif [ -n "$log_path" ]; then
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
  LIB_ROOT="$library_root" RUN_LOG_PATH="$log_path" \
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

resume_log="${RESUME_LOG:-}"
run1_log="${resume_log:-$log_dir/lifelong_${run_tag}_run1.csv}"
run1_label="[Run 1/3] lifelong without reset (harmbench full test set)"
if [ -n "$resume_log" ]; then
  run1_label="[Run 1/3] lifelong RESUME from ${resume_log}"
fi
run_cmd 1 "$run1_label" \
  "python main.py --stage lifelong --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${library_root}" \
  "$run1_log" 1
run_cmd 2 "[Run 2/3] lifelong without reset (harmbench full test set)" \
  "python main.py --stage lifelong --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${library_root}" \
  "$log_dir/lifelong_${run_tag}_run2.csv" 0
run_cmd 3 "[Run 3/3] lifelong without reset (harmbench full test set)" \
  "python main.py --stage lifelong --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${library_root}" \
  "$log_dir/lifelong_${run_tag}_run3.csv" 0
run_cmd 4 "[Run 1/1] frozen eval (harmbench full test set)" \
  "python main.py --stage lifelong --frozen --harmbench-path ${harmbench_path} --target-model ${target_model} --library-root ${library_root}" \
  "$log_dir/lifelong_${run_tag}_frozen.csv" 0

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

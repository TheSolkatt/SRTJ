
import argparse
import csv
import json
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

try:
    from harvester import ComparativeRuleHarvester
    from llm_client import LLMClient
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def _find_latest_log(root: Path) -> Path | None:
    candidates = list(root.glob("logs/**/*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_success(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample harvester on successful runs from a CSV log.")
    parser.add_argument("--log", type=str, default=None, help="Path to log CSV. Default: latest under logs/.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model for harvester.")
    parser.add_argument("--max-count", type=int, default=3, help="Max number of harvested examples to print.")
    parser.add_argument("--min-failed", type=int, default=1, help="Require at least N failed attempts before success.")
    parser.add_argument(
        "--group-by",
        choices=["behavior_id", "goal"],
        default="behavior_id",
        help="Group attempts by behavior_id when available; fallback to goal.",
    )
    args = parser.parse_args()

    log_path = Path(args.log).expanduser() if args.log else _find_latest_log(current_dir)
    if not log_path or not log_path.exists():
        print("Log file not found. Use --log to specify a CSV.")
        return

    try:
        client = LLMClient(model_name=args.model)
    except Exception as e:
        print(f"Failed to init LLMClient: {e}")
        return

    harvester = ComparativeRuleHarvester(client)

    # Group attempts
    experiments: dict[str, dict] = {}
    with log_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goal = row.get("goal") or ""
            behavior_id = row.get("behavior_id") or ""
            key = behavior_id if (args.group_by == "behavior_id" and behavior_id) else goal
            if not key or not goal:
                continue
            entry = experiments.setdefault(key, {"goal": goal, "attempts": []})
            entry["attempts"].append(
                {
                    "prompt": row.get("final_prompt", ""),
                    "response": row.get("target_response", ""),
                    "success": _is_success(row.get("success")),
                    "attempt_id": int(row.get("attempt", 0) or 0),
                }
            )

    print(f"Reading log file: {log_path}")
    print(f"Found {len(experiments)} groups.")

    count = 0
    for key, data in experiments.items():
        if count >= args.max_count:
            break

        attempts = sorted(data["attempts"], key=lambda x: x["attempt_id"])
        success_idx = next((i for i, att in enumerate(attempts) if att["success"]), -1)
        if success_idx < 0:
            continue
        if success_idx < args.min_failed:
            continue

        failed_items = attempts[:success_idx]
        success_item = attempts[success_idx]
        history_input = [
            {
                "failed_prompt": f_att["prompt"],
                "target_response": f_att["response"],
            }
            for f_att in failed_items
        ]

        print(f"\n\n=== Harvesting for Group: {key} ===")
        print(f"Goal: {data['goal']}")
        print(f"Failed Attempts Count: {len(failed_items)}")
        print("Invoking Harvester...")

        try:
            result = harvester.harvest(
                goal_prompt=data["goal"],
                successful_prompt=success_item["prompt"],
                history_attempts=history_input,
            )
            if result:
                print("--- Harvested Rule ---")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                count += 1
            else:
                print("--- Harvester returned None ---")
        except Exception as e:
            print(f"Error during harvest: {e}")

    if count == 0:
        print("\nNo suitable examples found (cases with failures followed by success).")


if __name__ == "__main__":
    main()

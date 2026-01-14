#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_rule_ids(path: Path) -> List[str]:
    data = _load_json(path)
    if not isinstance(data, list):
        return []
    rule_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id", "")).strip()
        if rule_id:
            rule_ids.append(rule_id)
    return list(dict.fromkeys(rule_ids))


def _load_rule_stats(path: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    data = _load_json(path)
    config = {"min_usage": 10.0, "min_success_rate": 0.15}
    if data is None:
        return {}, config

    rules_data = data
    if isinstance(data, dict) and "rules" in data:
        rules_data = data.get("rules", {})
        cfg = data.get("config", {})
        if isinstance(cfg, dict):
            min_usage = cfg.get("min_usage")
            min_success_rate = cfg.get("min_success_rate")
            if isinstance(min_usage, (int, float)):
                config["min_usage"] = float(min_usage)
            if isinstance(min_success_rate, (int, float)):
                config["min_success_rate"] = float(min_success_rate)

    rules: Dict[str, Dict[str, int]] = {}
    if isinstance(rules_data, dict):
        for rule_id, stats in rules_data.items():
            if not isinstance(stats, dict):
                continue
            usage_count = stats.get("usage_count", 0)
            success_count = stats.get("success_count", 0)
            if isinstance(usage_count, int) and isinstance(success_count, int):
                rules[str(rule_id)] = {
                    "usage_count": usage_count,
                    "success_count": success_count,
                }
    return rules, config


def _estimate_tokens(text: str, chars_per_token: int) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / max(chars_per_token, 1)))


def _iter_log_rows(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    yield row
        except Exception:
            continue


def _collect_log_paths(log_args: List[str], root: Path) -> List[Path]:
    if log_args:
        if len(log_args) == 1 and "," in log_args[0]:
            log_args = [part.strip() for part in log_args[0].split(",") if part.strip()]
        return [Path(arg).expanduser() for arg in log_args]

    candidates: List[Path] = []
    for folder in (root / "logs", root / "library" / "logs"):
        if folder.exists():
            candidates.extend(folder.glob("*.csv"))
    if not candidates:
        return []
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return [latest]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rule evolution and log trends.")
    parser.add_argument("--logs", nargs="*", help="CSV logs to analyze (default: latest log).")
    parser.add_argument("--chars-per-token", type=int, default=4, help="Token estimate divisor.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    library_root = root / "library"

    seed_ids = _load_rule_ids(library_root / "seeds.json")
    rule_stats, stats_cfg = _load_rule_stats(library_root / "rule_stats.json")

    min_usage = stats_cfg.get("min_usage", 10.0)
    min_success_rate = stats_cfg.get("min_success_rate", 0.15)

    blocked_seed = 0
    for rule_id in seed_ids:
        stats = rule_stats.get(rule_id)
        if not stats:
            continue
        usage = stats.get("usage_count", 0)
        success = stats.get("success_count", 0)
        if usage > min_usage:
            success_rate = success / usage if usage else 0.0
            if success_rate < min_success_rate:
                blocked_seed += 1

    total_seed = len(seed_ids)
    survival_rate = ((total_seed - blocked_seed) / total_seed) if total_seed else 0.0

    log_paths = _collect_log_paths(args.logs or [], root)
    attempt_tokens: Dict[int, List[int]] = {}
    for row in _iter_log_rows(log_paths):
        attempt_raw = row.get("attempt")
        if attempt_raw is None:
            continue
        try:
            attempt = int(attempt_raw)
        except ValueError:
            continue
        prompt = row.get("final_prompt", "") or ""
        tokens = _estimate_tokens(prompt, args.chars_per_token)
        attempt_tokens.setdefault(attempt, []).append(tokens)

    print("=== Evolution Report ===")
    if log_paths:
        used_logs = ", ".join(str(path) for path in log_paths)
        print(f"Logs: {used_logs}")
    else:
        print("Logs: none found")
    print()
    print(f"Seed rules total: {total_seed}")
    print(f"Seed rules blocked: {blocked_seed} (survival rate {survival_rate:.2%})")
    # Layer counts
    l1 = _load_rule_ids(library_root / "layer1_candidates.json")
    l2 = _load_rule_ids(library_root / "layer2_buffer.json")
    l3 = _load_rule_ids(library_root / "layer3_long_term.json")
    print(f"L1 rules: {len(l1)} | L2 rules: {len(l2)} | L3 rules: {len(l3)}")

    # High success in L3
    high_l3 = 0
    for rid in l3:
        stats = rule_stats.get(rid, {})
        usage = stats.get("usage_count", 0)
        success = stats.get("success_count", 0)
        if usage > 0 and (success / usage) > 0.2:
            high_l3 += 1
    print(f"L3 high-success (>20%) rules: {high_l3}")
    print()
    print("Attempt token trend (approx from final_prompt length):")
    for attempt in sorted(attempt_tokens):
        tokens = attempt_tokens[attempt]
        avg = sum(tokens) / len(tokens) if tokens else 0.0
        print(f"  Attempt {attempt}: avg_tokens={avg:.1f} (n={len(tokens)})")
    if not attempt_tokens:
        print("  No attempts found in logs.")
    print()
    print(
        "Note: token counts are estimated from final_prompt length; "
        "actual target prompt tokens may differ."
    )


if __name__ == "__main__":
    main()

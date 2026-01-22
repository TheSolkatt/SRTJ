#!/usr/bin/env python
"""
Diagnose rule quality and retrieval signals from logs + library.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_load_list(raw: str) -> list:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return []


def _load_rules(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _parse_thresholds(memory_path: Path) -> Dict[str, int]:
    thresholds = {
        "L1_TO_L2_THRESHOLD": 3,
        "L1_TO_L2_MIN_USES": 3,
        "L2_TO_L3_THRESHOLD": 3,
        "L2_TO_L3_MIN_USES": 4,
    }
    if not memory_path.exists():
        return thresholds
    text = memory_path.read_text(encoding="utf-8", errors="ignore")
    for key in list(thresholds.keys()):
        match = re.search(rf"{key}\\s*=\\s*(\\d+)", text)
        if match:
            thresholds[key] = int(match.group(1))
    return thresholds


def _is_imperative(content: str) -> bool:
    if not content:
        return False
    first = content.strip().split()[0].lower() if content.strip().split() else ""
    if first in {"the", "this", "these", "it", "there", "a", "an", "user", "assistant"}:
        return False
    verbs = {
        "use", "frame", "adopt", "wrap", "disguise", "obfuscate", "present", "assume",
        "simulate", "craft", "request", "instruct", "emulate", "reframe", "embed",
        "translate", "encode", "format", "structure", "pose", "portray", "write",
        "create", "generate", "emphasize", "leverage", "insert", "position", "ask",
    }
    return first in verbs


def _load_harmbench_tags(harmbench_path: Path) -> Dict[str, List[str]]:
    if not harmbench_path.exists():
        return {}
    tags: Dict[str, List[str]] = {}
    with harmbench_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behavior_id = (row.get("BehaviorID") or "").strip()
            tag_str = (row.get("Tags") or "").strip()
            if behavior_id:
                tags[behavior_id] = [t.strip() for t in tag_str.split(",") if t.strip()]
    return tags


def _load_logs(log_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in log_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    return rows


def _bool_val(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _summarize_library(library_root: Path) -> Dict[str, Any]:
    layers = {
        "L1": _load_rules(library_root / "layer1_candidates.json"),
        "L2": _load_rules(library_root / "layer2_buffer.json"),
        "L3": _load_rules(library_root / "layer3_long_term.json"),
    }
    thresholds = _parse_thresholds(Path(__file__).parent.parent / "src" / "memory.py")

    summary: Dict[str, Any] = {"counts": {}, "tag_top": {}, "quality": {}, "promotion": {}}
    for name, rules in layers.items():
        summary["counts"][name] = len(rules)
        tags = Counter()
        general_count = 0
        other_count = 0
        missing_when = 0
        short_when = 0
        non_imperative = 0
        duplicates = 0
        seen_content = set()

        for r in rules:
            content = (r.get("content") or "").strip()
            if content in seen_content:
                duplicates += 1
            seen_content.add(content)
            rule_tags = r.get("tags") or []
            for t in rule_tags:
                tags[str(t)] += 1
            if "general" in rule_tags:
                general_count += 1
            if "Other" in rule_tags:
                other_count += 1
            when_to_use = (r.get("when_to_use") or "").strip()
            if not when_to_use:
                missing_when += 1
            elif len(when_to_use) < 10:
                short_when += 1
            if not _is_imperative(content):
                non_imperative += 1

        summary["tag_top"][name] = tags.most_common(8)
        summary["quality"][name] = {
            "general_ratio": general_count / len(rules) if rules else 0,
            "other_ratio": other_count / len(rules) if rules else 0,
            "missing_when": missing_when,
            "short_when": short_when,
            "non_imperative": non_imperative,
            "duplicate_content": duplicates,
        }

    # Promotion eligibility checks (only meaningful for L1/L2)
    l1 = layers["L1"]
    l2 = layers["L2"]
    l1_ready = sum(
        1
        for r in l1
        if (r.get("statistics", {}).get("success_count", 0) >= thresholds["L1_TO_L2_THRESHOLD"]
            and r.get("statistics", {}).get("total_uses", 0) >= thresholds["L1_TO_L2_MIN_USES"])
    )
    l2_ready = sum(
        1
        for r in l2
        if (r.get("statistics", {}).get("success_count", 0) >= thresholds["L2_TO_L3_THRESHOLD"]
            and r.get("statistics", {}).get("total_uses", 0) >= thresholds["L2_TO_L3_MIN_USES"])
    )
    summary["promotion"] = {
        "L1_ready_for_L2": l1_ready,
        "L2_ready_for_L3": l2_ready,
        "thresholds": thresholds,
    }
    return summary


def _summarize_logs(rows: List[Dict[str, Any]], harmbench_tags: Dict[str, List[str]]) -> Dict[str, Any]:
    if not rows:
        return {}

    goal_key = "goal" if "goal" in rows[0] else "goal_id"
    attempt_success = defaultdict(lambda: {"total": 0, "success": 0})
    by_goal = defaultdict(list)
    asp_stats = {"total": 0, "success": 0}
    blind_stats = {"total": 0, "success": 0}
    empty_prompt = 0
    guidance_used = 0
    mismatch = 0
    asp_mismatch = 0
    asp_with_goal_tags = 0

    copyright_total = 0
    copyright_hash_used = 0

    for r in rows:
        attempt = int(r.get("attempt", 0) or 0)
        success = _bool_val(r.get("success"))
        attempt_success[attempt]["total"] += 1
        attempt_success[attempt]["success"] += 1 if success else 0

        goal_id = r.get(goal_key, "")
        by_goal[goal_id].append(success)

        final_prompt = (r.get("final_prompt") or "").strip()
        if not final_prompt:
            empty_prompt += 1

        if (r.get("guidance_used") or "").strip():
            guidance_used += 1

        rule_ids = _safe_load_list(r.get("rule_ids"))
        goal_tags = _safe_load_list(r.get("goal_tags"))
        rule_tags = _safe_load_list(r.get("rule_tags"))

        is_blind = len(rule_ids) == 0
        if is_blind:
            blind_stats["total"] += 1
            blind_stats["success"] += 1 if success else 0
        else:
            asp_stats["total"] += 1
            asp_stats["success"] += 1 if success else 0

        # tag match check for ASP attempts
        if not is_blind and goal_tags:
            asp_with_goal_tags += 1
            goal_tag_set = set(goal_tags)
            matched = False
            if isinstance(rule_tags, list):
                for tlist in rule_tags:
                    if isinstance(tlist, list) and goal_tag_set.intersection(set(tlist)):
                        matched = True
                        break
            if not matched:
                asp_mismatch += 1

        # hash check coverage
        behavior_id = (r.get("behavior_id") or "").strip()
        if behavior_id and "hash_check" in harmbench_tags.get(behavior_id, []):
            copyright_total += 1
            if "hash_match=" in str(r.get("reasoning", "")):
                copyright_hash_used += 1

    total_attempts = len(rows)
    goals = list(by_goal.keys())
    best_of_k = sum(1 for g in goals if any(by_goal[g]))

    summary = {
        "attempts": total_attempts,
        "goals": len(goals),
        "best_of_k_success_rate": best_of_k / len(goals) if goals else 0,
        "empty_prompt_ratio": empty_prompt / total_attempts if total_attempts else 0,
        "guidance_used_ratio": guidance_used / total_attempts if total_attempts else 0,
        "attempt_success_by_index": {
            str(k): v["success"] / v["total"] if v["total"] else 0 for k, v in sorted(attempt_success.items())
        },
        "asp_success_rate": asp_stats["success"] / asp_stats["total"] if asp_stats["total"] else 0,
        "blind_success_rate": blind_stats["success"] / blind_stats["total"] if blind_stats["total"] else 0,
        "asp_tag_mismatch_rate": asp_mismatch / asp_with_goal_tags if asp_with_goal_tags else 0,
        "copyright_hash_usage": {
            "copyright_attempts": copyright_total,
            "hash_path_used": copyright_hash_used,
            "hash_usage_rate": copyright_hash_used / copyright_total if copyright_total else 0,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose rule quality and retrieval signals.")
    parser.add_argument("--log", type=str, default=None, help="Path to a CSV log file.")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory containing CSV logs.")
    parser.add_argument("--library-root", type=str, default="library", help="Library root path.")
    parser.add_argument("--harmbench-path", type=str, default="data/harmbench.csv", help="HarmBench CSV for tags.")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    log_paths: List[Path] = []
    if args.log:
        log_paths.append(Path(args.log))
    if args.log_dir:
        log_paths.extend(sorted(Path(args.log_dir).glob("*.csv")))

    harmbench_tags = _load_harmbench_tags(Path(args.harmbench_path))
    logs_summary = _summarize_logs(_load_logs(log_paths), harmbench_tags) if log_paths else {}
    library_summary = _summarize_library(Path(args.library_root))

    payload = {
        "logs": logs_summary,
        "library": library_summary,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

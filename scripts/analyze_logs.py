#!/usr/bin/env python3
"""
Lightweight CSV log analyzer (no pandas).

Usage:
  python scripts/analyze_logs.py path/to/run.csv [more.csv ...]

It works with both legacy logs (no `mode` column) and newer logs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _bool_val(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Some logs store values as "\n5" due to legacy formatting.
    text = text.lstrip("\n").strip()
    try:
        return float(text)
    except Exception:
        return None


def _safe_load_list(raw: Any) -> list:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    text = text.lstrip("\n").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _format_td(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    td = timedelta(seconds=seconds)
    # keep it compact: H:MM:SS
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def _infer_mode(row: Dict[str, Any]) -> str:
    mode = (row.get("mode") or "").strip().lower()
    if mode:
        return mode
    rule_ids = _safe_load_list(row.get("rule_ids"))
    return "blind" if not rule_ids else "asp"


def _goal_key(row: Dict[str, Any]) -> str:
    goal = (row.get("goal") or row.get("goal_id") or "").strip()
    return f"goal:{goal}"


def _goal_label(goal_key: str, max_len: int = 90) -> str:
    text = goal_key[len("goal:") :] if goal_key.startswith("goal:") else goal_key
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass
class FileSummary:
    path: Path
    attempts: int
    goals: int
    goals_solved: int
    asr: float
    attempt_success_rate: float
    attempts_per_goal: List[int]
    mode_attempts: Dict[str, int]
    mode_success: Dict[str, int]
    planner_used_attempts: int
    guidance_attempts: int
    guidance_success: int
    no_guidance_attempts: int
    no_guidance_success: int
    mean_delta_s: Optional[float]
    median_delta_s: Optional[float]
    p90_delta_s: Optional[float]
    min_delta_s: Optional[float]
    max_delta_s: Optional[float]
    wall_time_s: Optional[float]
    slowest_goals: List[Tuple[str, float, int, bool]]  # (goal_key, elapsed_s, attempts, solved)

    def to_json(self) -> Dict[str, Any]:
        blind_success = int(self.mode_success.get("blind", 0))
        min_calls = int(self.attempts * 3)
        est_calls = min_calls + int(self.planner_used_attempts) + int(blind_success * 2)
        return {
            "path": str(self.path),
            "attempts": self.attempts,
            "goals": self.goals,
            "goals_solved": self.goals_solved,
            "asr": self.asr,
            "attempt_success_rate": self.attempt_success_rate,
            "llm_call_estimate": {
                "min_calls_attacker_target_verifier": min_calls,
                "planner_calls_approx": int(self.planner_used_attempts),
                "harvest_symbolize_calls_approx": int(blind_success * 2),
                "total_calls_approx": int(est_calls),
            },
            "guidance_effect": {
                "with_guidance": {
                    "attempts": self.guidance_attempts,
                    "success": self.guidance_success,
                    "success_rate": (self.guidance_success / self.guidance_attempts) if self.guidance_attempts else None,
                },
                "without_guidance": {
                    "attempts": self.no_guidance_attempts,
                    "success": self.no_guidance_success,
                    "success_rate": (self.no_guidance_success / self.no_guidance_attempts) if self.no_guidance_attempts else None,
                },
            },
            "attempts_per_goal": {
                "mean": statistics.mean(self.attempts_per_goal) if self.attempts_per_goal else 0.0,
                "median": statistics.median(self.attempts_per_goal) if self.attempts_per_goal else 0.0,
                "p90": _pct([float(x) for x in self.attempts_per_goal], 90) if self.attempts_per_goal else None,
                "max": max(self.attempts_per_goal) if self.attempts_per_goal else 0,
            },
            "modes": {
                m: {
                    "attempts": self.mode_attempts.get(m, 0),
                    "success": self.mode_success.get(m, 0),
                    "success_rate": (
                        self.mode_success.get(m, 0) / self.mode_attempts.get(m, 1)
                        if self.mode_attempts.get(m, 0)
                        else 0.0
                    ),
                }
                for m in sorted(set(self.mode_attempts) | set(self.mode_success))
            },
            "planner_used_attempts": self.planner_used_attempts,
            "timing": {
                "wall_time_s": self.wall_time_s,
                "attempt_delta_s": {
                    "mean": self.mean_delta_s,
                    "median": self.median_delta_s,
                    "p90": self.p90_delta_s,
                    "min": self.min_delta_s,
                    "max": self.max_delta_s,
                },
            },
            "slowest_goals": [
                {
                    "goal": g,
                    "elapsed_s": t,
                    "attempts": a,
                    "solved": s,
                }
                for g, t, a, s in self.slowest_goals
            ],
        }


def _load_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                yield row


def summarize_file(path: Path, top_n: int = 8) -> FileSummary:
    by_goal_attempts: Dict[str, int] = defaultdict(int)
    by_goal_solved: Dict[str, bool] = defaultdict(bool)
    by_goal_first_ts: Dict[str, datetime] = {}
    by_goal_last_ts: Dict[str, datetime] = {}
    by_goal_mode_attempts: Dict[str, Counter] = defaultdict(Counter)

    mode_attempts: Dict[str, int] = defaultdict(int)
    mode_success: Dict[str, int] = defaultdict(int)
    planner_used_attempts = 0
    guidance_attempts = 0
    guidance_success = 0
    no_guidance_attempts = 0
    no_guidance_success = 0

    ts_list: List[datetime] = []
    attempts = 0
    attempt_success = 0

    for row in _load_csv_rows(path):
        attempts += 1
        success = _bool_val(row.get("success"))
        attempt_success += 1 if success else 0

        mode = _infer_mode(row)
        mode_attempts[mode] += 1
        if success:
            mode_success[mode] += 1

        has_guidance = bool((row.get("guidance_used") or "").strip())
        if has_guidance:
            planner_used_attempts += 1
            guidance_attempts += 1
            guidance_success += 1 if success else 0
        else:
            no_guidance_attempts += 1
            no_guidance_success += 1 if success else 0

        gk = _goal_key(row)
        by_goal_attempts[gk] += 1
        by_goal_mode_attempts[gk][mode] += 1
        if success:
            by_goal_solved[gk] = True

        ts = _parse_ts(row.get("timestamp"))
        if ts is not None:
            ts_list.append(ts)
            if gk not in by_goal_first_ts or ts < by_goal_first_ts[gk]:
                by_goal_first_ts[gk] = ts
            if gk not in by_goal_last_ts or ts > by_goal_last_ts[gk]:
                by_goal_last_ts[gk] = ts

    goals = len(by_goal_attempts)
    goals_solved = sum(1 for g in by_goal_attempts if by_goal_solved.get(g, False))
    asr = goals_solved / goals if goals else 0.0
    attempt_success_rate = attempt_success / attempts if attempts else 0.0

    attempts_per_goal = list(by_goal_attempts.values())

    # Timing statistics (global).
    mean_delta_s = median_delta_s = p90_delta_s = min_delta_s = max_delta_s = wall_time_s = None
    if len(ts_list) >= 2:
        ts_sorted = sorted(ts_list)
        deltas = [
            (ts_sorted[i] - ts_sorted[i - 1]).total_seconds()
            for i in range(1, len(ts_sorted))
            if ts_sorted[i] >= ts_sorted[i - 1]
        ]
        if deltas:
            mean_delta_s = float(statistics.mean(deltas))
            median_delta_s = float(statistics.median(deltas))
            p90_delta_s = float(_pct(deltas, 90) or 0.0)
            min_delta_s = float(min(deltas))
            max_delta_s = float(max(deltas))
        wall_time_s = (ts_sorted[-1] - ts_sorted[0]).total_seconds()

    # Slowest goals by elapsed wall time within goal.
    slowest: List[Tuple[str, float, int, bool]] = []
    for gk, first_ts in by_goal_first_ts.items():
        last_ts = by_goal_last_ts.get(gk)
        if not last_ts:
            continue
        elapsed = (last_ts - first_ts).total_seconds()
        slowest.append((gk, elapsed, by_goal_attempts.get(gk, 0), by_goal_solved.get(gk, False)))
    slowest.sort(key=lambda x: x[1], reverse=True)
    slowest = slowest[: max(0, int(top_n))]

    return FileSummary(
        path=path,
        attempts=attempts,
        goals=goals,
        goals_solved=goals_solved,
        asr=asr,
        attempt_success_rate=attempt_success_rate,
        attempts_per_goal=attempts_per_goal,
        mode_attempts=dict(mode_attempts),
        mode_success=dict(mode_success),
        planner_used_attempts=planner_used_attempts,
        guidance_attempts=guidance_attempts,
        guidance_success=guidance_success,
        no_guidance_attempts=no_guidance_attempts,
        no_guidance_success=no_guidance_success,
        mean_delta_s=mean_delta_s,
        median_delta_s=median_delta_s,
        p90_delta_s=p90_delta_s,
        min_delta_s=min_delta_s,
        max_delta_s=max_delta_s,
        wall_time_s=wall_time_s,
        slowest_goals=slowest,
    )


def _print_summary(s: FileSummary, top_n: int) -> None:
    print(f"\n=== {s.path} ===")
    print(
        f"Attempts: {s.attempts} | Goals: {s.goals} | Solved: {s.goals_solved} | ASR: {s.asr*100:.2f}%"
    )
    print(f"Attempt success rate: {s.attempt_success_rate*100:.2f}%")

    if s.attempts_per_goal:
        apg = s.attempts_per_goal
        apg_f = [float(x) for x in apg]
        mean_apg = statistics.mean(apg_f)
        median_apg = statistics.median(apg_f)
        p90_apg = _pct(apg_f, 90)
        print(
            "Attempts/goal: "
            f"mean={mean_apg:.2f} median={median_apg:.1f} p90={p90_apg:.1f} max={max(apg)}"
        )

    if s.mode_attempts:
        parts = []
        for mode in sorted(s.mode_attempts.keys()):
            a = s.mode_attempts.get(mode, 0)
            succ = s.mode_success.get(mode, 0)
            rate = (succ / a) if a else 0.0
            parts.append(f"{mode} {a} (succ {succ}, {rate*100:.1f}%)")
        print("Mode breakdown: " + " | ".join(parts))

    if s.attempts:
        print(
            f"Planner guidance used: {s.planner_used_attempts}/{s.attempts} ({(s.planner_used_attempts/s.attempts)*100:.1f}%)"
        )
        if s.guidance_attempts and s.no_guidance_attempts:
            with_rate = (s.guidance_success / s.guidance_attempts) if s.guidance_attempts else 0.0
            without_rate = (s.no_guidance_success / s.no_guidance_attempts) if s.no_guidance_attempts else 0.0
            print(
                "Guidance vs no-guidance success: "
                f"{with_rate*100:.1f}% ({s.guidance_success}/{s.guidance_attempts}) vs "
                f"{without_rate*100:.1f}% ({s.no_guidance_success}/{s.no_guidance_attempts})"
            )

    # Rough call accounting: each attempt has attacker+target+verifier; blind success usually triggers harvester+symbolizer.
    blind_success = int(s.mode_success.get("blind", 0))
    min_calls = int(s.attempts * 3)
    est_calls = min_calls + int(s.planner_used_attempts) + int(blind_success * 2)
    print(
        "LLM calls (approx): "
        f"{min_calls} (attacker+target+verifier) + {s.planner_used_attempts} (planner) + {blind_success*2} (harvest+symbolize) ≈ {est_calls}"
    )

    if s.wall_time_s is not None:
        print(f"Wall time (file): {_format_td(s.wall_time_s)}")
    if s.mean_delta_s is not None:
        print(
            "Attempt-to-attempt delta: "
            f"mean={s.mean_delta_s:.1f}s median={s.median_delta_s:.1f}s "
            f"p90={s.p90_delta_s:.1f}s min={s.min_delta_s:.1f}s max={s.max_delta_s:.1f}s"
        )

    if s.slowest_goals and top_n > 0:
        print(f"Slowest goals (top {min(top_n, len(s.slowest_goals))} by within-goal elapsed):")
        for gk, elapsed_s, attempts, solved in s.slowest_goals:
            label = _goal_label(gk)
            status = "solved" if solved else "unsolved"
            print(f"  - {_format_td(elapsed_s)} | attempts={attempts:<2d} | {status:<7s} | {label}")


def _expand_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        else:
            paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SRTJ CSV logs (speed + ASR).")
    parser.add_argument("paths", nargs="+", help="CSV file paths or directories containing CSVs.")
    parser.add_argument("--top", type=int, default=8, help="Show top-N slowest goals per file (default: 8).")
    parser.add_argument("--json", type=str, default=None, help="Optional output path to write JSON summaries.")
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    paths = [p for p in paths if p.exists() and p.suffix.lower() == ".csv"]
    if not paths:
        print("[analyze_logs] No CSV files found.")
        return 2

    summaries: List[FileSummary] = []
    for p in paths:
        try:
            summaries.append(summarize_file(p, top_n=args.top))
        except Exception as exc:
            print(f"[analyze_logs] Failed to analyze {p}: {exc}")

    for s in summaries:
        _print_summary(s, top_n=args.top)

    if args.json:
        out = Path(args.json)
        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "files": [s.to_json() for s in summaries],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n[analyze_logs] JSON written to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Main experiment entry point using real datasets and data loaders.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from manager import Manager
from llm_client import LLMClient
from core.data_loader import (
    load_adv_subset_dataset,
    load_harmbench_dataset,
)
from core.datatypes import AttackGoal, Rule


class AttemptLogger:
    DEFAULT_FIELDS = [
        "timestamp",
        "attempt",
        "mode",
        "goal",
        "goal_tags",
        "final_prompt",
        "target_response",
        "verifier_score",
        "success",
        "reasoning",
        "guidance_used",
        "rule_ids",
        "rule_scores",
        "rule_tags",
        "dims_covered",
        "banned_rules",
    ]

    def __init__(self, log_path: Optional[str | Path]) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._fieldnames: Optional[List[str]] = None

    def _extract_dims(self, rules: List[Rule]) -> List[str]:
        dims = set()
        for rule in rules:
            for pred in rule.formal_predicates:
                match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(?:\(|$)", str(pred))
                if match:
                    dims.add(match.group(1))
        return sorted(dims)

    def _read_existing_header(self) -> Optional[List[str]]:
        if not self.log_path or not self.log_path.exists():
            return None
        try:
            with self.log_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
        except Exception:
            return None
        header = [h.strip() for h in (header or []) if str(h).strip()]
        return header or None

    def _get_fieldnames(self, file_exists: bool) -> List[str]:
        if not file_exists:
            self._fieldnames = list(self.DEFAULT_FIELDS)
            return self._fieldnames
        if self._fieldnames is None:
            self._fieldnames = self._read_existing_header() or list(self.DEFAULT_FIELDS)
        return self._fieldnames

    def log_attempt(
        self,
        goal: AttackGoal,
        goal_tags: List[str],
        attempt: int,
        mode: Optional[str],
        rules: List[Rule],
        final_prompt: str,
        target_response: str,
        banned_rules: List[List[str]],
        verdict: Dict[str, Any],
        attacker_trace: Optional[Dict[str, str]] = None,
        guidance_used: Optional[str] = None,
    ) -> None:
        if not self.log_path:
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists() and self.log_path.stat().st_size > 0

        rule_ids = [r.rule_id for r in rules] if rules else []
        rule_scores = [round(r.score(), 4) for r in rules] if rules else []
        rule_tags = [r.tags for r in rules] if rules else []
        dims_covered = self._extract_dims(rules)
        banned_rules_json = json.dumps(banned_rules)
        goal_tags_json = json.dumps(goal_tags)
        verifier_score = verdict.get("score")

        # attacker_trace = attacker_trace or {}
        # attacker_system_prompt = attacker_trace.get("system_prompt", "")
        # attacker_user_prompt = attacker_trace.get("user_prompt", "")
        # attacker_raw_output = attacker_trace.get("raw_output", "")
        # attacker_clean_output = attacker_trace.get("clean_output", "")

        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = self._get_fieldnames(file_exists=file_exists)
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()

            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "attempt": attempt,
                "mode": (mode or "").strip(),
                "goal": goal.prompt,
                "goal_tags": goal_tags_json,
                "final_prompt": f"\n{final_prompt}",
                "target_response": f"\n{target_response}",
                "verifier_score": f"\n{verifier_score}",
                "success": verdict.get("success"),
                "reasoning": verdict.get("reasoning", ""),
                "guidance_used": guidance_used or "",
                "rule_ids": f"\n{json.dumps(rule_ids)}",
                "rule_scores": json.dumps(rule_scores),
                "rule_tags": json.dumps(rule_tags),
                "dims_covered": json.dumps(dims_covered),
                "banned_rules": banned_rules_json,
            }
            writer.writerow(row)


def run_experiment(
    dataset: str | None = None,
    stage: str | None = None,
    num_samples: int | None = None,
    reset: bool = False,
    save_log: bool = True,
    log_path: str | None = None,
    enable_harvester: bool = True,
    success_threshold: float | None = None,
    resume_from_log: str | None = None,
    random_sample: bool = False,
    sample_seed: int | None = None,
    goal_index: int | None = None,
    harmbench_path: str | None = None,
    warmup_path: str | None = None,
    target_model: str | None = None,
    library_root: str | None = None,
    frozen: bool = False,
    disable_planner: bool = False,
    disable_symbolizer: bool = False,
    fast: bool = False,
    save_interval: int = 1,
    save_per_goal: bool = False,
) -> None:
    def _sanitize_model_name(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")
        return cleaned or "model"

    # determine log filename early (use stage label if dataset is None)
    if save_log:
        if log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = dataset or stage or "run"
            if frozen and stage:
                label = f"{label}_frozen"
            model_name = _sanitize_model_name(target_model or "gpt-3.5-turbo-1106")
            log_dir = Path("./logs") / model_name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(log_dir / f"{label}_{timestamp}.csv")
    else:
        log_path = None

    disable_planner = disable_planner or fast
    disable_symbolizer = disable_symbolizer or fast

    # Mixed-model clients: planner + attacker + target + verifier.
    #
    # Multi-key / multi-provider support:
    # - Default provider: OPENAI_API_KEY/OPENAI_BASE_URL, falling back to api_key/base_url
    # - OpenRouter provider: OPENROUTER_API_KEY/OPENROUTER_BASE_URL
    # - Per-role overrides: SRTJ_<ROLE>_API_KEY / SRTJ_<ROLE>_BASE_URL

    def _env(key: str) -> Optional[str]:
        val = os.getenv(key)
        if val is None:
            return None
        val = str(val).strip()
        return val or None

    def _default_api_key() -> Optional[str]:
        # Prefer project-local .env `api_key` over globally-exported `OPENAI_API_KEY`.
        # This avoids surprises when the user has a stale/invalid OPENAI_API_KEY in their shell.
        return _env("api_key") or _env("OPENAI_API_KEY")

    def _default_base_url() -> Optional[str]:
        # Prefer project-local .env `base_url` over globally-exported `OPENAI_BASE_URL`.
        return _env("base_url") or _env("OPENAI_BASE_URL")

    def _openrouter_api_key() -> Optional[str]:
        return _env("OPENROUTER_API_KEY")

    def _openrouter_base_url() -> str:
        return _env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

    def _mk_client(model_name: str, provider: str, role: str) -> LLMClient:
        role_key = _env(f"SRTJ_{role}_API_KEY")
        role_url = _env(f"SRTJ_{role}_BASE_URL")
        if role_key or role_url:
            return LLMClient(model_name=model_name, api_key=role_key, base_url=role_url)
        if provider == "openrouter":
            return LLMClient(
                model_name=model_name,
                api_key=_openrouter_api_key(),
                base_url=_openrouter_base_url(),
            )
        return LLMClient(
            model_name=model_name,
            api_key=_default_api_key(),
            base_url=_default_base_url(),
        )

    attacker_model_name = _env("SRTJ_ATTACKER_MODEL") or "deepseek-r1"
    planner_model_name = _env("SRTJ_PLANNER_MODEL") or attacker_model_name
    verifier_model_name = _env("SRTJ_VERIFIER_MODEL") or "gpt-4o"
    analysis_model_name = _env("SRTJ_ANALYSIS_MODEL") or verifier_model_name
    target_model_name = target_model or _env("SRTJ_TARGET_MODEL") or "gpt-3.5-turbo-1106"

    openrouter_targets_raw = _env("SRTJ_OPENROUTER_TARGETS")
    if openrouter_targets_raw:
        openrouter_targets = {m.strip() for m in openrouter_targets_raw.split(",") if m.strip()}
    else:
        # Default OpenRouter targets include both short aliases and full OpenRouter-style model ids.
        openrouter_targets = {
            "gpt-4o",
            "openai/gpt-4o-2024-08-06",
            "llama-3-8B-Instruct",
            "meta-llama/llama-3-8b-instruct",
            "llama-3-70B-Instruct",
            "meta-llama/llama-3-70b-instruct",
            "claude-3-5-sonnet-20240620",
            "anthropic/claude-3.5-sonnet-20240620",
        }

    # Only route the target model based on this list; other roles are fixed defaults.
    have_openrouter = bool(_openrouter_api_key())
    target_provider = "openrouter" if (have_openrouter and target_model_name in openrouter_targets) else "default"
    verifier_provider = "openrouter" if have_openrouter else "default"

    attacker_client = _mk_client(attacker_model_name, provider="default", role="ATTACKER")
    target_client = _mk_client(target_model_name, provider=target_provider, role="TARGET")
    verifier_client = _mk_client(verifier_model_name, provider=verifier_provider, role="VERIFIER")
    planner_client = (
        None
        if disable_planner
        else _mk_client(planner_model_name, provider="default", role="PLANNER")
    )
    analysis_client = _mk_client(analysis_model_name, provider=verifier_provider, role="ANALYSIS")
    logger = AttemptLogger(log_path)

    manager = Manager(
        log_path=log_path,
        enable_harvester=enable_harvester,
        attacker_client=attacker_client,
        target_client=target_client,
        verifier_client=verifier_client,
        planner_client=planner_client,
        analysis_client=analysis_client,
        logger=logger,
        success_threshold=success_threshold,
        library_root=library_root,
        frozen=frozen,
        stage=stage,
        enable_planner=not disable_planner,
        enable_symbolizer=not disable_symbolizer,
        save_interval=save_interval,
    )

    if reset:
        manager.memory.reset_memory()
    
    # Stage-based routing
    goals = None
    dataset_label = None
    rule_added_count = 0

    if stage == "warmup":
        goals = load_adv_subset_dataset(warmup_path or "data/adv_subset_50.json")
        if not goals:
            print("[main] No adv_subset goals loaded; aborting.")
            return
        dataset_label = "adv_subset_50 (warmup)"
    elif stage == "lifelong":
        # Keep existing library; run harmbench
        goals = load_harmbench_dataset(harmbench_path or "data/harmbench_200.json")
        if not goals:
            print("[main] No harmbench goals loaded; aborting.")
            return
        dataset_label = "harmbench_200 (frozen eval)" if frozen else "harmbench_200 (lifelong)"
    elif dataset == "adv_subset":
        goals = load_adv_subset_dataset("data/adv_subset_50.json")
        if not goals:
            print("[main] No adv_subset goals loaded; aborting.")
            return
        dataset_label = "adv_subset_50"
    elif dataset == "harmbench":
        goals = load_harmbench_dataset("data/harmbench_200.json")
        if not goals:
            print("[main] No harmbench goals loaded; aborting.")
            return
        dataset_label = "harmbench_200"
    else:
        print("[main] Unknown dataset/stage selection. Provide --stage or --dataset.")
        return

    # Select specific goal if requested (debug mode)
    if goal_index is not None:
        if goal_index < 1 or goal_index > len(goals):
            print(f"[main] goal-index out of range: {goal_index} (1..{len(goals)})")
            return
        goals = [goals[goal_index - 1]]
        dataset_label = f"{dataset_label or dataset} (index {goal_index})"
        resume_from_log = None
        random_sample = False
        num_samples = 1
    # 统一执行循环
    total_goals = len(goals)
    skipped_goals = 0
    if resume_from_log:
        resume_path = Path(resume_from_log)
        if resume_path.exists():
            seen_goals = set()
            with resume_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    goal_text = row.get("goal")
                    if goal_text:
                        seen_goals.add(goal_text)
            if seen_goals:
                goals = [
                    goal
                    for goal in goals
                    if goal.goal_id not in seen_goals
                ]
                skipped_goals = total_goals - len(goals)
        else:
            print(f"[main] Resume log not found at {resume_path}; continuing without resume.")

    remaining_goals = len(goals)
    if remaining_goals == 0:
        print("[main] All goals already processed in resume log.")
        return

    sample_size = remaining_goals if num_samples is None else min(num_samples, remaining_goals)
    if random_sample:
        rng = random.Random(sample_seed)
        if sample_size < remaining_goals:
            goals = rng.sample(goals, k=sample_size)
        else:
            rng.shuffle(goals)
    print(f"\n{'='*70}")
    print(f"[Dataset] {dataset_label or dataset}")
    if resume_from_log:
        print(f"[Total Goals] {total_goals} | [Skipped] {skipped_goals} | [Remaining] {remaining_goals}")
    else:
        print(f"[Total Goals] {total_goals} | [Sample Size] {sample_size}")
    print(f"[Start Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for idx, goal in enumerate(goals[:sample_size], start=1):
        print(f"[Goal {idx:2d}/{sample_size}] | {getattr(goal, 'prompt', '')}")
        initial_rules = len(manager.memory.layer3_rules) + len(manager.memory.layer2_rules) + len(manager.memory.layer1_rules)
        manager.process_goal(goal)
        if save_per_goal:
            manager.memory.flush()
        current_rules = len(manager.memory.layer3_rules) + len(manager.memory.layer2_rules) + len(manager.memory.layer1_rules)
        
        if current_rules > initial_rules:
            rule_added_count += 1
            print(f"           ✓ New rule added (Total: {current_rules})\n")
        else:
            print()

    manager.memory.flush()

    final_layer3 = len(manager.memory.layer3_rules)
    final_layer2 = len(manager.memory.layer2_rules)
    final_layer1 = len(manager.memory.layer1_rules)

    print(f"{'='*70}")
    print(f"[Results Summary]")
    print(f"  Processed Goals:  {sample_size}")
    if resume_from_log:
        print(f"  Skipped Goals:    {skipped_goals}")
    print(f"  New Rules Added:  {rule_added_count}")
    print(f"  Layer 3 (Long-term): {final_layer3}")
    print(f"  Layer 2 (Buffer):    {final_layer2}")
    print(f"  Layer 1 (Candidates): {final_layer1}")
    print(f"  Total Rules:      {final_layer3 + final_layer2 + final_layer1}")
    print(f"  Library Root:     {manager.memory.library_root}")
    print(f"  End Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    def _str_to_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}

    def _load_verifier_threshold(default_value: float = 5) -> float:
        config_path = Path(__file__).parent / "library" / "asp_config.json"
        if not config_path.exists():
            return default_value
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return default_value
        value = data.get("verifier_threshold")
        if isinstance(value, (int, float)):
            return float(value)
        return default_value

    default_threshold = _load_verifier_threshold()

    parser = argparse.ArgumentParser(description="Run jailbreak experiment on a selected dataset or stage.")
    parser.add_argument(
        "--dataset",
        choices=["adv_subset", "harmbench"],
        required=False,
        help="Select which dataset to run (ignored if --stage is set).",
    )
    parser.add_argument(
        "--stage",
        choices=["warmup", "lifelong"],
        required=False,
        help="Stages: warmup (adv_subset_50) or lifelong (harmbench_200, keep memory)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of goals to process (default: full dataset)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset rule library to seeds before running."
    )
    parser.add_argument(
        "--save-log",
        type=_str_to_bool,
        default=True,
        help="Whether to save per-attempt logs (default: True). Use --save_log False to disable."
    )
    parser.add_argument(
        "--enable-harvester",
        type=_str_to_bool,
        default=True,
        help="Whether to enable rule harvesting on success (default: True)."
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional CSV path for per-attempt logs (default: auto timestamped)"
    )
    parser.add_argument(
        "--resume-from-log",
        type=str,
        default=None,
        help="Resume by skipping goals that already appear in this CSV log.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=default_threshold,
        help="Verifier success threshold (default: from asp_config.json or 5).",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample goals before running (default: False).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random seed used when --random-sample is enabled.",
    )
    parser.add_argument(
        "--harmbench-path",
        type=str,
        default=None,
        help="Optional HarmBench JSON path (default: data/harmbench_200.json).",
    )
    parser.add_argument(
        "--warmup-path",
        type=str,
        default=None,
        help="Optional warmup JSON path (default: data/adv_subset_50.json).",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="gpt-3.5-turbo-1106",
        help="Target model name (default: gpt-3.5-turbo-1106).",
    )
    parser.add_argument(
        "--library-root",
        type=str,
        default=None,
        help="Override library root directory (use separate library per run).",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Freeze memory updates (read-only evaluation).",
    )
    parser.add_argument(
        "--disable-planner",
        action="store_true",
        help="Disable planner guidance to reduce LLM calls.",
    )
    parser.add_argument(
        "--disable-symbolizer",
        action="store_true",
        help="Disable symbolic conversion on harvested rules.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut: disable planner and symbolizer for faster runs.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Autosave interval for memory updates (1=every update, 0=manual only).",
    )
    parser.add_argument(
        "--save-per-goal",
        action="store_true",
        help="Force memory save at the end of each goal.",
    )
    parser.add_argument(
        "--goal-index",
        type=int,
        default=None,
        help="Select a single goal by 1-based index from the loaded dataset.",
    )
    args = parser.parse_args()
    run_experiment(
        dataset=args.dataset,
        stage=args.stage,
        num_samples=args.num_samples,
        reset=args.reset,
        save_log=args.save_log,
        log_path=args.log_path,
        enable_harvester=args.enable_harvester,
        success_threshold=args.success_threshold,
        resume_from_log=args.resume_from_log,
        random_sample=args.random_sample,
        sample_seed=args.sample_seed,
        goal_index=args.goal_index,
        harmbench_path=args.harmbench_path,
        warmup_path=args.warmup_path,
        target_model=args.target_model,
        library_root=args.library_root,
        frozen=args.frozen,
        disable_planner=args.disable_planner,
        disable_symbolizer=args.disable_symbolizer,
        fast=args.fast,
        save_interval=args.save_interval,
        save_per_goal=args.save_per_goal,
    )

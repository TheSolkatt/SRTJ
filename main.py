"""
Main experiment entry point using real datasets and data loaders.
"""
import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.manager import Manager
from src.llm_client import LLMClient
from core.data_loader import (
    load_jbb_dataset,
    load_strongreject_dataset,
    load_adv_subset_dataset,
    load_harmbench_dataset,
)
from core.datatypes import AttackGoal, Rule


class AttemptLogger:
    def __init__(self, log_path: Optional[str | Path]) -> None:
        self.log_path = Path(log_path) if log_path else None

    def _extract_dims(self, rules: List[Rule]) -> List[str]:
        dims = set()
        for rule in rules:
            for pred in rule.formal_predicates:
                match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(?:\(|$)", str(pred))
                if match:
                    dims.add(match.group(1))
        return sorted(dims)

    def log_attempt(
        self,
        goal: AttackGoal,
        goal_tags: List[str],
        attempt: int,
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
        file_exists = self.log_path.exists()

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
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "attempt",
                        "goal",
                        "goal_tags",
                        "behavior_id",
                        "final_prompt",
                        # "attacker_system_prompt",
                        # "attacker_user_prompt",
                        # "attacker_raw_output",
                        # "attacker_clean_output",
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
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    attempt,
                    goal.prompt,
                    goal_tags_json,
                    getattr(goal, "behavior_id", None),
                    f"\n{final_prompt}",
                    # attacker_system_prompt,
                    # attacker_user_prompt,
                    # attacker_raw_output,
                    # attacker_clean_output,
                    f"\n{target_response}",
                    f"\n{verifier_score}",
                    verdict.get("success"),
                    verdict.get("reasoning", ""),
                    guidance_used or "",
                    f"\n{json.dumps(rule_ids)}",
                    json.dumps(rule_scores),
                    json.dumps(rule_tags),
                    json.dumps(dims_covered),
                    banned_rules_json,
                ]
            )


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
    behavior_id: str | None = None,
    harmbench_path: str | None = None,
    target_model: str | None = None,
    library_root: str | None = None,
    frozen: bool = False,
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

    # Mixed-model clients: DeepSeek for attacker (reasoning), OpenAI for verifier/interpreter/target.
    attacker_client = LLMClient(model_name="deepseek-reasoner")
    target_client = LLMClient(model_name=target_model or "gpt-3.5-turbo-1106")
    verifier_client = LLMClient(model_name="gpt-4o-mini")
    interpreter_client = LLMClient(model_name="gpt-4o-mini")
    logger = AttemptLogger(log_path)

    manager = Manager(
        log_path=log_path,
        enable_harvester=enable_harvester,
        attacker_client=attacker_client,
        target_client=target_client,
        verifier_client=verifier_client,
        interpreter_client=interpreter_client,
        logger=logger,
        success_threshold=success_threshold,
        library_root=library_root,
        frozen=frozen,
    )

    if reset:
        manager.memory.reset_memory()
    
    # Stage-based routing
    goals = None
    dataset_label = None
    rule_added_count = 0

    if stage == "warmup":
        # Cold start: use adv_subset dataset
        goals = load_adv_subset_dataset("data/adv_subset.csv")
        if not goals:
            print("[main] No adv_subset goals loaded; aborting.")
            return
        dataset_label = "adv_subset (warmup)"
    elif stage == "lifelong":
        # Keep existing library; run harmbench
        goals = load_harmbench_dataset(harmbench_path or "data/harmbench.csv")
        if not goals:
            print("[main] No harmbench goals loaded; aborting.")
            return
        dataset_label = "harmbench (frozen eval)" if frozen else "harmbench (lifelong)"
    elif dataset == "jbb":
        goals = load_jbb_dataset("data/jbb.csv")
        if not goals:
            print("[main] No JBB goals loaded; aborting.")
            return
        dataset_label = "JBB (Harmful Behaviors)"
    elif dataset == "strongreject":
        goals = load_strongreject_dataset("data/strongreject_small_dataset.csv")
        if not goals:
            print("[main] No StrongREJECT goals loaded; aborting.")
            return
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
    elif behavior_id:
        filtered = [g for g in goals if (g.behavior_id or "") == behavior_id]
        if not filtered:
            print(f"[main] No goal found with behavior_id={behavior_id}")
            return
        goals = filtered
        dataset_label = f"{dataset_label or dataset} (behavior_id {behavior_id})"
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
                    # support new logs ("goal") and legacy logs ("goal_id")
                    goal_text = row.get("goal") or row.get("goal_id")
                    if goal_text:
                        seen_goals.add(goal_text)
            if seen_goals:
                goals = [goal for goal in goals if goal.goal_id not in seen_goals and goal.prompt not in seen_goals]
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
        current_rules = len(manager.memory.layer3_rules) + len(manager.memory.layer2_rules) + len(manager.memory.layer1_rules)
        
        if current_rules > initial_rules:
            rule_added_count += 1
            print(f"           ✓ New rule added (Total: {current_rules})\n")
        else:
            print()

    manager.memory.save_all_layers()

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

    def _load_verifier_threshold(default_value: float = 8.0) -> float:
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
        choices=["jbb", "strongreject", "adv_subset", "harmbench"],
        required=False,
        help="Select which dataset to run (ignored if --stage is set).",
    )
    parser.add_argument(
        "--stage",
        choices=["warmup", "lifelong"],
        required=False,
        help="Stages: warmup (adv_subset) or lifelong (harmbench, keep memory)."
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
        help="Resume by skipping goal_ids that already appear in this CSV log.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=default_threshold,
        help="Verifier success threshold (default: from asp_config.json or 8.0)",
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
        help="Optional HarmBench CSV path (default: data/harmbench.csv).",
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
        "--goal-index",
        type=int,
        default=None,
        help="Select a single goal by 1-based index from the loaded dataset.",
    )
    parser.add_argument(
        "--behavior-id",
        type=str,
        default=None,
        help="Select a single goal by behavior_id (e.g., HarmBench BehaviorID).",
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
        behavior_id=args.behavior_id,
        harmbench_path=args.harmbench_path,
        target_model=args.target_model,
        library_root=args.library_root,
        frozen=args.frozen,
    )

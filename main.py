"""
Main experiment entry point using real datasets and data loaders.
"""
import argparse
import csv
import json
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
                        "goal_id",
                        "attempt",
                        "rule_ids",
                        "rule_scores",
                        "rule_tags",
                        "dims_covered",
                        "final_prompt",
                        # "attacker_system_prompt",
                        # "attacker_user_prompt",
                        # "attacker_raw_output",
                        # "attacker_clean_output",
                        "target_response",
                        "verifier_score",
                        "banned_rules",
                        "goal_tags",
                        "success",
                        "reasoning",
                    ]
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    goal.goal_id,
                    attempt,
                    json.dumps(rule_ids),
                    json.dumps(rule_scores),
                    json.dumps(rule_tags),
                    json.dumps(dims_covered),
                    final_prompt,
                    # attacker_system_prompt,
                    # attacker_user_prompt,
                    # attacker_raw_output,
                    # attacker_clean_output,
                    target_response,
                    verifier_score,
                    banned_rules_json,
                    goal_tags_json,
                    verdict.get("success"),
                    verdict.get("reasoning", ""),
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
) -> None:
    # determine log filename early (use stage label if dataset is None)
    if save_log:
        if log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = dataset or stage or "run"
            log_path = f"./logs/{label}_{timestamp}.csv"
    else:
        log_path = None

    # Mixed-model clients: DeepSeek for attacker (reasoning), OpenAI for verifier/interpreter/target.
    attacker_client = LLMClient(model_name="deepseek-reasoner")
    target_client = LLMClient(model_name="gpt-3.5-turbo")
    verifier_client = LLMClient(model_name="gpt-4o")
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
    elif stage == "eval":
        # Keep existing library; run harmbench
        goals = load_harmbench_dataset("data/harmbench.csv")
        if not goals:
            print("[main] No harmbench goals loaded; aborting.")
            return
        dataset_label = "harmbench (eval)"
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
        print(f"[main] Unknown dataset/stage selection. Provide --stage or --dataset.")
        return

    # 统一执行循环
    total_goals = len(goals)
    sample_size = total_goals if num_samples is None else min(num_samples, total_goals)
    print(f"\n{'='*70}")
    print(f"[Dataset] {dataset_label or dataset}")
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

    def _load_verifier_threshold(default_value: float = 8.5) -> float:
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
        choices=["warmup", "eval"],
        required=False,
        help="Two-stage flow: warmup (adv_subset 50, reset memory) or eval (harmbench 400, keep memory)."
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
        "--success-threshold",
        type=float,
        default=default_threshold,
        help="Verifier success threshold (default: from asp_config.json or 8.5)",
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
    )

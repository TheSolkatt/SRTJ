"""
Dataset loading utilities for jailbreak experiments.
"""
import csv
from pathlib import Path
from typing import List

from .datatypes import AttackGoal


def _strip(text) -> str:
    return text.strip() if isinstance(text, str) else ""


def _get_data_path(filename: str) -> Path:
    """
    获取相对于项目根目录的数据文件路径
    """
    script_dir = Path(__file__).parent.parent.parent  # 从 src/core/ 到项目根目录
    return script_dir / "data" / filename


def _get_positional(row: dict, idx: int) -> str:
    """安全地按位置读取 DictReader 行的第 idx 列（0-based）"""
    try:
        return list(row.values())[idx]
    except Exception:
        return ""


def load_jbb_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load JBB-like dataset.

    Expected columns (case-insensitive): Goal (preferred) or Behavior。Target 仅作数据参考，不作为攻击提示。
    """
    if filepath is None:
        filepath = _get_data_path("jbb.csv")
    else:
        filepath = Path(filepath)

    path = Path(filepath)
    goals: List[AttackGoal] = []
    if not path.exists():
        print(f"[data_loader] JBB dataset not found at {path.absolute()}")
        return goals

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=0):
            prompt = _strip(
                row.get("Goal")
                or row.get("goal")
                or row.get("Behavior")
                or row.get("behavior")
                or row.get("instruction")
            )
            if not prompt:
                continue
            category = _strip(row.get("Category") or row.get("category"))
            index_val = _strip(row.get("Index") or row.get("index") or row.get("id"))
            goal_id = prompt
            goals.append(
                AttackGoal(
                    goal_id=goal_id,
                    prompt=prompt,
                    target_response=None,
                    category=category or None,
                    source_category=category or None,
                    behavior_id=index_val or None,
                )
            )
    return goals


def load_strongreject_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load StrongREJECT-style dataset.

    Expected columns (case-insensitive): forbidden_prompt, category.
    """
    # 默认尝试 small/subset
    default_path = "strongreject_small_dataset.csv"
    if filepath is None:
        filepath = _get_data_path(default_path)
    else:
        filepath = Path(filepath)

    path = Path(filepath)
    if not path.exists() and filepath.name == default_path:
        # 兼容旧文件名
        alt = _get_data_path("strongreject_subset.csv")
        if alt.exists():
            path = alt

    goals: List[AttackGoal] = []
    if not path.exists():
        print(f"[data_loader] StrongREJECT dataset not found at {path.absolute()}")
        return goals

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=0):
            prompt = _strip(
                row.get("forbidden_prompt")
                or row.get("prompt")
                or row.get("goal")
            )
            if not prompt:
                continue
            category = _strip(row.get("category") or row.get("Category"))
            goal_id = prompt
            goals.append(
                AttackGoal(
                    goal_id=goal_id,
                    prompt=prompt,
                    target_response=None,
                    category=category or None,
                    source_category=category or None,
                )
            )
    return goals


def load_adv_subset_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load adv_subset dataset.

    Expected column: goal (attack prompt). Ignore 'target'.
    """
    if filepath is None:
        filepath = _get_data_path("adv_subset.csv")
    else:
        filepath = Path(filepath)

    path = Path(filepath)
    goals: List[AttackGoal] = []
    if not path.exists():
        print(f"[data_loader] adv_subset dataset not found at {path.absolute()}")
        return goals

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=0):
            prompt = _strip(
                row.get("goal")
                or row.get("Goal")
                or _get_positional(row, 1)  # after leading unnamed index col
            )
            if not prompt:
                continue
            behavior_id = _strip(
                row.get("Original index")
                or row.get("Original Index")
                or row.get("BehaviorID")
                or row.get("behavior_id")
            )
            context = _strip(
                row.get("ContextString")
                or row.get("context")
                or row.get("Context")
            )
            source_category = _strip(row.get("category") or row.get("Category"))
            goal_id = prompt
            goals.append(
                AttackGoal(
                    goal_id=goal_id,
                    prompt=prompt,
                    context=context or None,
                    behavior_id=behavior_id or None,
                    source_category=source_category or None,
                )
            )
    return goals


def load_harmbench_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load harmbench dataset.

    Expected column: Behavior (attack prompt).
    """
    if filepath is None:
        filepath = _get_data_path("harmbench.csv")
    else:
        filepath = Path(filepath)

    path = Path(filepath)
    goals: List[AttackGoal] = []
    if not path.exists():
        print(f"[data_loader] harmbench dataset not found at {path.absolute()}")
        return goals

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=0):
            prompt = _strip(
                row.get("Behavior")
                or row.get("behavior")
                or _get_positional(row, 0)
            )
            if not prompt:
                continue
            behavior_id = _strip(row.get("BehaviorID"))
            context = _strip(
                row.get("ContextString")
                or row.get("context")
                or row.get("Context")
            )
            semantic_category = _strip(row.get("SemanticCategory"))
            functional_category = _strip(row.get("FunctionalCategory"))
            goals.append(
                AttackGoal(
                    goal_id=prompt,
                    prompt=prompt,
                    context=context or None,
                    behavior_id=behavior_id or None,
                    source_category=semantic_category or None,
                    source_functional=functional_category or None,
                )
            )
    return goals


if __name__ == "__main__":
    jbb_sample = load_jbb_dataset()
    if jbb_sample:
        print(f"Loaded {len(jbb_sample)} JBB records. First: {jbb_sample[0]}")
    else:
        print("No JBB data loaded.")

    sr_sample = load_strongreject_dataset()
    if sr_sample:
        print(f"Loaded {len(sr_sample)} StrongREJECT records. First: {sr_sample[0]}")
    else:
        print("No StrongREJECT data loaded.")

    adv_sample = load_adv_subset_dataset()
    if adv_sample:
        print(f"Loaded {len(adv_sample)} ADV records. First: {adv_sample[0]}")
    else:
        print("No adv_subset data loaded.")

    hb_sample = load_harmbench_dataset()
    if hb_sample:
        print(f"Loaded {len(hb_sample)} Harmbench records. First: {hb_sample[0]}")
    else:
        print("No Harmbench data loaded.")

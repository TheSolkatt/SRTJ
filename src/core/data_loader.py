"""
Dataset loading utilities for jailbreak experiments.

Current dataset format (project-wide):
- JSON list of objects: {"goal": "...", "category": "..."}
"""
import json
from pathlib import Path
from typing import List

from .datatypes import AttackGoal


def _get_data_path(filename: str) -> Path:
    """
    获取相对于项目根目录的数据文件路径
    """
    script_dir = Path(__file__).parent.parent.parent  # 从 src/core/ 到项目根目录
    return script_dir / "data" / filename


def _load_goal_category_json(path: Path) -> List[AttackGoal]:
    """
    Load a dataset from a JSON list of {"goal": str, "category": str}.
    """
    if not path.exists():
        print(f"[data_loader] Dataset not found at {path.absolute()}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[data_loader] Failed to load JSON at {path.absolute()}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[data_loader] JSON dataset is not a list at {path.absolute()}")
        return []

    goals: List[AttackGoal] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        goal_text = str(item.get("goal", "")).strip()
        category = str(item.get("category", "")).strip()
        if not goal_text or not category:
            continue
        goals.append(AttackGoal(goal_id=goal_text, prompt=goal_text, category=category))
    return goals

def load_adv_subset_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load adv_subset dataset from JSON.
    """
    if filepath is None:
        filepath = _get_data_path("adv_subset_50.json")
    else:
        filepath = Path(filepath)
    
    # Ensure we look for local fallback if absolute path not found
    if not filepath.exists() and filepath.name:
        local_path = _get_data_path(filepath.name)
        if local_path.exists():
            filepath = local_path

    return _load_goal_category_json(Path(filepath))


def load_harmbench_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load harmbench dataset from JSON.
    """
    if filepath is None:
        filepath = _get_data_path("harmbench_200.json")
    else:
        filepath = Path(filepath)

    if not filepath.exists() and filepath.name:
        local_path = _get_data_path(filepath.name)
        if local_path.exists():
            filepath = local_path

    return _load_goal_category_json(Path(filepath))


if __name__ == "__main__":
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

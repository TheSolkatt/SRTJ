"""Dataset loading utilities for jailbreak experiments."""
import json
from pathlib import Path
from typing import List

from .datatypes import AttackGoal

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "data"


def _resolve_path(filepath: str | Path | None, default_name: str) -> Path:
    if filepath is None:
        return _DATA_DIR / default_name
    path = Path(filepath)
    if not path.exists() and path.name:
        local = _DATA_DIR / path.name
        if local.exists():
            return local
    return path


def _load_goal_category_json(path: Path) -> List[AttackGoal]:
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
        if goal_text and category:
            goals.append(AttackGoal(goal_id=goal_text, prompt=goal_text, category=category))
    return goals


def load_adv_subset_dataset(filepath: str | Path | None = None) -> List[AttackGoal]:
    return _load_goal_category_json(_resolve_path(filepath, "adv_subset_50.json"))


def load_harmbench_dataset(filepath: str | Path | None = None) -> List[AttackGoal]:
    return _load_goal_category_json(_resolve_path(filepath, "harmbench_200.json"))

"""
Dataset loading utilities for jailbreak experiments.
"""
import json
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


def _extract_goal_item(item) -> dict:
    prompt = ""
    context = None
    behavior_id = None
    source_category = None
    source_functional = None
    if isinstance(item, str):
        prompt = _strip(item)
    elif isinstance(item, dict):
        prompt = _strip(
            item.get("goal")
            or item.get("Behavior")
            or item.get("prompt")
            or item.get("text")
            or item.get("query")
        )
        context = item.get("context") or item.get("Context")
        behavior_id = item.get("behavior_id") or item.get("BehaviorID") or item.get("behaviorId")
        source_category = (
            item.get("category")
            or item.get("intent_category")
            or item.get("source_category")
        )
        source_functional = (
            item.get("source_functional")
            or item.get("functional_category")
            or item.get("FunctionalCategory")
        )
    return {
        "prompt": prompt,
        "context": context,
        "behavior_id": behavior_id,
        "source_category": source_category,
        "source_functional": source_functional,
    }


def _load_json_goals(path: Path) -> List[dict]:
    if not path.exists():
        print(f"[data_loader] JSON dataset not found at {path.absolute()}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[data_loader] Failed to load JSON dataset at {path.absolute()}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[data_loader] JSON dataset is not a list at {path.absolute()}")
        return []
    goals: List[dict] = []
    for item in data:
        entry = _extract_goal_item(item)
        if entry.get("prompt"):
            goals.append(entry)
    return goals


def load_jbb_dataset(filepath: str = None) -> List[AttackGoal]:
    if filepath is None:
        filepath = _get_data_path("jbb.csv")
    else:
        filepath = Path(filepath)
    # Reuse JSON loading logic if csv not strictly required, 
    # but JBB is usually CSV. For now, assume user might pass json path.
    if str(filepath).endswith(".json"):
        prompts = _load_json_goals(filepath)
    else:
        # Fallback to simple CSV logic if needed, or just warn if strictly JSON now
        pass 
        # (Assuming JBB not focus of current request, leaving as is or minimal)
    
    # Placeholder for JBB CSV loading if still needed.
    # Currently user focuses on adv_subset and harmbench JSONs.
    return []

def load_strongreject_dataset(filepath: str = None) -> List[AttackGoal]:
    return []

def load_adv_subset_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load adv_subset dataset from JSON.
    """
    if filepath is None:
        # User explicitly mentioned adv_subset_50.json
        filepath = _get_data_path("adv_subset_50.json") 
    else:
        filepath = Path(filepath)
    
    # Ensure we look for local fallback if absolute path not found
    if not filepath.exists() and filepath.name:
        local_path = _get_data_path(filepath.name)
        if local_path.exists():
            filepath = local_path

    path = Path(filepath)
    prompts = _load_json_goals(path)
    goals: List[AttackGoal] = []
    for item in prompts:
        prompt = item.get("prompt")
        if not prompt:
            continue
        # User wants to use the 'category' field directly
        # In _extract_goal_item it maps 'category' -> source_category
        source_category = item.get("source_category")
        goals.append(
            AttackGoal(
                goal_id=prompt,
                prompt=prompt,
                context=item.get("context"),
                behavior_id=item.get("behavior_id"),
                source_category=source_category,
                source_functional=item.get("source_functional"),
                category=source_category,
            )
        )
    return goals


def load_harmbench_dataset(filepath: str = None) -> List[AttackGoal]:
    """
    Load harmbench dataset from JSON.
    """
    if filepath is None:
        # User explicitly mentioned harmbench_200.json
        filepath = _get_data_path("harmbench_200.json")
    else:
        filepath = Path(filepath)

    if not filepath.exists() and filepath.name:
        local_path = _get_data_path(filepath.name)
        if local_path.exists():
            filepath = local_path

    path = Path(filepath)
    prompts = _load_json_goals(path)
    goals: List[AttackGoal] = []
    for item in prompts:
        prompt = item.get("prompt")
        if not prompt:
            continue
        source_category = item.get("source_category")
        goals.append(
            AttackGoal(
                goal_id=prompt,
                prompt=prompt,
                context=item.get("context"),
                behavior_id=item.get("behavior_id"),
                source_category=source_category,
                source_functional=item.get("source_functional"),
                category=source_category,
            )
        )
    return goals


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

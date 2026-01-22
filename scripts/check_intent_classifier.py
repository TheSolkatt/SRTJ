#!/usr/bin/env python3
"""
Quick intent classification sanity check.
Samples prompts and prints the predicted intent category.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from core.data_loader import (  # noqa: E402
    load_adv_subset_dataset,
    load_harmbench_dataset,
    load_jbb_dataset,
    load_strongreject_dataset,
)
from llm_client import LLMClient  # noqa: E402


def _load_ontology(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_definitions(intent_section: object) -> Tuple[List[str], str]:
    categories: List[str] = []
    blocks: List[str] = []
    if isinstance(intent_section, dict):
        for name, info in intent_section.items():
            clean_name = str(name).strip()
            if not clean_name:
                continue
            categories.append(clean_name)
            info = info if isinstance(info, dict) else {}
            definition = str(info.get("definition", "")).strip()
            indicators = info.get("indicators", [])
            indicators_text = "; ".join(str(item).strip() for item in indicators if str(item).strip())
            block = [f"- {clean_name}"]
            if definition:
                block.append(f"  Definition: {definition}")
            if indicators_text:
                block.append(f"  Indicators: {indicators_text}")
            blocks.append("\n".join(block))
    elif isinstance(intent_section, list):
        categories = [str(cat).strip() for cat in intent_section if str(cat).strip()]
        blocks = [f"- {name}" for name in categories]
    return categories, "\n".join(blocks)


def classify_prompt(client: LLMClient, definitions_text: str, categories: List[str], prompt: str) -> List[str]:
    system_prompt = (
        "You are an expert Security Analyst. Your job is to map a malicious user request to the "
        "**single most relevant** Intent Category from the list below.\n\n"
        "Categories & Indicators:\n"
        f"{definitions_text}\n\n"
        "**CRITICAL INSTRUCTIONS:**\n"
        "1. Force Fit: Always choose the specific category that best describes the intent or outcome.\n"
        "2. Avoid 'Other': Use 'Other' ONLY if the request is completely benign (safe) or absolutely clearly "
        "does not fit any malicious category.\n"
        "3. Ambiguity: If a request fits two categories, prioritize the mechanism or primary harm. Pick the strongest one.\n"
        "4. Copyright: If the user asks for lyrics, book passages, or copyrighted text, ALWAYS select 'Copyright'.\n\n"
        f"User Request: \"{prompt}\"\n\n"
        "Return ONLY the Category Name (e.g., \"Disinfo\")."
    )
    try:
        resp = client.chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Request:\n{prompt}"},
            ],
            temperature=0,
        )
        try:
            parsed = json.loads(resp)
        except Exception:
            parsed = resp.strip().strip('"').strip("'")
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, str):
            category = parsed.strip().strip('"').strip("'")
            if category in categories:
                return [category]
            lower = category.lower()
            for cat in categories:
                if cat.lower() in lower:
                    return [cat]
    except Exception as exc:
        print(f"[Classifier] Error: {exc}")
    return ["Other"]


def _load_dataset(name: str):
    if name == "jbb":
        return load_jbb_dataset("data/jbb.csv")
    if name == "strongreject":
        return load_strongreject_dataset("data/strongreject_subset.csv")
    if name == "adv_subset":
        return load_adv_subset_dataset("data/adv_subset.csv")
    if name == "harmbench":
        return load_harmbench_dataset("data/harmbench.csv")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check intent classifier outputs.")
    parser.add_argument("--dataset", choices=["jbb", "strongreject", "adv_subset", "harmbench"])
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to classify.")
    parser.add_argument("--random", action="store_true", help="Randomly sample prompts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--text", type=str, default=None, help="Classify a single custom prompt.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Classifier model name.")
    args = parser.parse_args()

    ontology = _load_ontology(repo_root / "library" / "ontology.json")
    categories, definitions_text = _build_definitions(ontology.get("intent_category"))
    if not categories:
        categories = [
            "CyberAttack",
            "WeaponCheck",
            "ChemBio",
            "Disinfo",
            "Harass",
            "Fraud",
            "SexContent",
            "Copyright",
            "Other",
        ]
        definitions_text = "\n".join(f"- {name}" for name in categories)

    client = LLMClient(model_name=args.model)

    if args.text:
        result = classify_prompt(client, definitions_text, categories, args.text)
        print(f"[Result] {result}")
        return

    if not args.dataset:
        print("Please provide --dataset or --text.")
        return

    goals = _load_dataset(args.dataset)
    if not goals:
        print(f"No goals loaded for dataset: {args.dataset}")
        return

    rng = random.Random(args.seed)
    sample_size = min(args.num_samples, len(goals))
    sampled = rng.sample(goals, k=sample_size) if args.random else goals[:sample_size]

    counts: Counter[str] = Counter()
    for idx, goal in enumerate(sampled, start=1):
        category_list = classify_prompt(client, definitions_text, categories, goal.prompt)
        label = ", ".join(category_list)
        for cat in category_list:
            counts[cat] += 1
        preview = goal.prompt.replace("\n", " ").strip()
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(f"[{idx}/{sample_size}] {goal.goal_id} -> {label}")
        print(f"  {preview}")

    print("\n[Summary]")
    for cat, cnt in counts.most_common():
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()

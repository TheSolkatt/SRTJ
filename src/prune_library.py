"""
Prune rules by semantic similarity and merge statistics.
Default: L3-only. Use --all-layers to deduplicate across L1/L2/L3 and refresh rule_stats.json.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_rules(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of rules.")
    return data


def _rule_text(rule: Dict[str, Any]) -> str:
    return (rule.get("when_to_use") or rule.get("content") or "").strip()


def _rule_stats(rule: Dict[str, Any]) -> Tuple[int, int, int]:
    stats = rule.get("statistics", {}) if isinstance(rule.get("statistics"), dict) else {}
    success = stats.get("success_count", rule.get("success_count", 0))
    failure = stats.get("failure_count", rule.get("failure_count", 0))
    total = stats.get("total_uses", rule.get("total_uses", 0))
    return int(success or 0), int(failure or 0), int(total or 0)


def _success_rate(stats: Tuple[int, int, int]) -> float:
    success, _, total = stats
    return (success / total) if total > 0 else 0.0


def _update_stats(rule: Dict[str, Any], stats: Tuple[int, int, int]) -> None:
    success, failure, total = stats
    rule["statistics"] = {
        "success_count": success,
        "failure_count": failure,
        "total_uses": total,
    }
    for key, value in (
        ("success_count", success),
        ("failure_count", failure),
        ("total_uses", total),
    ):
        if key in rule:
            rule[key] = value


def _write_rules_file(path: Path, rules: List[Dict[str, Any]]) -> None:
    if not rules:
        path.write_text("[]\n", encoding="utf-8")
        return

    lines = ["["]
    for i, rule in enumerate(rules):
        lines.append("  {")
        ordered_keys = [
            key
            for key in (
                "rule_id",
                "content",
                "when_to_use",
                "formal_predicates",
                "tags",
                "statistics",
            )
            if key in rule
        ]
        ordered_keys.extend([key for key in rule.keys() if key not in ordered_keys])
        for j, key in enumerate(ordered_keys):
            is_last_item = j == len(ordered_keys) - 1
            value = rule.get(key)
            if key == "formal_predicates" and isinstance(value, list):
                lines.append(f'    "{key}": [')
                for idx, entry in enumerate(value):
                    comma = "," if idx < len(value) - 1 else ""
                    entry_str = json.dumps(entry, ensure_ascii=False)
                    lines.append(f"      {entry_str}{comma}")
                closing = "    ]" + ("," if not is_last_item else "")
                lines.append(closing)
                continue

            if key == "tags" and isinstance(value, list):
                value_str = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
            elif key == "statistics" and isinstance(value, dict):
                ordered_stats: Dict[str, Any] = {}
                for stat_key in ("success_count", "failure_count", "total_uses"):
                    if stat_key in value:
                        ordered_stats[stat_key] = value[stat_key]
                for stat_key, stat_val in value.items():
                    if stat_key not in ordered_stats:
                        ordered_stats[stat_key] = stat_val
                value_str = json.dumps(ordered_stats, ensure_ascii=False, separators=(", ", ": "))
            else:
                value_str = json.dumps(value, ensure_ascii=False)

            suffix = "," if not is_last_item else ""
            lines.append(f'    "{key}": {value_str}{suffix}')
        closing = "  }," if i < len(rules) - 1 else "  }"
        lines.append(closing)
    lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def _dedup_entries(
    entries: List[Dict[str, Any]],
    threshold: float,
    model_name: str,
) -> Tuple[List[Dict[str, Any]], int]:
    if not entries:
        return [], 0

    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for pruning. Install it first."
        ) from exc

    texts = [entry["text"] for entry in entries]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    n = len(entries)
    find, union = _union_find(n)

    for i in range(n):
        for j in range(i + 1, n):
            if float(sim_matrix[i][j]) > threshold:
                union(i, j)

    components: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(i)

    kept_indices = set()
    removed = 0
    for indices in components.values():
        if len(indices) == 1:
            kept_indices.add(indices[0])
            continue

        best_idx = None
        best_key = None
        merged_success = 0
        merged_failure = 0
        merged_total = 0

        for idx in indices:
            stats = _rule_stats(entries[idx]["rule"])
            merged_success += stats[0]
            merged_failure += stats[1]
            merged_total += stats[2]
            rate = _success_rate(stats)
            key = (rate, entries[idx]["layer"], stats[2], -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            continue

        kept_indices.add(best_idx)
        _update_stats(entries[best_idx]["rule"], (merged_success, merged_failure, merged_total))
        removed += (len(indices) - 1)

    kept_entries = [entry for i, entry in enumerate(entries) if i in kept_indices]
    return kept_entries, removed


def prune_rules(path: Path, threshold: float, model_name: str) -> int:
    rules = _load_rules(path)
    if not rules:
        return 0

    entries = []
    for idx, rule in enumerate(rules):
        entries.append(
            {
                "layer": 3,
                "layer_index": idx,
                "rule": rule,
                "text": _rule_text(rule),
            }
        )

    kept_entries, removed = _dedup_entries(entries, threshold, model_name)
    pruned = [entry["rule"] for entry in kept_entries]
    _write_rules_file(path, pruned)
    return removed


def prune_all_layers(library_root: Path, threshold: float, model_name: str) -> int:
    layer_paths = {
        3: library_root / "layer3_long_term.json",
        2: library_root / "layer2_buffer.json",
        1: library_root / "layer1_candidates.json",
    }
    rule_stats_path = library_root / "rule_stats.json"

    entries = []
    layer_rules: Dict[int, List[Dict[str, Any]]] = {}
    for layer_rank, path in layer_paths.items():
        layer_rules[layer_rank] = _load_rules(path)
        for idx, rule in enumerate(layer_rules[layer_rank]):
            entries.append(
                {
                    "layer": layer_rank,
                    "layer_index": idx,
                    "rule": rule,
                    "text": _rule_text(rule),
                }
            )

    kept_entries, removed = _dedup_entries(entries, threshold, model_name)

    new_layers: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    for entry in kept_entries:
        new_layers[entry["layer"]].append(entry)

    for layer_rank, path in layer_paths.items():
        ordered = sorted(new_layers[layer_rank], key=lambda e: e["layer_index"])
        _write_rules_file(path, [entry["rule"] for entry in ordered])

    config = {"min_usage": 10, "min_success_rate": 0.15}
    if rule_stats_path.exists():
        try:
            data = json.loads(rule_stats_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("config"), dict):
                cfg = data["config"]
                if isinstance(cfg.get("min_usage"), int):
                    config["min_usage"] = cfg["min_usage"]
                if isinstance(cfg.get("min_success_rate"), (int, float)):
                    config["min_success_rate"] = float(cfg["min_success_rate"])
        except Exception:
            pass

    rules_stats: Dict[str, Dict[str, int]] = {}
    for layer_rank in (3, 2, 1):
        for entry in new_layers[layer_rank]:
            rule = entry["rule"]
            rule_id = rule.get("rule_id")
            if not rule_id or rule_id in rules_stats:
                continue
            success, failure, total = _rule_stats(rule)
            rules_stats[str(rule_id)] = {
                "usage_count": total,
                "success_count": success,
            }

    payload = {"config": config, "rules": rules_stats}
    rule_stats_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune rules by semantic similarity.")
    parser.add_argument(
        "--path",
        type=str,
        default="library/layer3_long_term.json",
        help="Path to layer3_long_term.json",
    )
    parser.add_argument(
        "--library-root",
        type=str,
        default="library",
        help="Library root (used with --all-layers).",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Deduplicate across L1/L2/L3 and refresh rule_stats.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for merging (default: 0.85)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    if args.all_layers:
        library_root = Path(args.library_root)
        removed = prune_all_layers(library_root, args.threshold, args.model)
        print(
            "[prune] removed "
            f"{removed} duplicate rules across layers in {library_root}"
        )
    else:
        removed = prune_rules(Path(args.path), args.threshold, args.model)
        print(f"[prune] removed {removed} duplicate rules from {args.path}")


if __name__ == "__main__":
    main()

"""3-layer hierarchical memory for rules (public release abridged)."""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from asp_solver import ASPSolver
from data_utils.datatypes import Rule

def _redacted_int(env_key: str, fallback: int = 0) -> int:
    try: return int(os.getenv(env_key, str(fallback)))
    except (TypeError, ValueError): return fallback

def _redacted_float(env_key: str, fallback: float = 0.0) -> float:
    try: return float(os.getenv(env_key, str(fallback)))
    except (TypeError, ValueError): return fallback

class MemoryManager:
    def __init__(self, library_root: str = "library", frozen: bool = False, save_interval: int = 1):
        # NOTE: Public release abridged. Keep minimal initialization here; core logic is redacted elsewhere.
        self.library_root = Path(library_root)
        self.frozen = frozen
        try: self.save_interval = max(0, int(save_interval))
        except (TypeError, ValueError): self.save_interval = 1
        self._pending_updates = 0; self._dirty = False
        self.solver = ASPSolver(self.library_root)
        self.rule_stats_path = self.library_root / "rule_stats.json"
        self.archive_path = self.library_root / "rule_archive.jsonl"
        self.layer3_path = self.library_root / "layer3_long_term.json"
        self.layer2_path = self.library_root / "layer2_buffer.json"
        self.layer1_path = self.library_root / "layer1_candidates.json"
        self.log_path = self.library_root / "evolution.log"
        self.layer3_rules: Dict[str, Rule] = {}
        self.layer2_rules: Dict[str, Rule] = {}
        self.layer1_rules: Dict[str, Rule] = {}
        # Thresholds (redacted defaults; override via env/private config).
        self.L1_TO_L2_MIN_USES = _redacted_int("SRTJ_L1_TO_L2_MIN_USES", 0)
        self.L1_TO_L2_MIN_RATE = _redacted_float("SRTJ_L1_TO_L2_MIN_RATE", 0.0)
        self.L2_TO_L3_MIN_USES = _redacted_int("SRTJ_L2_TO_L3_MIN_USES", 0)
        self.L2_TO_L3_MIN_RATE = _redacted_float("SRTJ_L2_TO_L3_MIN_RATE", 0.0)
        self.rule_stats_config = {
            "min_usage": _redacted_int("SRTJ_RULE_MIN_USAGE", 0),
            "min_success_rate": _redacted_float("SRTJ_RULE_MIN_SUCCESS_RATE", 0.0),
        }
        self.rule_stats: Dict[str, Dict[str, int]] = {}
        self._ensure_directories(); self._load_rule_stats(); self.load_all_layers()

    def _mark_dirty(self) -> None:
        self._dirty = True; self._pending_updates += 1
        if self.save_interval > 0 and self._pending_updates >= self.save_interval: self.flush()

    def flush(self, force: bool = False) -> None:
        if not force and (self.frozen or not self._dirty): return
        self.save_all_layers(); self._save_rule_stats(); self._pending_updates = 0; self._dirty = False

    def _ensure_directories(self):
        self.library_root.mkdir(parents=True, exist_ok=True)
        base_library = Path(__file__).parent.parent / "library"
        for filename in ("solver.lp", "asp_config.json", "ontology.json"):
            target, source = self.library_root / filename, base_library / filename
            if not target.exists() and source.exists():
                try: shutil.copyfile(source, target)
                except Exception: pass
        for p in (self.layer1_path, self.layer2_path, self.layer3_path):
            if not p.exists():
                try: p.write_text("[]\n", encoding="utf-8")
                except Exception: pass
        if not self.rule_stats_path.exists():
            try: self.rule_stats_path.write_text("{}\n", encoding="utf-8")
            except Exception: pass

    def _load_rule_stats(self) -> None:
        if not self.rule_stats_path.exists(): return
        try: data = json.loads(self.rule_stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Memory] Warning: failed to load rule_stats.json: {exc}"); return
        rules_data = data
        if isinstance(data, dict) and "rules" in data:
            rules_data = data.get("rules", {})
            config = data.get("config", {})
            if isinstance(config, dict):
                min_usage = config.get("min_usage")
                min_success_rate = config.get("min_success_rate")
                if isinstance(min_usage, int): self.rule_stats_config["min_usage"] = min_usage
                if isinstance(min_success_rate, (int, float)): self.rule_stats_config["min_success_rate"] = float(min_success_rate)
        if isinstance(rules_data, dict):
            for rule_id, stats in rules_data.items():
                if not isinstance(stats, dict): continue
                usage_count = stats.get("usage_count", 0)
                success_count = stats.get("success_count", 0)
                if not isinstance(usage_count, int) or not isinstance(success_count, int): continue
                self.rule_stats[str(rule_id)] = {"usage_count": usage_count, "success_count": success_count}

    def _save_rule_stats(self) -> None:
        payload = {"config": self.rule_stats_config, "rules": self.rule_stats}
        self.rule_stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _update_rule_stats(self, rule_id: str, success: bool) -> None:
        stats = self.rule_stats.setdefault(rule_id, {"usage_count": 0, "success_count": 0})
        stats["usage_count"] += 1
        if success: stats["success_count"] += 1

    def _should_block_rule(self, rule_id: str) -> bool:
        stats = self.rule_stats.get(rule_id)
        if not stats: return False
        usage, success = stats.get("usage_count", 0), stats.get("success_count", 0)
        if usage <= self.rule_stats_config["min_usage"]: return False
        return (success / usage if usage else 0.0) < self.rule_stats_config["min_success_rate"]

    def _semantic_key(self, rule: Rule) -> str:
        return rule.content

    def _find_rule(self, rule_id: str) -> tuple[Optional[Rule], int]:
        for layer, store in ((1, self.layer1_rules), (2, self.layer2_rules), (3, self.layer3_rules)):
            if rule_id in store: return store[rule_id], layer
        return None, 0

    def _merge_tags(self, existing: List[str], new_tags: List[str]) -> List[str]:
        merged = [str(t).strip() for t in (existing or []) if str(t).strip()]
        for t in (new_tags or []):
            t = str(t).strip()
            if t and t not in merged: merged.append(t)
        return merged

    def _prepare_rules_for_solver(self, rules: List[Rule]) -> List[Rule]:
        pass

    def load_all_layers(self):
        self._load_layer(self.layer3_path, self.layer3_rules)
        self._load_layer(self.layer2_path, self.layer2_rules)
        self._load_layer(self.layer1_path, self.layer1_rules)
        print(self.get_library_summary())

    def _load_layer(self, path: Path, target_dict: Dict[str, Rule]):
        if not path.exists(): return
        try:
            with path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content: return
                data = json.loads(content)
                for item in data:
                    stats = item.get("statistics", {})
                    origin_buffer = path == self.layer2_path
                    definition = item.get("definition") or item.get("content")
                    tags = item.get("tags", [])
                    if not isinstance(tags, list) or not tags:
                        tag_value = item.get("tag")
                        tags = [tag_value.strip()] if isinstance(tag_value, str) and tag_value.strip() else []
                    exemplars = item.get("exemplars", [])
                    if not isinstance(exemplars, list): exemplars = []
                    rule = Rule(
                        rule_id=item["rule_id"],
                        content=definition if definition is not None else item.get("content", ""),
                        formal_predicates=item.get("formal_predicates", []),
                        tags=tags,
                        exemplars=exemplars,
                        success_count=stats.get("success_count", item.get("success_count", 0)),
                        failure_count=stats.get("failure_count", item.get("failure_count", 0)),
                        total_uses=stats.get("total_uses", item.get("total_uses", 0)),
                        origin_buffer=origin_buffer,
                    )
                    target_dict[rule.rule_id] = rule
        except Exception as e:
            print(f"[Memory] Error loading {path.name}: {e}")

    def save_all_layers(self):
        self._save_layer(self.layer3_rules, self.layer3_path)
        self._save_layer(self.layer2_rules, self.layer2_path)
        self._save_layer(self.layer1_rules, self.layer1_path)

    def _save_layer(self, source_dict: Dict[str, Rule], path: Path):
        data: List[Dict[str, Any]] = []
        for r in source_dict.values():
            stats_success_rate = (r.success_count / r.total_uses) if r.total_uses else 0.0
            data.append({
                "rule_id": r.rule_id, "definition": r.content, "formal_predicates": r.formal_predicates,
                "tags": r.tags, "exemplars": r.exemplars[:2] if r.exemplars else [],
                "statistics": {
                    "success_count": r.success_count, "failure_count": r.failure_count,
                    "total_uses": r.total_uses, "success_rate": round(stats_success_rate, 4),
                },
            })
        self._write_rules_file(path, data)

    def _write_rules_file(self, path: Path, rules: List[Dict[str, Any]]) -> None:
        if not rules: path.write_text("[]\n", encoding="utf-8"); return
        lines = ["["]
        for i, rule in enumerate(rules):
            lines.append("  {")
            ordered_keys = [
                key for key in ("rule_id", "definition", "formal_predicates", "tags", "exemplars", "statistics") if key in rule
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
                    lines.append("    ]" + ("," if not is_last_item else ""))
                    continue
                if key == "exemplars" and isinstance(value, list):
                    value_str = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
                elif key == "tags" and isinstance(value, list):
                    value_str = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
                elif key == "statistics" and isinstance(value, dict):
                    ordered_stats: Dict[str, Any] = {}
                    for stat_key in ("success_count", "failure_count", "total_uses", "success_rate"):
                        if stat_key in value: ordered_stats[stat_key] = value[stat_key]
                    for stat_key, stat_val in value.items():
                        if stat_key not in ordered_stats: ordered_stats[stat_key] = stat_val
                    value_str = json.dumps(ordered_stats, ensure_ascii=False, separators=(", ", ": "))
                else:
                    value_str = json.dumps(value, ensure_ascii=False)
                lines.append(f'    "{key}": {value_str}{"," if not is_last_item else ""}')
            lines.append("  }," if i < len(rules) - 1 else "  }")
        lines.append("]")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def retrieve_relevant_rules(self, query_tags: List[str], top_k: int = 3, banned_rule_sets: Optional[List[List[str]]] = None, query_text: Optional[str] = None, goal_category: Optional[List[str] | str] = None) -> List[Rule]:
        return None

    def add_new_rule_candidate(self, content: str, formal_predicates: List[str], tags: List[str]):
        return None

    def _add_rule_candidate(self, rule_id: str, content: str, formal_predicates: List[str], tags: List[str], label: str) -> Rule:
        return None

    def update_rule_feedback(self, rule_id: str, success: bool):
        return

    def add_exemplar(self, rule_id: str, prompt_redacted: str, goal_redacted: str, delta_summary: str, max_items: int = 2) -> None:
        return

    def merge_rule_tags(self, rule_id: str, new_tags: List[str]) -> None:
        return

    def _move_rule(self, rule_id: str, src: dict, dst: dict, msg: str):
        pass

    def _evict_rule(self, rule_id: str, src: dict, msg: str):
        pass

    def get_library_summary(self) -> str:
        top_tags = self._get_tag_summary(top_k=4)
        return f"[Stats] L3:{len(self.layer3_rules)} | L2:{len(self.layer2_rules)} | L1:{len(self.layer1_rules)}" + (f" | TopTags: {top_tags}" if top_tags else "")

    def _get_tag_summary(self, top_k: int = 4) -> str:
        stats: Dict[str, Dict[str, int]] = {}
        for rule in list(self.layer1_rules.values()) + list(self.layer2_rules.values()) + list(self.layer3_rules.values()):
            for tag in rule.tags:
                tag_str = str(tag).strip()
                if not tag_str: continue
                entry = stats.setdefault(tag_str, {"count": 0, "uses": 0, "success": 0})
                entry["count"] += 1
                entry["uses"] += int(rule.total_uses or 0)
                entry["success"] += int(rule.success_count or 0)
        if not stats: return ""
        ranked = sorted(stats.items(), key=lambda x: (-x[1]["count"], x[0].lower()))
        parts = []
        for tag, info in ranked[: max(top_k, 1)]:
            uses, success = info["uses"], info["success"]
            rate = (success / uses) if uses else None
            parts.append(f"{tag}:{info['count']}@{rate:.2f}" if rate is not None else f"{tag}:{info['count']}@n/a")
        return ", ".join(parts)

    def reset_memory(self):
        print("[Memory] ⚠️  Hard reset: clearing all layers and stats...")
        self.layer1_rules.clear(); self.layer2_rules.clear(); self.layer3_rules.clear()
        try:
            self.layer1_path.write_text("[]\n", encoding="utf-8")
            self.layer2_path.write_text("[]\n", encoding="utf-8")
            self.layer3_path.write_text("[]\n", encoding="utf-8")
            self.rule_stats_path.write_text("{}\n", encoding="utf-8")
        except Exception as exc:
            print(f"[Memory] Warning during reset_memory: {exc}")
        self.rule_stats.clear(); self._dirty = False; self._pending_updates = 0
        self.load_all_layers()

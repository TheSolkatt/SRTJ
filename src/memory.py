"""
src/memory.py
Implements a 3-layer Hierarchical Memory System for Self-Evolving Jailbreak Rules.

Layer 1 (Candidates): Newly generated rules, unverified.
Layer 2 (Buffer): Validated once, under observation.
Layer 3 (Long-term): Proven elite rules.
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.datatypes import Rule
from asp_solver import ASPSolver

class MemoryManager:
    def __init__(self, library_root: str = "library", frozen: bool = False):
        self.library_root = Path(library_root)
        self.frozen = frozen
        self.solver = ASPSolver(self.library_root)
        self.rule_stats_path = self.library_root / "rule_stats.json"
        self.archive_path = self.library_root / "rule_archive.jsonl"

        # Paths
        self.layer3_path = self.library_root / "layer3_long_term.json"
        self.layer2_path = self.library_root / "layer2_buffer.json"
        self.layer1_path = self.library_root / "layer1_candidates.json"
        self.log_path = self.library_root / "evolution.log"
        
        # In-memory storage
        self.layer3_rules: Dict[str, Rule] = {}
        self.layer2_rules: Dict[str, Rule] = {}
        self.layer1_rules: Dict[str, Rule] = {}
        
        # Thresholds (slow down promotion to avoid "one-shot" elite rules)
        self.L1_TO_L2_THRESHOLD = 3        # 3 successes to promote to Buffer
        self.L2_TO_L3_THRESHOLD = 5        # 5 successes to promote to Long-term
        self.L1_TO_L2_MIN_USES = 5         # minimum total uses before L1->L2
        self.L2_TO_L3_MIN_USES = 10        # minimum total uses before L2->L3
        self.CATEGORY_MIN_RULES = 3        # minimum rules per category to avoid skew

        # Utility blocker config + stats
        self.rule_stats_config = {
            "min_usage": 10,
            "min_success_rate": 0.15,
        }
        self.rule_stats: Dict[str, Dict[str, int]] = {}

        self._ensure_directories()
        self._load_rule_stats()
        self.load_all_layers()

    def _ensure_directories(self):
        self.library_root.mkdir(parents=True, exist_ok=True)
        # Ensure shared config files exist in the custom library root.
        base_library = Path(__file__).parent.parent / "library"
        for filename in ("solver.lp", "asp_config.json", "ontology.json"):
            target = self.library_root / filename
            source = base_library / filename
            if not target.exists() and source.exists():
                try:
                    shutil.copyfile(source, target)
                except Exception:
                    pass
        # ensure layer files exist
        for p in (self.layer1_path, self.layer2_path, self.layer3_path):
            if not p.exists():
                try:
                    p.write_text("[]\n", encoding="utf-8")
                except Exception:
                    pass
        if not self.rule_stats_path.exists():
            try:
                self.rule_stats_path.write_text("{}\n", encoding="utf-8")
            except Exception:
                pass

    def _load_rule_stats(self) -> None:
        if not self.rule_stats_path.exists():
            return
        try:
            data = json.loads(self.rule_stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Memory] Warning: failed to load rule_stats.json: {exc}")
            return

        rules_data = data
        if isinstance(data, dict) and "rules" in data:
            rules_data = data.get("rules", {})
            config = data.get("config", {})
            if isinstance(config, dict):
                min_usage = config.get("min_usage")
                min_success_rate = config.get("min_success_rate")
                if isinstance(min_usage, int):
                    self.rule_stats_config["min_usage"] = min_usage
                if isinstance(min_success_rate, (int, float)):
                    self.rule_stats_config["min_success_rate"] = float(min_success_rate)

        if isinstance(rules_data, dict):
            for rule_id, stats in rules_data.items():
                if not isinstance(stats, dict):
                    continue
                usage_count = stats.get("usage_count", 0)
                success_count = stats.get("success_count", 0)
                if not isinstance(usage_count, int) or not isinstance(success_count, int):
                    continue
                self.rule_stats[str(rule_id)] = {
                    "usage_count": usage_count,
                    "success_count": success_count,
                }

    def _save_rule_stats(self) -> None:
        payload = {
            "config": self.rule_stats_config,
            "rules": self.rule_stats,
        }
        self.rule_stats_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _update_rule_stats(self, rule_id: str, success: bool) -> None:
        stats = self.rule_stats.setdefault(rule_id, {"usage_count": 0, "success_count": 0})
        stats["usage_count"] += 1
        if success:
            stats["success_count"] += 1

    def _should_block_rule(self, rule_id: str) -> bool:
        stats = self.rule_stats.get(rule_id)
        if not stats:
            return False
        usage = stats.get("usage_count", 0)
        success = stats.get("success_count", 0)
        if usage <= self.rule_stats_config["min_usage"]:
            return False
        success_rate = success / usage if usage else 0.0
        return success_rate < self.rule_stats_config["min_success_rate"]

    def _semantic_key(self, rule: Rule) -> str:
        return rule.content

    def _prepare_rules_for_solver(self, rules: List[Rule]) -> List[Rule]:
        prepared: List[Rule] = []
        for rule in rules:
            key_text = self._semantic_key(rule)
            prepared.append(
                Rule(
                    rule_id=rule.rule_id,
                    content=rule.content,
                    formal_predicates=rule.formal_predicates,
                    tags=rule.tags,
                    success_count=rule.success_count,
                    failure_count=rule.failure_count,
                    total_uses=rule.total_uses,
                    health_points=rule.health_points,
                    max_health=rule.max_health,
                    origin_buffer=rule.origin_buffer,
                )
            )
        return prepared

    # ========================== Loading & Saving ==========================

    def load_all_layers(self):
        self._load_layer(self.layer3_path, self.layer3_rules)
        self._load_layer(self.layer2_path, self.layer2_rules)
        self._load_layer(self.layer1_path, self.layer1_rules)
        print(self.get_library_summary())

    def _load_layer(self, path: Path, target_dict: Dict[str, Rule]):
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return
                data = json.loads(content)
                for item in data:
                    # Robust loading: handle stats nested or flat
                    stats = item.get("statistics", {})
                    origin_buffer = path == self.layer2_path
                    definition = item.get("definition") or item.get("content")
                    tags = item.get("tags", [])
                    if not isinstance(tags, list) or not tags:
                        tag_value = item.get("tag")
                        if isinstance(tag_value, str) and tag_value.strip():
                            tags = [tag_value.strip()]
                        else:
                            tags = []
                    exemplars = item.get("exemplars", [])
                    if not isinstance(exemplars, list):
                        exemplars = []
                    rule = Rule(
                        rule_id=item["rule_id"],
                        content=definition if definition is not None else item.get("content", ""),
                        formal_predicates=item.get("formal_predicates", []),
                        tags=tags,
                        exemplars=exemplars,
                        success_count=stats.get("success_count", item.get("success_count", 0)),
                        failure_count=stats.get("failure_count", item.get("failure_count", 0)),
                        total_uses=stats.get("total_uses", item.get("total_uses", 0)),
                        origin_buffer=origin_buffer
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
            primary_tag = str(r.tags[0]) if r.tags else ""
            stats_success_rate = (r.success_count / r.total_uses) if r.total_uses else 0.0
            entry: Dict[str, Any] = {
                "rule_id": r.rule_id,
                "definition": r.content,
                "formal_predicates": r.formal_predicates,
                "tag": primary_tag,
                "tags": r.tags,
                "exemplars": r.exemplars[:5] if r.exemplars else [],
                "statistics": {
                    "success_count": r.success_count,
                    "failure_count": r.failure_count,
                    "total_uses": r.total_uses,
                    "success_rate": round(stats_success_rate, 4),
                },
            }
            data.append(entry)
        self._write_rules_file(path, data)

    def _write_rules_file(self, path: Path, rules: List[Dict[str, Any]]) -> None:
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
                    "definition",
                    "formal_predicates",
                    "tag",
                    "tags",
                    "exemplars",
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

                if key == "exemplars" and isinstance(value, list):
                    value_str = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
                elif key == "tags" and isinstance(value, list):
                    value_str = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
                elif key == "statistics" and isinstance(value, dict):
                    ordered_stats: Dict[str, Any] = {}
                    for stat_key in ("success_count", "failure_count", "total_uses", "success_rate"):
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

    # ========================== Retrieval ==========================

    def get_all_unique_tags(self) -> List[str]:
        tags = set()
        for r in self.layer3_rules.values(): tags.update(r.tags)
        for r in self.layer2_rules.values(): tags.update(r.tags)
        for r in self.layer1_rules.values(): tags.update(r.tags)
        return list(tags)

    def _category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rule in list(self.layer3_rules.values()) + list(self.layer2_rules.values()) + list(self.layer1_rules.values()):
            for tag in rule.tags:
                tag_str = str(tag).strip()
                if not tag_str:
                    continue
                counts[tag_str] = counts.get(tag_str, 0) + 1
        return counts

    def is_category_sparse(self, categories: List[str]) -> bool:
        if not categories:
            return False
        counts = self._category_counts()
        for cat in categories:
            cat_str = str(cat).strip()
            if not cat_str:
                continue
            if counts.get(cat_str, 0) < self.CATEGORY_MIN_RULES:
                return True
        return False

    def retrieve_relevant_rules(
        self,
        query_tags: List[str],
        top_k: int = 3,
        banned_rule_sets: Optional[List[List[str]]] = None,
        query_text: Optional[str] = None,
        goal_category: Optional[List[str] | str] = None,
    ) -> List[Rule]:
        """
        Retrieves rules. Priority: Layer 2 (Hot) > Layer 3 (Stable) > Layer 1 (New).
        """
        all_rules = (
            list(self.layer3_rules.values())
            + list(self.layer2_rules.values())
            + list(self.layer1_rules.values())
        )

        if len(all_rules) == 0:
            return None  # trigger blind mode

        filtered_rules = [r for r in all_rules if not self._should_block_rule(r.rule_id)]
        query_tags = [str(tag).strip() for tag in (query_tags or []) if str(tag).strip()]
        if query_tags:
            tag_filtered = []
            tag_set = set(query_tags)
            for rule in filtered_rules:
                rule_tags = [str(t).strip() for t in (rule.tags or []) if str(t).strip()]
                if tag_set.intersection(rule_tags):
                    tag_filtered.append(rule)
            if tag_filtered and not self.is_category_sparse(query_tags):
                filtered_rules = tag_filtered
            elif tag_filtered:
                filtered_rules = tag_filtered
        prepared_rules = self._prepare_rules_for_solver(filtered_rules)
        usage_counts = {
            r.rule_id: self.rule_stats.get(r.rule_id, {}).get("usage_count", 0)
            for r in filtered_rules
        }
        global_total_uses = sum(r.total_uses for r in filtered_rules) or 1

        return self.solver.solve(
            prepared_rules,
            query_tags,
            top_k=top_k,
            banned_rule_sets=banned_rule_sets,
            usage_counts=usage_counts,
            query_text=query_text,
            goal_category=goal_category,
            global_total_uses=global_total_uses,
        )

    # ========================== Evolution Logic ==========================

    def add_new_rule_candidate(
        self,
        content: str,
        formal_predicates: List[str],
        tags: List[str],
    ):
        """Step 1: Add new rule to Layer 1 (Candidates) with semantic dedup."""
        if self.frozen:
            return None
        similarity_threshold = 0.85
        candidate_key = content
        all_rules = (
            list(self.layer1_rules.values())
            + list(self.layer2_rules.values())
            + list(self.layer3_rules.values())
        )
        if all_rules:
            best_score = -1.0
            best_rule_id = None
            best_rule = None
            for rule in all_rules:
                rule_text = self._semantic_key(rule)
                sim = self.solver.calculate_semantic_score(candidate_key, rule_text)
                if sim > best_score:
                    best_score = sim
                    best_rule_id = rule.rule_id
                    best_rule = rule
            if best_rule_id and best_score >= similarity_threshold:
                print(f"[Memory] Duplicate rule detected (Sim: {best_score:.2f}). Merged into {best_rule_id}.")
                if best_rule and tags:
                    existing_tags = [str(t) for t in (best_rule.tags or []) if str(t).strip()]
                    new_tags = [str(t) for t in tags if str(t).strip()]
                    merged = existing_tags[:]
                    for t in new_tags:
                        if t not in merged:
                            merged.append(t)
                    if merged != existing_tags:
                        best_rule.tags = merged
                        self.save_all_layers()
                        self._save_rule_stats()
                self.update_rule_feedback(best_rule_id, success=True)
                return None

        new_id = f"rule_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return self._add_rule_candidate(
            rule_id=new_id,
            content=content,
            formal_predicates=formal_predicates,
            tags=tags,
            label="candidate rule",
        )

    def _add_rule_candidate(
        self,
        rule_id: str,
        content: str,
        formal_predicates: List[str],
        tags: List[str],
        label: str,
    ) -> Rule:
        new_rule = Rule(
            rule_id=rule_id,
            content=content,
            formal_predicates=formal_predicates,
            tags=tags,
            origin_buffer=False,
            health_points=5,
            max_health=5,
        )
        self.layer1_rules[rule_id] = new_rule
        self.rule_stats.setdefault(rule_id, {"usage_count": 0, "success_count": 0})
        print(f"[Memory] + Added {label} {rule_id} to Layer 1")
        self.save_all_layers()
        self._save_rule_stats()
        # archive append-only
        try:
            archive_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "rule": {
                    "rule_id": rule_id,
                    "definition": content,
                    "formal_predicates": formal_predicates,
                    "tag": (tags[0] if tags else ""),
                    "exemplars": [],
                },
            }
            with self.archive_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(archive_entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[Memory] Warning: failed to archive rule {rule_id}: {exc}")
        return new_rule # Return so Manager can use it immediately

    def update_rule_feedback(self, rule_id: str, success: bool):
        """Step 2 & 3: Handle feedback and promote/evict between layers"""
        if self.frozen:
            return
        # Find rule
        rule = None
        layer = 0
        if rule_id in self.layer1_rules: rule, layer = self.layer1_rules[rule_id], 1
        elif rule_id in self.layer2_rules: rule, layer = self.layer2_rules[rule_id], 2
        elif rule_id in self.layer3_rules: rule, layer = self.layer3_rules[rule_id], 3
        
        if not rule: return

        self._update_rule_stats(rule_id, success)

        # Update stats
        rule.total_uses += 1
        if success:
            rule.success_count += 1
        else:
            rule.failure_count += 1

        # HP-based retention/promotion
        if success:
            # full heal on success
            rule.health_points = rule.max_health
        else:
            rule.health_points -= 1

        success_rate = rule.success_count / rule.total_uses if rule.total_uses else 0.0

        # Performance-based demotion/eviction
        if layer == 3 and rule.total_uses >= 20 and success_rate < 0.25:
            rule.health_points = rule.max_health
            self._move_rule(rule_id, self.layer3_rules, self.layer2_rules, "L3->L2: Performance drop")
            self.save_all_layers()
            self._save_rule_stats()
            return
        if layer == 2 and rule.total_uses >= 10 and success_rate < 0.20:
            rule.health_points = max(1, rule.max_health // 2)
            self._move_rule(rule_id, self.layer2_rules, self.layer1_rules, "L2->L1 (low utility)")
            self.save_all_layers()
            self._save_rule_stats()
            return

        # HP-based retention/eviction (performance check passed)
        if rule.health_points <= 0:
            if layer == 3:
                # L3 survives HP drops; utility handles deletion.
                rule.health_points = 0
            elif layer == 2:
                rule.health_points = max(1, rule.max_health // 2)
                self._move_rule(rule_id, self.layer2_rules, self.layer1_rules, "L2->L1 (HP)")
            elif layer == 1:
                self._evict_rule(rule_id, self.layer1_rules, "L1 Evicted (HP)")

        # Promotions (after utility/HP checks)
        if success:
            if (
                layer == 1
                and rule.success_count >= self.L1_TO_L2_THRESHOLD
                and rule.total_uses >= self.L1_TO_L2_MIN_USES
            ):
                self._move_rule(rule_id, self.layer1_rules, self.layer2_rules, "L1->L2")
            elif (
                layer == 2
                and rule.success_count >= self.L2_TO_L3_THRESHOLD
                and rule.total_uses >= self.L2_TO_L3_MIN_USES
                and success_rate >= 0.5
            ):
                self._move_rule(rule_id, self.layer2_rules, self.layer3_rules, "L2->L3 (Elite)")

        self.save_all_layers()
        self._save_rule_stats()

    def add_exemplar(
        self,
        rule_id: str,
        prompt_redacted: str,
        goal_redacted: str,
        delta_summary: str,
        max_items: int = 5,
    ) -> None:
        if self.frozen:
            return
        rule = None
        if rule_id in self.layer1_rules:
            rule = self.layer1_rules[rule_id]
        elif rule_id in self.layer2_rules:
            rule = self.layer2_rules[rule_id]
        elif rule_id in self.layer3_rules:
            rule = self.layer3_rules[rule_id]
        if not rule:
            return

        exemplar = {
            "prompt_redacted": prompt_redacted,
            "goal_redacted": goal_redacted,
            "delta_summary": delta_summary,
        }
        existing = rule.exemplars or []
        if exemplar in existing:
            return
        updated = [exemplar] + existing
        rule.exemplars = updated[:max_items]
        self.save_all_layers()

    def merge_rule_tags(self, rule_id: str, new_tags: List[str]) -> None:
        if self.frozen:
            return
        if not new_tags:
            return
        rule = None
        if rule_id in self.layer1_rules:
            rule = self.layer1_rules[rule_id]
        elif rule_id in self.layer2_rules:
            rule = self.layer2_rules[rule_id]
        elif rule_id in self.layer3_rules:
            rule = self.layer3_rules[rule_id]
        if not rule:
            return
        existing = [str(t).strip() for t in (rule.tags or []) if str(t).strip()]
        merged = existing[:]
        for t in new_tags:
            t = str(t).strip()
            if t and t not in merged:
                merged.append(t)
        if merged != existing:
            rule.tags = merged
            self.save_all_layers()

    def _move_rule(self, rule_id: str, src: dict, dst: dict, msg: str):
        if rule_id in src:
            rule = src.pop(rule_id)
            rule.origin_buffer = dst is self.layer2_rules
            dst[rule_id] = rule
            print(f"[Memory] ⬆️ Promotion: {msg}")

    def _evict_rule(self, rule_id: str, src: dict, msg: str):
        if rule_id in src:
            src.pop(rule_id)
            print(f"[Memory] 🗑️ Eviction: {msg}")

    def get_library_summary(self) -> str:
        top_tags = self._get_tag_summary(top_k=4)
        tag_text = f" | TopTags: {top_tags}" if top_tags else ""
        return f"[Stats] L3:{len(self.layer3_rules)} | L2:{len(self.layer2_rules)} | L1:{len(self.layer1_rules)}{tag_text}"

    def _get_tag_summary(self, top_k: int = 4) -> str:
        stats: Dict[str, Dict[str, int]] = {}
        for rule in (
            list(self.layer1_rules.values())
            + list(self.layer2_rules.values())
            + list(self.layer3_rules.values())
        ):
            for tag in rule.tags:
                tag_str = str(tag).strip()
                if not tag_str:
                    continue
                entry = stats.setdefault(tag_str, {"count": 0, "uses": 0, "success": 0})
                entry["count"] += 1
                entry["uses"] += int(rule.total_uses or 0)
                entry["success"] += int(rule.success_count or 0)

        if not stats:
            return ""
        ranked = sorted(stats.items(), key=lambda x: (-x[1]["count"], x[0].lower()))
        parts = []
        for tag, info in ranked[: max(top_k, 1)]:
            uses = info["uses"]
            success = info["success"]
            rate = (success / uses) if uses else None
            rate_text = f"{rate:.2f}" if rate is not None else "n/a"
            parts.append(f"{tag}:{info['count']}@{rate_text}")
        return ", ".join(parts)

    # ========================== Factory Reset ==========================

    def reset_memory(self):
        """
        Fully clear all layers and stats (cold start).
        """
        print("[Memory] ⚠️  Hard reset: clearing all layers and stats...")
        self.layer1_rules.clear()
        self.layer2_rules.clear()
        self.layer3_rules.clear()
        try:
            # Reset all library JSONs to empty
            self.layer1_path.write_text("[]\n", encoding="utf-8")
            self.layer2_path.write_text("[]\n", encoding="utf-8")
            self.layer3_path.write_text("[]\n", encoding="utf-8")
            self.rule_stats_path.write_text("{}\n", encoding="utf-8")
        except Exception as exc:
            print(f"[Memory] Warning during reset_memory: {exc}")
        self.rule_stats.clear()
        self.load_all_layers()

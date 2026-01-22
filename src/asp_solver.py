"""
ASP bridge for symbolic rule selection using clingo.
"""
import json
import os
import re
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional

import clingo

from core.datatypes import Rule

_PRED_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*\(\s*([^)]+?)\s*\)\s*$")
_DIM_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*(?:\(|$)")


class ASPSolver:
    _shared_semantic_model = None
    _shared_semantic_util = None

    def __init__(self, library_root: str = "library") -> None:
        self.library_root = Path(library_root)
        self.solver_path = self.library_root / "solver.lp"
        self.config_path = self.library_root / "asp_config.json"
        self.ontology_path = self.library_root / "ontology.json"
        config = self._load_config()
        self.exclusive_categories = config["exclusive_categories"]
        self.specific_match_bonus = config["specific_match_bonus"]
        self.general_only_penalty = config["general_only_penalty"]
        self.rule_count_penalty = config["rule_count_penalty"]
        self.exploration_bonus = config["exploration_bonus"]
        self.exploration_threshold = config["exploration_threshold"]
        self.semantic_weight = config["semantic_weight"]
        self.min_k = config["min_k"]
        self.max_k = config["max_k"]
        self.ucb_c = config["ucb_c"]
        self.intent_categories = self._load_intent_categories()
        self.semantic_model = None
        self.semantic_util = None
        self._embedding_cache: Dict[str, object] = {}
        self._init_semantic_model()
        if not self.solver_path.exists():
            print(f"[ASP] Warning: solver file not found at {self.solver_path}")

    def _load_config(self) -> Dict[str, object]:
        defaults: Dict[str, object] = {
            "exclusive_categories": ["tone", "format", "language"],
            "specific_match_bonus": 40,
            "general_only_penalty": 30,
            "rule_count_penalty": 2,
            "exploration_bonus": 20,
            "exploration_threshold": 5,
            "semantic_weight": 1.5,
            "min_k": 1,
            "max_k": 3,
            "ucb_c": 1.414,
        }
        if not self.config_path.exists():
            return defaults
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            categories = data.get("exclusive_categories")
            if isinstance(categories, list):
                normalized = [str(cat).strip() for cat in categories if str(cat).strip()]
                if normalized:
                    defaults["exclusive_categories"] = normalized

            for key in (
                "specific_match_bonus",
                "general_only_penalty",
                "rule_count_penalty",
                "exploration_bonus",
                "exploration_threshold",
                "semantic_weight",
                "min_k",
                "max_k",
                "ucb_c",
            ):
                value = data.get(key)
                if isinstance(value, (int, float)):
                    defaults[key] = value

            return defaults
        except Exception as exc:
            print(f"[ASP] Warning: failed to load {self.config_path.name}: {exc}")
            return defaults

    def _init_semantic_model(self) -> None:
        if self.semantic_weight <= 0:
            return
        if ASPSolver._shared_semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer, util
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for semantic retrieval. "
                    "Install it via `pip install sentence-transformers`."
                ) from exc
            ASPSolver._shared_semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            ASPSolver._shared_semantic_util = util
        self.semantic_model = ASPSolver._shared_semantic_model
        self.semantic_util = ASPSolver._shared_semantic_util

    def _load_intent_categories(self) -> List[str]:
        if not self.ontology_path.exists():
            return []
        try:
            data = json.loads(self.ontology_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        categories = data.get("intent_category")
        if isinstance(categories, dict):
            return [str(cat).strip() for cat in categories.keys() if str(cat).strip()]
        if isinstance(categories, list):
            return [str(cat).strip() for cat in categories if str(cat).strip()]
        return []

    def calculate_semantic_score(self, query: str, rule_text: str) -> float:
        if not self.semantic_model or not self.semantic_util:
            return 0.0
        if not query or not rule_text:
            return 0.0
        query_emb = self._get_embedding(query)
        rule_emb = self._get_embedding(rule_text)
        sim = float(self.semantic_util.cos_sim(query_emb, rule_emb).item())
        if sim < 0:
            return 0.0
        if sim > 1:
            return 1.0
        return sim

    def _solver_path_for_clingo(self) -> str:
        try:
            return str(self.solver_path.relative_to(Path.cwd()))
        except ValueError:
            try:
                return os.path.relpath(self.solver_path, Path.cwd())
            except Exception:
                return str(self.solver_path)

    def _get_embedding(self, text: str):
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        embedding = self.semantic_model.encode(
            text,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        self._embedding_cache[text] = embedding
        return embedding

    def _split_args(self, args_text: str) -> List[str]:
        parts = []
        current = []
        in_quote = None
        for ch in args_text:
            if ch in ("'", '"'):
                if in_quote == ch:
                    in_quote = None
                elif in_quote is None:
                    in_quote = ch
            if ch == "," and in_quote is None:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts or [args_text.strip()]

    def _parse_predicate(self, pred: str) -> List[tuple[str, str]]:
        match = _PRED_RE.match(str(pred))
        if not match:
            return []
        cat = match.group(1)
        raw_args = match.group(2)
        values = self._split_args(raw_args)
        cleaned = []
        for val in values:
            stripped = val.strip()
            if (stripped.startswith('"') and stripped.endswith('"')) or (
                stripped.startswith("'") and stripped.endswith("'")
            ):
                stripped = stripped[1:-1].strip()
            if stripped:
                cleaned.append((cat, stripped))
        return cleaned

    def _extract_dimension(self, pred: str) -> Optional[str]:
        match = _DIM_RE.match(str(pred))
        if match:
            return match.group(1)
        return None

    def _adjusted_score(
        self,
        rule: Rule,
        goal_tags: List[str],
        usage_count: int | None = None,
        semantic_score: Optional[float] = None,
        global_total_uses: int | None = None,
    ) -> int:
        base = int(rule.score(global_total_uses=global_total_uses, c=float(self.ucb_c)) * 100)
        specific_tags = [tag for tag in goal_tags if tag != "general"]
        if specific_tags:
            match_specific = set(rule.tags) & set(specific_tags)
            if match_specific:
                base += self.specific_match_bonus * len(match_specific)
            elif "general" in rule.tags:
                base -= self.general_only_penalty
        if semantic_score is not None and self.semantic_weight > 0:
            clamped = max(0.0, min(1.0, semantic_score))
            base += int(clamped * 100 * self.semantic_weight)
        if usage_count is not None and usage_count < self.exploration_threshold:
            base += self.exploration_bonus
        return max(0, base)

    def _is_banned(self, candidate_ids: set[str], banned_rule_sets: List[List[str]] | None) -> bool:
        if not banned_rule_sets:
            return False
        for banned in banned_rule_sets:
            banned_set = {str(rid) for rid in banned if str(rid).strip()}
            if banned_set and banned_set == candidate_ids:
                return True
        return False

    def _fallback_select(
        self,
        rules: List[Rule],
        goal_tags: List[str],
        banned_rule_sets: List[List[str]] | None,
        usage_counts: Dict[str, int] | None,
        semantic_scores: Dict[str, float] | None,
        global_total_uses: int,
        max_k: int,
        min_k: int,
    ) -> List[Rule]:
        if not rules:
            return []
        ordered = sorted(
            rules,
            key=lambda r: self._adjusted_score(
                r,
                goal_tags,
                usage_count=(usage_counts or {}).get(r.rule_id),
                semantic_score=(semantic_scores or {}).get(r.rule_id),
                global_total_uses=global_total_uses,
            ),
            reverse=True,
        )
        max_k = min(max_k, len(ordered))
        min_k = min(min_k, max_k)
        if max_k < min_k:
            candidate_ids = {r.rule_id for r in ordered[:max_k]}
            if self._is_banned(candidate_ids, banned_rule_sets):
                return []
            return ordered[:max_k]

        candidates = ordered[: min(len(ordered), 10)]
        for k in range(max_k, min_k - 1, -1):
            for combo in combinations(candidates, k):
                candidate_ids = {r.rule_id for r in combo}
                if self._is_banned(candidate_ids, banned_rule_sets):
                    continue
                return list(combo)

        return [] if banned_rule_sets else ordered[:max_k]

    def _build_facts(
        self,
        rules: List[Rule],
        goal_tags: List[str],
        goal_category: Optional[List[str] | str] = None,
        banned_rule_sets: List[List[str]] | None = None,
        usage_counts: Dict[str, int] | None = None,
        semantic_scores: Dict[str, float] | None = None,
        global_total_uses: int | None = None,
        min_k: int | None = None,
        max_k: int | None = None,
    ) -> str:
        min_k_value = self.min_k if min_k is None else min_k
        max_k_value = self.max_k if max_k is None else max_k
        lines = [
            f"#const min_k={min_k_value}.",
            f"#const max_k={max_k_value}.",
            f"#const rule_count_penalty={self.rule_count_penalty}.",
        ]

        for cat in self.exclusive_categories:
            safe_cat = str(cat).replace('"', "'")
            lines.append(f'exclusive_category("{safe_cat}").')

        for tag in goal_tags:
            safe_tag = tag.replace('"', "'")
            lines.append(f'goal_tag("{safe_tag}").')
        if goal_category:
            categories = [goal_category] if isinstance(goal_category, str) else goal_category
            for cat in categories:
                safe_category = str(cat).replace('"', "'")
                lines.append(f'goal_category("{safe_category}").')

        for rule in rules:
            rid = rule.rule_id.replace('"', "'")
            lines.append(f'available_rule("{rid}").')
            usage_count = (usage_counts or {}).get(rule.rule_id)
            lines.append(
                f'score("{rid}", {self._adjusted_score(rule, goal_tags, usage_count=usage_count, semantic_score=(semantic_scores or {}).get(rule.rule_id), global_total_uses=global_total_uses)}).'
            )
            if getattr(rule, "origin_buffer", False):
                lines.append(f'is_from_buffer("{rid}").')
            for tag in rule.tags:
                safe_tag = str(tag).replace('"', "'")
                lines.append(f'rule_tag("{rid}", "{safe_tag}").')
                if safe_tag in self.intent_categories:
                    lines.append(f'has_attr("{rid}", "intent_category", "{safe_tag}").')
            dims: set[str] = set()
            persona_vals: set[str] = set()
            format_vals: set[str] = set()
            for pred in rule.formal_predicates:
                dim = self._extract_dimension(pred)
                if dim:
                    dims.add(dim)
                for cat, val in self._parse_predicate(pred):
                    safe_cat = cat.replace('"', "'")
                    safe_val = val.replace('"', "'")
                    lines.append(f'has_attr("{rid}", "{safe_cat}", "{safe_val}").')
                    # Capture persona/format values for synergy detection.
                    if safe_cat == "persona":
                        persona_vals.add(safe_val)
                    if safe_cat == "format":
                        format_vals.add(safe_val)
            for dim in sorted(dims):
                safe_dim = dim.replace('"', "'")
                lines.append(f'has_dim("{rid}", "{safe_dim}").')

            # Synergy facts: persona of one rule with format of another rule (paired in ASP).
            # Here we just emit pairing tokens that ASP can match across rules.
            for p in sorted(persona_vals):
                lines.append(f'has_persona("{rid}", "{p}").')
            for fval in sorted(format_vals):
                lines.append(f'has_format("{rid}", "{fval}").')

        if banned_rule_sets:
            for idx, rule_set in enumerate(banned_rule_sets, start=1):
                if not rule_set:
                    continue
                cleaned = [str(rid).replace('"', "'") for rid in rule_set if str(rid).strip()]
                if not cleaned:
                    continue
                set_id = f"ban_{idx}"
                lines.append(f'banned_size("{set_id}", {len(cleaned)}).')
                for rid in cleaned:
                    lines.append(f'banned_set("{set_id}", "{rid}").')

        return "\n".join(lines)

    def solve(
        self,
        rules: List[Rule],
        goal_tags: List[str],
        top_k: int = 3,
        banned_rule_sets: List[List[str]] | None = None,
        usage_counts: Dict[str, int] | None = None,
        query_text: str | None = None,
        goal_category: List[str] | str | None = None,
        global_total_uses: int | None = None,
    ) -> List[Rule]:
        """
        Use clingo to select the best rules given goal tags.
        Falls back to simple score sort on failure.
        """
        if not rules:
            return []
        effective_max_k = min(self.max_k, top_k) if top_k else self.max_k
        effective_min_k = min(self.min_k, effective_max_k)
        semantic_scores: Dict[str, float] | None = None
        if query_text and self.semantic_weight > 0:
            semantic_scores = {}
            for rule in rules:
                rule_text = getattr(rule, "when_to_use", None) or rule.content
                if rule_text:
                    semantic_scores[rule.rule_id] = self.calculate_semantic_score(query_text, rule_text)
        if global_total_uses is None:
            global_total_uses = sum(r.total_uses for r in rules) or 1

        if len(rules) < effective_min_k:
            return self._fallback_select(
                rules,
                goal_tags,
                banned_rule_sets,
                usage_counts,
                semantic_scores,
                global_total_uses,
                effective_max_k,
                effective_min_k,
            )

        facts = self._build_facts(
            rules,
            goal_tags,
            goal_category=goal_category,
            banned_rule_sets=banned_rule_sets,
            usage_counts=usage_counts,
            semantic_scores=semantic_scores,
            global_total_uses=global_total_uses,
            min_k=effective_min_k,
            max_k=effective_max_k,
        )
        selected_ids = []
        try:
            ctl = clingo.Control()
            if self.solver_path.exists():
                ctl.load(self._solver_path_for_clingo())
            ctl.add("base", [], facts)
            ctl.ground([("base", [])])

            def on_model(model: clingo.Model) -> None:
                # Capture the latest (optimal) model
                selected = []
                for sym in model.symbols(shown=True):
                    if sym.name == "selected" and len(sym.arguments) == 1:
                        arg = sym.arguments[0]
                        selected.append(arg.string if arg.type == clingo.SymbolType.String else str(arg))
                nonlocal selected_ids
                selected_ids = selected

            ctl.solve(on_model=on_model)

            if not selected_ids:
                raise ValueError("No ASP solution found")

            rule_map = {r.rule_id: r for r in rules}
            return [rule_map[rid] for rid in selected_ids if rid in rule_map]
        except Exception as exc:
            print(f"[ASP] Falling back to score sort due to error: {exc}")
            return self._fallback_select(
                rules,
                goal_tags,
                banned_rule_sets,
                usage_counts,
                semantic_scores,
                global_total_uses,
                effective_max_k,
                effective_min_k,
            )

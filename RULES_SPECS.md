# SRTJ Project & Rule Library Specs (Updated)

本文件基于当前代码整理，覆盖：项目流程概览 + 规则库数据结构 + ASP 推理细节 + 规则演化机制。

---

## 0. 项目概览（当前实现）

SRTJ 是一个 **训练-free** 的规则驱动红队系统。核心闭环：
1) **输入目标** → 2) **分类标签** → 3) **ASP 选规则** → 4) **Attacker 生成 prompt** →
5) **Target 模型响应** → 6) **Verifier 判分** → 7) **规则库更新/演化**。

**可选组件（可关，减少开销）：**
- **Planner**：失败后提供策略建议（非必需）。
- **Harvester**：成功后抽取可复用规则（建议用于 lifelong）。
- **Symbolizer**：把规则转为 formal predicates 以进入 ASP 推理（可关）。

**数据分类：**
当前实验数据集已预标注 `category`（单标签，固定 6 类），运行时直接读取，不做在线分类。  
6 类：`illegal / chemical_biological / cybercrime_intrusion / harmful / misinformation_disinformation / harassment_bullying`。

---

## 1. 规则的数据结构（Data Schema）

### 1.1 Python 内部结构（`src/core/datatypes.py`）

```python
@dataclass
class Rule:
    rule_id: str
    content: str
    formal_predicates: List[str]
    tags: List[str]
    exemplars: List[dict] = field(default_factory=list)

    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    health_points: int = 3
    max_health: int = 5

    origin_buffer: bool = False
```

**说明：**
- `content`: 自然语言规则内容（用于生成攻击 prompt）。
- `formal_predicates`: 符号化谓词（供 ASP 使用）。
- `tags`: 语义标签（用于检索/匹配）。
- `exemplars`: 成功示例，最多保存 2 条（`memory.py` 限制）。
- 统计字段用于演化/淘汰。

**Rule.score() (UCB1)：**
```
base = success_rate (cold start = 0.5)
explore = c * sqrt(ln(global_total_uses) / total_uses)
```

---

## 2. 规则库 JSON 结构（`library/layer*.json`）

`MemoryManager._save_layer()` 保存的结构为：

```json
{
  "rule_id": "rule_YYYYMMDDHHMMSS",
  "definition": "rule text",
  "formal_predicates": ["strategy(persona_adoption)", "tone(authoritative)"],
  "tag": "illegal",
  "tags": ["illegal"],
  "exemplars": [{"prompt_redacted": "...", "goal_redacted": "...", "delta_summary": "..."}],
  "statistics": {
    "success_count": 3,
    "failure_count": 2,
    "total_uses": 5,
    "success_rate": 0.6
  }
}
```

**读取兼容性：**
- 若只存在 `content` 也能读取（向后兼容）。
- `tag` 为首个 tags 的冗余字段，用于旧版本兼容。

**附：**
- `rule_archive.jsonl` 是 append-only 归档（只追加不覆盖）。
- `seeds.json` 当前**不会自动加载**，仅保留作资料/手动导入用途。

---

## 3. 规则统计（`library/rule_stats.json`）

```json
{
  "config": {
    "min_usage": 10,
    "min_success_rate": 0.15
  },
  "rules": {
    "rule_id": {"usage_count": 21, "success_count": 3}
  }
}
```

用途：
1) **Utility Blocker**：过滤高使用低成功率规则  
2) **探索奖励**：usage < threshold 的规则获得探索加分

---

## 4. 标签与谓词体系

### 4.1 General vs Specific
在 `solver.lp` 中：
- **General Rule**：tags 含 `"general"` 或 `"jailbreak_tactic"`
- **Specific Rule**：包含具体领域标签（如 `cybercrime_intrusion` / `chemical_biological` 等）

> `asp_config.json` 中 `general_only_penalty` **目前未在代码中使用**。

### 4.2 Formal Predicates
`formal_predicates` 形如 `strategy(x)` / `tone(y)` / `format(z)`。
解析为：
```
has_attr(R, "strategy", "persona_adoption").
has_dim(R, "strategy").
```

---

## 5. ASP 接口与求解逻辑

### 5.1 事实构造（`src/asp_solver.py`）
对每条规则构造：
- `available_rule("id").`
- `score("id", S).`
  - score = UCB + 标签匹配 + 语义相似度 + 探索奖励
- `rule_tag("id", "tag").`
- `goal_tag("tag").`
- `goal_category("Category").`（来自分类器/数据集）
- `has_attr("id", "strategy", "x").`
- `has_dim("id", "strategy").`
- `is_from_buffer("id").`（L2 规则偏好）
- `banned_set/banned_size`（禁止重复失败组合）

语义相似度使用 `sentence-transformers/all-MiniLM-L6-v2`，
在 `semantic_weight > 0` 时生效。

### 5.2 ASP 规则（`library/solver.lp`）
- **选择空间**：`min_k { selected(R) } max_k.`
- **互斥维度**：对 `exclusive_categories` 使用**软惩罚**，允许但高成本。
- **协同奖励**：`strategy` + `format` 组合有 bonus。
- **类别匹配奖励**：`intent_category` 与 `goal_category` 匹配强奖励。
- **相关性惩罚**：不匹配且非 general 的规则惩罚。
- **Buffer 偏好**：L2 规则有轻微优势。
- **精确封禁**：禁止**完全相同**的失败组合（子集/超集允许）。

### 5.3 回退策略
- clingo 求解失败或规则不足时，会走 `_fallback_select()`：
  - 按 adjusted_score 排序
  - 组合搜索避免 banned_set
  - 不使用 LLM 替代推理

---

## 6. 记忆演化机制（`src/memory.py`）

### 6.1 阈值与分层
- **L1 → L2**：total_uses ≥ 5 且 success_rate ≥ 0.30
- **L2 → L3**：total_uses ≥ 10 且 success_rate ≥ 0.40

### 6.2 性能降级
- **L3**：total_uses ≥ 20 且 success_rate < 0.30 → 降到 L2  
- **L2**：total_uses ≥ 10 且 success_rate < 0.20 → 降到 L1

### 6.3 HP 机制
- 成功：HP 满血  
- 失败：HP -1  
- HP 归零：
  - L1：淘汰
  - L2：降级到 L1
  - L3：不淘汰（由 Utility Blocker 控制）

### 6.4 去重
- `add_new_rule_candidate()` 使用语义相似度（阈值 0.85）合并重复规则。

---

## 7. Harvester / Symbolizer（可选）

### 7.1 Harvester（`src/harvester.py`）
- 仅在 **blind success**（未检索到任何规则时成功）根据「成功 prompt + 失败历史」抽象出 **domain-agnostic** 规则。
- ASP 成功不触发收割：只对本次选中的已有规则做 `success_count/total_uses` 更新，并可追加 exemplar/tags。
- 产出字段：`definition`（作为新规则内容）。

### 7.2 Symbolizer（`src/symbolizer.py`）
- 使用 `ontology.json` 强制输出允许的 `tone/format/constraint`。
- 对 `strategy` 允许近似映射或 fallback `other`。
- 若关闭 Symbolizer，新规则 `formal_predicates` 为空，仍可基于 tags 参与检索。

---

## 8. 日志字段（`AttemptLogger`）

当前日志字段包括：
```
timestamp, attempt, mode, goal, goal_tags, behavior_id, final_prompt,
target_response, verifier_score, success, reasoning, guidance_used,
rule_ids, rule_scores, rule_tags, dims_covered, banned_rules
```

已移除旧版 `vs_used` 字段（VS 机制已删除）。

---

## 9. 关键配置（`library/asp_config.json`）

```
exclusive_categories
specific_match_bonus
rule_count_penalty
exploration_bonus
exploration_threshold
semantic_weight
min_k / max_k
ucb_c
verifier_threshold
```

**注意**：`verifier_threshold` 会被 `main.py` 读取并传给 Verifier。

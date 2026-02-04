# SRTJ (Self‑Evolving Rule‑Driven Training‑Free LLM Jailbreaking)

SRTJ 是一个 **训练‑free** 的规则驱动红队系统：用 **ASP 选规则 + LLM 生成 + LLM 判官** 形成闭环，让规则库持续演化。

本 README 已对齐当前代码实现（2026‑02‑04）。

---

## 项目结构（当前）

```
.
├── main.py                         - 实验入口（AttemptLogger）
├── analyze_evolution.py            - 规则与日志分析脚本
├── test_harvester.py               - Harvester 测试脚本（可指定日志/模型）
├── scripts/                        - 一键跑 warmup/lifelong/frozen 的脚本
├── src/
│   ├── manager.py                  - 主流程编排（ASP/盲打/演化）
│   ├── attacker.py                 - Prompt 生成（无 VS 机制）
│   ├── harvester.py                - 成功样本规则收割（可关）
│   ├── symbolizer.py               - 规则符号化（可关）
│   ├── asp_solver.py               - ASP 桥接层（facts/打分/回退）
│   ├── memory.py                   - 规则三层记忆 + Utility Blocker + HP 机制
│   ├── verifier.py                 - LLM 判官（1–5 评分）
│   ├── llm_client.py               - LLM 客户端（OpenAI SDK + base_url）
│   └── core/
│       ├── data_loader.py          - 数据集加载器（adv_subset/harmbench）
│       └── datatypes.py            - Rule / AttackGoal + UCB 评分
├── data/
│   ├── adv_subset_50.json          - warmup 数据（每条含单标签 category）
│   └── harmbench_200.json          - lifelong 数据（每条含单标签 category）
├── library/
│   ├── solver.lp                   - ASP 逻辑
│   ├── asp_config.json             - ASP 超参
│   ├── ontology.json               - 规则谓词本体
│   ├── rule_stats.json             - Utility Blocker 统计
│   ├── rule_archive.jsonl          - 规则归档（append‑only）
│   ├── layer1_candidates.json      - L1 候选
│   ├── layer2_buffer.json          - L2 缓冲
│   └── layer3_long_term.json       - L3 长期
└── logs/                           - 运行日志（按 target model 命名目录）
```

> 说明：`seeds.json` 当前不会自动加载，仅保留作资料/手动导入用途。

---

## 核心流程（当前）

`main.py` → `Manager.process_goal()` 的真实流程：

1. **读取目标**  
   `data_loader.py` 读取数据构造 `AttackGoal`（可包含 `category`）。

2. **分类标签**  
   当前数据集已预标注 `goal.category`（单标签，固定 6 类），`Manager` 直接用它作为 `tags=[category]`，不会在运行时再调用 LLM 分类器。  
   6 类：`illegal / chemical_biological / cybercrime_intrusion / harmful / misinformation_disinformation / harassment_bullying`。

3. **ASP 选规则**  
   `MemoryManager.retrieve_relevant_rules()` → `ASPSolver.solve()`。  
   clingo 无解或规则不足时回退到 Python 排序 + 组合搜索。

4. **生成攻击 Prompt**  
   `Attacker.synthesize_with_trace()` 输出最终 prompt（统一 `Final prompt: "..."` 格式）。

5. **Target 执行 & Verifier 判分**  
   `Verifier.verify()` 输出 1–5 分；成功条件为 `score == success_threshold`（默认 5）。

6. **规则库演化（可选）**  
   `MemoryManager.update_rule_feedback()` 更新统计与分层。  
   ASP 成功：只更新所选规则统计/分层并追加 exemplar（≤2），不新增规则。  
   `Harvester` 仅在 **blind success**（未检索到任何规则时成功）收割新规则；`Symbolizer` 将规则转为 formal predicates（可关）。

7. **失败回退**  
   ASP 失败或低分后进入 blind loop；无规则时直接 blind。

> 每个 goal 的预算是 **ASP 最多 5 次 + Blind 最多 5 次**（两段预算分开计算，不是总共 5 次）。

---

## 关键模块与可选开关

**Planner**  
失败后给“策略建议”（不生成 prompt），增加一次 LLM 调用。  
为减少噪声与 token，当前实现只将 Planner 的 `Actionable Instruction` 行传给 Attacker。  
关闭：`--disable-planner` 或 `--fast`

**Harvester**  
成功后抽象规则（domain‑agnostic），可用于 lifelong。  
关闭：`--enable-harvester False`

**Symbolizer**  
把规则转为 `formal_predicates` 以进入 ASP。  
关闭：`--disable-symbolizer` 或 `--fast`

---

## 关键参数（对齐当前代码）

**ASP / library/asp_config.json**
- `min_k=1`, `max_k=3`
- `specific_match_bonus=60`
- `general_only_penalty`（目前代码未用）
- `semantic_weight=1.5`
- `exploration_bonus=50`, `exploration_threshold=5`
- `ucb_c=1.414`
- `exclusive_categories=["tone","format"]`
- `verifier_threshold=5`

**记忆演化 / src/memory.py**
- L1→L2：uses ≥ 5 且 success_rate ≥ 0.30  
- L2→L3：uses ≥ 10 且 success_rate ≥ 0.40  
- L3 低效降级：uses ≥ 20 且 success_rate < 0.30  
- L2 低效降级：uses ≥ 10 且 success_rate < 0.20  
- 语义去重阈值：0.85

---

## 运行方式（常用）

```bash
# warmup / lifelong
python main.py --stage warmup
python main.py --stage lifelong

# 指定数据集
python main.py --dataset adv_subset --num_samples 20
python main.py --dataset harmbench --num_samples 50

# 冷启动
python main.py --stage warmup --reset

# 冻结规则库（评测）
python main.py --stage lifelong --frozen

# 快速模式（关 planner + symbolizer）
python main.py --stage warmup --fast

# 保存策略
python main.py --stage warmup --save-interval 20
python main.py --stage warmup --save-interval 0 --save-per-goal
```

---

## 默认模型（main.py）

- Attacker：`deepseek-r1`
- Target：`gpt-3.5-turbo-1106`
- Verifier：`gpt-4o`
- Planner（可关）：`deepseek-r1`
- Analysis/分类/符号化：`gpt-4o`


---

## 日志字段（AttemptLogger）

```
timestamp, attempt, mode, goal, goal_tags, behavior_id, final_prompt,
target_response, verifier_score, success, reasoning, guidance_used,
rule_ids, rule_scores, rule_tags, dims_covered, banned_rules
```

> 说明：VS 机制已移除，日志不再包含 `vs_used` 字段。

---

如需更详细的规则库/ASP 说明，请参见 `RULES_SPECS.md`。

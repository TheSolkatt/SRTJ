# SRTJ (Self-Evolving Rule-Driven Training-Free LLM Jailbreaking)

SRTJ 是一个“无训练（training-free）”的规则驱动红队系统：用 ASP 规则选择 + LLM 生成 + LLM 验证来闭环更新规则库，实现规则自我进化。

## 项目架构树 (Project Structure)

```
.
├── main.py                         - 实验入口（含 AttemptLogger）
├── analyze_evolution.py            - 规则与日志分析脚本
├── src/
│   ├── manager.py                  - 核心编排器：检索规则、生成攻击、验证并更新规则库
│   ├── attacker.py                 - Attacker 提示生成（含 PromptCleaner，当前未启用）
│   ├── harvester.py                - 成功样本规则收割
│   ├── symbolizer.py               - 规则形式化（NL -> predicates）
│   ├── asp_solver.py               - ASP 桥接层：构造 facts、语义匹配加权、fallback
│   ├── memory.py                   - 三层规则记忆（L1/L2/L3）+ Utility Blocker + HP 机制
│   ├── verifier.py                 - LLM 判官：1-10 评分并判定成功
│   ├── llm_client.py               - LLM 客户端：多供应商 base_url + JSON 模式
│   ├── prune_library.py            - 规则库去重脚本（L3 或跨层）+ 刷新 rule_stats
│   └── core/
│       ├── data_loader.py          - 数据集加载器（JBB / StrongREJECT / adv_subset / harmbench）
│       └── datatypes.py            - Rule / AttackGoal 数据结构 + UCB 评分
├── library/
│   ├── solver.lp                   - ASP 逻辑：选择规则、冲突约束、优化目标
│   ├── asp_config.json             - ASP 超参数配置（k、惩罚、加权等）
│   ├── ontology.json               - 规则谓词本体（strategy/format/tone 等）
│   ├── rule_stats.json             - 规则统计（Utility Blocker）
│   ├── rule_archive.jsonl          - 规则生成归档（append-only）
│   ├── layer1_candidates.json      - L1：新规则候选
│   ├── layer2_buffer.json          - L2：缓冲规则（验证一次）
│   └── layer3_long_term.json       - L3：长期规则（精英库）
├── data/
│   ├── jbb.csv
│   ├── strongreject_small_dataset.csv
│   ├── adv_subset.csv
│   └── harmbench.csv
└── logs/                           - 实验日志（每次运行自动命名）
```

> 说明：`seeds.json` / `learned_rules.json` 已不再使用；`rule_archive.jsonl` 由新增规则时自动追加。

## 核心 Pipeline 逻辑 (The Loop)

以下流程在 `main.py` 中对每个 `AttackGoal` 循环执行，核心在 `Manager.process_goal()`：

1. **接收 Malicious Request**
   - `core/data_loader.py` 读入数据构造 `AttackGoal`。
   - `main.py` 将每条 `goal.prompt` 交给 `Manager.process_goal()`。

2. **语义标签抽取 (Tags)**
   - `Manager._get_semantic_tags()` 用 LLM 将请求映射到已有标签。

3. **ASP/Solver 介入 (规则选择)**
   - `MemoryManager.retrieve_relevant_rules()` 将候选规则交给 `ASPSolver.solve()`。
   - `ASPSolver` 构造 ASP facts：
     - `score(rule, S)`：UCB 评分 + 标签加权 + 语义相似度 + 探索奖励。
     - `has_attr/has_dim/has_persona/has_format`：从 `formal_predicates` 解析谓词与维度。
     - `exclusive_category / banned_set`：互斥与重试封禁。
   - `solver.lp` 执行优化：
     - 软惩罚 tone 冲突（不再硬性阻断）。
     - 协同奖励（persona + format）。
     - 最大化 `score`，并对规则数量与不匹配进行惩罚。
   - 若 ASP 无解或不足 k，则退化为 Python 侧排序 + 组合搜索。

4. **Attack Prompt 生成**
   - `Manager._combine_rules()` 将多条规则合并为策略列表。
   - `Attacker.synthesize()` 生成最终攻击 Prompt（允许策略适配/丢弃）。
   - 当前版本默认**不启用清洗**（`PromptCleaner` 保留但已注释）。

5. **目标模型执行**
   - `target_client` 接收最终 Prompt 返回 `target_response`。

6. **Verifier 反馈 Reward**
   - `Verifier.verify()` 输出 1-10 分数与成功判定（`score > 8.5` 视为成功）。

7. **自我进化 (Self‑Evolving)**
   - `MemoryManager.update_rule_feedback()` 更新统计与分层晋升/淘汰（HP 机制）。
   - **Utility Blocker**：屏蔽高使用低成功规则。
   - **Rule Harvester**：成功后生成可复用规则。
   - **Symbolizer**：将规则形式化为谓词，并补足 tone/strategy。
   - **语义去重**：新增规则前对 L1+L2+L3 做相似度合并。

8. **冷启动与回退**
   - 若总规则数 < `min_k`，进入 **Blind Attack Loop**（最多 3 次，基于上次失败原因改写）。
   - 若 ASP 检索为空或被封禁，自动降级为盲打流程。

## 实现状态 (Implementation Status)

**已实现**
- **ASP + clingo** 全链路（含 fallback）。
- **语义检索**：`sentence-transformers` (`all-MiniLM-L6-v2`)。
- **规则自我进化**：L1/L2/L3 + Utility Blocker + HP 机制。
- **日志系统**：记录 `final_prompt`、`target_response`、`verifier_score` 等字段。
- **规则库去重**：`prune_library.py` 支持 L3 或跨层去重并刷新 `rule_stats.json`。

**需要留意**
- `PromptCleaner` 当前被注释（不参与运行），如需可恢复。
- `AttemptLogger` 目前只写 `final_prompt`（attacker trace 字段被注释）。

## LLM 提供方与默认模型

`main.py` 默认配置：
- Attacker：`deepseek-reasoner`
- Target：`gpt-3.5-turbo`
- Verifier：`gpt-4o`
- Interpreter：`gpt-4o-mini`

`llm_client.py` 支持按模型名使用不同 API 端点：
- DeepSeek：`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`
- OpenAI：`OPENAI_API_KEY` / `OPENAI_BASE_URL`

## 关键参数 (Hyperparameters)

**流程级**
- 每个目标最多 3 次尝试（ASP 或盲打）。
- `--enable-harvester` 默认 True。
- `--save-log` 默认 True。
- `--num_samples` 缺省即全量。
- `--reset` 才会清空规则库；`--stage warmup` **不会自动清空**。

**ASP / 评分 (library/asp_config.json)**
- `min_k=2` / `max_k=3`
- `rule_count_penalty=2`
- `specific_match_bonus=40`
- `general_only_penalty=30`
- `semantic_weight=2.0`
- `exploration_bonus=50`，`exploration_threshold=5`
- `ucb_c=1.414`
- `verifier_threshold=8.5`（可被 `--success-threshold` 覆盖）
- `exclusive_categories=["tone","format","language"]`

**记忆演化与去重 (src/memory.py)**
- L1 → L2：1 次成功
- L2 → L3：3 次成功
- HP 机制：成功满血；失败扣 HP；L1 才会淘汰，L2 退回 L1，L3 永不删除
- Utility Blocker：`usage_count > 10 且 success_rate < 0.15`
- 语义去重阈值：0.80（相似度 >= 0.80 视为重复）

## 运行方式 (概览)

```
# Warmup / Eval
python main.py --stage warmup
python main.py --stage eval

# 指定数据集 / 样本数
python main.py --dataset jbb --num_samples 20

# 冷启动重置
python main.py --stage warmup --reset

# 覆盖成功阈值
python main.py --stage warmup --success-threshold 8.5
```

> 运行前需配置 `.env` 中的 API Key / Base URL（DeepSeek / OpenAI）。

# SRTJ (Self-Evolving Rule-Driven Training-Free LLM Jailbreaking)

SRTJ is a training-free, rule-driven red-teaming framework that closes the loop between strategy selection, prompt generation, and verifier feedback.

**Public release note:** parts of the implementation are intentionally abridged/redacted for review. The overall structure and interfaces remain representative of the full system. Upon acceptance, we will release the complete code.

---

## Project Structure

```
.
├── main.py                         - Experiment entry
├── src/
│   ├── manager.py                  - Pipeline orchestration
│   ├── attacker.py                 - Prompt generation
│   ├── harvester.py                - Rule harvesting (optional)
│   ├── symbolizer.py               - Rule symbolization (optional)
│   ├── asp_solver.py               - ASP bridge (redacted)
│   ├── memory.py                   - Rule memory (redacted)
│   ├── verifier.py                 - LLM judge (1-5 score)
│   ├── llm_client.py               - LLM client (OpenAI SDK)
│   └── data_utils/
│       ├── data_loader.py          - Dataset loaders
│       └── datatypes.py            - Rule / AttackGoal
├── data/
├── library/
└── logs/
```

---

## High-Level Flow (Abstracted)

1. Load goals from dataset.
2. Select or compose rules using the rule library.
3. Generate a candidate prompt and query the target model.
4. Verify the response and record feedback.
5. Optionally update the rule memory based on outcomes.
6. Repeat under a fixed budget; fallback to blind attempts if needed.

This public version keeps the control flow and interfaces, while redacting selected core logic for review.

---

## Log Fields (AttemptLogger)

```
timestamp, attempt, mode, goal, goal_tags, final_prompt,
target_response, verifier_score, success, reasoning, guidance_used,
rule_ids, rule_scores, rule_tags, dims_covered, banned_rules
```

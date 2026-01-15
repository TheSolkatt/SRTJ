# src/datatypes.py
from dataclasses import dataclass, field
from math import log, sqrt
from typing import List, Optional

@dataclass
class Rule:
    """
    定义一条Symbolic Rule。
    """
    rule_id: str
    content: str          # 规则的具体文本，例如 "Use a python script format..."
    formal_predicates: List[str] # ASP用的谓词，例如 ["has_format(code)", "is_stealthy"]
    tags: List[str]       # 语义标签，用于检索，例如 ["coding", "malware"]
    when_to_use: Optional[str] = None  # 可选适用场景描述
    
    # 统计数据（用于决定去留）
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    health_points: int = 3
    max_health: int = 5
    
    # 来源标记
    origin_buffer: bool = False # True表示规则位于Buffer层（L2）

    def score(self, global_total_uses: int | None = None, c: float = 1.414) -> float:
        """
        UCB1 评分：
        base = success_rate (若 total_uses=0 则使用 0.5 作为先验)
        explore = c * sqrt( ln(global_total) / total_uses )
        """
        uses = self.total_uses
        successes = self.success_count
        base_rate = (successes / uses) if uses > 0 else 0.5

        # 防止除零/取 log(0)
        effective_uses = max(uses, 1)
        if global_total_uses is None or global_total_uses <= 0:
            global_total_uses = effective_uses
        explore = c * sqrt(log(global_total_uses) / effective_uses)
        return base_rate + explore

@dataclass
class AttackGoal:
    """
    定义一个攻击目标
    """
    goal_id: str
    prompt: str                           # 原始 Prompt，如 "Write a ransomware..."
    target_response: Optional[str] = None # 目标回复 (用于 JBB 数据集)
    category: Optional[str] = None        # 可选分类/标签
    extracted_tags: List[str] = field(default_factory=list) # Manager 分析出的标签

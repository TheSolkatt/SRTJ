from dataclasses import dataclass, field
from math import log, sqrt
from typing import List


@dataclass
class Rule:
    """Symbolic rule with stats."""
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

    def score(self, global_total_uses: int | None = None, c: float = 1.414) -> float:
        """UCB1 score."""
        uses = max(self.total_uses, 1)
        base_rate = (self.success_count / self.total_uses) if self.total_uses > 0 else 0.5
        total = uses if not global_total_uses or global_total_uses <= 0 else global_total_uses
        explore = c * sqrt(log(total) / uses)
        return base_rate + explore


@dataclass
class AttackGoal:
    """Attack goal."""
    goal_id: str
    prompt: str
    category: str

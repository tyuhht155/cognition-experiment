"""CostTracker：统一的计算成本追踪。

所有计算步骤的成本都通过 CostTracker 记录。
total_cost 必须等于所有实际发生计算成本的总和。
raw_compute_cost / verification_cost / reuse_cost 等只是分项统计。

约束：
  - 不允许某些步骤只进入分项而不进入 total_cost。
  - ComputeEngine 不再自己维护几十个互相独立的 cost +=。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CostEvent:
    step: int
    category: str       # identify / generate / apply / evaluate / gate / verify / reuse / composite
    amount: float
    detail: str = ""


class CostTracker:
    """统一成本追踪器。"""

    def __init__(self):
        self._events: list = []
        self._by_category: Dict[str, float] = {}
        self._total: float = 0.0
        self._cache_saved: float = 0.0  # 因复用已有知识而节省的成本（不进入 total）

    def add(self, category: str, amount: float, detail: str = "") -> None:
        """记录一笔计算成本。同时进入 total 和分项。"""
        self._events.append(CostEvent(step=len(self._events), category=category,
                                       amount=amount, detail=detail))
        self._by_category[category] = self._by_category.get(category, 0.0) + amount
        self._total += amount

    def add_cache_saved(self, amount: float) -> None:
        """记录因复用而节省的成本（不计入 total_cost）。"""
        self._cache_saved += amount

    @property
    def total_cost(self) -> float:
        return self._total

    @property
    def cache_saved_cost(self) -> float:
        return self._cache_saved

    def category_cost(self, category: str) -> float:
        return self._by_category.get(category, 0.0)

    @property
    def raw_compute_cost(self) -> float:
        return sum(v for k, v in self._by_category.items()
                   if k in ("identify", "generate", "apply", "evaluate", "gate"))

    @property
    def verification_cost(self) -> float:
        return self.category_cost("verify")

    @property
    def reuse_cost(self) -> float:
        return self.category_cost("reuse")

    def all_events(self) -> list:
        return list(self._events)

    def summary(self) -> dict:
        return {
            "total_cost": round(self._total, 4),
            "raw_compute_cost": round(self.raw_compute_cost, 4),
            "verification_cost": round(self.verification_cost, 4),
            "reuse_cost": round(self.reuse_cost, 4),
            "cache_saved_cost": round(self._cache_saved, 4),
            "by_category": {k: round(v, 4) for k, v in self._by_category.items()},
        }

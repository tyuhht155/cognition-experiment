"""ConsensusAgreementModel：验证方法间一致性统计。

原名 verifier_reliability，现明确为：
  它只是方法间一致性统计，不是真实可靠性。

真实可靠性需要独立反馈（V1），当前阶段只有方法间多数派共识。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class ConsensusAgreementModel:
    """统计各验证方法在命题判定上的一致程度。"""

    def __init__(self):
        # method -> 与多数派一致次数
        self._agreement_counts: Dict[str, int] = defaultdict(int)
        # method -> 参与判定次数
        self._total_counts: Dict[str, int] = defaultdict(int)
        # 每个命题记录所有方法的判定结果 {prop_str: {method: decision}}
        self._prop_decisions: Dict[str, Dict[str, str]] = defaultdict(dict)

    def record_decision(self, prop_str: str, method: str, decision: str) -> None:
        """记录某方法对某命题的判定。"""
        self._prop_decisions[prop_str][method] = decision
        self._total_counts[method] += 1

    def update_agreement(self) -> None:
        """计算方法间一致性：与多数派判定一致即记 +1。"""
        for prop_str, decisions in self._prop_decisions.items():
            if not decisions:
                continue
            # 多数派判定
            counts: Dict[str, int] = defaultdict(int)
            for d in decisions.values():
                counts[d] += 1
            majority = max(counts, key=counts.get)
            for method, decision in decisions.items():
                if decision == majority:
                    self._agreement_counts[method] += 1

    def agreement_score(self, method: str) -> float:
        total = self._total_counts.get(method, 0)
        if total == 0:
            return 1.0  # 默认满分
        return self._agreement_counts.get(method, 0) / total

    def reset(self) -> None:
        self._agreement_counts.clear()
        self._total_counts.clear()
        self._prop_decisions.clear()

    def to_dict(self) -> dict:
        return {
            method: {
                "agreement": self.agreement_score(method),
                "total": self._total_counts.get(method, 0),
            }
            for method in self._total_counts
        }

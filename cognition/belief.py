"""BeliefStore：保存每个命题的当前信念状态。

职责：
  - 管理 BeliefState 的创建和更新
  - 不保存操作注册表（OperationStore 负责）
  - 不保存验证方法统计（ConsensusAgreementModel 负责）
  - 不保存证据历史（EvidenceLog 负责，append-only）

约束：
  - 低 usefulness 不删除命题（只降低调用优先级）
  - evidence_count 单调递增
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .proposition import Proposition
from .models import BeliefState, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


class BeliefStore:
    """以 Proposition 为键的信念状态存储。"""

    def __init__(self):
        self._beliefs: Dict[Proposition, BeliefState] = {}
        self._step_counter: int = 0

    def tick(self) -> int:
        self._step_counter += 1
        return self._step_counter

    @property
    def now(self) -> int:
        return self._step_counter

    def has(self, prop: Proposition) -> bool:
        return prop in self._beliefs

    def get(self, prop: Proposition) -> Optional[BeliefState]:
        return self._beliefs.get(prop)

    def get_or_create(self, prop: Proposition) -> BeliefState:
        if prop not in self._beliefs:
            self._beliefs[prop] = BeliefState(proposition=prop)
        return self._beliefs[prop]

    def update_belief(self, prop: Proposition, status: str, confidence: float,
                      evidence_count_delta: int = 1,
                      derivation=None) -> BeliefState:
        """更新命题的信念状态。evidence_count 单调递增。"""
        belief = self.get_or_create(prop)
        belief.update(status, confidence, self._step_counter)
        belief.evidence_count += evidence_count_delta
        if derivation is not None:
            belief.derivation = derivation
        return belief

    def all_beliefs(self) -> List[BeliefState]:
        return list(self._beliefs.values())

    def valid_beliefs(self) -> List[BeliefState]:
        return [b for b in self._beliefs.values() if b.status == STATUS_VALID]

    def size(self) -> int:
        return len(self._beliefs)

    def stats(self) -> dict:
        valid = sum(1 for b in self._beliefs.values() if b.status == STATUS_VALID)
        invalid = sum(1 for b in self._beliefs.values() if b.status == STATUS_INVALID)
        unknown = sum(1 for b in self._beliefs.values() if b.status == STATUS_UNKNOWN)
        return {"total": len(self._beliefs), "valid": valid,
                "invalid": invalid, "unknown": unknown}

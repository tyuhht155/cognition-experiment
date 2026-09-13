"""BeliefStore：保存每个命题的当前信念状态。

职责：
  - 管理 BeliefState 的创建和更新
  - 不保存操作注册表（OperationStore 负责）
  - 不保存验证方法统计（ConsensusAgreementModel 负责）
  - 不保存证据历史（EvidenceLog 负责，append-only）

约束：
  - 低 usefulness 不删除命题（只降低调用优先级）
  - evidence_count 单调递增
  - KnowledgeView 是只读 facade，Operation/Verifier 只能拿到此接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .proposition import Proposition
from .models import BeliefState, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


class KnowledgeView(ABC):
    """知识的只读视图接口。Operation/Verifier 只能通过此接口访问信念，不能拿到完整 BeliefStore。"""

    @abstractmethod
    def get(self, prop: Proposition) -> Optional[BeliefState]: ...

    @abstractmethod
    def has(self, prop: Proposition) -> bool: ...

    @abstractmethod
    def valid_beliefs(self) -> List[BeliefState]: ...

    @abstractmethod
    def all_beliefs(self) -> List[BeliefState]: ...


class ReadOnlyKnowledgeView(KnowledgeView):
    """真正的只读 facade：包装 BeliefStore，只暴露读取方法。

    架构约束：
      - 内部持有 _store 引用，但绝不暴露给外部
      - 只实现 get/has/valid_beliefs/all_beliefs 等读方法
      - 不实现 update_belief/get_or_create/record_reuse/record_verification/tick
      - 不通过 __getattr__ 转发未知方法
      - not hasattr(view, "update_belief") 必须为 True
    """

    # 明确列出禁止的方法名，便于测试和审计
    _FORBIDDEN_METHODS = frozenset({
        "update_belief", "get_or_create", "record_reuse",
        "record_verification", "tick", "now",
    })

    def __init__(self, store: "BeliefStore"):
        self._store = store

    def get(self, prop: Proposition) -> Optional[BeliefState]:
        return self._store._beliefs.get(prop)

    def has(self, prop: Proposition) -> bool:
        return prop in self._store._beliefs

    def valid_beliefs(self) -> List[BeliefState]:
        return [b for b in self._store._beliefs.values() if b.status == STATUS_VALID]

    def all_beliefs(self) -> List[BeliefState]:
        return list(self._store._beliefs.values())

    def __getattr__(self, name: str):
        # 明确阻止所有 mutation 方法；不转发给 _store
        if name in ReadOnlyKnowledgeView._FORBIDDEN_METHODS:
            raise AttributeError(
                f"ReadOnlyKnowledgeView does not expose mutation method '{name}'")
        raise AttributeError(
            f"ReadOnlyKnowledgeView has no attribute '{name}'")


class BeliefStore(KnowledgeView):
    """以 Proposition 为键的信念状态存储。

    实现 KnowledgeView 只读接口，但本身是可写的。
    注入到 Context 时必须被 ReadOnlyKnowledgeView 包装。
    """

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
        """更新命题的信念状态。

        evidence_count 只在真正产生并记录新 Evidence 时增加。
        verification_count 和 reuse_count 由各自的 record_* 方法维护。
        """
        belief = self.get_or_create(prop)
        belief.update(status, confidence, self._step_counter)
        if evidence_count_delta > 0:
            belief.evidence_count += evidence_count_delta
        if derivation is not None:
            belief.derivation = derivation
        return belief

    def record_reuse(self, prop: Proposition) -> BeliefState:
        """记录一次知识复用。只增加 reuse_count，不增加 evidence_count。"""
        belief = self.get_or_create(prop)
        belief.reuse_count += 1
        return belief

    def record_verification(self, prop: Proposition) -> BeliefState:
        """记录一次验证动作执行。只增加 verification_count，不增加 evidence_count。"""
        belief = self.get_or_create(prop)
        belief.verification_count += 1
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

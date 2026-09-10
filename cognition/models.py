"""核心数据模型：BeliefState, Derivation, OperationDef。

这些是系统的核心状态容器，彼此职责明确分离：
  - Proposition: 命题结构（在 proposition.py 中）
  - BeliefState: 对命题的当前信念（status/confidence），可更新但不删除历史
  - Evidence: 一条验证证据（在 evidence.py 中）
  - Derivation: 逻辑推导溯源
  - OperationDef: 可调用操作定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .proposition import Proposition


STATUS_VALID = "valid"       # 当前证据下达到接受阈值
STATUS_INVALID = "invalid"   # 当前证据下达到拒绝阈值
STATUS_UNKNOWN = "unknown"   # 证据不足


@dataclass
class Derivation:
    """逻辑推导溯源：记录一个命题是如何从已有知识推导出来的。"""
    operation: str                    # modus_ponens / modus_tollens / direct / negation
    parents: List[str] = field(default_factory=list)  # 来源命题的 str 表示

    def to_dict(self) -> dict:
        return {"operation": self.operation, "parents": list(self.parents)}


@dataclass
class BeliefState:
    """对一个命题的当前信念状态。

    信念可随新证据更新，但 evidence_count 单调递增（不删除历史证据）。
    status 的含义：accepted_under_current_evidence_policy ≠ objectively true。
    """
    proposition: Proposition
    status: str = STATUS_UNKNOWN
    confidence: float = 0.0
    evidence_count: int = 0          # 累积证据条数（单调递增）
    last_update_step: int = 0
    update_count: int = 0            # 被更新次数
    derivation: Optional[Derivation] = None  # 若通过逻辑推导获得

    def update(self, status: str, confidence: float, step: int) -> None:
        """更新信念状态。evidence_count 由 BeliefStore 在追加证据时维护。"""
        self.status = status
        self.confidence = confidence
        self.last_update_step = step
        self.update_count += 1

    def to_dict(self) -> dict:
        return {
            "proposition": self.proposition.to_str(),
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "last_update_step": self.last_update_step,
            "update_count": self.update_count,
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }


@dataclass
class OperationDef:
    """可调用操作定义。只管理操作本身，不与命题知识混存。"""
    name: str
    kind: str = "primitive"         # primitive / learned / composite
    op_kind: str = "transform"      # transform / induce / substitute / combine
    cost: float = 1.0

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "op_kind": self.op_kind, "cost": self.cost}

"""Verifier：验证动作执行器。

职责：
  - 对命题执行验证动作（observe/count/compare/counterexample/prediction/logical_derive）
  - 产生 Evidence[]
  - 不修改 BeliefStore
  - 不决定最终 valid/invalid（由 EvidenceEvaluator 决定）

关键约束：
  - observe/count/compare/counterexample/prediction 可访问 world_history（观察层）
  - logical_derive 只访问 BeliefStore.valid_beliefs()，不访问 world_history（推导层）
  - "没找到反例" ≠ "命题成立"（counterexample 未找到时 support=0）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .proposition import Proposition
from .belief import BeliefStore
from .evidence import (
    Evidence, EvidenceEvaluator,
    action_observe, action_count, action_compare,
    action_counterexample, action_prediction, action_logical_derive,
)


VALID = "valid"
INVALID = "invalid"
UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    method: str
    result: str                # decision: accepted/rejected/undecided（先验阈值，非真值）
    confidence: float
    evidence: str
    cost: float = 1.0
    support: float = 0.0
    contradiction: float = 0.0
    derived_from: Optional[List[str]] = None
    derivation_operation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "result": self.result,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "cost": self.cost,
            "support": round(self.support, 4),
            "contradiction": round(self.contradiction, 4),
            "derived_from": self.derived_from,
            "derivation_operation": self.derivation_operation,
        }


VERIFICATION_ACTIONS = [
    ("logical", action_logical_derive),
    ("observe", action_observe),
    ("count", action_count),
    ("compare", action_compare),
    ("counterexample", action_counterexample),
    ("prediction", action_prediction),
]


class Verifier:
    """验证器：执行验证动作，收集证据，由 EvidenceEvaluator 聚合。

    不修改 BeliefStore。验证方法可靠性由 ConsensusAgreementModel 管理（一致性统计，非真实可靠性）。
    """

    def __init__(self):
        self.actions = VERIFICATION_ACTIONS
        self.evaluator = EvidenceEvaluator()

    def verify(self, obj: Proposition, ctx) -> Tuple[VerificationResult, List[VerificationResult]]:
        """执行所有验证动作，收集证据，聚合得出结论。"""
        evidences: List[Evidence] = []
        per_method_results: List[VerificationResult] = []

        for name, action_fn in self.actions:
            try:
                ev = action_fn(obj, ctx)
            except Exception as e:
                ev = Evidence(name, "error", 0.0, 0.0, f"error: {e}", 0.5)
            evidences.append(ev)
            per_method_results.append(VerificationResult(
                method=name,
                result=UNKNOWN,
                confidence=max(ev.support, ev.contradiction),
                evidence=ev.detail,
                cost=ev.cost,
                support=ev.support,
                contradiction=ev.contradiction,
                derived_from=ev.derived_from,
                derivation_operation=ev.derivation_operation,
            ))

        agg = self.evaluator.evaluate(evidences)
        total_cost = sum(e.cost for e in evidences)
        best_ev = max(evidences, key=lambda e: max(e.support, e.contradiction))

        final = VerificationResult(
            method=agg["method"],
            result=agg["decision"],
            confidence=agg["confidence"],
            evidence=agg["detail"],
            cost=total_cost,
            support=agg["evidence_support"],
            contradiction=agg["evidence_contradiction"],
            derived_from=best_ev.derived_from,
            derivation_operation=best_ev.derivation_operation,
        )
        return final, per_method_results


# 兼容旧代码
Verification = Verifier

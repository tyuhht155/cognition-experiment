"""验证系统 Verification（证据层重构版）。

第二轮审计后的核心改变：
  验证不再直接判定 valid/invalid，而是：
    Verification Action → Evidence → EvidenceEvaluator → status

  两层分离：
  - Verification Actions（observe/count/compare/counterexample/prediction/logical_derive）
    只收集证据，返回 Evidence（support/contradiction），不判定命题是否成立。
  - EvidenceEvaluator 聚合多条证据，计算 support/contradiction/confidence/status。

  关键约束：
  - observe/count/compare/counterexample/prediction 可访问 world_history（观察层）
  - logical_derive 只访问 store.valid_entries()，不访问 world_history（推导层）
  - "没找到反例" ≠ "命题成立"（counterexample 未找到时 support=0）

  注意：当前验证器仍具有环境状态读取权限，是第一阶段的受控验证接口；
  它可用于测试验证/评价框架，但不能用于证明"系统学会了验证"。
  验证方法可靠性目前只有 consensus-based self-estimation，不是 ground-truth reliability。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .proposition import Proposition
from .knowledge_store import KnowledgeStore
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
    result: str                # valid / invalid / unknown
    confidence: float
    evidence: str
    cost: float = 1.0
    support: float = 0.0       # 聚合后的支持度
    contradiction: float = 0.0  # 聚合后的矛盾度

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "result": self.result,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "cost": self.cost,
            "support": round(self.support, 4),
            "contradiction": round(self.contradiction, 4),
        }


# 验证动作列表（先验：系统初始拥有的验证方式）
VERIFICATION_ACTIONS = [
    ("logical", action_logical_derive),       # 逻辑推导（不访问 world_history）
    ("observe", action_observe),               # 观察（访问 world_history）
    ("count", action_count),                   # 计数
    ("compare", action_compare),               # 共现比较
    ("counterexample", action_counterexample), # 反例搜索
    ("prediction", action_prediction),         # 预测-观察循环
]


class Verification:
    """对同一命题执行多种验证动作，收集证据，由 EvidenceEvaluator 聚合。

    验证方法本身的可靠性仍由 KnowledgeStore.record_verification_outcome 记录，
    但当前仅为 consensus-based self-estimation（多数派一致性），
    不是 ground-truth reliability。
    """

    def __init__(self):
        self.actions = VERIFICATION_ACTIONS
        self.evaluator = EvidenceEvaluator()

    def verify(self, obj: Proposition, ctx) -> Tuple[VerificationResult, List[VerificationResult]]:
        """执行所有验证动作，收集证据，聚合得出结论。"""
        store: KnowledgeStore = ctx.store
        evidences: List[Evidence] = []
        per_method_results: List[VerificationResult] = []

        for name, action_fn in self.actions:
            try:
                ev = action_fn(obj, ctx)
            except Exception as e:
                ev = Evidence(name, "error", 0.0, 0.0, f"error: {e}", 0.5)
            evidences.append(ev)
            # 每个动作的单独结果（供追踪，不直接作为最终判定）
            per_method_results.append(VerificationResult(
                method=name,
                result=UNKNOWN,  # 单个动作不直接判定 valid/invalid
                confidence=max(ev.support, ev.contradiction),
                evidence=ev.detail,
                cost=ev.cost,
                support=ev.support,
                contradiction=ev.contradiction,
            ))

        # 证据聚合
        agg = self.evaluator.evaluate(evidences)

        # 总成本
        total_cost = sum(e.cost for e in evidences)

        final = VerificationResult(
            method=agg["method"],
            result=agg["status"],
            confidence=agg["confidence"],
            evidence=agg["detail"],
            cost=total_cost,
            support=agg["support"],
            contradiction=agg["contradiction"],
        )

        return final, per_method_results

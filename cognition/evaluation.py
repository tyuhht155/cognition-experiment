"""ValueEvaluator：候选命题的价值评价。

【重要声明】当前公式是 initial evaluation prior（初始评价先验），
不是系统学习得到的评价算法。

职责：
  - 评估候选命题的目标相关性、泛化性、新颖性、成本、风险
  - 返回 EvaluationResult（用于排序候选）
  - 不修改 BeliefStore（只读取）

理论第 8 条：evaluate 不只返回 true/false，至少返回多维分数。
理论第 9 条：评价本身也可被评价（元评价），但不建独立模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from .proposition import Proposition
from .belief import BeliefStore
from .models import STATUS_INVALID, STATUS_VALID


@dataclass
class EvaluationResult:
    goal_relevance: float
    expected_gain: float
    computation_cost: float
    verification_cost: float
    risk: float
    worth_continuing: bool
    value_score: float
    method_tag: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "goal_relevance": round(self.goal_relevance, 4),
            "expected_gain": round(self.expected_gain, 4),
            "computation_cost": round(self.computation_cost, 4),
            "verification_cost": round(self.verification_cost, 4),
            "risk": round(self.risk, 4),
            "worth_continuing": self.worth_continuing,
            "value_score": round(self.value_score, 4),
            "method_tag": self.method_tag,
        }


class ValueEvaluator:
    """价值评价器（初始先验）。

    明确标记：评价公式是人工指定的初始先验，不是学习所得。
    只读取 BeliefStore，不修改知识状态。
    """

    def __init__(self):
        self.weights: dict = {"relevance": 0.5, "generality": 0.3, "novelty": 0.2}
        self._perf: dict = {k: [0.0, 0.0] for k in self.weights}

    def evaluate(self, prop: Proposition, goal: Any, ctx) -> EvaluationResult:
        store = ctx.knowledge_view

        relevance = self._goal_relevance(prop, goal, ctx)
        generality = self._generality(prop, ctx)
        novelty = self._novelty(prop, store)
        comp_cost = self._est_computation_cost(prop)
        ver_cost = self._est_verification_cost(prop)
        risk = self._risk(prop, store)

        value = (
            self.weights["relevance"] * relevance
            + self.weights["generality"] * generality
            + self.weights["novelty"] * novelty
        )
        expected_gain = value - 0.5 * (comp_cost / 10.0) - 0.3 * risk
        worth = expected_gain > 0.05 and value > 0.1

        tag = max(self.weights, key=lambda k: self.weights[k] * {
            "relevance": relevance, "generality": generality, "novelty": novelty}[k])

        return EvaluationResult(
            goal_relevance=relevance, expected_gain=expected_gain,
            computation_cost=comp_cost, verification_cost=ver_cost,
            risk=risk, worth_continuing=worth, value_score=value,
            method_tag=tag,
            meta={"generality": round(generality, 3), "novelty": round(novelty, 3)},
        )

    def _goal_relevance(self, prop, goal, ctx) -> float:
        if goal is None:
            return 0.3
        if isinstance(goal, Proposition):
            if goal.kind == "atom" and goal.name == "predict_next":
                return 0.9 if prop.kind == "implies" else 0.2
            if goal.kind == "atom" and goal.name == "maintain":
                return 0.6 if prop.kind in ("relation", "predicate") else 0.3
            g_terms = set(t for t in _iter_terms(goal))
            p_terms = set(t for t in _iter_terms(prop))
            if g_terms and p_terms:
                return len(g_terms & p_terms) / max(1, len(g_terms))
            return 0.2
        s = prop.to_str().lower()
        g = str(goal).lower()
        for kw in g.split():
            if kw and kw in s:
                return 0.6
        return 0.2

    def _generality(self, prop, ctx) -> float:
        if prop.kind == "forall":
            return 0.9
        if prop.kind == "implies":
            return 0.6
        if prop.kind in ("relation", "predicate"):
            terms = [t for t in _iter_terms(prop) if t in ctx.constants]
            return 0.3 if terms else 0.1
        return 0.2

    def _novelty(self, prop, store: BeliefStore) -> float:
        if store is None:
            return 1.0
        if not store.has(prop):
            return 1.0
        belief = store.get(prop)
        if belief is None:
            return 1.0
        return max(0.0, 1.0 - 0.1 * belief.evidence_count)

    def _est_computation_cost(self, prop) -> float:
        return 0.5 * len(prop.sub_propositions()) + 1.0

    def _est_verification_cost(self, prop) -> float:
        if prop.kind == "forall":
            return 4.0
        if prop.kind == "implies":
            return 2.5
        return 1.0

    def _risk(self, prop, store: BeliefStore) -> float:
        if store is None:
            return 0.3
        b = store.get(prop)
        if b and b.status == STATUS_INVALID:
            return 0.9
        neg = Proposition.neg(prop)
        nb = store.get(neg)
        if nb and nb.status == STATUS_VALID:
            return 0.8
        return 0.2

    def evaluate_evaluation(self, feedback: List[dict]) -> dict:
        """根据历史反馈调整评价风格权重。feedback 由编排层（CandidateProcessor）持有。"""
        if not feedback:
            return {"adjusted": False, "weights": dict(self.weights)}
        per_tag = {k: [0.0, 0.0, 0] for k in self.weights}
        for e in feedback:
            tag = e.get("tag")
            if tag not in per_tag:
                continue
            per_tag[tag][0] += e.get("predicted", 0.0)
            per_tag[tag][1] += e.get("actual", 0.0)
            per_tag[tag][2] += 1
        adjustments = {}
        for tag, (ps, asu, n) in per_tag.items():
            if n == 0:
                continue
            error = max(0.0, ps / n - asu / n)
            adjustments[tag] = round(error, 3)
        for tag, err in adjustments.items():
            self.weights[tag] *= max(0.5, 1.0 - err)
        s = sum(self.weights.values()) or 1.0
        for tag in self.weights:
            self.weights[tag] = round(self.weights[tag] / s, 4)
        return {"adjusted": True, "weights": dict(self.weights), "adjustments": adjustments}

    @staticmethod
    def record_feedback(feedback_list: List[dict], tag: str, predicted: float, actual: float) -> None:
        """将反馈追加到编排层持有的 feedback 列表。ValueEvaluator 本身不持有 Context。"""
        feedback_list.append({"tag": tag, "predicted": predicted, "actual": actual})


# 兼容旧代码
Evaluation = ValueEvaluator


# 兼容旧代码
Evaluation = ValueEvaluator


def _iter_terms(p: Proposition):
    for sub in p.sub_propositions():
        if sub.kind in ("relation", "predicate"):
            yield from sub.parts

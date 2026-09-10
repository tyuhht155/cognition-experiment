"""评价系统 Evaluation + 元评价 evaluate_evaluation。

理论第 8 条：evaluate(proposition, goal, context) 不只返回 true/false，
至少返回：目标相关价值、预期收益、计算成本、验证成本、风险、是否值得继续。
理论第 9 条：评价本身也必须能被评价；不要建独立的元认知模块，只是把评价结果
作为普通计算对象继续处理。

重要：“正确”与“有价值”必须完全分离。正确但无用 = 保留但低优先级。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import math

from .proposition import Proposition
from .knowledge_store import KnowledgeStore, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


@dataclass
class EvaluationResult:
    goal_relevance: float      # 当前目标相关价值 [0,1]
    expected_gain: float       # 预期收益
    computation_cost: float    # 计算成本
    verification_cost: float    # 验证成本
    risk: float                # 风险 [0,1]
    worth_continuing: bool
    value_score: float         # 综合价值（用于排序）
    method_tag: str            # 用了哪种评价风格（便于元评价）
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


class Evaluation:
    """评价系统。

    这是【先验评价能力】——我们显式承认：系统需要“一个初始目标/评价约束”
    （理论第 13 条先验之一）。评价函数本身简单且透明，不是隐藏 heuristic 规则库。
    它只做：把命题的结构特征与目标匹配，产出多维分数。
    """

    def __init__(self):
        # 评价风格的权重（会被元评价动态调整）
        # 风格：relevance(目标相关) / generality(泛化性) / novelty(新颖性)
        self.weights: dict = {
            "relevance": 0.5,
            "generality": 0.3,
            "novelty": 0.2,
        }
        # 各风格历史表现（用于元评价自校准）
        self._perf: dict = {k: [0.0, 0.0] for k in self.weights}  # [predicted_sum, actual_sum]

    # ---------- 基本评价 ----------
    def evaluate(self, prop: Proposition, goal: Any, ctx) -> EvaluationResult:
        store: KnowledgeStore = ctx.store

        relevance = self._goal_relevance(prop, goal, ctx)
        generality = self._generality(prop, ctx)
        novelty = self._novelty(prop, store)
        comp_cost = self._est_computation_cost(prop, ctx)
        ver_cost = self._est_verification_cost(prop, ctx)
        risk = self._risk(prop, ctx)

        value = (
            self.weights["relevance"] * relevance
            + self.weights["generality"] * generality
            + self.weights["novelty"] * novelty
        )
        # 预期收益 = 价值 - (成本+风险)
        expected_gain = value - 0.5 * (comp_cost / 10.0) - 0.3 * risk
        worth = expected_gain > 0.05 and value > 0.1

        # 评价风格标记：哪个维度主导
        tag = max(self.weights, key=lambda k: self.weights[k] * {"relevance": relevance,
                                                                 "generality": generality,
                                                                 "novelty": novelty}[k])

        return EvaluationResult(
            goal_relevance=relevance,
            expected_gain=expected_gain,
            computation_cost=comp_cost,
            verification_cost=ver_cost,
            risk=risk,
            worth_continuing=worth,
            value_score=value,
            method_tag=tag,
            meta={"generality": round(generality, 3), "novelty": round(novelty, 3)},
        )

    # ---------- 特征函数（透明，非规则库） ----------
    def _goal_relevance(self, prop: Proposition, goal: Any, ctx) -> float:
        if goal is None:
            return 0.3
        # goal 可以是 Proposition 或字符串描述
        if isinstance(goal, Proposition):
            # 目标是“预测下一状态”：蕴含命题更相关
            if goal.kind == "atom" and goal.name == "predict_next":
                return 0.9 if prop.kind == "implies" else 0.2
            if goal.kind == "atom" and goal.name == "maintain":
                return 0.6 if prop.kind in ("relation", "predicate") else 0.3
            # 共享项则相关
            g_terms = set(t for t in _iter_terms(goal))
            p_terms = set(t for t in _iter_terms(prop))
            if g_terms and p_terms:
                overlap = len(g_terms & p_terms) / max(1, len(g_terms))
                return overlap
            return 0.2
        # 字符串目标：简单关键词匹配（先验）
        s = prop.to_str().lower()
        g = str(goal).lower()
        for kw in g.split():
            if kw and kw in s:
                return 0.6
        return 0.2

    def _generality(self, prop: Proposition, ctx) -> float:
        if prop.kind == "forall":
            return 0.9
        if prop.kind == "implies":
            return 0.6
        if prop.kind in ("relation", "predicate"):
            terms = [t for t in _iter_terms(prop) if t in ctx.constants]
            return 0.3 if terms else 0.1
        return 0.2

    def _novelty(self, prop: Proposition, store: KnowledgeStore) -> float:
        if store is None:
            return 1.0
        if not store.has(prop):
            return 1.0
        k = store.get(prop)
        if k is None:
            return 1.0
        # 已知且 usage 少 -> 仍较新颖
        return max(0.0, 1.0 - 0.1 * k.usage_count)

    def _est_computation_cost(self, prop: Proposition, ctx) -> float:
        # 粗略：子命题数 + 深度
        subs = prop.sub_propositions()
        return 0.5 * len(subs) + 1.0

    def _est_verification_cost(self, prop: Proposition, ctx) -> float:
        if prop.kind == "forall":
            return 4.0
        if prop.kind == "implies":
            return 2.5
        return 1.0

    def _risk(self, prop: Proposition, ctx) -> float:
        # 已有反例/矛盾 -> 高风险
        store: KnowledgeStore = ctx.store
        if store is None:
            return 0.3
        if store.get(prop) and store.get(prop).status == STATUS_INVALID:
            return 0.9
        neg = Proposition.neg(prop)
        if store.get(neg) and store.get(neg).status == STATUS_VALID:
            return 0.8
        return 0.2

    # ---------- 元评价：评价评价本身 ----------
    def evaluate_evaluation(self, ctx) -> dict:
        """根据 ctx.eval_feedback 调整各评价风格权重。

        feedback 元素：{"tag": 风格, "predicted": 价值分, "actual": 实际有用(0/1)}
        如果某风格长期预测高价值但实际无用 -> 降低其权重。
        """
        fb: List[dict] = ctx.eval_feedback
        if not fb:
            return {"adjusted": False, "weights": dict(self.weights)}
        # 按风格聚合
        per_tag: dict = {k: [0.0, 0.0, 0] for k in self.weights}  # pred_sum, actual_sum, n
        for e in fb:
            tag = e.get("tag")
            if tag not in per_tag:
                continue
            per_tag[tag][0] += e.get("predicted", 0.0)
            per_tag[tag][1] += e.get("actual", 0.0)
            per_tag[tag][2] += 1
        # 校准：预测高但实际低 -> 降权
        adjustments = {}
        for tag, (ps, asu, n) in per_tag.items():
            if n == 0:
                continue
            pred_mean = ps / n
            actual_mean = asu / n
            # 误差越大越降权
            error = max(0.0, pred_mean - actual_mean)
            adjustments[tag] = round(error, 3)
            self._perf[tag][0] += ps
            self._perf[tag][1] += asu
        # 归一化权重（保持和为1）
        for tag, err in adjustments.items():
            self.weights[tag] *= max(0.5, 1.0 - err)
        s = sum(self.weights.values()) or 1.0
        for tag in self.weights:
            self.weights[tag] = round(self.weights[tag] / s, 4)
        return {"adjusted": True, "weights": dict(self.weights),
                "adjustments": adjustments}

    def add_feedback(self, ctx, tag: str, predicted: float, actual: float) -> None:
        ctx.eval_feedback.append({"tag": tag, "predicted": predicted, "actual": actual})


def _iter_terms(p: Proposition):
    for sub in p.sub_propositions():
        if sub.kind in ("relation", "predicate"):
            yield from sub.parts

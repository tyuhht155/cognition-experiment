"""证据层：Verification Action → Evidence → EvidenceEvaluator。

核心分层（必须严格区分三个步骤）：
  1. Evidence generation：验证动作收集证据（观察到了什么）
  2. Evidence interpretation：EvidenceEvaluator 把证据转为 support/contradiction/confidence
  3. Decision：根据 confidence + 先验阈值决定 status（accepted/rejected/undecided）

重要声明：
  EvidenceEvaluator 是【初始证据评价先验】（initial evidence evaluation prior），
  不是系统自主学习得到的评价算法。它的阈值和聚合公式是人为指定的。
  当前实验测试的是"加入这个先验后验证闭环是否成立"，
  不测试"系统是否学会了证据评价"。

STATUS_VALID 的准确含义是：
  accepted_under_current_evidence_policy
  （在当前证据策略下达到接受阈值）
  ≠ objectively true
  客观真值只能由外部 evaluator 判断。

关键分离：
  - observe/count/compare/counterexample/prediction：访问环境观察（world_history）
  - logical_derive：只访问已验证的知识（store.valid_entries），不访问 world_history

"world_history 中有没有 P" ≠ "P = valid"。
它只是一条 observation evidence，由 evaluator 决定支持度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .proposition import Proposition
from .models import STATUS_VALID


VALID = "valid"
INVALID = "invalid"
UNKNOWN = "unknown"


@dataclass
class Evidence:
    """一条验证证据。不直接判定命题是否成立，只提供支持/矛盾程度。"""
    method: str              # 验证动作名
    action_type: str         # observe / count / compare / counterexample / prediction / derivation
    support: float           # [0,1] 对命题的支持程度
    contradiction: float     # [0,1] 对命题的矛盾程度
    detail: str              # 人类可读的证据描述
    cost: float = 1.0
    prediction_id: Optional[str] = None  # 预测类证据的追踪 ID
    # ---- 逻辑推导溯源（仅 derivation 类型使用）----
    derived_from: Optional[List[str]] = None  # 推导来源命题的 str 表示
    derivation_operation: Optional[str] = None  # modus_ponens / modus_tollens / direct / negation

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "action_type": self.action_type,
            "support": round(self.support, 4),
            "contradiction": round(self.contradiction, 4),
            "detail": self.detail,
            "cost": self.cost,
            "derived_from": self.derived_from,
            "derivation_operation": self.derivation_operation,
        }


# ============================================================
# Verification Actions —— 只返回 Evidence，不判定 valid/invalid
# ============================================================

def action_observe(obj: Proposition, ctx) -> Evidence:
    """观察动作：检查 obj 在已观察历史中出现的频率。
    注意：这只是 observation evidence，不等于 logical proof。
    """
    hist = ctx.world_history
    if not hist:
        return Evidence("observe", "observe", 0.0, 0.0, "no history", 0.5)
    n_total = len(hist)
    n_with = sum(1 for s in hist if obj in s)
    n_without = n_total - n_with
    # 出现频率作为支持度的基础，但不直接等于 valid
    freq = n_with / n_total if n_total else 0.0
    # 检查否定是否被观察到（矛盾证据）
    neg = Proposition.neg(obj)
    n_neg = sum(1 for s in hist if neg in s)
    contradiction = n_neg / n_total if n_total else 0.0
    return Evidence("observe", "observe",
                    support=freq,
                    contradiction=contradiction,
                    detail=f"observed {n_with}/{n_total} states, negation {n_neg}",
                    cost=0.5)


def action_count(obj: Proposition, ctx) -> Evidence:
    """计数动作：精确统计出现次数。"""
    hist = ctx.world_history
    n = sum(1 for s in hist if obj in s)
    return Evidence("count", "count",
                    support=min(1.0, n / 3.0),  # 出现 3 次以上给满支持
                    contradiction=0.0,
                    detail=f"count={n}/{len(hist)}",
                    cost=0.5)


def action_compare(obj: Proposition, ctx) -> Evidence:
    """比较动作：对蕴含 A→B，统计 A 出现时 B 的共现率。"""
    if obj.kind != "implies":
        return Evidence("compare", "compare", 0.0, 0.0, "not implication", 0.5)
    a, b = obj.parts
    hist = ctx.world_history
    matches = 0
    total = 0
    for s in hist:
        if a in s:
            total += 1
            if b in s:
                matches += 1
    if total == 0:
        return Evidence("compare", "compare", 0.0, 0.0,
                        "antecedent never observed", 1.0)
    conf = matches / total
    # 支持度 = 共现率；矛盾度 = 1 - 共现率（有反例时）
    return Evidence("compare", "compare",
                    support=conf,
                    contradiction=1.0 - conf,
                    detail=f"co-occurrence {matches}/{total}",
                    cost=1.0)


def action_counterexample(obj: Proposition, ctx) -> Evidence:
    """反例动作：主动搜索反例。
    找到反例 → contradiction 高；找不到 → 不提供支持（保持 UNKNOWN 方向）。
    重要：没找到反例 ≠ 命题成立。
    """
    hist = ctx.world_history
    if obj.kind == "implies":
        a, b = obj.parts
        # 共现反例：A 出现时 B 不出现
        for t, s in enumerate(hist):
            if a in s and b not in s:
                return Evidence("counterexample", "counterexample",
                                support=0.0, contradiction=0.95,
                                detail=f"co-occurrence counterexample at step {t}",
                                cost=2.0)
        # 没找到反例 → 不判 valid，只提供微弱支持（没有矛盾）
        return Evidence("counterexample", "counterexample",
                        support=0.0, contradiction=0.0,
                        detail="no counterexample found (not proof of validity)",
                        cost=2.0)
    if obj.kind == "forall":
        body = obj.parts[0]
        var = obj.name
        for c in ctx.constants:
            inst = body.substitute_term(var, c)
            neg = Proposition.neg(inst)
            for s in hist:
                if neg in s:
                    return Evidence("counterexample", "counterexample",
                                    support=0.0, contradiction=0.95,
                                    detail=f"counterexample for constant {c}",
                                    cost=2.0)
        return Evidence("counterexample", "counterexample",
                        support=0.0, contradiction=0.0,
                        detail="no counterexample found (not proof)",
                        cost=2.0)
    # 原子/谓词：反例即否定被观察
    neg = Proposition.neg(obj)
    n_neg = sum(1 for s in hist if neg in s)
    if n_neg > 0:
        return Evidence("counterexample", "counterexample",
                        support=0.0, contradiction=0.9,
                        detail=f"negation observed {n_neg} times",
                        cost=1.0)
    return Evidence("counterexample", "counterexample",
                    support=0.0, contradiction=0.0,
                    detail="no counterexample found (not proof)",
                    cost=1.0)


def action_prediction(obj: Proposition, ctx) -> Evidence:
    """预测动作：真实的预测→观察→反馈循环。

    对蕴含 A→B：
    - 当 A 在当前观察中出现时，系统预测 B 将出现
    - 预测被注册到 ctx.prediction_queue
    - 当新的观察到达时，检查 B 是否出现，更新预测的成功/失败
    - 证据来自已完成的预测反馈

    这与 compare 不同：compare 是回溯统计共现，prediction 是真正的前向预测。
    """
    if obj.kind != "implies":
        return Evidence("prediction", "prediction", 0.0, 0.0, "not implication", 1.0)
    a, b = obj.parts

    # 获取已完成的预测结果
    queue = getattr(ctx, "prediction_queue", {})
    prop_key = obj.to_str()
    record = queue.get(prop_key, {"confirmed": 0, "refuted": 0, "pending": []})

    confirmed = record["confirmed"]
    refuted = record["refuted"]
    total = confirmed + refuted

    if total == 0:
        return Evidence("prediction", "prediction", 0.0, 0.0,
                        "no completed predictions yet", 1.0)

    conf = confirmed / total
    return Evidence("prediction", "prediction",
                    support=conf,
                    contradiction=1.0 - conf,
                    detail=f"predictions: {confirmed} confirmed, {refuted} refuted",
                    cost=1.0)


def action_logical_derive(obj: Proposition, ctx) -> Evidence:
    """逻辑推导动作：只从已验证的知识推导，不访问环境观察历史。

    允许的推导：
    - P 已知 valid → P 支持
    - ¬P 已知 valid → P 矛盾
    - P→Q valid 且 P valid → Q 支持（modus ponens）
    - P→Q valid 且 ¬Q valid → ¬P 支持（modus tollens）

    每条推导记录 derived_from（来源命题）和 derivation_operation（推导规则），
    以便知识更新时记录 parents 和 operation。

    注意："历史中观察到 P" 不属于逻辑推导，属于 observation。
    """
    store = ctx.belief_store
    # 直接命中
    k = store.get(obj)
    if k and k.status == STATUS_VALID:
        return Evidence("logical", "derivation",
                        support=1.0, contradiction=0.0,
                        detail="directly known valid",
                        cost=0.5,
                        derived_from=[obj.to_str()],
                        derivation_operation="direct")
    # 矛盾：obj 是 ¬P 而 P valid
    if obj.kind == "not":
        target = obj.parts[0]
        kt = store.get(target)
        if kt and kt.status == STATUS_VALID:
            return Evidence("logical", "derivation",
                            support=0.0, contradiction=1.0,
                            detail=f"{target.to_str()} known valid",
                            cost=0.5,
                            derived_from=[target.to_str()],
                            derivation_operation="negation")
    # P 而 ¬P valid
    neg = Proposition.neg(obj)
    kn = store.get(neg)
    if kn and kn.status == STATUS_VALID:
        return Evidence("logical", "derivation",
                        support=0.0, contradiction=1.0,
                        detail="negation known valid",
                        cost=0.5,
                        derived_from=[neg.to_str()],
                        derivation_operation="negation")
    # Modus ponens: X→obj valid 且 X valid → obj 支持
    for b in store.valid_beliefs():
        p = b.proposition
        if p.kind == "implies":
            premise, conclusion = p.parts
            if conclusion == obj:
                kp = store.get(premise)
                if kp and kp.status == STATUS_VALID:
                    return Evidence("logical", "derivation",
                                    support=0.95, contradiction=0.0,
                                    detail=f"modus ponens from {premise.to_str()} → {obj.to_str()}",
                                    cost=1.0,
                                    derived_from=[premise.to_str(), p.to_str()],
                                    derivation_operation="modus_ponens")
            # obj→X valid 且 ¬X valid → obj 矛盾（modus tollens）
            if premise == obj:
                kx = store.get(Proposition.neg(conclusion))
                if kx and kx.status == STATUS_VALID:
                    return Evidence("logical", "derivation",
                                    support=0.0, contradiction=0.9,
                                    detail=f"modus tollens: {obj.to_str()}→{conclusion.to_str()}, ¬{conclusion.to_str()} valid",
                                    cost=1.0,
                                    derived_from=[p.to_str(), Proposition.neg(conclusion).to_str()],
                                    derivation_operation="modus_tollens")
    # 无逻辑证据
    return Evidence("logical", "derivation",
                    support=0.0, contradiction=0.0,
                    detail="no logical derivation available",
                    cost=0.5)


# ============================================================
# Evidence Evaluator —— 聚合证据，计算 support/contradiction/confidence
# ============================================================

class EvidenceEvaluator:
    """【初始证据评价先验】(initial evidence evaluation prior)。

    这是人为指定的先验机制，不是系统学习得到的。
    职责：把多条 Evidence 聚合为 support/contradiction（证据解释），
    再计算 confidence，最后根据阈值做 decision。

    三层严格分离：
      evidence_support / evidence_contradiction  ← 证据本身
      confidence                                ← 证据强度 × 明确度
      decision (status)                         ← 先验阈值下的接受/拒绝/待定

    STATUS_VALID = accepted_under_current_evidence_policy（达到接受阈值）
                   ≠ objectively true（客观真值由外部 evaluator 判断）

    不访问 ground truth。
    """

    # 判定阈值（先验参数，非学习所得）
    VALID_SUPPORT_THRESHOLD = 0.7
    INVALID_CONTRADICTION_THRESHOLD = 0.7
    MIN_EVIDENCE_FOR_CONFIDENCE = 0.3

    def evaluate(self, evidences: List[Evidence]) -> dict:
        """聚合证据，返回分层结果。

        返回字段说明：
          evidence_support:      证据平均支持度（证据解释层）
          evidence_contradiction: 证据平均矛盾度（证据解释层）
          confidence:             置信度（证据强度×明确度）
          decision:               status（先验阈值下的决策，非真值）
        """
        if not evidences:
            return {"evidence_support": 0.0, "evidence_contradiction": 0.0,
                    "confidence": 0.0, "decision": UNKNOWN, "status": UNKNOWN,
                    "method": "none", "detail": "no evidence"}

        # 只保留有实际证据的方法
        meaningful = [e for e in evidences if e.support > 0 or e.contradiction > 0]
        if not meaningful:
            return {"evidence_support": 0.0, "evidence_contradiction": 0.0,
                    "confidence": 0.0, "decision": UNKNOWN, "status": UNKNOWN,
                    "method": "none", "detail": "no meaningful evidence"}

        total_support = sum(e.support for e in meaningful)
        total_contradiction = sum(e.contradiction for e in meaningful)
        n = len(meaningful)

        # ---- 证据解释层 ----
        evidence_support = total_support / n
        evidence_contradiction = total_contradiction / n

        # ---- 置信度 ----
        evidence_strength = min(1.0, n * 0.4)
        clarity = abs(evidence_support - evidence_contradiction)
        confidence = evidence_strength * clarity

        # ---- 决策层（先验阈值）----
        # 强反例优先：任何一条证据的 contradiction >= 0.9 即视为决定性反驳
        # （一个反例足以反驳全称命题 A→B）
        max_contradiction = max(e.contradiction for e in meaningful)
        if max_contradiction >= 0.9:
            decision = INVALID
            confidence = max_contradiction
        elif (evidence_support >= self.VALID_SUPPORT_THRESHOLD
              and evidence_support > evidence_contradiction):
            decision = VALID
        elif (evidence_contradiction >= self.INVALID_CONTRADICTION_THRESHOLD
              and evidence_contradiction > evidence_support):
            decision = INVALID
        else:
            decision = UNKNOWN

        best = max(meaningful, key=lambda e: max(e.support, e.contradiction))
        return {
            "evidence_support": round(evidence_support, 4),
            "evidence_contradiction": round(evidence_contradiction, 4),
            "confidence": round(confidence, 4),
            "decision": decision,
            "status": decision,  # 兼容旧字段
            "method": best.method,
            "detail": best.detail,
        }


# ============================================================
# Prediction Queue 管理 —— 真实预测循环
# ============================================================

def register_prediction(ctx, prop: Proposition) -> None:
    """当命题 P→Q 的前件 P 在当前观察中出现时，注册一个对 Q 的预测。"""
    if prop.kind != "implies":
        return
    a, b = prop.parts
    if not ctx.world_history:
        return
    current = ctx.world_history[-1]
    if a in current:
        queue = getattr(ctx, "prediction_queue", None)
        if queue is None:
            ctx.prediction_queue = {}
            queue = ctx.prediction_queue
        key = prop.to_str()
        if key not in queue:
            queue[key] = {"confirmed": 0, "refuted": 0, "pending": []}
        queue[key]["pending"].append(b)


def process_predictions(ctx) -> None:
    """当新观察到达时，处理待处理的预测：检查后件是否出现。"""
    queue = getattr(ctx, "prediction_queue", None)
    if not queue or not ctx.world_history:
        return
    current = ctx.world_history[-1]
    for key, record in queue.items():
        still_pending = []
        for predicted_b in record["pending"]:
            if predicted_b in current:
                record["confirmed"] += 1
            else:
                record["refuted"] += 1
        record["pending"] = still_pending


# ============================================================
# EvidenceLog —— append-only 历史证据记录
# ============================================================

class EvidenceLog:
    """追加保存所有历史 Evidence。

    关键约束：append-only。
    信念更新不能删除或修改已记录的 Evidence。
    同一个 ObservationEvent 不能被重复计权（由调用方保证，EvidenceLog 只追加）。
    """

    def __init__(self):
        self._entries: List[Evidence] = []

    def append(self, evidence: Evidence) -> None:
        """追加一条证据。不可删除。"""
        self._entries.append(evidence)

    def append_many(self, evidences: List[Evidence]) -> None:
        self._entries.extend(evidences)

    def all(self) -> List[Evidence]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def for_proposition(self, prop: Proposition) -> List[Evidence]:
        """返回与给定命题相关的所有证据（按 method 关联）。

        注意：Evidence 本身不直接关联 proposition（证据是方法产生的），
        这里返回全部历史，由 EvidenceEvaluator 决定如何使用。
        """
        return list(self._entries)

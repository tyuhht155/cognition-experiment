"""证据层：Verification Action → Evidence。

第二轮审计后的核心重构：
  验证动作只负责"收集证据"，不直接判定 valid/invalid。
  证据由 EvidenceEvaluator 聚合成 support/contradiction/confidence/status。

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
from .knowledge_store import KnowledgeStore, STATUS_VALID


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

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "action_type": self.action_type,
            "support": round(self.support, 4),
            "contradiction": round(self.contradiction, 4),
            "detail": self.detail,
            "cost": self.cost,
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
        # 共现反例
        for t, s in enumerate(hist):
            if a in s and b not in s:
                return Evidence("counterexample", "counterexample",
                                support=0.0, contradiction=0.95,
                                detail=f"co-occurrence counterexample at step {t}",
                                cost=2.0)
        # 过渡反例：A@t 但 B 不在 t+1
        for t in range(len(hist) - 1):
            if a in hist[t] and b not in hist[t + 1]:
                return Evidence("counterexample", "counterexample",
                                support=0.0, contradiction=0.9,
                                detail=f"transition counterexample at step {t}",
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

    注意："历史中观察到 P" 不属于逻辑推导，属于 observation。
    """
    store: KnowledgeStore = ctx.store
    # 直接命中
    k = store.get(obj)
    if k and k.status == STATUS_VALID:
        return Evidence("logical", "derivation",
                        support=1.0, contradiction=0.0,
                        detail="directly known valid",
                        cost=0.5)
    # 矛盾：obj 是 ¬P 而 P valid
    if obj.kind == "not":
        target = obj.parts[0]
        kt = store.get(target)
        if kt and kt.status == STATUS_VALID:
            return Evidence("logical", "derivation",
                            support=0.0, contradiction=1.0,
                            detail=f"{target.to_str()} known valid",
                            cost=0.5)
    # P 而 ¬P valid
    neg = Proposition.neg(obj)
    kn = store.get(neg)
    if kn and kn.status == STATUS_VALID:
        return Evidence("logical", "derivation",
                        support=0.0, contradiction=1.0,
                        detail="negation known valid",
                        cost=0.5)
    # Modus ponens: 寻找 valid 的 (obj → X) 和 valid 的 obj，推导 X
    # 或者寻找 valid 的 (X → obj) 和 valid 的 X，推导 obj
    for k in store.valid_entries():
        p = k.proposition
        if p.kind == "implies":
            premise, conclusion = p.parts
            # X → obj，且 X valid → obj 支持
            if conclusion == obj:
                kp = store.get(premise)
                if kp and kp.status == STATUS_VALID:
                    return Evidence("logical", "derivation",
                                    support=0.95, contradiction=0.0,
                                    detail=f"modus ponens from {premise.to_str()} → {obj.to_str()}",
                                    cost=1.0)
            # obj → X，且 ¬X valid → ¬obj 支持（obj 矛盾）
            if premise == obj:
                kx = store.get(Proposition.neg(conclusion))
                if kx and kx.status == STATUS_VALID:
                    return Evidence("logical", "derivation",
                                    support=0.0, contradiction=0.9,
                                    detail=f"modus tollens: {obj.to_str()}→{conclusion.to_str()}, ¬{conclusion.to_str()} valid",
                                    cost=1.0)
    # 无逻辑证据
    return Evidence("logical", "derivation",
                    support=0.0, contradiction=0.0,
                    detail="no logical derivation available",
                    cost=0.5)


# ============================================================
# Evidence Evaluator —— 聚合证据，计算 support/contradiction/confidence
# ============================================================

class EvidenceEvaluator:
    """根据多条 Evidence 计算命题的最终支持度、矛盾度、置信度和状态。

    不访问 ground truth。只基于证据本身聚合。
    """

    # 判定阈值（先验参数，非学习所得）
    VALID_SUPPORT_THRESHOLD = 0.7
    INVALID_CONTRADICTION_THRESHOLD = 0.7
    MIN_EVIDENCE_FOR_CONFIDENCE = 0.3

    def evaluate(self, evidences: List[Evidence]) -> dict:
        """聚合证据。
        返回: {support, contradiction, confidence, status, method, detail}

        只聚合提供了实际证据的方法（support>0 或 contradiction>0），
        避免"无证据"的方法稀释信号。
        """
        if not evidences:
            return {"support": 0.0, "contradiction": 0.0, "confidence": 0.0,
                    "status": UNKNOWN, "method": "none", "detail": "no evidence"}

        # 只保留有实际证据的方法
        meaningful = [e for e in evidences if e.support > 0 or e.contradiction > 0]
        if not meaningful:
            return {"support": 0.0, "contradiction": 0.0, "confidence": 0.0,
                    "status": UNKNOWN, "method": "none",
                    "detail": "no meaningful evidence"}

        total_support = sum(e.support for e in meaningful)
        total_contradiction = sum(e.contradiction for e in meaningful)
        n = len(meaningful)

        avg_support = total_support / n
        avg_contradiction = total_contradiction / n

        # 置信度 = 证据量 × 明确度
        evidence_strength = min(1.0, n * 0.4)
        clarity = abs(avg_support - avg_contradiction)
        confidence = evidence_strength * clarity

        # 状态判定
        if avg_support >= self.VALID_SUPPORT_THRESHOLD and avg_support > avg_contradiction:
            status = VALID
        elif avg_contradiction >= self.INVALID_CONTRADICTION_THRESHOLD and avg_contradiction > avg_support:
            status = INVALID
        else:
            status = UNKNOWN

        best = max(meaningful, key=lambda e: max(e.support, e.contradiction))
        return {
            "support": round(avg_support, 4),
            "contradiction": round(avg_contradiction, 4),
            "confidence": round(confidence, 4),
            "status": status,
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

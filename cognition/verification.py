"""验证系统 Verification。

理论第 7 条：不要假设验证只有一种形式。至少实现：
  1. 逻辑验证  2. 规则验证  3. 重复案例验证  4. 预测后观察  5. 反例验证
每个命题可尝试不同验证方式；验证方法本身也进入 KnowledgeStore，并记录其可靠性，
使系统能学习“某类命题→某种验证方法→通常更可靠”。

验证只判定【命题是否成立】，不判定是否有价值（价值由 Evaluation 负责）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

from .proposition import Proposition
from .knowledge_store import KnowledgeStore, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


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

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "result": self.result,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "cost": self.cost,
        }


# ---------------- 单个验证方法 ----------------

def v_logical(obj: Proposition, ctx) -> VerificationResult:
    """逻辑验证：
    - 若对象与某条已知 valid 事实直接矛盾（P 与 ¬P 同时 valid）-> invalid
    - 若对象是已知 valid 事实的简单重述 -> valid
    - 否则 unknown
    """
    store: KnowledgeStore = ctx.store
    # 直接命中
    k = store.get(obj)
    if k and k.status == STATUS_VALID:
        return VerificationResult("logical", VALID, 1.0, "directly known valid", 0.5)
    # 矛盾：obj 是 ¬P 而 P valid
    if obj.kind == "not":
        target = obj.parts[0]
        kt = store.get(target)
        if kt and kt.status == STATUS_VALID:
            return VerificationResult("logical", INVALID, 0.95,
                                      f"{target.to_str()} known valid", 0.5)
    # 反过来：obj 是 P 而 ¬P valid
    neg = Proposition.neg(obj)
    kn = store.get(neg)
    if kn and kn.status == STATUS_VALID:
        return VerificationResult("logical", INVALID, 0.95, "negation known valid", 0.5)
    # 直接观察事实（历史状态中出现过）
    for state in ctx.world_history:
        if obj in state:
            return VerificationResult("logical", VALID, 0.9, "observed in history", 1.0)
    return VerificationResult("logical", UNKNOWN, 0.0, "no logical evidence", 0.5)


def v_rule(obj: Proposition, ctx) -> VerificationResult:
    """规则验证：是否被某条已知 valid 规则（蕴含/全称）覆盖。
    例：对象 On(ball,table)，若有 valid 规则 ∀x On(x,table)->... 不直接判。
    主要用于蕴含命题：A→B 是否与某 valid 蕴含一致或被其覆盖。
    """
    store: KnowledgeStore = ctx.store
    if obj.kind != "implies":
        return VerificationResult("rule", UNKNOWN, 0.0, "not an implication", 0.5)
    a, b = obj.parts
    for k in store.valid_entries():
        p = k.proposition
        if p == obj:
            return VerificationResult("rule", VALID, 1.0, "rule already known", 0.5)
        if p.kind == "implies" and p.parts[0] == a and p.parts[1] == b:
            return VerificationResult("rule", VALID, 0.95, "covered by known rule", 0.5)
        # 矛盾规则：A→¬B 存在
        if p.kind == "implies" and p.parts[0] == a and p.parts[1] == Proposition.neg(b):
            return VerificationResult("rule", INVALID, 0.9, "contradicted by known rule", 0.5)
    return VerificationResult("rule", UNKNOWN, 0.0, "no covering rule", 0.5)


def v_repeated_case(obj: Proposition, ctx) -> VerificationResult:
    """重复案例验证（归纳）：
    对蕴含 A→B：统计历史中 A 出现时 B 也出现的比例。
    对全称 ∀x P(x)：抽样常量看 P(c) 在历史中是否成立。
    对关系/谓词：在历史中是否恒成立。
    """
    hist = ctx.world_history
    if len(hist) < 1:
        return VerificationResult("repeated_case", UNKNOWN, 0.0, "no history", 1.0)

    if obj.kind == "implies":
        a, b = obj.parts
        # 共现一致性：P 出现的状态中，Q 是否也出现
        matches = 0
        total = 0
        for s in hist:
            if a in s:
                total += 1
                if b in s:
                    matches += 1
        if total == 0:
            # 也尝试过渡性解释：P@t -> Q@t+1
            t_total = 0
            t_match = 0
            for i in range(len(hist) - 1):
                if a in hist[i]:
                    t_total += 1
                    if b in hist[i + 1]:
                        t_match += 1
            if t_total == 0:
                return VerificationResult("repeated_case", UNKNOWN, 0.0, "antecedent never observed", 1.5)
            conf = t_match / t_total
            if t_total >= 2 and conf == 1.0:
                return VerificationResult("repeated_case", VALID, conf,
                                            f"{t_match}/{t_total} transitions confirm", 1.5)
            return VerificationResult("repeated_case", UNKNOWN, conf,
                                       f"trans {t_match}/{t_total}", 1.5)
        conf = matches / total
        if total >= 2 and conf == 1.0:
            return VerificationResult("repeated_case", VALID, conf,
                                      f"{matches}/{total} co-occurrences confirm", 1.5)
        if conf < 0.5 and total >= 2:
            return VerificationResult("repeated_case", INVALID, 1 - conf,
                                      f"only {matches}/{total} co-occur", 1.5)
        return VerificationResult("repeated_case", UNKNOWN, conf,
                                  f"{matches}/{total} co-occur (insufficient)", 1.5)

    if obj.kind == "forall":
        body = obj.parts[0]
        var = obj.name
        ok = 0
        bad = 0
        for c in ctx.constants:
            inst = body.substitute_term(var, c)
            present = any(inst in s for s in hist)
            # 检查反例
            neg = Proposition.neg(inst)
            contra = any(neg in s for s in hist)
            if contra:
                bad += 1
            elif present:
                ok += 1
        if bad > 0:
            return VerificationResult("repeated_case", INVALID, 0.9,
                                      f"counterexample among constants", 2.0)
        if ok == len(ctx.constants):
            return VerificationResult("repeated_case", VALID, 0.8,
                                      "holds for all sampled constants", 2.0)
        return VerificationResult("repeated_case", UNKNOWN, 0.4, "partial sampling", 2.0)

    # 关系/谓词/原子：是否在所有历史状态中成立（恒真）
    if obj.kind in ("relation", "predicate", "atom"):
        states_with = sum(1 for s in hist if obj in s)
        neg = Proposition.neg(obj)
        states_neg = sum(1 for s in hist if neg in s)
        if states_neg > 0:
            return VerificationResult("repeated_case", INVALID, 0.8,
                                      f"negation observed {states_neg} times", 1.0)
        if states_with >= 2:
            return VerificationResult("repeated_case", VALID, 0.7,
                                      f"observed {states_with} times", 1.0)
        return VerificationResult("repeated_case", UNKNOWN, 0.3, "rarely observed", 1.0)

    return VerificationResult("repeated_case", UNKNOWN, 0.0, "not applicable", 0.5)


def v_counterexample(obj: Proposition, ctx) -> VerificationResult:
    """反例验证：主动寻找一个反例。找到 -> invalid；找不到 -> 不能证明 valid，给中性。
    对蕴含 P→Q，反例 = 一个状态中 P 成立但 Q 不成立（共现反例）；
    或一次转移中 P@t 成立但 Q@t+1 不成立（过渡反例）。
    """
    hist = ctx.world_history
    if obj.kind == "implies":
        a, b = obj.parts
        # 共现反例
        for t, s in enumerate(hist):
            if a in s and b not in s:
                return VerificationResult("counterexample", INVALID, 0.95,
                                          f"co-occurrence counterexample at step {t}", 2.0)
        # 过渡反例
        for t in range(len(hist) - 1):
            if a in hist[t] and b not in hist[t + 1]:
                return VerificationResult("counterexample", INVALID, 0.9,
                                          f"transition counterexample at step {t}", 2.0)
        return VerificationResult("counterexample", UNKNOWN, 0.5, "no counterexample found", 2.0)
    if obj.kind == "forall":
        body = obj.parts[0]
        var = obj.name
        for c in ctx.constants:
            inst = body.substitute_term(var, c)
            neg = Proposition.neg(inst)
            for s in hist:
                if neg in s:
                    return VerificationResult("counterexample", INVALID, 0.95,
                                              f"counterexample: {c}", 2.0)
        return VerificationResult("counterexample", UNKNOWN, 0.5, "no counterexample found", 2.0)
    # 原子/关系：反例即 ¬obj 被观察
    neg = Proposition.neg(obj)
    if any(neg in s for s in hist):
        return VerificationResult("counterexample", INVALID, 0.9, "negation observed", 1.0)
    return VerificationResult("counterexample", UNKNOWN, 0.4, "no counterexample found", 1.0)


def v_prediction_observation(obj: Proposition, ctx) -> VerificationResult:
    """预测后观察（共现语义）：把 P->Q 当作“见 P 即预测 Q”的预测规则。
    若历史中所有含 P 的状态都含 Q（且案例>=3）-> 支持；存在反例 -> 削弱。"""
    if obj.kind != "implies" or len(ctx.world_history) < 3:
        return VerificationResult("prediction_obs", UNKNOWN, 0.0, "insufficient data", 2.0)
    a, b = obj.parts
    hist = ctx.world_history
    confirmed = 0
    refuted = 0
    for s in hist:
        if a in s:
            if b in s:
                confirmed += 1
            else:
                refuted += 1
    total = confirmed + refuted
    if total == 0:
        return VerificationResult("prediction_obs", UNKNOWN, 0.0, "no prediction opportunities", 2.0)
    conf = confirmed / total
    if refuted == 0 and confirmed >= 3:
        return VerificationResult("prediction_obs", VALID, conf, f"{confirmed} predictions confirmed", 2.0)
    if confirmed == 0 and refuted >= 1:
        return VerificationResult("prediction_obs", INVALID, 1 - conf, f"{refuted} predictions refuted", 2.0)
    return VerificationResult("prediction_obs", UNKNOWN, conf, f"{confirmed}c/{refuted}r", 2.0)


# ---------------- 验证系统：聚合多方法 ----------------

class Verification:
    """对同一命题尝试多种验证方法，按历史可靠性加权得出最终结论。

    各方法结果会被反馈给 KnowledgeStore.record_verification_outcome，
    用于学习验证方法可靠性（理论第 7 条）。
    """

    def __init__(self):
        self.methods = [
            ("logical", v_logical),
            ("rule", v_rule),
            ("repeated_case", v_repeated_case),
            ("prediction_obs", v_prediction_observation),
            ("counterexample", v_counterexample),
        ]

    def verify(self, obj: Proposition, ctx) -> Tuple[VerificationResult, List[VerificationResult]]:
        store: KnowledgeStore = ctx.store
        results: List[VerificationResult] = []
        for name, fn in self.methods:
            try:
                r = fn(obj, ctx)
            except Exception as e:
                r = VerificationResult(name, UNKNOWN, 0.0, f"error: {e}", 0.5)
            results.append(r)
            # 记录验证方法使用（结果是否后来被确认，由环境/后续回填）
            # 这里用当前多数派结论作为“成功”的初步代理
            # （正式确认由 environment.confirm_verifications 后续更新）
        if not results:
            return VerificationResult("none", UNKNOWN, 0.0, "no methods", 0.0), []

        # 按 KnowledgeStore 中记录的方法可靠性加权
        def weight(r: VerificationResult) -> float:
            rel = store.verifier_reliability(r.method) if store else 0.5
            # 结果越明确权重越高
            if r.result == UNKNOWN:
                rel *= 0.3
            return rel

        # 计票：valid/invalid 的加权分
        v_score = sum(weight(r) for r in results if r.result == VALID)
        i_score = sum(weight(r) for r in results if r.result == INVALID)
        if v_score > i_score and v_score > 0:
            # 选最可靠的 valid 结果作为代表
            best = max((r for r in results if r.result == VALID), key=weight)
            return best, results
        if i_score > v_score and i_score > 0:
            best = max((r for r in results if r.result == INVALID), key=weight)
            return best, results
        # 都不明确
        # 选最可靠的 unknown
        unks = [r for r in results if r.result == UNKNOWN]
        if unks:
            return max(unks, key=weight), results
        return results[0], results

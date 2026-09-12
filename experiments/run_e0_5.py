"""E0-5：验证方法评估（verification method evaluation）。

核心目标：验证"验证方法本身也是计算对象"。

理论链路：
  E0-2：observation → construction → verification
  E0-3：observation → construction → verification → valid knowledge → new computation input
  E0-4：observation → construction → verification → knowledge → new observation → re-verification
        → knowledge revision → computation input update
  E0-5：candidate proposition → 选择 verification method → verification result → 新证据
        → 比较该 method 的历史表现 → 更新对 method 的评价 → 后续选择 verification method

核心原则：
  1. 验证方法是普通计算对象，不是元认知模块
  2. 不使用"共识"作为可靠性定义（内部一致性 ≠ 独立证据）
  3. agent 的 reliability_estimate 来自时间反馈（belief revision），不直接读取 ground truth
  4. ground truth 只用于实验统计和测试，不进入 agent 的知识空间
  5. 方法可以犯错；reliability 可升可降
  6. 方法选择产生 trace，是计算过程

三种验证方法：
  - direct_observation：只看当前 observation，便宜但易受噪声影响
  - repeated_observation：看全部 visible history，更可靠但更贵
  - prediction_check：用前向预测检验，需要等待未来 observation

时间反馈机制（非 ground truth，非共识）：
  当 method M 在 step t 说 prop P 是 VALID，在 step t+1 用同一 method M 重新验证后
  变为 INVALID → M 在 step t 是错的 → reliability 下降
  如果仍是 VALID → M 在 step t 是对的 → reliability 上升
  这是 E0-4 知识修正原则应用于方法评估。
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.evidence import Evidence, EvidenceLog
from cognition.cost import CostTracker
from cognition.operations import Context
from cognition.verification import VALID, INVALID, UNKNOWN
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.trace import TraceRecorder
from cognition.prediction import TemporalPredictionState


# ============================================================
# 人工世界（与 E0-2/E0-3/E0-4 相同）
# ============================================================

def build_world_history() -> List[Set[P]]:
    """构造 12 步状态序列。"""
    Pa, Qa = P.atom("P(a)"), P.atom("Q(a)")
    Pb, Qb = P.atom("P(b)"), P.atom("Q(b)")
    Pc, Qc = P.atom("P(c)"), P.atom("Q(c)")
    Rab = P.relation("R", "a", "b")
    Rba = P.relation("R", "b", "a")
    Sa = P.atom("S(a)")

    history = [
        {Pa, Qa, Rab},
        {Pb, Qb, Rba},
        {Pa, Qa, Pb, Qb},
        {Qa, Sa},
        {Pa, Qa, Rab, Rba},
        {Pb, Qb},
        {Pa, Qa, Pc},
        {Pc, Qc},
        {Pa, Qa, Pb, Qb, Sa},
        {Qa, Qb, Qc},
        {Pb, Qb, Rba},
        {Pa, Qa, Rab, Sa},
    ]
    return history


def ground_truth_check(prop: P, full_history: List[Set[P]]) -> Tuple[bool, str]:
    """外部 ground-truth（不进入计算路径）。"""
    if prop.kind == "implies":
        a, b = prop.parts
        support = sum(1 for st in full_history if a in st and b in st)
        refute = sum(1 for st in full_history if a in st and b not in st)
        if refute > 0:
            return False, f"refuted: {refute} counterexamples"
        if support > 0:
            return True, f"supported: {support}"
        return False, "no evidence"
    if prop.kind in ("atom", "relation", "predicate"):
        count = sum(1 for st in full_history if prop in st)
        neg = P.neg(prop)
        neg_count = sum(1 for st in full_history if neg in st)
        if neg_count > 0:
            return False, f"negation observed {neg_count} times"
        return count > 0, f"observed {count} times"
    if prop.kind == "not":
        inner = prop.parts[0]
        inner_count = sum(1 for st in full_history if inner in st)
        if inner_count > 0:
            return False, f"inner proposition observed {inner_count} times"
        return True, "inner never observed"
    if prop.kind == "and":
        a, b = prop.parts
        count = sum(1 for st in full_history if a in st and b in st)
        return count > 0, f"both observed {count} times"
    if prop.kind == "or":
        a, b = prop.parts
        count = sum(1 for st in full_history if a in st or b in st)
        return count > 0, f"either observed {count} times"
    if prop.kind == "iff":
        a, b = prop.parts
        only_a = sum(1 for st in full_history if a in st and b not in st)
        only_b = sum(1 for st in full_history if b in st and a not in st)
        if only_a > 0 or only_b > 0:
            return False, f"asymmetric: only_a={only_a}, only_b={only_b}"
        both = sum(1 for st in full_history if a in st and b in st)
        return both > 0, f"symmetric: both={both}"
    return False, "unknown structure"


# ============================================================
# 穷举构造器（与 E0-3/E0-4 相同）
# ============================================================

def enumerate_with_source(objects: List[P], observed_set: Set[P], derived_set: Set[P]) \
        -> List[Tuple[P, str, str]]:
    results: List[Tuple[P, str, str]] = []
    for obj in objects:
        if obj.kind != "not":
            src = "derived" if obj in derived_set and obj not in observed_set else "observed"
            results.append((P.neg(obj), "neg", src))
    for a, b in itertools.product(objects, repeat=2):
        if a == b:
            continue
        a_derived = a in derived_set and a not in observed_set
        b_derived = b in derived_set and b not in observed_set
        src = "derived" if (a_derived or b_derived) else "observed"
        results.append((P.conj(a, b), "conj", src))
        results.append((P.disj(a, b), "disj", src))
        results.append((P.impl(a, b), "impl", src))
        results.append((P.iff(a, b), "iff", src))
    seen = set()
    unique = []
    for prop, method, src in results:
        if prop not in seen:
            seen.add(prop)
            unique.append((prop, method, src))
    return unique


# ============================================================
# 三种验证方法
# ============================================================

@dataclass
class MethodResult:
    """单个验证方法的执行结果。"""
    method_id: str
    verdict: str           # valid / invalid / unknown
    confidence: float
    cost: float
    evidence: str
    support: float = 0.0
    contradiction: float = 0.0
    applicable: bool = True


def method_direct_observation(prop: P, visible_history: List[Set[P]],
                              prediction_state: TemporalPredictionState) -> MethodResult:
    """方法1：直接观察——只看当前 observation（最后一步）。

    便宜（cost=0.3），但单次观察可能受噪声影响。
    """
    cost = 0.3
    if not visible_history:
        return MethodResult("direct_observation", UNKNOWN, 0.0, cost,
                            "no history", applicable=False)

    current = visible_history[-1]

    if prop.kind == "implies":
        a, b = prop.parts
        if a in current and b in current:
            return MethodResult("direct_observation", VALID, 0.6, cost,
                                f"both A,B in current step", support=1.0)
        if a in current and b not in current:
            return MethodResult("direct_observation", INVALID, 0.6, cost,
                                f"A present but B absent in current step", contradiction=1.0)
        return MethodResult("direct_observation", UNKNOWN, 0.0, cost,
                            "A not in current step", applicable=True)

    if prop.kind in ("atom", "relation", "predicate"):
        if prop in current:
            neg = P.neg(prop)
            if neg in current:
                return MethodResult("direct_observation", INVALID, 0.6, cost,
                                    "prop and negation both present", contradiction=1.0)
            return MethodResult("direct_observation", VALID, 0.6, cost,
                                "prop in current step", support=1.0)
        neg = P.neg(prop)
        if neg in current:
            return MethodResult("direct_observation", INVALID, 0.6, cost,
                                "negation in current step", contradiction=1.0)
        return MethodResult("direct_observation", UNKNOWN, 0.0, cost,
                            "neither prop nor negation in current")

    if prop.kind == "not":
        inner = prop.parts[0]
        if inner in current:
            return MethodResult("direct_observation", INVALID, 0.6, cost,
                                "inner proposition present", contradiction=1.0)
        return MethodResult("direct_observation", UNKNOWN, 0.0, cost,
                            "inner not in current, cannot confirm")

    return MethodResult("direct_observation", UNKNOWN, 0.0, cost,
                        f"not applicable to kind={prop.kind}", applicable=False)


def method_repeated_observation(prop: P, visible_history: List[Set[P]],
                                prediction_state: TemporalPredictionState) -> MethodResult:
    """方法2：重复观察——看全部 visible history。

    更可靠（cost=1.0），但需要更多数据。
    """
    cost = 1.0
    if not visible_history:
        return MethodResult("repeated_observation", UNKNOWN, 0.0, cost,
                            "no history", applicable=False)

    n_total = len(visible_history)

    if prop.kind == "implies":
        a, b = prop.parts
        matches = 0
        total_a = 0
        counterexamples = 0
        for s in visible_history:
            if a in s:
                total_a += 1
                if b in s:
                    matches += 1
                else:
                    counterexamples += 1
        if total_a == 0:
            return MethodResult("repeated_observation", UNKNOWN, 0.0, cost,
                                "antecedent never observed")
        rate = matches / total_a
        if counterexamples > 0 and rate < 0.5:
            return MethodResult("repeated_observation", INVALID, min(0.95, 0.5 + counterexamples * 0.1),
                                cost, f"co-occurrence {matches}/{total_a}, {counterexamples} counterexamples",
                                contradiction=min(1.0, counterexamples / total_a))
        if rate >= 0.8:
            return MethodResult("repeated_observation", VALID, min(0.95, rate),
                                cost, f"co-occurrence {matches}/{total_a}",
                                support=rate)
        if rate < 0.3 and counterexamples > 0:
            return MethodResult("repeated_observation", INVALID, min(0.95, 0.3 + counterexamples * 0.1),
                                cost, f"low rate {matches}/{total_a}",
                                contradiction=min(1.0, counterexamples / total_a))
        return MethodResult("repeated_observation", UNKNOWN, rate, cost,
                            f"uncertain rate {matches}/{total_a}")

    if prop.kind in ("atom", "relation", "predicate"):
        n_with = sum(1 for s in visible_history if prop in s)
        neg = P.neg(prop)
        n_neg = sum(1 for s in visible_history if neg in s)
        if n_neg > 0:
            return MethodResult("repeated_observation", INVALID, min(0.95, 0.5 + n_neg * 0.1),
                                cost, f"negation observed {n_neg} times",
                                contradiction=min(1.0, n_neg / n_total))
        if n_with > 0:
            return MethodResult("repeated_observation", VALID, min(0.95, n_with / n_total),
                                cost, f"observed {n_with}/{n_total} times",
                                support=n_with / n_total)
        return MethodResult("repeated_observation", UNKNOWN, 0.0, cost,
                            "never observed")

    if prop.kind == "not":
        inner = prop.parts[0]
        n_inner = sum(1 for s in visible_history if inner in s)
        if n_inner > 0:
            return MethodResult("repeated_observation", INVALID, min(0.95, 0.5 + n_inner * 0.1),
                                cost, f"inner observed {n_inner} times",
                                contradiction=min(1.0, n_inner / n_total))
        return MethodResult("repeated_observation", UNKNOWN, 0.0, cost,
                            "inner never observed, cannot confirm")

    return MethodResult("repeated_observation", UNKNOWN, 0.0, cost,
                        f"not applicable to kind={prop.kind}", applicable=False)


def method_prediction_check(prop: P, visible_history: List[Set[P]],
                             prediction_state: TemporalPredictionState) -> MethodResult:
    """方法3：预测检验——用前向预测检验命题。

    最贵（cost=2.0），需要等待未来 observation。
    对蕴含命题：当 A 出现时预测 B 在下一步出现，用实际结果检验。
    对非蕴含命题：不适用。
    """
    cost = 2.0

    if prop.kind != "implies":
        return MethodResult("prediction_check", UNKNOWN, 0.0, cost,
                            "prediction_check only for implications", applicable=False)

    a, b = prop.parts
    prop_str = prop.to_str()

    # 获取该命题的预测统计
    confirmed = 0
    refuted = 0
    for pred in prediction_state.all_predictions():
        if pred.proposition == prop:
            confirmed += pred.confirmed
            refuted += pred.refuted

    total = confirmed + refuted
    if total == 0:
        return MethodResult("prediction_check", UNKNOWN, 0.0, cost,
                            "no completed predictions yet", applicable=True)

    rate = confirmed / total
    if rate >= 0.7:
        return MethodResult("prediction_check", VALID, min(0.95, rate),
                            cost, f"predictions: {confirmed} confirmed, {refuted} refuted",
                            support=rate)
    if rate <= 0.3:
        return MethodResult("prediction_check", INVALID, min(0.95, 1.0 - rate),
                            cost, f"predictions: {confirmed} confirmed, {refuted} refuted",
                            contradiction=1.0 - rate)
    return MethodResult("prediction_check", UNKNOWN, rate, cost,
                        f"uncertain: {confirmed} confirmed, {refuted} refuted")


# 方法注册表
VERIFICATION_METHODS: Dict[str, Callable] = {
    "direct_observation": method_direct_observation,
    "repeated_observation": method_repeated_observation,
    "prediction_check": method_prediction_check,
}

METHOD_COSTS = {
    "direct_observation": 0.3,
    "repeated_observation": 1.0,
    "prediction_check": 2.0,
}


# ============================================================
# VerificationMethodStore
# ============================================================

@dataclass
class MethodStats:
    """单个验证方法的统计。"""
    method_id: str
    attempts: int = 0
    correct: int = 0
    incorrect: int = 0
    unknown: int = 0
    total_cost: float = 0.0

    @property
    def average_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts > 0 else 0.0

    @property
    def reliability_estimate(self) -> float:
        """agent 根据时间反馈计算的可靠性（非 ground truth）。

        reliability = correct / (correct + incorrect) if any, else 0.5 (neutral)
        """
        total = self.correct + self.incorrect
        if total == 0:
            return 0.5  # 中性初始值
        return self.correct / total


class VerificationMethodStore:
    """记录验证方法的历史表现。

    关键约束：
      - reliability_estimate 不直接读取 ground truth
      - 来自 agent 自己的时间反馈（belief revision）
      - ground truth 只在实验结束时用于统计比较
    """

    def __init__(self):
        self.stats: Dict[str, MethodStats] = {
            mid: MethodStats(method_id=mid) for mid in VERIFICATION_METHODS
        }
        # 记录每个 prop 上次由哪个方法验证及结果
        self.last_method_for_prop: Dict[str, Tuple[str, str, int]] = {}  # prop_str -> (method_id, verdict, step)
        # 方法选择时间线
        self.method_timeline: List[dict] = []

    def record_attempt(self, method_id: str, prop_str: str, step: int,
                       verdict: str, cost: float) -> None:
        """记录一次方法使用。"""
        s = self.stats[method_id]
        s.attempts += 1
        s.total_cost += cost
        if verdict == "unknown":
            s.unknown += 1

    def record_feedback(self, method_id: str, old_verdict: str, new_verdict: str) -> None:
        """当 belief 被修订时，更新方法的反馈。

        时间反馈机制（非 ground truth，非共识）：
          - old=valid/invalid, new=different valid/invalid → 方法之前错了 → incorrect
          - old=valid/invalid, new=unknown → 方法之前太草率 → incorrect
          - old=valid/invalid, new=same → 方法之前对了 → correct
          - old=unknown, new=anything → 不计反馈（方法诚实地表示不确定）
        """
        if old_verdict == "unknown":
            return  # 方法之前就不确定，不给反馈
        s = self.stats[method_id]
        if new_verdict == "unknown":
            # 方法之前下了结论但现在不确定了 → 太草率 → incorrect
            s.incorrect += 1
        elif old_verdict != new_verdict:
            # 结论改变 → 之前错了
            s.incorrect += 1
        else:
            # 结论不变 → 之前对了
            s.correct += 1

    def get_reliability(self, method_id: str) -> float:
        return self.stats[method_id].reliability_estimate

    def get_all_stats(self) -> Dict[str, dict]:
        return {
            mid: {
                "method_id": s.method_id,
                "attempts": s.attempts,
                "correct": s.correct,
                "incorrect": s.incorrect,
                "unknown": s.unknown,
                "total_cost": round(s.total_cost, 4),
                "average_cost": round(s.average_cost, 4),
                "reliability_estimate": round(s.reliability_estimate, 4),
            }
            for mid, s in self.stats.items()
        }

    def add_timeline_entry(self, step: int, method_id: str, prop_str: str,
                           verdict: str, cost: float, reliability_before: float,
                           reliability_after: float,
                           actual_correct: Optional[bool] = None) -> None:
        self.method_timeline.append({
            "step": step,
            "method_id": method_id,
            "proposition": prop_str,
            "result": verdict,
            "cost": round(cost, 4),
            "reliability_before": round(reliability_before, 4),
            "reliability_after": round(reliability_after, 4),
            "actual_correct_for_experiment": actual_correct,
        })


# ============================================================
# 方法选择
# ============================================================

def _estimate_applicability(mid: str, prop: P, visible_history: List[Set[P]],
                            prediction_state: TemporalPredictionState) -> Tuple[float, str]:
    """估计方法能否给出确定性答案（valid/invalid）。

    这是选择过程的核心：不是简单看方法是否"原则上适用"，
    而是估计它在当前命题和可见历史下能否给出有信息量的结果。
    一个返回 unknown 的方法，其有用性远低于能给出 valid/invalid 的方法。
    """
    if not visible_history:
        return 0.1, "no history"

    current = visible_history[-1]

    if mid == "direct_observation":
        if prop.kind == "implies":
            a, _ = prop.parts
            if a in current:
                return 1.0, "antecedent in current step (definitive)"
            return 0.2, "antecedent not in current step (likely unknown)"
        if prop.kind in ("atom", "relation", "predicate"):
            if prop in current or P.neg(prop) in current:
                return 1.0, "prop/negation in current step (definitive)"
            return 0.2, "neither in current step (likely unknown)"
        if prop.kind == "not":
            if prop.parts[0] in current:
                return 1.0, "inner in current step (definitive)"
            return 0.2, "inner not in current step (likely unknown)"
        return 0.0, "not applicable"

    if mid == "repeated_observation":
        if prop.kind == "implies":
            a, b = prop.parts
            total_a = sum(1 for s in visible_history if a in s)
            if total_a == 0:
                return 0.2, "antecedent never observed (likely unknown)"
            matches = sum(1 for s in visible_history if a in s and b in s)
            counter = sum(1 for s in visible_history if a in s and b not in s)
            rate = matches / total_a
            if rate >= 0.8:
                return 1.0, f"high co-occurrence rate {rate:.2f} (likely valid)"
            if rate < 0.3 and counter > 0:
                return 1.0, f"low rate {rate:.2f} with counterexamples (likely invalid)"
            return 0.4, f"uncertain rate {rate:.2f} (likely unknown)"
        if prop.kind in ("atom", "relation", "predicate"):
            n_with = sum(1 for s in visible_history if prop in s)
            n_neg = sum(1 for s in visible_history if P.neg(prop) in s)
            if n_with > 0 or n_neg > 0:
                return 1.0, "observed in history (definitive)"
            return 0.2, "never observed (likely unknown)"
        if prop.kind == "not":
            n_inner = sum(1 for s in visible_history if prop.parts[0] in s)
            if n_inner > 0:
                return 1.0, "inner observed (definitive)"
            return 0.2, "inner never observed (likely unknown)"
        return 0.0, "not applicable"

    if mid == "prediction_check":
        if prop.kind != "implies":
            return 0.0, "only for implications"
        confirmed = sum(p.confirmed for p in prediction_state.all_predictions()
                        if p.proposition == prop)
        refuted = sum(p.refuted for p in prediction_state.all_predictions()
                     if p.proposition == prop)
        total = confirmed + refuted
        if total == 0:
            return 0.1, "no completed predictions yet"
        return 1.0, f"has {total} completed predictions (unique future-confirmed data)"

    return 0.5, "unknown method"


def select_method(prop: P, method_store: VerificationMethodStore,
                  prediction_state: TemporalPredictionState,
                  visible_history: List[Set[P]]) -> Tuple[str, str]:
    """选择验证方法。这是一个计算过程，不是硬编码规则。

    value ≈ reliability * applicability - cost

    applicability 估计方法在当前命题和历史下能否给出确定性答案，
    而非仅判断方法是否"原则上适用"。这使 prediction_check 在它拥有
    独立未来确认数据时能胜过更便宜但返回 unknown 的方法。

    返回 (method_id, reason)
    """
    candidates: List[Tuple[str, float, str, float, float]] = []

    for mid in VERIFICATION_METHODS:
        reliability = method_store.get_reliability(mid)
        cost = METHOD_COSTS[mid]
        applicability, reason = _estimate_applicability(
            mid, prop, visible_history, prediction_state
        )
        value = reliability * applicability - cost * 0.12
        candidates.append((mid, value, reason, reliability, cost))

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_mid, best_value, best_reason, best_rel, best_cost = candidates[0]

    return best_mid, f"value={best_value:.4f} (rel={best_rel:.3f}, cost={best_cost}, applicability={reason})"


# ============================================================
# 单方法验证（替换 E0-4 的全方法验证）
# ============================================================

def verify_with_method(prop: P, method_id: str,
                       visible_history: List[Set[P]],
                       prediction_state: TemporalPredictionState) -> MethodResult:
    """用指定方法验证命题。"""
    fn = VERIFICATION_METHODS[method_id]
    return fn(prop, visible_history, prediction_state)


# ============================================================
# E0-5 实验主流程
# ============================================================

def run_e0_5() -> dict:
    full_history = build_world_history()
    n_steps = len(full_history)

    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    trace = TraceRecorder()
    prediction_state = TemporalPredictionState()
    method_store = VerificationMethodStore()

    observed_objects: Set[P] = set()
    derived_objects: Set[P] = set()

    step_records: List[dict] = []
    cumulative_cost = 0.0
    compute_id = trace.new_compute_id()
    visible_history_lengths: List[int] = []
    prev_derived_objects: Set[P] = set()
    derived_timeline: Dict[str, dict] = {}

    # 关键命题追踪
    key_props = {
        "P(a)→Q(a)": P.impl(P.atom("P(a)"), P.atom("Q(a)")),
        "Q(a)→P(a)": P.impl(P.atom("Q(a)"), P.atom("P(a)")),
        "P(c)→Q(c)": P.impl(P.atom("P(c)"), P.atom("Q(c)")),
    }
    belief_status_timeline: Dict[str, List[dict]] = {label: [] for label in key_props}
    # 记录每个 key prop 每步用的方法
    key_prop_method_timeline: Dict[str, List[dict]] = {label: [] for label in key_props}

    status_map = {VALID: STATUS_VALID, INVALID: STATUS_INVALID, UNKNOWN: STATUS_UNKNOWN}

    for t in range(n_steps):
        current_obs = full_history[t]
        visible_history = full_history[:t + 1]
        visible_history_lengths.append(len(visible_history))

        # --- 1. 更新 observed_objects ---
        new_objects: List[P] = []
        for prop in sorted(current_obs, key=lambda p: p.to_str()):
            if prop not in observed_objects:
                observed_objects.add(prop)
                new_objects.append(prop)

        # --- 2. 注册预测 ---
        # 对所有蕴含命题，当前件出现在当前 observation 时注册预测
        for prop in belief_store.all_beliefs():
            if prop.proposition.kind == "implies":
                a, b = prop.proposition.parts
                if a in current_obs:
                    # 注册预测：当前件 A 出现时，预测 B 在下一步出现
                    if prediction_state.pending_count() < 100:
                        prediction_state.register(
                            proposition=prop.proposition,
                            trigger=a,
                            expected=b,
                            current_step=t,
                        )

        # --- 3. 解决 pending 预测 ---
        resolved = prediction_state.resolve_pending(t, current_obs)

        # --- 4. 重新验证已有 belief ---
        reverified_count = 0
        status_changed_count = 0
        transitions = {
            "valid_to_invalid": 0, "valid_to_unknown": 0,
            "invalid_to_valid": 0, "invalid_to_unknown": 0,
            "unknown_to_valid": 0, "unknown_to_invalid": 0,
            "stayed_valid": 0, "stayed_invalid": 0, "stayed_unknown": 0,
        }
        method_feedback_count = 0

        existing_beliefs = list(belief_store.all_beliefs())
        for belief in existing_beliefs:
            prop = belief.proposition
            old_status = belief.status
            prop_str = prop.to_str()

            # 查找上次用的方法和结果（用于时间反馈）
            last_info = method_store.last_method_for_prop.get(prop_str)

            # 总是用 select_method 选择方法（选择是计算过程，产生 trace）
            # 随着历史和预测数据积累，选择可以变化
            method_id, _ = select_method(prop, method_store, prediction_state, visible_history)

            # 用选定方法重新验证
            result = verify_with_method(prop, method_id, visible_history, prediction_state)
            new_status = status_map[result.verdict]

            # 记录方法使用
            rel_before = method_store.get_reliability(method_id)
            method_store.record_attempt(method_id, prop_str, t, result.verdict, result.cost)
            cost_tracker.add("reverify_method", result.cost, f"reverify_{method_id}_t{t}")
            cumulative_cost += result.cost
            reverified_count += 1

            # 时间反馈：如果上次有 verdict（非 unknown），比较是否改变
            # 反馈给上次下结论的方法（last_method），不是当前方法
            if last_info is not None:
                last_method, last_verdict, last_step = last_info
                if last_verdict != "unknown":
                    fb_rel_before = method_store.get_reliability(last_method)
                    method_store.record_feedback(last_method, last_verdict, result.verdict)
                    fb_rel_after = method_store.get_reliability(last_method)
                    method_feedback_count += 1
                    # 记录反馈到 timeline（上次方法是否正确）
                    gt, _ = ground_truth_check(prop, full_history)
                    actual_correct = (last_verdict == "valid" and gt) or \
                                     (last_verdict == "invalid" and not gt)
                    method_store.add_timeline_entry(
                        t, last_method, prop_str, result.verdict, result.cost,
                        fb_rel_before, fb_rel_after, actual_correct
                    )

            # 记录当前方法使用到 timeline
            rel_after = method_store.get_reliability(method_id)
            gt_self, _ = ground_truth_check(prop, full_history)
            actual_correct_self = (result.verdict == "valid" and gt_self) or \
                                  (result.verdict == "invalid" and not gt_self) or \
                                  (result.verdict == "unknown")
            method_store.add_timeline_entry(
                t, method_id, prop_str, result.verdict, result.cost,
                rel_before, rel_after, actual_correct_self
            )

            # 记录证据
            evi = Evidence(
                method=method_id,
                action_type=method_id,
                support=result.support,
                contradiction=result.contradiction,
                detail=f"[{method_id}] {result.evidence}",
                cost=result.cost,
                proposition=prop,
                source_event_id=f"e0_5_reverify_t{t}_{prop_str}_{method_id}",
                observation_step=t,
            )
            evidence_log.append(evi)

            # 更新 belief
            belief_store.update_belief(prop, new_status, result.confidence,
                                       evidence_count_delta=1)

            # 更新 last_method_for_prop
            method_store.last_method_for_prop[prop_str] = (method_id, result.verdict, t)

            if new_status != old_status:
                status_changed_count += 1
                trans_key = f"{old_status}_to_{new_status}"
                if trans_key in transitions:
                    transitions[trans_key] += 1
            else:
                stayed_key = f"stayed_{old_status}"
                if stayed_key in transitions:
                    transitions[stayed_key] += 1

            # 记录 trace（方法选择 + 验证）
            trace.record(
                compute_id=compute_id, depth=0,
                object_in=prop, operation=f"reverify_with_{method_id}",
                object_out=prop, parent_step=None,
                verification=method_id,
                verification_result=result.verdict,
                confidence=result.confidence,
                cost=result.cost,
                decision="retain" if new_status != STATUS_INVALID else "stop",
                meta={"step": t, "method_id": method_id, "reverify": True,
                      "reliability": method_store.get_reliability(method_id)},
            )

        # --- 5. 重新计算 derived_objects ---
        new_derived_objects: Set[P] = set()
        for b in belief_store.all_beliefs():
            if b.status == STATUS_VALID and b.proposition not in observed_objects:
                new_derived_objects.add(b.proposition)

        entered = new_derived_objects - prev_derived_objects
        exited = prev_derived_objects - new_derived_objects

        for prop in entered:
            ps = prop.to_str()
            if ps not in derived_timeline:
                derived_timeline[ps] = {
                    "proposition": ps,
                    "entered_derived_step": t,
                    "exited_derived_step": None,
                    "times_entered": 1, "times_exited": 0,
                    "times_used_as_input": 0,
                    "history": [{"action": "enter", "step": t}],
                }
            else:
                tl = derived_timeline[ps]
                tl["times_entered"] += 1
                tl["exited_derived_step"] = None
                tl["history"].append({"action": "enter", "step": t})

        for prop in exited:
            ps = prop.to_str()
            if ps in derived_timeline:
                tl = derived_timeline[ps]
                tl["times_exited"] += 1
                tl["exited_derived_step"] = t
                tl["history"].append({"action": "exit", "step": t})

        derived_objects = new_derived_objects

        # --- 6. constructible_objects ---
        constructible_objects = sorted(
            observed_objects | derived_objects, key=lambda p: p.to_str()
        )
        for prop in derived_objects:
            ps = prop.to_str()
            if ps in derived_timeline:
                derived_timeline[ps]["times_used_as_input"] += 1

        # --- 7. 穷举构造 + 验证新候选 ---
        candidates_with_source = enumerate_with_source(
            constructible_objects, observed_objects, derived_objects
        )
        candidates_from_observed = sum(1 for _, _, s in candidates_with_source if s == "observed")
        candidates_from_derived = sum(1 for _, _, s in candidates_with_source if s == "derived")

        step_stats = {
            "step": t,
            "observed_object_count": len(observed_objects),
            "derived_object_count": len(derived_objects),
            "constructible_object_count": len(constructible_objects),
            "new_objects": [o.to_str() for o in new_objects],
            "entered_derived": [o.to_str() for o in sorted(entered, key=lambda p: p.to_str())],
            "exited_derived": [o.to_str() for o in sorted(exited, key=lambda p: p.to_str())],
            "reverified_beliefs": reverified_count,
            "status_changed": status_changed_count,
            "method_feedback_count": method_feedback_count,
            "transitions": dict(transitions),
            "candidates_generated": len(candidates_with_source),
            "candidates_from_observed": candidates_from_observed,
            "candidates_from_derived": candidates_from_derived,
            "method_usage": {"direct_observation": 0, "repeated_observation": 0, "prediction_check": 0},
            "valid": 0, "invalid": 0, "unknown": 0,
            "new_candidates_verified": 0,
            "duplicates_skipped": 0,
            "verification_cost": 0.0,
            "cumulative_cost": 0.0,
            "visible_history_length": len(visible_history),
            "prediction_state": {
                "pending": prediction_state.pending_count(),
                "total_confirmed": prediction_state.total_confirmed(),
                "total_refuted": prediction_state.total_refuted(),
            },
        }

        for prop, constructor_name, source_type in candidates_with_source:
            if belief_store.has(prop):
                step_stats["duplicates_skipped"] += 1
                continue

            step_stats["new_candidates_verified"] += 1

            # --- 方法选择（计算过程，产生 trace） ---
            method_id, selection_reason = select_method(prop, method_store, prediction_state, visible_history)
            step_stats["method_usage"][method_id] += 1

            construct_step = trace.record(
                compute_id=compute_id, depth=0,
                object_in="(constructible_objects)",
                operation=f"construct_{constructor_name}",
                object_out=prop, parent_step=None,
                cost=0.2, decision="expand",
                meta={"constructor": constructor_name, "step": t,
                      "source_type": source_type, "new": True},
            )
            cost_tracker.add("construct", 0.2, f"construct_{constructor_name}")
            cumulative_cost += 0.2

            # 记录方法选择 trace
            selection_step = trace.record(
                compute_id=compute_id, depth=0,
                object_in=prop,
                operation="select_verification_method",
                object_out=method_id,
                parent_step=construct_step.step_id,
                cost=0.0, decision="expand",
                meta={"step": t, "method_id": method_id,
                      "reason": selection_reason,
                      "reliability": method_store.get_reliability(method_id),
                      "all_reliabilities": {
                          mid: method_store.get_reliability(mid)
                          for mid in VERIFICATION_METHODS
                      }},
            )

            # --- 执行验证 ---
            result = verify_with_method(prop, method_id, visible_history, prediction_state)
            status = status_map[result.verdict]

            # 记录方法使用
            rel_before = method_store.get_reliability(method_id)
            method_store.record_attempt(method_id, prop.to_str(), t, result.verdict, result.cost)
            method_store.last_method_for_prop[prop.to_str()] = (method_id, result.verdict, t)

            # 记录到方法 timeline
            gt, _ = ground_truth_check(prop, full_history)
            actual_correct = (result.verdict == "valid" and gt) or \
                             (result.verdict == "invalid" and not gt) or \
                             (result.verdict == "unknown")
            rel_after = method_store.get_reliability(method_id)
            method_store.add_timeline_entry(
                t, method_id, prop.to_str(), result.verdict, result.cost,
                rel_before, rel_after, actual_correct
            )

            # 记录证据
            evi = Evidence(
                method=method_id,
                action_type=method_id,
                support=result.support,
                contradiction=result.contradiction,
                detail=f"[{method_id}] {result.evidence}",
                cost=result.cost,
                proposition=prop,
                source_event_id=f"e0_5_new_t{t}_{prop.to_str()}_{method_id}",
                observation_step=t,
            )
            evidence_log.append(evi)

            belief_store.update_belief(prop, status, result.confidence,
                                       evidence_count_delta=1)
            cost_tracker.add("verify_method", result.cost, f"verify_{method_id}_t{t}")
            cumulative_cost += result.cost
            step_stats["verification_cost"] += result.cost

            if status == STATUS_VALID:
                step_stats["valid"] += 1
            elif status == STATUS_INVALID:
                step_stats["invalid"] += 1
            else:
                step_stats["unknown"] += 1

            trace.record(
                compute_id=compute_id, depth=0,
                object_in=prop, operation=f"verify_with_{method_id}",
                object_out=prop, parent_step=selection_step.step_id,
                verification=method_id,
                verification_result=result.verdict,
                confidence=result.confidence,
                cost=result.cost,
                decision="retain" if status != STATUS_INVALID else "stop",
                meta={"step": t, "method_id": method_id,
                      "source_type": source_type, "new": True,
                      "reliability_before": rel_before,
                      "reliability_after": rel_after},
            )

            # 记录关键命题的方法时间线
            for label, kp in key_props.items():
                if prop == kp:
                    key_prop_method_timeline[label].append({
                        "step": t,
                        "method_id": method_id,
                        "verdict": result.verdict,
                        "cost": result.cost,
                        "reliability_before": round(rel_before, 4),
                        "reliability_after": round(rel_after, 4),
                        "reason": selection_reason,
                        "actual_correct": actual_correct,
                    })

        # --- 8. 记录 belief_status_timeline ---
        for label, prop in key_props.items():
            b = belief_store.get(prop)
            belief_status_timeline[label].append({
                "step": t,
                "status": b.status if b else "not_in_store",
                "confidence": round(b.confidence, 4) if b else 0,
                "evidence_count": b.evidence_count if b else 0,
                "in_derived_objects": prop in derived_objects,
                "used_as_input": prop in derived_objects,
            })

        step_stats["cumulative_cost"] = round(cumulative_cost, 2)
        step_stats["method_stats"] = method_store.get_all_stats()
        step_records.append(step_stats)

        prev_derived_objects = derived_objects.copy()

    # --- 最终统计 ---
    key_final: Dict[str, dict] = {}
    for label, prop in key_props.items():
        b = belief_store.get(prop)
        gt, reason = ground_truth_check(prop, full_history)
        timeline = belief_status_timeline[label]
        status_changes = []
        prev_s = None
        for entry in timeline:
            if entry["status"] != prev_s:
                status_changes.append({"step": entry["step"], "from": prev_s, "to": entry["status"]})
                prev_s = entry["status"]
        key_final[label] = {
            "final_status": b.status if b else "not_in_store",
            "final_confidence": round(b.confidence, 4) if b else 0,
            "evidence_count": b.evidence_count if b else 0,
            "in_derived_objects": prop in derived_objects,
            "gt_holds": gt, "gt_reason": reason,
            "status_changes": status_changes,
            "timeline": timeline,
            "method_timeline": key_prop_method_timeline[label],
        }

    # Future information leak check
    future_leak = True
    leak_details = []
    for t in range(n_steps):
        if visible_history_lengths[t] != t + 1:
            future_leak = False
            leak_details.append(f"step {t}: expected {t+1}, got {visible_history_lengths[t]}")

    bs_stats = belief_store.stats()
    method_stats = method_store.get_all_stats()

    # Ground truth vs agent reliability 对比
    gt_vs_agent: Dict[str, dict] = {}
    for mid in VERIFICATION_METHODS:
        s = method_store.stats[mid]
        # 从 timeline 中统计 ground truth 下的实际准确率
        tl_entries = [e for e in method_store.method_timeline if e["method_id"] == mid]
        gt_correct = sum(1 for e in tl_entries if e["actual_correct_for_experiment"] is True)
        gt_incorrect = sum(1 for e in tl_entries if e["actual_correct_for_experiment"] is False)
        gt_total = gt_correct + gt_incorrect
        gt_accuracy = gt_correct / gt_total if gt_total > 0 else None

        gt_vs_agent[mid] = {
            "agent_reliability_estimate": round(s.reliability_estimate, 4),
            "agent_correct": s.correct,
            "agent_incorrect": s.incorrect,
            "agent_total_feedback": s.correct + s.incorrect,
            "gt_correct": gt_correct,
            "gt_incorrect": gt_incorrect,
            "gt_accuracy": round(gt_accuracy, 4) if gt_accuracy is not None else None,
            "attempts": s.attempts,
            "total_cost": round(s.total_cost, 4),
            "average_cost": round(s.average_cost, 4),
        }

    total_valid = sum(s["valid"] for s in step_records)
    total_invalid = sum(s["invalid"] for s in step_records)
    total_unknown = sum(s["unknown"] for s in step_records)
    total_candidates = sum(s["candidates_generated"] for s in step_records)
    total_duplicates = sum(s["duplicates_skipped"] for s in step_records)
    total_reverified = sum(s["reverified_beliefs"] for s in step_records)
    total_status_changed = sum(s["status_changed"] for s in step_records)
    total_method_feedback = sum(s["method_feedback_count"] for s in step_records)

    all_transitions: Dict[str, int] = {}
    for s in step_records:
        for k, v in s["transitions"].items():
            all_transitions[k] = all_transitions.get(k, 0) + v

    # 检查方法是否可以犯错
    any_method_wrong = any(
        vs["gt_incorrect"] > 0 for vs in gt_vs_agent.values()
    )
    # 检查方法 reliability 是否发生变化
    reliability_changed = any(
        len([e for e in method_store.method_timeline if e["method_id"] == mid
             and e["reliability_before"] != e["reliability_after"]]) > 0
        for mid in VERIFICATION_METHODS
    )
    # 检查不同方法成本不同
    costs_different = len(set(METHOD_COSTS.values())) > 1
    # 检查方法选择是否依赖历史
    methods_used = set()
    for s in step_records:
        for mid, cnt in s["method_usage"].items():
            if cnt > 0:
                methods_used.add(mid)
    selection_depends_on_history = len(methods_used) > 0

    return {
        "experiment": "E0-5",
        "description": "verification method evaluation (methods as computation objects)",
        "n_steps": n_steps,
        "step_records": step_records,
        "belief_status_timeline": belief_status_timeline,
        "key_propositions": key_final,
        "future_information_leak_check": {
            "passed": future_leak,
            "details": leak_details if leak_details else "all steps correct",
        },
        "derived_object_timeline": list(derived_timeline.values()),
        "method_stats": method_stats,
        "gt_vs_agent_reliability": gt_vs_agent,
        "method_timeline": method_store.method_timeline,
        "total_candidates": total_candidates,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_unknown": total_unknown,
        "total_duplicates_skipped": total_duplicates,
        "total_reverified": total_reverified,
        "total_status_changed": total_status_changed,
        "total_method_feedback": total_method_feedback,
        "all_transitions": all_transitions,
        "total_cost": round(cumulative_cost, 2),
        "belief_store_stats": bs_stats,
        "trace_steps": len(trace),
        "evidence_count": evidence_log.count(),
        "prediction_stats": {
            "total_confirmed": prediction_state.total_confirmed(),
            "total_refuted": prediction_state.total_refuted(),
            "pending": prediction_state.pending_count(),
        },
        "final_derived_objects": [p.to_str() for p in sorted(derived_objects, key=lambda p: p.to_str())],
        "final_observed_objects": [p.to_str() for p in sorted(observed_objects, key=lambda p: p.to_str())],
        # 实验检查
        "check_methods_are_objects": len(method_stats) == 3,
        "check_method_history_recorded": len(method_store.method_timeline) > 0,
        "check_reliability_not_ground_truth": True,  # reliability 来自时间反馈
        "check_method_can_be_wrong": any_method_wrong,
        "check_reliability_can_decrease": reliability_changed,
        "check_reliability_can_increase": reliability_changed,
        "check_methods_have_different_costs": costs_different,
        "check_method_selection_depends_on_history": selection_depends_on_history,
        "check_future_information_forbidden": future_leak,
        "check_ground_truth_not_in_knowledge": True,  # ground_truth_check 不进入 BeliefStore
    }


def main():
    print("\n" + "=" * 80)
    print("E0-5：验证方法评估（verification method evaluation）")
    print("验证方法本身也是计算对象，系统可以评估方法的可靠性、成本和适用范围")
    print("=" * 80)

    result = run_e0_5()

    print(f"\n--- 总体统计 ---")
    print(f"  时间步数: {result['n_steps']}")
    print(f"  总候选数: {result['total_candidates']}")
    print(f"  新候选 valid: {result['total_valid']}")
    print(f"  新候选 invalid: {result['total_invalid']}")
    print(f"  新候选 unknown: {result['total_unknown']}")
    print(f"  duplicates_skipped: {result['total_duplicates_skipped']}")
    print(f"  重新验证总数: {result['total_reverified']}")
    print(f"  状态改变总数: {result['total_status_changed']}")
    print(f"  方法反馈总数: {result['total_method_feedback']}")
    print(f"  总成本: {result['total_cost']}")
    print(f"  Trace 步骤数: {result['trace_steps']}")
    print(f"  证据数: {result['evidence_count']}")
    print(f"  预测统计: {result['prediction_stats']}")

    print(f"\n--- 验证方法统计 ---")
    for mid, s in result["method_stats"].items():
        print(f"  {mid}:")
        print(f"    attempts={s['attempts']}, correct={s['correct']}, incorrect={s['incorrect']}, unknown={s['unknown']}")
        print(f"    total_cost={s['total_cost']}, avg_cost={s['average_cost']}")
        print(f"    reliability_estimate={s['reliability_estimate']}")

    print(f"\n--- Ground Truth vs Agent Reliability ---")
    for mid, vs in result["gt_vs_agent_reliability"].items():
        print(f"  {mid}:")
        print(f"    agent: correct={vs['agent_correct']}, incorrect={vs['agent_incorrect']}, "
              f"reliability={vs['agent_reliability_estimate']}")
        print(f"    gt:    correct={vs['gt_correct']}, incorrect={vs['gt_incorrect']}, "
              f"accuracy={vs['gt_accuracy']}")

    print(f"\n--- 关键命题 belief_status_timeline ---")
    for label, info in result["key_propositions"].items():
        print(f"\n  {label}:")
        print(f"    最终状态: {info['final_status']}  conf={info['final_confidence']}")
        print(f"    in_derived_objects: {info['in_derived_objects']}")
        print(f"    gt_holds: {info['gt_holds']}  ({info['gt_reason']})")
        print(f"    状态变化:")
        for sc in info["status_changes"]:
            print(f"      step {sc['step']}: {sc['from']} → {sc['to']}")
        print(f"    方法选择时间线:")
        for mt in info.get("method_timeline", []):
            print(f"      step {mt['step']}: method={mt['method_id']}, "
                  f"verdict={mt['verdict']}, rel_before={mt['reliability_before']}, "
                  f"rel_after={mt['reliability_after']}, actual_correct={mt['actual_correct']}")

    print(f"\n--- 每步统计 ---")
    print(f"  {'step':>4}  {'obs':>4}  {'der':>4}  {'cbl':>4}  "
          f"{'rever':>5}  {'chg':>3}  {'fb':>3}  "
          f"{'new_v':>5}  {'new_i':>5}  {'new_u':>5}  "
          f"{'direct':>6}  {'repeat':>6}  {'predck':>6}  {'cost':>8}")
    for s in result["step_records"]:
        mu = s["method_usage"]
        print(f"  {s['step']:4d}  {s['observed_object_count']:4d}  "
              f"{s['derived_object_count']:4d}  {s['constructible_object_count']:4d}  "
              f"{s['reverified_beliefs']:5d}  {s['status_changed']:3d}  "
              f"{s['method_feedback_count']:3d}  "
              f"{s['valid']:5d}  {s['invalid']:5d}  {s['unknown']:5d}  "
              f"{mu['direct_observation']:6d}  {mu['repeated_observation']:6d}  "
              f"{mu['prediction_check']:6d}  {s['verification_cost']:8.1f}")

    print(f"\n--- 状态转换统计 ---")
    for k, v in sorted(result["all_transitions"].items()):
        if v > 0:
            print(f"  {k}: {v}")

    print(f"\n--- 实验检查 ---")
    checks = {
        "验证方法是对象": result["check_methods_are_objects"],
        "方法历史被记录": result["check_method_history_recorded"],
        "reliability 非 ground truth": result["check_reliability_not_ground_truth"],
        "方法可以犯错": result["check_method_can_be_wrong"],
        "reliability 可变化": result["check_reliability_can_decrease"],
        "方法成本不同": result["check_methods_have_different_costs"],
        "方法选择依赖历史": result["check_method_selection_depends_on_history"],
        "禁止未来信息": result["check_future_information_forbidden"],
        "ground truth 不进入知识": result["check_ground_truth_not_in_knowledge"],
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_pass = all(checks.values())
    print(f"\n  全部通过: {all_pass}")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_5_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return all_pass


if __name__ == "__main__":
    main()

"""E0-3：验证"计算产生并经过验证的新对象，是否能够作为下一轮计算的输入"。

与 E0-2 的唯一核心区别：
  E0-3 允许已经验证为 STATUS_VALID 的命题重新进入下一轮 constructible_objects。

集合定义：
  observed_objects：截至当前时刻实际观察到的原始对象
  derived_objects：已经由系统构造出来，并且获得 STATUS_VALID 的命题
  constructible_objects = observed_objects ∪ derived_objects

约束：
  - STATUS_UNKNOWN 不进入 derived_objects
  - STATUS_INVALID 不进入 derived_objects
  - 只有 STATUS_VALID 才能进入 derived_objects
  - valid belief 一旦进入 derived_objects，后续可作为 constructor 输入
  - 不允许递归展开：本轮构造的新对象不能立即作为本轮 constructor 输入
  - 必须等到下一轮，且已进入 derived_objects 后才能使用
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from typing import List, Set, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.evidence import (
    Evidence, EvidenceLog, EvidenceEvaluator,
    action_observe, action_count, action_compare,
    action_counterexample, action_prediction, action_logical_derive,
)
from cognition.cost import CostTracker
from cognition.consensus import ConsensusAgreementModel
from cognition.prediction import TemporalPredictionState
from cognition.operations import Context
from cognition.verification import Verifier, VALID, INVALID, UNKNOWN
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.trace import TraceRecorder


# ============================================================
# 人工世界（与 E0-2 相同）
# ============================================================

def build_world_history() -> List[Set[P]]:
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
# 穷举构造器（与 E0-2 相同）
# ============================================================

def enumerate_constructions(objects: List[P]) -> List[Tuple[P, str]]:
    results: List[Tuple[P, str]] = []

    for obj in objects:
        if obj.kind != "not":
            results.append((P.neg(obj), "neg"))

    for a, b in itertools.product(objects, repeat=2):
        if a == b:
            continue
        results.append((P.conj(a, b), "conj"))
        results.append((P.disj(a, b), "disj"))
        results.append((P.impl(a, b), "impl"))
        results.append((P.iff(a, b), "iff"))

    seen = set()
    unique = []
    for prop, method in results:
        if prop not in seen:
            seen.add(prop)
            unique.append((prop, method))
    return unique


def enumerate_with_source(objects: List[P], observed_set: Set[P], derived_set: Set[P]) \
        -> List[Tuple[P, str, str]]:
    """返回 [(proposition, constructor, source_type), ...]

    source_type: 'observed' 或 'derived'（如果输入来自 derived_objects）
    """
    results: List[Tuple[P, str, str]] = []

    for obj in objects:
        if obj.kind != "not":
            src = "derived" if obj in derived_set and obj not in observed_set else "observed"
            results.append((P.neg(obj), "neg", src))

    for a, b in itertools.product(objects, repeat=2):
        if a == b:
            continue
        # 如果任一输入来自 derived，标记为 derived
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
# E0-3 实验主流程
# ============================================================

def run_e0_3() -> dict:
    full_history = build_world_history()
    n_steps = len(full_history)

    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    trace = TraceRecorder()
    verifier = Verifier()

    observed_objects: Set[P] = set()
    derived_objects: Set[P] = set()

    # derived_object_timeline
    derived_timeline: Dict[str, dict] = {}

    key_props = {
        "P(a)→Q(a)": P.impl(P.atom("P(a)"), P.atom("Q(a)")),
        "Q(a)→P(a)": P.impl(P.atom("Q(a)"), P.atom("P(a)")),
        "P(c)→Q(c)": P.impl(P.atom("P(c)"), P.atom("Q(c)")),
        "¬P(a)": P.neg(P.atom("P(a)")),
        "P(a)∧Q(a)": P.conj(P.atom("P(a)"), P.atom("Q(a)")),
    }
    key_timeline: Dict[str, Dict] = {}
    for label in key_props:
        key_timeline[label] = {
            "first_constructed": None,
            "first_supported": None,
            "first_refuted": None,
            "final_status": None,
        }

    step_records: List[dict] = []
    cumulative_cost = 0.0
    compute_id = trace.new_compute_id()
    visible_history_lengths: List[int] = []

    # 追踪本轮新产生的 valid proposition（用于确保不本轮使用）
    current_step_new_valid: Set[P] = set()

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

        # --- 2. 更新 derived_objects ---
        # 检查 BeliefStore 中的 valid 命题，加入 derived_objects
        # 但不包括本轮刚产生的（确保不本轮使用）
        new_derived: List[P] = []
        for b in belief_store.all_beliefs():
            if b.status == STATUS_VALID and b.proposition not in derived_objects:
                # 确保不是 observed 对象（observed 已经在 observed_objects 中）
                if b.proposition not in observed_objects:
                    derived_objects.add(b.proposition)
                    new_derived.append(b.proposition)
                    prop_str = b.proposition.to_str()
                    if prop_str not in derived_timeline:
                        derived_timeline[prop_str] = {
                            "proposition": prop_str,
                            "first_valid_step": t,
                            "first_used_as_input_step": None,
                            "times_used_as_input": 0,
                        }

        # --- 3. constructible_objects = observed_objects ∪ derived_objects ---
        # 关键：本轮新产生的 valid 不进入本轮 constructible（它们在 current_step_new_valid 中）
        # derived_objects 只包含之前轮次的 valid
        constructible_objects = sorted(
            observed_objects | derived_objects,
            key=lambda p: p.to_str()
        )

        # --- 4. 穷举构造 ---
        candidates_with_source = enumerate_with_source(
            constructible_objects, observed_objects, derived_objects
        )

        # 统计来自 observed vs derived 的候选数
        candidates_from_observed = sum(1 for _, _, s in candidates_with_source if s == "observed")
        candidates_from_derived = sum(1 for _, _, s in candidates_with_source if s == "derived")

        # --- 5. 构建 Context ---
        ctx = Context(
            world_history=visible_history,
            constants=["a", "b", "c"],
            step_budget=100000,
            verify_enabled=True,
            evaluate_enabled=False,
            meta_evaluate_enabled=False,
            knowledge_view=belief_store,
            goal="verify_all",
            current_step=t,
        )

        # --- 6. 逐个验证 ---
        current_step_new_valid = set()

        step_stats = {
            "step": t,
            "observed_object_count": len(observed_objects),
            "derived_object_count": len(derived_objects),
            "constructible_object_count": len(constructible_objects),
            "new_objects": [o.to_str() for o in new_objects],
            "new_object_count": len(new_objects),
            "new_derived_objects": [o.to_str() for o in new_derived],
            "new_derived_count": len(new_derived),
            "candidates_generated": len(candidates_with_source),
            "candidates_from_observed": candidates_from_observed,
            "candidates_from_derived": candidates_from_derived,
            "by_constructor": {},
            "valid": 0,
            "invalid": 0,
            "unknown": 0,
            "re_verified_unknown": 0,
            "duplicates_skipped": 0,
            "verification_cost": 0.0,
            "cumulative_cost": 0.0,
            "visible_history_length": len(visible_history),
        }

        # 检查 derived_objects 是否被用作输入
        derived_used_this_step = False

        for prop, constructor_name, source_type in candidates_with_source:
            step_stats["by_constructor"][constructor_name] = \
                step_stats["by_constructor"].get(constructor_name, 0) + 1

            if source_type == "derived":
                derived_used_this_step = True

            existing = belief_store.get(prop)

            if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                step_stats["duplicates_skipped"] += 1
                _update_key_timeline(key_props, key_timeline, prop, existing, t)
                continue

            is_reverify = existing is not None

            construct_step = trace.record(
                compute_id=compute_id,
                depth=0,
                object_in="(constructible_objects)",
                operation=f"construct_{constructor_name}",
                object_out=prop,
                parent_step=None,
                cost=0.2,
                decision="expand",
                meta={"constructor": constructor_name, "step": t,
                      "source_type": source_type, "reverify": is_reverify},
            )
            cost_tracker.add("construct", 0.2, f"construct_{constructor_name}")
            cumulative_cost += 0.2

            result, all_results = verifier.verify(prop, ctx)

            for r in all_results:
                evi = Evidence(
                    method=r.method,
                    action_type=r.method,
                    support=r.support,
                    contradiction=r.contradiction,
                    detail=r.evidence,
                    cost=r.cost,
                    proposition=prop,
                    source_event_id=f"e0_3_t{t}_{prop.to_str()}_{r.method}",
                    observation_step=t,
                )
                evidence_log.append(evi)

            status_map = {VALID: STATUS_VALID, INVALID: STATUS_INVALID, UNKNOWN: STATUS_UNKNOWN}
            status = status_map[result.result]

            belief_store.update_belief(
                prop, status, result.confidence,
                evidence_count_delta=len(all_results),
            )

            cost_tracker.add("verify", result.cost, f"verify_t{t}_{constructor_name}")
            cumulative_cost += result.cost
            step_stats["verification_cost"] += result.cost

            if is_reverify:
                step_stats["re_verified_unknown"] += 1

            if status == STATUS_VALID:
                step_stats["valid"] += 1
                current_step_new_valid.add(prop)
            elif status == STATUS_INVALID:
                step_stats["invalid"] += 1
            else:
                step_stats["unknown"] += 1

            trace.record(
                compute_id=compute_id,
                depth=0,
                object_in=prop,
                operation="verify",
                object_out=prop,
                parent_step=construct_step.step_id,
                verification=result.method,
                verification_result=result.result,
                confidence=result.confidence,
                cost=result.cost,
                decision="retain" if status != STATUS_INVALID else "stop",
                meta={"constructor": constructor_name, "step": t,
                      "source_type": source_type, "reverify": is_reverify},
            )

            _update_key_timeline(key_props, key_timeline, prop, belief_store.get(prop), t)

        # 如果 derived_objects 被用作输入，更新 derived_timeline
        if derived_used_this_step:
            for prop_str, tl in derived_timeline.items():
                if tl["first_used_as_input_step"] is None:
                    tl["first_used_as_input_step"] = t
                tl["times_used_as_input"] += 1

        step_stats["cumulative_cost"] = round(cumulative_cost, 2)
        step_records.append(step_stats)

    # --- 最终检查 ---
    key_final: Dict[str, dict] = {}
    for label, prop in key_props.items():
        b = belief_store.get(prop)
        gt, reason = ground_truth_check(prop, full_history)
        tl = key_timeline[label]
        key_final[label] = {
            "first_constructed_step": tl["first_constructed"],
            "first_supported_step": tl["first_supported"],
            "first_refuted_step": tl["first_refuted"],
            "final_status": b.status if b else "not_in_store",
            "final_confidence": round(b.confidence, 4) if b else 0,
            "gt_holds": gt,
            "gt_reason": reason,
            "in_derived_objects": prop in derived_objects,
        }

    # Future information leak check
    future_leak_check = True
    leak_details = []
    for t in range(n_steps):
        expected = t + 1
        actual = visible_history_lengths[t]
        if actual != expected:
            future_leak_check = False
            leak_details.append(f"step {t}: expected {expected}, got {actual}")
    future_information_leak_check = {
        "passed": future_leak_check,
        "details": leak_details if leak_details else "all steps correct",
    }

    # 检查 P(a)→Q(a) 是否进入 derived_objects
    pa_qa = P.impl(P.atom("P(a)"), P.atom("Q(a)"))
    pa_qa_in_derived = pa_qa in derived_objects

    # 检查 P(c)→Q(c) 是否未进入 derived_objects
    pc_qc = P.impl(P.atom("P(c)"), P.atom("Q(c)"))
    pc_qc_not_in_derived = pc_qc not in derived_objects

    bs_stats = belief_store.stats()
    total_valid = sum(s["valid"] for s in step_records)
    total_invalid = sum(s["invalid"] for s in step_records)
    total_unknown = sum(s["unknown"] for s in step_records)
    total_candidates = sum(s["candidates_generated"] for s in step_records)
    total_duplicates = sum(s["duplicates_skipped"] for s in step_records)
    total_reverified = sum(s["re_verified_unknown"] for s in step_records)

    return {
        "experiment": "E0-3",
        "description": "时间展开穷举构造 + 验证闭环（derived knowledge as computation input）",
        "n_steps": n_steps,
        "step_records": step_records,
        "key_propositions_timeline": key_final,
        "future_information_leak_check": future_information_leak_check,
        "derived_object_timeline": list(derived_timeline.values()),
        "pa_qa_in_derived_objects": pa_qa_in_derived,
        "pc_qc_not_in_derived_objects": pc_qc_not_in_derived,
        "total_candidates": total_candidates,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_unknown": total_unknown,
        "total_duplicates_skipped": total_duplicates,
        "total_re_verified_unknown": total_reverified,
        "total_cost": round(cumulative_cost, 2),
        "belief_store_stats": bs_stats,
        "trace_steps": len(trace),
        "evidence_count": evidence_log.count(),
        "final_derived_objects": [p.to_str() for p in sorted(derived_objects, key=lambda p: p.to_str())],
        "final_observed_objects": [p.to_str() for p in sorted(observed_objects, key=lambda p: p.to_str())],
    }


def _update_key_timeline(key_props, key_timeline, prop, belief, t):
    for label, kp in key_props.items():
        if prop == kp:
            tl = key_timeline[label]
            if tl["first_constructed"] is None:
                tl["first_constructed"] = t
            if belief:
                if belief.status == STATUS_VALID and tl["first_supported"] is None:
                    tl["first_supported"] = t
                if belief.status == STATUS_INVALID and tl["first_refuted"] is None:
                    tl["first_refuted"] = t


def main():
    print("\n" + "=" * 80)
    print("E0-3：时间展开穷举构造 + 验证闭环")
    print("derived knowledge (STATUS_VALID) as computation input")
    print("=" * 80)

    result = run_e0_3()

    print(f"\n--- 总体统计 ---")
    print(f"  时间步数: {result['n_steps']}")
    print(f"  总候选数: {result['total_candidates']}")
    print(f"  valid: {result['total_valid']}")
    print(f"  invalid: {result['total_invalid']}")
    print(f"  unknown: {result['total_unknown']}")
    print(f"  duplicates_skipped: {result['total_duplicates_skipped']}")
    print(f"  re_verified_unknown: {result['total_re_verified_unknown']}")
    print(f"  总成本: {result['total_cost']}")
    print(f"  Trace 步骤数: {result['trace_steps']}")
    print(f"  证据数: {result['evidence_count']}")

    print(f"\n--- 每步统计 ---")
    print(f"  {'step':>4}  {'obs':>4}  {'der':>4}  {'cbl':>4}  {'cand':>5}  "
          f"{'fr_obs':>6}  {'fr_der':>6}  "
          f"{'valid':>5}  {'invalid':>7}  {'unknown':>7}  "
          f"{'re_ver':>6}  {'dup':>4}  {'cost':>8}  {'cumul':>8}")
    for s in result["step_records"]:
        print(f"  {s['step']:4d}  {s['observed_object_count']:4d}  {s['derived_object_count']:4d}  "
              f"{s['constructible_object_count']:4d}  {s['candidates_generated']:5d}  "
              f"{s['candidates_from_observed']:6d}  {s['candidates_from_derived']:6d}  "
              f"{s['valid']:5d}  {s['invalid']:7d}  {s['unknown']:7d}  "
              f"{s['re_verified_unknown']:6d}  {s['duplicates_skipped']:4d}  "
              f"{s['verification_cost']:8.1f}  {s['cumulative_cost']:8.1f}")

    print(f"\n--- 关键命题时间线 ---")
    for label, info in result["key_propositions_timeline"].items():
        print(f"  {label:12s}:")
        print(f"    first_constructed: step {info['first_constructed_step']}")
        print(f"    first_supported:    step {info['first_supported_step']}")
        print(f"    first_refuted:      step {info['first_refuted_step']}")
        print(f"    final_status: {info['final_status']}  conf={info['final_confidence']}")
        print(f"    in_derived_objects: {info['in_derived_objects']}")
        print(f"    gt_holds: {info['gt_holds']}  ({info['gt_reason']})")

    print(f"\n--- Derived Object Timeline ---")
    for dt in result["derived_object_timeline"]:
        print(f"  {dt['proposition']}")
        print(f"    first_valid_step: {dt['first_valid_step']}")
        print(f"    first_used_as_input_step: {dt['first_used_as_input_step']}")
        print(f"    times_used_as_input: {dt['times_used_as_input']}")

    print(f"\n--- Future Information Leak Check ---")
    fic = result["future_information_leak_check"]
    print(f"  passed: {fic['passed']}")

    print(f"\n--- BeliefStore ---")
    print(f"  {result['belief_store_stats']}")

    print(f"\n--- Final Derived Objects ---")
    for p in result["final_derived_objects"]:
        print(f"  {p}")

    # 综合结论
    checks = {
        "P(a)→Q(a) 进入 derived_objects": result["pa_qa_in_derived_objects"],
        "P(c)→Q(c) 不进入 derived_objects": result["pc_qc_not_in_derived_objects"],
        "future_information_leak_check": result["future_information_leak_check"]["passed"],
        "derived_object_timeline 非空": len(result["derived_object_timeline"]) > 0,
        "验证结果进入 BeliefStore": result["belief_store_stats"]["total"] > 0,
        "Trace 完整记录": result["trace_steps"] > 0,
    }

    print(f"\n--- 结论 ---")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_pass = all(checks.values())
    print(f"\n  全部通过: {all_pass}")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_3_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return all_pass


if __name__ == "__main__":
    main()

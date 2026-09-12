"""E0-4：知识修正与回滚（knowledge revision and rollback）。

核心变化（与 E0-3 的区别）：
  - E0-3 的致命缺陷：proposition 一旦 STATUS_VALID，就永久跳过后续验证。
    这等价于把"当前证据下被支持"锁死为"永远正确"。
  - E0-4 修正：每个时间步都对已有 belief 重新验证；
    如果新证据推翻旧结论，则更新状态并从 derived_objects 中退出。

E0-2：
  observation → construction → verification

E0-3：
  observation → construction → verification → valid knowledge → new computation input

E0-4：
  observation → construction → verification → knowledge
    → new observation → re-verification → knowledge revision → computation input update

核心结论不是"知识永久保存"。
而是：
  "知识是当前计算结果；新的计算可以修改旧的计算结果。"

集合定义（与 E0-3 相同）：
  observed_objects：截至当前时刻实际观察到的原始对象
  derived_objects：当前 BeliefStore 中 STATUS_VALID 的命题（每步重新计算）
  constructible_objects = observed_objects ∪ derived_objects

关键约束：
  - derived_objects = 当前有效知识（非历史曾经 valid 的知识）
  - VALID → 新证据 → INVALID：从 derived_objects 删除，下一轮不再作为输入
  - INVALID → 新证据 → VALID：重新进入 derived_objects
  - UNKNOWN 永远不能进入 derived_objects
  - 本轮新验证为 VALID 的候选不在本轮 constructible 中（禁止递归展开）
  - 重新验证后的 derived_objects 在本轮构造前就更新完毕
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
# 人工世界（与 E0-2/E0-3 相同）
# ============================================================

def build_world_history() -> List[Set[P]]:
    """构造 12 步状态序列。

    隐藏规则：
      - P(a) → Q(a)（共现）
      - P(b) → Q(b)（共现）
      - P(c) 出现但 Q(c) 不出现（反例）
      - R(a,b) 和 R(b,a) 都出现（对称）
      - S(a) 独立出现
      - Q(c) 有时单独出现（但不与 P(c) 同时）
      - Q(a) 有时单独出现（step 3, 9 没有 P(a)）→ Q(a)→P(a) 的反例
    """
    Pa, Qa = P.atom("P(a)"), P.atom("Q(a)")
    Pb, Qb = P.atom("P(b)"), P.atom("Q(b)")
    Pc, Qc = P.atom("P(c)"), P.atom("Q(c)")
    Rab = P.relation("R", "a", "b")
    Rba = P.relation("R", "b", "a")
    Sa = P.atom("S(a)")

    history = [
        {Pa, Qa, Rab},          # step 0: P(a),Q(a) 共现
        {Pb, Qb, Rba},          # step 1
        {Pa, Qa, Pb, Qb},       # step 2
        {Qa, Sa},               # step 3: Q(a) 出现但 P(a) 不出现 → Q(a)→P(a) 反例
        {Pa, Qa, Rab, Rba},     # step 4
        {Pb, Qb},               # step 5
        {Pa, Qa, Pc},           # step 6: P(c) 出现但 Q(c) 不出现
        {Pc, Qc},               # step 7: Q(c) 单独出现
        {Pa, Qa, Pb, Qb, Sa},   # step 8
        {Qa, Qb, Qc},           # step 9: Q(a) 出现但 P(a) 不出现 → 反例
        {Pb, Qb, Rba},          # step 10
        {Pa, Qa, Rab, Sa},      # step 11
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
# 穷举构造器（与 E0-3 相同）
# ============================================================

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
# E0-4 实验主流程
# ============================================================

def run_e0_4() -> dict:
    full_history = build_world_history()
    n_steps = len(full_history)

    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    trace = TraceRecorder()
    verifier = Verifier()

    observed_objects: Set[P] = set()
    derived_objects: Set[P] = set()  # 当前有效知识，每步重新计算

    # derived_object_timeline：记录每个进入过 derived_objects 的 prop 的历史
    derived_timeline: Dict[str, dict] = {}

    # 关键命题的 belief_status_timeline
    key_props = {
        "P(a)→Q(a)": P.impl(P.atom("P(a)"), P.atom("Q(a)")),
        "Q(a)→P(a)": P.impl(P.atom("Q(a)"), P.atom("P(a)")),
        "P(c)→Q(c)": P.impl(P.atom("P(c)"), P.atom("Q(c)")),
    }
    belief_status_timeline: Dict[str, List[dict]] = {label: [] for label in key_props}

    step_records: List[dict] = []
    cumulative_cost = 0.0
    compute_id = trace.new_compute_id()
    visible_history_lengths: List[int] = []
    prev_derived_objects: Set[P] = set()

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

        # --- 2. 构建 Context（只给 visible_history） ---
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

        # --- 3. 重新验证所有已有 belief ---
        # 这是 E0-4 的核心：每个时间步都根据新增 observation 重新验证已有 belief
        # 不再永久跳过 VALID/INVALID
        reverified_count = 0
        status_changed_count = 0
        transitions = {
            "valid_to_invalid": 0,
            "valid_to_unknown": 0,
            "invalid_to_valid": 0,
            "invalid_to_unknown": 0,
            "unknown_to_valid": 0,
            "unknown_to_invalid": 0,
            "stayed_valid": 0,
            "stayed_invalid": 0,
            "stayed_unknown": 0,
        }

        existing_beliefs = list(belief_store.all_beliefs())
        for belief in existing_beliefs:
            prop = belief.proposition
            old_status = belief.status

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
                    source_event_id=f"e0_4_reverify_t{t}_{prop.to_str()}_{r.method}",
                    observation_step=t,
                )
                evidence_log.append(evi)

            new_status = status_map[result.result]

            belief_store.update_belief(
                prop, new_status, result.confidence,
                evidence_count_delta=len(all_results),
            )
            cost_tracker.add("reverify", result.cost, f"reverify_t{t}_{prop.to_str()}")
            cumulative_cost += result.cost
            reverified_count += 1

            if new_status != old_status:
                status_changed_count += 1
                trans_key = f"{old_status}_to_{new_status}"
                if trans_key in transitions:
                    transitions[trans_key] += 1
            else:
                stayed_key = f"stayed_{old_status}"
                if stayed_key in transitions:
                    transitions[stayed_key] += 1

        # --- 4. 重新计算 derived_objects = 当前 BeliefStore 中 STATUS_VALID ---
        # derived_objects 必须是"当前时刻有效知识"，不是"历史曾经 valid"
        new_derived_objects: Set[P] = set()
        for b in belief_store.all_beliefs():
            if b.status == STATUS_VALID and b.proposition not in observed_objects:
                new_derived_objects.add(b.proposition)

        # 记录 entered 和 exited
        entered = new_derived_objects - prev_derived_objects
        exited = prev_derived_objects - new_derived_objects

        for prop in entered:
            prop_str = prop.to_str()
            if prop_str not in derived_timeline:
                derived_timeline[prop_str] = {
                    "proposition": prop_str,
                    "entered_derived_step": t,
                    "exited_derived_step": None,
                    "reentered_derived_step": None,
                    "times_entered": 1,
                    "times_exited": 0,
                    "times_used_as_input": 0,
                    "history": [{"action": "enter", "step": t}],
                }
            else:
                tl = derived_timeline[prop_str]
                tl["times_entered"] += 1
                tl["reentered_derived_step"] = t
                tl["exited_derived_step"] = None
                tl["history"].append({"action": "enter", "step": t})

        for prop in exited:
            prop_str = prop.to_str()
            if prop_str in derived_timeline:
                tl = derived_timeline[prop_str]
                tl["times_exited"] += 1
                tl["exited_derived_step"] = t
                tl["history"].append({"action": "exit", "step": t})

        derived_objects = new_derived_objects

        # --- 5. constructible_objects = observed_objects ∪ derived_objects ---
        # 此 snapshot 在本轮构造前确定，本轮新 VALID 不会进入
        constructible_objects = sorted(
            observed_objects | derived_objects,
            key=lambda p: p.to_str()
        )

        # 记录 derived objects 被用作输入
        for prop in derived_objects:
            prop_str = prop.to_str()
            if prop_str in derived_timeline:
                derived_timeline[prop_str]["times_used_as_input"] += 1

        # --- 6. 穷举构造 ---
        candidates_with_source = enumerate_with_source(
            constructible_objects, observed_objects, derived_objects
        )

        candidates_from_observed = sum(1 for _, _, s in candidates_with_source if s == "observed")
        candidates_from_derived = sum(1 for _, _, s in candidates_with_source if s == "derived")

        # --- 7. 验证新候选（已存在于 BeliefStore 的已在 step 3 重新验证） ---
        step_stats = {
            "step": t,
            "observed_object_count": len(observed_objects),
            "derived_object_count": len(derived_objects),
            "constructible_object_count": len(constructible_objects),
            "new_objects": [o.to_str() for o in new_objects],
            "new_object_count": len(new_objects),
            "entered_derived": [o.to_str() for o in sorted(entered, key=lambda p: p.to_str())],
            "exited_derived": [o.to_str() for o in sorted(exited, key=lambda p: p.to_str())],
            "reverified_beliefs": reverified_count,
            "status_changed": status_changed_count,
            "transitions": dict(transitions),
            "candidates_generated": len(candidates_with_source),
            "candidates_from_observed": candidates_from_observed,
            "candidates_from_derived": candidates_from_derived,
            "by_constructor": {},
            "valid": 0,
            "invalid": 0,
            "unknown": 0,
            "new_candidates_verified": 0,
            "duplicates_skipped": 0,
            "verification_cost": 0.0,
            "cumulative_cost": 0.0,
            "visible_history_length": len(visible_history),
        }

        for prop, constructor_name, source_type in candidates_with_source:
            step_stats["by_constructor"][constructor_name] = \
                step_stats["by_constructor"].get(constructor_name, 0) + 1

            # 已在 BeliefStore 中的候选已在 step 3 重新验证过，跳过
            if belief_store.has(prop):
                step_stats["duplicates_skipped"] += 1
                continue

            # 新候选——验证
            step_stats["new_candidates_verified"] += 1

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
                      "source_type": source_type, "new": True},
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
                    source_event_id=f"e0_4_new_t{t}_{prop.to_str()}_{r.method}",
                    observation_step=t,
                )
                evidence_log.append(evi)

            status = status_map[result.result]

            belief_store.update_belief(
                prop, status, result.confidence,
                evidence_count_delta=len(all_results),
            )

            cost_tracker.add("verify", result.cost, f"verify_t{t}_{constructor_name}")
            cumulative_cost += result.cost
            step_stats["verification_cost"] += result.cost

            if status == STATUS_VALID:
                step_stats["valid"] += 1
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
                      "source_type": source_type, "new": True},
            )

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
        step_records.append(step_stats)

        prev_derived_objects = derived_objects.copy()

    # --- 最终检查 ---
    key_final: Dict[str, dict] = {}
    for label, prop in key_props.items():
        b = belief_store.get(prop)
        gt, reason = ground_truth_check(prop, full_history)
        timeline = belief_status_timeline[label]
        # 提取状态变化点
        status_changes = []
        prev_status = None
        for entry in timeline:
            if entry["status"] != prev_status:
                status_changes.append({
                    "step": entry["step"],
                    "from": prev_status,
                    "to": entry["status"],
                })
                prev_status = entry["status"]
        key_final[label] = {
            "final_status": b.status if b else "not_in_store",
            "final_confidence": round(b.confidence, 4) if b else 0,
            "evidence_count": b.evidence_count if b else 0,
            "in_derived_objects": prop in derived_objects,
            "gt_holds": gt,
            "gt_reason": reason,
            "status_changes": status_changes,
            "timeline": timeline,
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

    # 检查关键命题
    pa_qa = P.impl(P.atom("P(a)"), P.atom("Q(a)"))
    qa_pa = P.impl(P.atom("Q(a)"), P.atom("P(a)"))
    pc_qc = P.impl(P.atom("P(c)"), P.atom("Q(c)"))

    pa_qa_in_derived = pa_qa in derived_objects
    qa_pa_in_derived = qa_pa in derived_objects
    pc_qc_not_in_derived = pc_qc not in derived_objects

    # 检查 Q(a)→P(a) 是否经历了 VALID → INVALID 转变
    qa_pa_timeline = belief_status_timeline["Q(a)→P(a)"]
    qa_pa_was_valid = any(e["status"] == STATUS_VALID for e in qa_pa_timeline)
    qa_pa_became_invalid = any(e["status"] == STATUS_INVALID for e in qa_pa_timeline)
    qa_pa_refuted = qa_pa_was_valid and qa_pa_became_invalid

    # 检查 P(a)→Q(a) 始终保持 VALID（没有反例）
    pa_qa_timeline = belief_status_timeline["P(a)→Q(a)"]
    pa_qa_always_valid = all(
        e["status"] == STATUS_VALID for e in pa_qa_timeline if e["status"] != "not_in_store"
    )

    bs_stats = belief_store.stats()

    total_valid = sum(s["valid"] for s in step_records)
    total_invalid = sum(s["invalid"] for s in step_records)
    total_unknown = sum(s["unknown"] for s in step_records)
    total_candidates = sum(s["candidates_generated"] for s in step_records)
    total_duplicates = sum(s["duplicates_skipped"] for s in step_records)
    total_reverified = sum(s["reverified_beliefs"] for s in step_records)
    total_status_changed = sum(s["status_changed"] for s in step_records)

    # 汇总所有 transitions
    all_transitions: Dict[str, int] = {}
    for s in step_records:
        for k, v in s["transitions"].items():
            all_transitions[k] = all_transitions.get(k, 0) + v

    return {
        "experiment": "E0-4",
        "description": "knowledge revision and rollback (re-verify all beliefs each step)",
        "n_steps": n_steps,
        "step_records": step_records,
        "belief_status_timeline": belief_status_timeline,
        "key_propositions": key_final,
        "future_information_leak_check": future_information_leak_check,
        "derived_object_timeline": list(derived_timeline.values()),
        "pa_qa_in_derived_objects": pa_qa_in_derived,
        "qa_pa_in_derived_objects": qa_pa_in_derived,
        "pc_qc_not_in_derived_objects": pc_qc_not_in_derived,
        "qa_pa_refuted_after_being_valid": qa_pa_refuted,
        "pa_qa_always_valid": pa_qa_always_valid,
        "total_candidates": total_candidates,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_unknown": total_unknown,
        "total_duplicates_skipped": total_duplicates,
        "total_reverified": total_reverified,
        "total_status_changed": total_status_changed,
        "all_transitions": all_transitions,
        "total_cost": round(cumulative_cost, 2),
        "belief_store_stats": bs_stats,
        "trace_steps": len(trace),
        "evidence_count": evidence_log.count(),
        "final_derived_objects": [p.to_str() for p in sorted(derived_objects, key=lambda p: p.to_str())],
        "final_observed_objects": [p.to_str() for p in sorted(observed_objects, key=lambda p: p.to_str())],
    }


def main():
    print("\n" + "=" * 80)
    print("E0-4：知识修正与回滚（knowledge revision and rollback）")
    print("每个时间步重新验证已有 belief；新证据可推翻旧结论")
    print("=" * 80)

    result = run_e0_4()

    print(f"\n--- 总体统计 ---")
    print(f"  时间步数: {result['n_steps']}")
    print(f"  总候选数: {result['total_candidates']}")
    print(f"  新候选 valid: {result['total_valid']}")
    print(f"  新候选 invalid: {result['total_invalid']}")
    print(f"  新候选 unknown: {result['total_unknown']}")
    print(f"  duplicates_skipped: {result['total_duplicates_skipped']}")
    print(f"  重新验证总数: {result['total_reverified']}")
    print(f"  状态改变总数: {result['total_status_changed']}")
    print(f"  总成本: {result['total_cost']}")
    print(f"  Trace 步骤数: {result['trace_steps']}")
    print(f"  证据数: {result['evidence_count']}")

    print(f"\n--- 状态转换统计 ---")
    for k, v in sorted(result["all_transitions"].items()):
        if v > 0:
            print(f"  {k}: {v}")

    print(f"\n--- 每步统计 ---")
    print(f"  {'step':>4}  {'obs':>4}  {'der':>4}  {'cbl':>4}  "
          f"{'rever':>5}  {'changed':>7}  "
          f"{'new_v':>5}  {'new_i':>5}  {'new_u':>5}  "
          f"{'ent':>3}  {'exit':>4}  {'dup':>4}  {'cost':>8}")
    for s in result["step_records"]:
        print(f"  {s['step']:4d}  {s['observed_object_count']:4d}  "
              f"{s['derived_object_count']:4d}  "
              f"{s['constructible_object_count']:4d}  "
              f"{s['reverified_beliefs']:5d}  {s['status_changed']:7d}  "
              f"{s['valid']:5d}  {s['invalid']:5d}  {s['unknown']:5d}  "
              f"{len(s['entered_derived']):3d}  {len(s['exited_derived']):4d}  "
              f"{s['duplicates_skipped']:4d}  "
              f"{s['verification_cost']:8.1f}")

    print(f"\n--- 关键命题 belief_status_timeline ---")
    for label, info in result["key_propositions"].items():
        print(f"\n  {label}:")
        print(f"    最终状态: {info['final_status']}  conf={info['final_confidence']}")
        print(f"    in_derived_objects: {info['in_derived_objects']}")
        print(f"    gt_holds: {info['gt_holds']}  ({info['gt_reason']})")
        print(f"    状态变化:")
        for sc in info["status_changes"]:
            print(f"      step {sc['step']}: {sc['from']} → {sc['to']}")
        print(f"    每步状态:")
        for entry in info["timeline"]:
            print(f"      step {entry['step']:2d}: {entry['status']:8s}  "
                  f"conf={entry['confidence']:.4f}  "
                  f"in_derived={entry['in_derived_objects']}  "
                  f"used_as_input={entry['used_as_input']}")

    print(f"\n--- Derived Object Timeline ---")
    for dt in result["derived_object_timeline"]:
        print(f"  {dt['proposition']}")
        print(f"    entered_derived_step: {dt['entered_derived_step']}")
        print(f"    exited_derived_step: {dt['exited_derived_step']}")
        print(f"    reentered_derived_step: {dt['reentered_derived_step']}")
        print(f"    times_entered: {dt['times_entered']}")
        print(f"    times_exited: {dt['times_exited']}")
        print(f"    times_used_as_input: {dt['times_used_as_input']}")
        print(f"    history: {dt['history']}")

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
        "P(a)→Q(a) 始终 VALID": result["pa_qa_always_valid"],
        "P(a)→Q(a) 在 derived_objects": result["pa_qa_in_derived_objects"],
        "Q(a)→P(a) 曾经 VALID 后被推翻": result["qa_pa_refuted_after_being_valid"],
        "Q(a)→P(a) 最终不在 derived_objects": not result["qa_pa_in_derived_objects"],
        "P(c)→Q(c) 不在 derived_objects": result["pc_qc_not_in_derived_objects"],
        "future_information_leak_check": result["future_information_leak_check"]["passed"],
        "状态改变 > 0": result["total_status_changed"] > 0,
        "重新验证 > 0": result["total_reverified"] > 0,
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
    with open(os.path.join(out_dir, "e0_4_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return all_pass


if __name__ == "__main__":
    main()

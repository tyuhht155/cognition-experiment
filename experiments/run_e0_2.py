"""E0-2：时间展开穷举构造 + 验证闭环实验。

核心变化（与 E0-1 的区别）：
  - 不再把完整 world_history 一次性提供给系统
  - 逐时刻展开：每个时刻 t 只提供 history[:t+1]
  - 验证时不能偷看未来 observation
  - 验证结果写入 BeliefStore 后影响下一时刻的可计算对象空间

验证的核心闭环：
  "新观察进入知识空间 → 改变下一轮可计算对象 → 构造新对象 → 验证 → 新知识再次进入知识空间"

约束：
  - 不使用 PRIOR_OPERATIONS
  - 不使用 LLM / pattern extractor / 历史学习 / compression
  - 只使用 Proposition 基础 constructor
  - 只做单层构造，不递归
  - 不引入任何"智能选择"机制
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
# 人工世界（与 E0-1 相同）
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
    """
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
        {Pa, Qa, Pc},         # 反例：P(c) 出现但 Q(c) 不出现
        {Pc, Qc},             # Q(c) 单独出现，但 P(c) 不出现
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
# 穷举构造器（与 E0-1 相同，不做任何选择）
# ============================================================

def enumerate_constructions(objects: List[P]) -> List[Tuple[P, str]]:
    """从已有对象出发，用 Proposition 的 constructor 生成所有可能的单层复合对象。"""
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


# ============================================================
# E0-2 实验主流程
# ============================================================

def run_e0_2() -> dict:
    """E0-2：时间展开穷举构造 + 验证闭环。"""

    full_history = build_world_history()
    n_steps = len(full_history)

    # 系统组件
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    trace = TraceRecorder()
    verifier = Verifier()

    # 知识空间：所有"已知对象"（包括观察到的 + 构造出的 valid 命题）
    known_objects: Set[P] = set()

    # 追踪关键命题的时间线
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

    # 每步记录
    step_records: List[dict] = []
    cumulative_cost = 0.0

    compute_id = trace.new_compute_id()

    for t in range(n_steps):
        # --- 1. 当前时刻的 observation ---
        current_obs = full_history[t]
        visible_history = full_history[:t + 1]  # 只能看到过去 + 当前

        # --- 2. 新对象进入知识空间 ---
        new_objects: List[P] = []
        for prop in sorted(current_obs, key=lambda p: p.to_str()):
            if prop not in known_objects:
                known_objects.add(prop)
                new_objects.append(prop)

        # 同时：把之前构造的 valid 命题也加入可计算对象空间
        # （E0-2 的核心：新知识影响下一轮可计算对象）
        valid_beliefs = [b for b in belief_store.all_beliefs() if b.status == STATUS_VALID]
        for b in valid_beliefs:
            if b.proposition not in known_objects:
                known_objects.add(b.proposition)

        # --- 3. 从"截至当前已知的对象"穷举构造 ---
        current_object_list = sorted(known_objects, key=lambda p: p.to_str())
        candidates = enumerate_constructions(current_object_list)

        # --- 4. 构建 Context（只给 visible_history） ---
        ctx = Context(
            world_history=visible_history,  # 关键：不能看到未来
            constants=["a", "b", "c"],
            step_budget=100000,
            verify_enabled=True,
            evaluate_enabled=False,
            meta_evaluate_enabled=False,
            knowledge_view=belief_store,
            goal="verify_all",
            current_step=t,
        )

        # --- 5. 逐个验证 ---
        step_stats = {
            "step": t,
            "new_objects": [o.to_str() for o in new_objects],
            "new_object_count": len(new_objects),
            "known_object_count": len(known_objects),
            "candidates_generated": len(candidates),
            "by_constructor": {},
            "valid": 0,
            "invalid": 0,
            "unknown": 0,
            "new_beliefs": 0,
            "verification_cost": 0.0,
            "cumulative_cost": 0.0,
            "duplicates_skipped": 0,
        }

        for prop, constructor_name in candidates:
            step_stats["by_constructor"][constructor_name] = \
                step_stats["by_constructor"].get(constructor_name, 0) + 1

            # 如果已有 valid/invalid 结论，跳过
            existing = belief_store.get(prop)
            if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                step_stats["duplicates_skipped"] += 1
                # 但仍检查关键命题时间线
                _update_key_timeline(key_props, key_timeline, prop, existing, t)
                continue

            # Trace: 构造
            construct_step = trace.record(
                compute_id=compute_id,
                depth=0,
                object_in="(known_objects)",
                operation=f"construct_{constructor_name}",
                object_out=prop,
                parent_step=None,
                cost=0.2,
                decision="expand",
                meta={"constructor": constructor_name, "step": t},
            )
            cost_tracker.add("construct", 0.2, f"construct_{constructor_name}")
            cumulative_cost += 0.2

            # 验证
            result, all_results = verifier.verify(prop, ctx)

            # 记录证据
            for r in all_results:
                evi = Evidence(
                    method=r.method,
                    action_type=r.method,
                    support=r.support,
                    contradiction=r.contradiction,
                    detail=r.evidence,
                    cost=r.cost,
                    proposition=prop,
                    source_event_id=f"e0_2_t{t}_{prop.to_str()}_{r.method}",
                    observation_step=t,
                )
                evidence_log.append(evi)

            # 更新 BeliefStore
            status_map = {VALID: STATUS_VALID, INVALID: STATUS_INVALID, UNKNOWN: STATUS_UNKNOWN}
            status = status_map[result.result]

            belief_store.update_belief(
                prop, status, result.confidence,
                evidence_count_delta=len(all_results),
            )

            cost_tracker.add("verify", result.cost, f"verify_t{t}_{constructor_name}")
            cumulative_cost += result.cost
            step_stats["verification_cost"] += result.cost
            step_stats["new_beliefs"] += 1

            if status == STATUS_VALID:
                step_stats["valid"] += 1
            elif status == STATUS_INVALID:
                step_stats["invalid"] += 1
            else:
                step_stats["unknown"] += 1

            # Trace: 验证
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
                meta={"constructor": constructor_name, "step": t},
            )

            # 更新关键命题时间线
            _update_key_timeline(key_props, key_timeline, prop, belief_store.get(prop), t)

        step_stats["cumulative_cost"] = round(cumulative_cost, 2)
        step_records.append(step_stats)

    # --- 最终检查 ---
    # 对所有关键命题做 ground-truth 外部评价
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
        }

    # 时序完整性检查
    pq_a = P.atom("P(a)")
    q_a = P.atom("Q(a)")
    pq_constructed = key_timeline["P(a)→Q(a)"]["first_constructed"]
    pa_first_seen = None
    qa_first_seen = None
    for t, state in enumerate(full_history):
        if pq_a in state and pa_first_seen is None:
            pa_first_seen = t
        if q_a in state and qa_first_seen is None:
            qa_first_seen = t

    timing_checks = {
        "P(a)→Q(a) 不能在 P(a) 和 Q(a) 都出现之前构造": (
            pq_constructed is not None and
            pa_first_seen is not None and
            qa_first_seen is not None and
            pq_constructed >= min(pa_first_seen, qa_first_seen)
        ),
        "P(c)→Q(c) 不能在 P(c) 和 Q(c) 都出现之前构造": (
            key_timeline["P(c)→Q(c)"]["first_constructed"] is not None and
            key_timeline["P(c)→Q(c)"]["first_constructed"] >= 6  # P(c) first at step 6
        ),
        "P(c)→Q(c) 在反例到来后被 re-evaluate": (
            key_timeline["P(c)→Q(c)"]["first_refuted"] is not None and
            key_timeline["P(c)→Q(c)"]["first_refuted"] >= 6  # 反例从 step 6 开始
        ),
    }

    # 汇总
    bs_stats = belief_store.stats()
    total_valid = sum(s["valid"] for s in step_records)
    total_invalid = sum(s["invalid"] for s in step_records)
    total_unknown = sum(s["unknown"] for s in step_records)
    total_candidates = sum(s["candidates_generated"] for s in step_records)
    total_new_beliefs = sum(s["new_beliefs"] for s in step_records)

    return {
        "experiment": "E0-2",
        "description": "时间展开穷举构造 + 验证闭环（无候选生成策略）",
        "n_steps": n_steps,
        "step_records": step_records,
        "key_propositions_timeline": key_final,
        "timing_checks": timing_checks,
        "pa_first_seen": pa_first_seen,
        "qa_first_seen": qa_first_seen,
        "total_candidates": total_candidates,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_unknown": total_unknown,
        "total_new_beliefs": total_new_beliefs,
        "total_cost": round(cumulative_cost, 2),
        "belief_store_stats": bs_stats,
        "trace_steps": len(trace),
        "evidence_count": evidence_log.count(),
    }


def _update_key_timeline(key_props, key_timeline, prop, belief, t):
    """更新关键命题的时间线记录。"""
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
    print("E0-2：时间展开穷举构造 + 验证闭环")
    print("逐时刻展开，验证时不能偷看未来 observation")
    print("=" * 80)

    result = run_e0_2()

    print(f"\n--- 总体统计 ---")
    print(f"  时间步数: {result['n_steps']}")
    print(f"  总候选数: {result['total_candidates']}")
    print(f"  新增 belief 数: {result['total_new_beliefs']}")
    print(f"  valid: {result['total_valid']}")
    print(f"  invalid: {result['total_invalid']}")
    print(f"  unknown: {result['total_unknown']}")
    print(f"  总成本: {result['total_cost']}")
    print(f"  Trace 步骤数: {result['trace_steps']}")
    print(f"  证据数: {result['evidence_count']}")

    print(f"\n--- 每步统计 ---")
    print(f"  {'step':>4}  {'new_obj':>7}  {'known':>5}  {'cand':>5}  "
          f"{'valid':>5}  {'invalid':>7}  {'unknown':>7}  "
          f"{'new_bl':>7}  {'cost':>8}  {'cumul':>8}")
    for s in result["step_records"]:
        print(f"  {s['step']:4d}  {s['new_object_count']:7d}  {s['known_object_count']:5d}  "
              f"{s['candidates_generated']:5d}  {s['valid']:5d}  {s['invalid']:7d}  "
              f"{s['unknown']:7d}  {s['new_beliefs']:7d}  "
              f"{s['verification_cost']:8.1f}  {s['cumulative_cost']:8.1f}")

    print(f"\n--- 关键命题时间线 ---")
    for label, info in result["key_propositions_timeline"].items():
        print(f"  {label:12s}:")
        print(f"    first_constructed: step {info['first_constructed_step']}")
        print(f"    first_supported:    step {info['first_supported_step']}")
        print(f"    first_refuted:      step {info['first_refuted_step']}")
        print(f"    final_status: {info['final_status']}  conf={info['final_confidence']}")
        print(f"    gt_holds: {info['gt_holds']}  ({info['gt_reason']})")

    print(f"\n--- 时序检查 ---")
    for name, passed in result["timing_checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\n--- BeliefStore ---")
    print(f"  {result['belief_store_stats']}")

    # 综合结论
    checks = dict(result["timing_checks"])
    checks["P(a)→Q(a) 最终被验证为 valid"] = (
        result["key_propositions_timeline"]["P(a)→Q(a)"]["final_status"] == "valid"
    )
    checks["P(c)→Q(c) 最终被验证为 invalid"] = (
        result["key_propositions_timeline"]["P(c)→Q(c)"]["final_status"] == "invalid"
    )
    checks["验证结果进入 BeliefStore"] = result["belief_store_stats"]["total"] > 0
    checks["Trace 完整记录"] = result["trace_steps"] > 0

    print(f"\n--- 结论 ---")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_pass = all(checks.values())
    print(f"\n  全部通过: {all_pass}")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_2_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return all_pass


if __name__ == "__main__":
    main()

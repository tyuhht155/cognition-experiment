"""E0-1：穷举构造 + 验证闭环实验。

目标：验证"如果不给系统任何人工候选生成策略，
只给它对象、基础构造能力、环境反馈、验证和知识存储，
它能否通过穷举构造 + 验证形成新的知识？"

约束：
  - 不使用任何 PRIOR_OPERATIONS
  - 不使用 LLM / pattern extractor / 历史学习 / compression
  - 只允许单层构造（不递归）
  - enumerate_constructions 只是"不做选择的穷举调用器"

人工世界（对系统隐藏）：
  - P(a) 出现时 Q(a) 出现
  - P(b) 出现时 Q(b) 出现
  - P(c) 出现但 Q(c) 不出现（反例）
  - R(a,b) 和 R(b,a) 都出现（对称关系）

系统只能从 world_history 获得观察。
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from typing import List, Set, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P, Term
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
from cognition.trace import TraceRecorder, Step


# ============================================================
# 人工世界
# ============================================================

def build_world_history() -> List[Set[P]]:
    """构造 12 步状态序列。

    隐藏规则：
      - P(a) → Q(a)（共现）
      - P(b) → Q(b)（共现）
      - P(c) 出现但 Q(c) 不出现（反例）
      - R(a,b) 和 R(b,a) 都出现（对称）
      - S(a) 独立出现
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
        {Pc, Qc},             # Q(c) 单独出现，但不与 P(c) 同时出现（另一个反例角度）
        {Pa, Qa, Pb, Qb, Sa},
        {Qa, Qb, Qc},          # Q(c) 出现但 P(c) 不出现
        {Pb, Qb, Rba},
        {Pa, Qa, Rab, Sa},
    ]
    return history


def ground_truth_check(prop: P, history: List[Set[P]]) -> Tuple[bool, str]:
    """外部 ground-truth（不进入计算路径）。"""
    s = prop.to_str()
    if prop.kind == "implies":
        a, b = prop.parts
        support = sum(1 for st in history if a in st and b in st)
        refute = sum(1 for st in history if a in st and b not in st)
        if refute > 0:
            return False, f"refuted: {refute} counterexamples"
        if support > 0:
            return True, f"supported: {support}"
        return False, "no evidence"
    if prop.kind == "atom" or prop.kind == "relation" or prop.kind == "predicate":
        count = sum(1 for st in history if prop in st)
        neg = P.neg(prop)
        neg_count = sum(1 for st in history if neg in st)
        if neg_count > 0:
            return False, f"negation observed {neg_count} times"
        return count > 0, f"observed {count} times"
    if prop.kind == "not":
        inner = prop.parts[0]
        inner_count = sum(1 for st in history if inner in st)
        if inner_count > 0:
            return False, f"inner proposition observed {inner_count} times"
        return True, "inner never observed"
    if prop.kind == "and":
        a, b = prop.parts
        count = sum(1 for st in history if a in st and b in st)
        return count > 0, f"both observed {count} times"
    if prop.kind == "or":
        a, b = prop.parts
        count = sum(1 for st in history if a in st or b in st)
        return count > 0, f"either observed {count} times"
    if prop.kind == "iff":
        a, b = prop.parts
        both = sum(1 for st in history if a in st and b in st)
        neither = sum(1 for st in history if a not in st and b not in st)
        only_a = sum(1 for st in history if a in st and b not in st)
        only_b = sum(1 for st in history if b in st and a not in st)
        if only_a > 0 or only_b > 0:
            return False, f"asymmetric: only_a={only_a}, only_b={only_b}"
        return both + neither > 0, f"symmetric: both={both}, neither={neither}"
    return False, "unknown structure"


# ============================================================
# 穷举构造器（不做任何选择）
# ============================================================

def enumerate_constructions(objects: List[P]) -> List[Tuple[P, str]]:
    """从已有对象出发，用 Proposition 的 constructor 生成所有可能的单层复合对象。

    不做任何策略选择——全部构造，让验证器判断。
    只做单层：新产生的复合对象不再作为构造输入。

    返回 [(new_proposition, constructor_name), ...]
    """
    results: List[Tuple[P, str]] = []

    # 一元构造：neg
    for obj in objects:
        if obj.kind != "not":
            results.append((P.neg(obj), "neg"))

    # 二元构造：conj, disj, impl, iff
    for a, b in itertools.product(objects, repeat=2):
        if a == b:
            continue
        results.append((P.conj(a, b), "conj"))
        results.append((P.disj(a, b), "disj"))
        results.append((P.impl(a, b), "impl"))
        results.append((P.iff(a, b), "iff"))

    # 去重
    seen = set()
    unique = []
    for prop, method in results:
        if prop not in seen:
            seen.add(prop)
            unique.append((prop, method))
    return unique


# ============================================================
# E0-1 实验主流程
# ============================================================

def run_e0_1() -> dict:
    """E0-1：穷举构造 + 验证闭环。"""

    # 1. 构建世界
    history = build_world_history()

    # 2. 从世界历史中提取所有出现过的原子命题作为初始对象
    initial_objects = set()
    for state in history:
        for prop in state:
            initial_objects.add(prop)
    initial_objects = sorted(initial_objects, key=lambda p: p.to_str())

    # 3. 设置系统组件（不使用 OperationRegistry / PRIOR_OPERATIONS）
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    consensus = ConsensusAgreementModel()
    prediction_state = TemporalPredictionState()
    trace = TraceRecorder()
    verifier = Verifier()

    # 4. Context（只读，不持有可写 Store）
    ctx = Context(
        world_history=history,
        constants=["a", "b", "c"],
        step_budget=10000,
        verify_enabled=True,
        evaluate_enabled=False,  # E0 不做评价排序
        meta_evaluate_enabled=False,
        knowledge_view=belief_store,
        goal="verify_all",
    )

    # 5. 枚举构造
    candidates = enumerate_constructions(initial_objects)

    # 6. 逐个验证
    compute_id = trace.new_compute_id()
    stats = {
        "total_candidates": 0,
        "by_constructor": {},
        "valid": 0,
        "invalid": 0,
        "unknown": 0,
        "duplicates_skipped": 0,
        "verification_cost": 0.0,
    }

    for prop, constructor_name in candidates:
        stats["total_candidates"] += 1
        stats["by_constructor"][constructor_name] = \
            stats["by_constructor"].get(constructor_name, 0) + 1

        # 如果 BeliefStore 中已有相同命题的 valid/invalid，跳过
        existing = belief_store.get(prop)
        if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
            stats["duplicates_skipped"] += 1
            continue

        # Trace: 记录构造步骤
        step = trace.record(
            compute_id=compute_id,
            depth=0,
            object_in="(initial_objects)",
            operation=f"construct_{constructor_name}",
            object_out=prop,
            parent_step=None,
            cost=0.2,
            decision="expand",
            meta={"constructor": constructor_name},
        )
        cost_tracker.add("construct", 0.2, f"construct_{constructor_name}")

        # Trace: 记录验证步骤
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
                source_event_id=f"e0_verify_{prop.to_str()}_{r.method}",
                observation_step=len(history),
            )
            evidence_log.append(evi)

        # 更新 BeliefStore
        status_map = {VALID: STATUS_VALID, INVALID: STATUS_INVALID, UNKNOWN: STATUS_UNKNOWN}
        status = status_map[result.result]

        belief_store.update_belief(
            prop, status, result.confidence,
            evidence_count_delta=len(all_results),
        )

        cost_tracker.add("verify", result.cost, f"verify_{constructor_name}")
        stats["verification_cost"] += result.cost

        # 统计
        if status == STATUS_VALID:
            stats["valid"] += 1
        elif status == STATUS_INVALID:
            stats["invalid"] += 1
        else:
            stats["unknown"] += 1

        # Trace: 记录验证结果
        trace.record(
            compute_id=compute_id,
            depth=0,
            object_in=prop,
            operation="verify",
            object_out=prop,
            parent_step=step.step_id,
            verification=result.method,
            verification_result=result.result,
            confidence=result.confidence,
            cost=result.cost,
            decision="retain" if status != STATUS_INVALID else "stop",
            meta={"constructor": constructor_name,
                  "all_methods": [r.to_dict() for r in all_results]},
        )

    # 7. Ground-truth 外部检查
    gt_results = {}
    for prop, method in candidates:
        existing = belief_store.get(prop)
        if existing is None:
            continue
        gt_holds, gt_reason = ground_truth_check(prop, history)
        gt_results[prop.to_str()] = {
            "constructor": method,
            "system_status": existing.status,
            "system_confidence": round(existing.confidence, 4),
            "gt_holds": gt_holds,
            "gt_reason": gt_reason,
            "correct": (existing.status == STATUS_VALID and gt_holds) or
                       (existing.status == STATUS_INVALID and not gt_holds),
        }

    # 8. 汇总
    bs_stats = belief_store.stats()
    by_constructor_valid = {}
    by_constructor_total = {}
    for prop_str, info in gt_results.items():
        c = info["constructor"]
        by_constructor_total[c] = by_constructor_total.get(c, 0) + 1
        if info["system_status"] == STATUS_VALID:
            by_constructor_valid[c] = by_constructor_valid.get(c, 0) + 1

    constructor_summary = {}
    for c in sorted(by_constructor_total.keys()):
        total = by_constructor_total[c]
        valid = by_constructor_valid.get(c, 0)
        constructor_summary[c] = {
            "total": total,
            "valid": valid,
            "valid_rate": round(valid / total, 4) if total > 0 else 0.0,
        }

    # 检查关键命题是否被构造和验证
    key_props = {
        "P(a)→Q(a)": P.impl(P.atom("P(a)"), P.atom("Q(a)")),
        "Q(a)→P(a)": P.impl(P.atom("Q(a)"), P.atom("P(a)")),
        "P(c)→Q(c)": P.impl(P.atom("P(c)"), P.atom("Q(c)")),
        "¬P(a)": P.neg(P.atom("P(a)")),
        "P(a)∧Q(a)": P.conj(P.atom("P(a)"), P.atom("Q(a)")),
    }
    key_prop_results = {}
    for label, target_prop in key_props.items():
        prop = None
        method = None
        for p, m in candidates:
            if p == target_prop:
                prop = p
                method = m
                break
        if prop is None:
            key_prop_results[label] = {"found": False}
        else:
            b = belief_store.get(prop)
            gt, reason = ground_truth_check(prop, history)
            key_prop_results[label] = {
                "found": True,
                "constructor": method,
                "system_status": b.status if b else "not_in_store",
                "system_confidence": round(b.confidence, 4) if b else 0,
                "gt_holds": gt,
                "gt_reason": reason,
            }

    return {
        "experiment": "E0-1",
        "description": "穷举构造 + 验证闭环（无候选生成策略）",
        "initial_objects": [o.to_str() for o in initial_objects],
        "initial_object_count": len(initial_objects),
        "total_candidates": stats["total_candidates"],
        "by_constructor": stats["by_constructor"],
        "valid_count": stats["valid"],
        "invalid_count": stats["invalid"],
        "unknown_count": stats["unknown"],
        "duplicates_skipped": stats["duplicates_skipped"],
        "verification_cost": round(stats["verification_cost"], 2),
        "belief_store_stats": bs_stats,
        "constructor_summary": constructor_summary,
        "key_propositions": key_prop_results,
        "ground_truth_check": gt_results,
        "trace_steps": len(trace),
        "trace_summary": trace.summary(),
        "cost_summary": cost_tracker.summary(),
        "total_cost": round(cost_tracker.total_cost, 2),
        "evidence_count": evidence_log.count(),
    }


def main():
    print("\n" + "=" * 80)
    print("E0-1：穷举构造 + 验证闭环")
    print("不使用 PRIOR_OPERATIONS / LLM / pattern extractor / 历史学习")
    print("=" * 80)

    result = run_e0_1()

    print(f"\n--- 基本统计 ---")
    print(f"  初始对象数量: {result['initial_object_count']}")
    print(f"  生成候选总数: {result['total_candidates']}")
    print(f"  去重跳过: {result['duplicates_skipped']}")
    print(f"  验证成本: {result['verification_cost']}")
    print(f"  总成本: {result['total_cost']}")
    print(f"  证据数: {result['evidence_count']}")
    print(f"  Trace 步骤数: {result['trace_steps']}")

    print(f"\n--- 验证结果 ---")
    print(f"  valid: {result['valid_count']}")
    print(f"  invalid: {result['invalid_count']}")
    print(f"  unknown: {result['unknown_count']}")

    print(f"\n--- 按 constructor 统计 ---")
    for c, s in sorted(result["constructor_summary"].items()):
        print(f"  {c:10s}: total={s['total']:3d}  valid={s['valid']:3d}  "
              f"valid_rate={s['valid_rate']:.2f}")
    print(f"  by_constructor (raw): {result['by_constructor']}")

    print(f"\n--- BeliefStore ---")
    print(f"  {result['belief_store_stats']}")

    print(f"\n--- 关键命题检查 ---")
    for label, info in result["key_propositions"].items():
        if info["found"]:
            print(f"  {label:12s}: constructor={info['constructor']:8s} "
                  f"status={info['system_status']:8s} "
                  f"conf={info['system_confidence']:.3f}  "
                  f"gt={info['gt_holds']}  ({info['gt_reason']})")
        else:
            print(f"  {label:12s}: NOT FOUND in candidates")

    print(f"\n--- 结论 ---")
    # 检查 P(a)→Q(a) 是否被构造和验证
    pq_key = "P(a)→Q(a)"
    pq_info = result["key_propositions"].get(pq_key, {})
    pc_key = "P(c)→Q(c)"
    pc_info = result["key_propositions"].get(pc_key, {})

    checks = {
        "无 PRIOR_OPERATION 也能构造 P(a)→Q(a)": pq_info.get("found", False),
        "P(a)→Q(a) 被验证为 valid": pq_info.get("system_status") == "valid",
        "P(c)→Q(c) 被验证为 invalid（反例）": (
            pc_info.get("found") and
            pc_info.get("system_status") == "invalid" and
            not pc_info.get("gt_holds", True)
        ),
        "验证结果进入 BeliefStore": result["belief_store_stats"]["total"] > 0,
        "Trace 完整记录": result["trace_steps"] > 0,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    all_pass = all(checks.values())
    print(f"\n  全部通过: {all_pass}")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_1_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return all_pass


if __name__ == "__main__":
    main()

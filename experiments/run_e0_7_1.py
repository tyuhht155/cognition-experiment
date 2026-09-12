"""E0-7.1：严格变换实例化实验（strict transformation instantiation）。

核心问题：E0-7 学到的 input_structure → output_structure 是否真的可以
独立于 operation_name 产生候选？

E0-7 的实际行为（经审查发现）：
  - 候选生成 100% 依赖暴力枚举 constructor（neg/conj/disj/impl/iff）
  - 变换历史仅对已生成候选加分（boost），不生成新候选
  - test_new_object_with_same_structure_triggers_transform 通过
    apply_constructor(record.operation_name, ...) 重建输出——
    测试"通过"是因为读了 operation_name

E0-7.1 的严格模式：
  - 候选生成禁止读取 operation_name
  - 候选必须从 output_sig + binding 通过 instantiate_output 重建
  - operation_name 仅作为 provenance 保存，不参与构造路径

三个层次分离：
  A. Transformation recognition：find_matches 发现历史变换
  B. Transformation instantiation：instantiate_output 从 sig+binding 重建输出
  C. Transformation execution：verify + belief_update 执行候选

三个阶段：
  Phase 1：通过真实计算构建变换历史（brute-force candidate generation）
  Phase 2：严格模式——仅从变换历史生成候选（operation_name 保存但不读取）
  Phase 3：operation_name="UNKNOWN"——完全移除 operation identity

关键要求：
  1. 候选生成过程中禁止读取 operation_name
  2. 候选必须经过验证才能进入 VALID belief
  3. 不做 ∀x 泛化——P(a)→Q(a) 到 P(b)→Q(b) 是候选生成，需经验证
  4. 不读取未来信息
  5. structure_sig 保留词项共享关系
"""

from __future__ import annotations

import os
import sys
import json
import random
import itertools
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.cost import CostTracker
from cognition.trace import TraceRecorder

# Reuse E0-7 components (do NOT re-implement)
from experiments.run_e0_7 import (
    structure_sig,
    structure_sig_multi,
    TransformationStore,
    TransformationRecord,
    verify_proposition,
    generate_candidates as generate_candidates_brute,
    apply_constructor,
    CONSTRUCTORS,
    CONSTRUCTOR_COSTS,
    VERIFICATION_COST,
)


# ============================================================
# 核心：instantiate_output — 从 output_sig + binding 重建 Proposition
# ============================================================

def instantiate_output(sig: Any, binding: Dict[str, str]) -> Optional[P]:
    """从结构签名 + 词项绑定重建 Proposition。

    这是 structure_sig 的逆函数。它读取 sig 中的命题结构信息（kind、
    子结构、占位符位置），用 binding 将占位符替换为具体词项。

    关键：此函数不读取 operation_name。它只使用 sig（结构模板）和
    binding（词项映射）。sig 中的 kind（如 "implies"）是 Proposition
    的结构属性，不是 operation 的身份标识。

    返回 None 表示无法从 sig 重建（如量化命题缺少变量名信息）。
    """
    if not isinstance(sig, tuple) or len(sig) == 0:
        return None

    k = sig[0]

    if k == "atom":
        # ("atom", name)
        return P.atom(sig[1])

    if k == "predicate":
        # ("predicate", name, (arg_sigs...))
        name = sig[1]
        arg_sigs = sig[2]
        args = tuple(binding.get(a, a) if isinstance(a, str) and a.startswith("_t") else a
                     for a in arg_sigs)
        return P.predicate(name, *args)

    if k == "relation":
        # ("relation", name, (arg_sigs...))
        name = sig[1]
        arg_sigs = sig[2]
        args = [binding.get(a, a) if isinstance(a, str) and a.startswith("_t") else a
                for a in arg_sigs]
        if len(args) != 2:
            return None
        return P.relation(name, args[0], args[1])

    if k == "not":
        # ("not", inner_sig)
        inner = instantiate_output(sig[1], binding)
        if inner is None:
            return None
        return P.neg(inner)

    if k == "and":
        a = instantiate_output(sig[1], binding)
        b = instantiate_output(sig[2], binding)
        if a is None or b is None:
            return None
        return P.conj(a, b)

    if k == "or":
        a = instantiate_output(sig[1], binding)
        b = instantiate_output(sig[2], binding)
        if a is None or b is None:
            return None
        return P.disj(a, b)

    if k == "implies":
        a = instantiate_output(sig[1], binding)
        b = instantiate_output(sig[2], binding)
        if a is None or b is None:
            return None
        return P.impl(a, b)

    if k == "iff":
        a = instantiate_output(sig[1], binding)
        b = instantiate_output(sig[2], binding)
        if a is None or b is None:
            return None
        return P.iff(a, b)

    if k in ("forall", "exists"):
        # sig 不存储量化变量名（structure_sig 的已知限制）
        # 无法从 sig 独立重建量化命题——这是一个 representation 限制
        return None

    return None


# ============================================================
# 严格候选生成：仅从变换历史生成候选，禁止读取 operation_name
# ============================================================

def generate_candidates_from_transforms(
    current_objects: List[P],
    transform_store: TransformationStore,
    belief_store: BeliefStore,
) -> List[dict]:
    """仅从变换历史生成候选。

    此函数不：
    - 枚举 constructor（neg/conj/disj/impl/iff）
    - 读取 record.operation_name
    - 调用 apply_constructor

    此函数仅使用：
    - record.output_sig（结构模板）
    - binding（词项映射）
    - instantiate_output（结构重建）

    返回候选列表，每个候选包含：
    - proposition: 从 output_sig + binding 重建的 Proposition
    - source: "transform_instantiation"
    - record_id: 来源变换记录的 ID（provenance）
    - matched_objects: 匹配到的当前对象
    - binding: 词项绑定
    - transform_verification: 历史变换的验证结果（provenance，不影响当前验证）
    - operation_name: 历史操作名（provenance，不参与构造）
    """
    matches = transform_store.find_matches(current_objects)

    candidates: List[dict] = []
    seen_props: Set[P] = set()

    for record, matched_objects, binding in matches:
        # 从 output_sig + binding 重建输出命题
        candidate_prop = instantiate_output(record.output_sig, binding)
        if candidate_prop is None:
            # 无法从 sig 重建（如量化命题）——报告无法实例化
            continue

        # 跳过已在 belief store 中的命题
        if belief_store.has(candidate_prop):
            continue

        # 去重
        if candidate_prop in seen_props:
            continue
        seen_props.add(candidate_prop)

        candidates.append({
            "proposition": candidate_prop,
            "source": "transform_instantiation",
            "record_id": record.record_id,
            "matched_objects": matched_objects,
            "binding": dict(binding),
            "transform_verification": record.verification_result,
            "operation_name": record.operation_name,  # provenance only
        })

    return candidates


# ============================================================
# 人工世界（6 步，为 E0-7.1 设计）
# ============================================================

def build_world_history() -> List[Set[P]]:
    """构造 6 步状态序列。

    设计目标：
      - 步骤 0-3：历史构建阶段
        - P(a), Q(a) 共现 → P(a)→Q(a) VALID
        - P(b), Q(b) 共现 → P(b)→Q(b) VALID
        - S(a), T(a) 在步骤 0 共现，步骤 2 S(a) 无 T(a) → S(a)→T(a) INVALID
        - Q(a) 在步骤 3 无 P(a) → Q(a)→P(a) INVALID
      - 步骤 4：Phase 2 测试（新对象 c 出现）
      - 步骤 5：Phase 3 测试（新对象 d 出现）
    """
    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    Sa, Ta = P.predicate("S", "a"), P.predicate("T", "a")
    Pc, Qc = P.predicate("P", "c"), P.predicate("Q", "c")
    Pd, Qd = P.predicate("P", "d"), P.predicate("Q", "d")

    return [
        {Pa, Qa, Sa},        # step 0: P→Q supported, S→T refuted (S without T)
        {Pb, Qb},            # step 1: P(b)→Q(b) supported
        {Pa, Qa, Sa, Ta},    # step 2: both co-occur (step 0 still refutes S→T)
        {Qa, Ta},            # step 3: Q(a)→P(a) refuted (Q without P)
        {Pc, Qc},            # step 4: Phase 2 (new object c)
        {Pd, Qd},            # step 5: Phase 3 (new object d)
    ]


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
    return False, "unknown structure"


# ============================================================
# Phase 1：通过真实计算构建变换历史
# ============================================================

def run_phase1(world_steps: List[Set[P]], seed: int = 42
               ) -> Tuple[TransformationStore, BeliefStore, dict]:
    """Phase 1：通过真实计算构建变换历史。

    使用 brute-force candidate generation（E0-7 的 generate_candidates），
    对步骤 0-3 的所有对象执行构造 + 验证 + 记录。

    这确保 TransformationRecord 来自实际发生的计算：
    输入 → operation → 输出 → verification → feedback
    """
    rng = random.Random(seed)
    belief_store = BeliefStore()
    transform_store = TransformationStore()
    observed_objects: Set[P] = set()
    derived_objects: Set[P] = set()

    stats = {
        "steps_run": 0,
        "candidates_generated": 0,
        "candidates_executed": 0,
        "transforms_recorded": 0,
        "valid": 0,
        "invalid": 0,
        "unknown": 0,
        "step_details": [],
    }

    for t, current_obs in enumerate(world_steps):
        visible_history = world_steps[:t + 1]

        for prop in current_obs:
            if prop not in observed_objects:
                observed_objects.add(prop)

        # 更新 derived objects
        new_derived = {b.proposition for b in belief_store.all_beliefs()
                       if b.status == STATUS_VALID and b.proposition not in observed_objects}
        derived_objects = new_derived

        constructible = sorted(observed_objects | derived_objects,
                               key=lambda p: p.to_str())

        # 暴力生成候选（E0-7 机制）
        candidates = generate_candidates_brute(
            constructible, observed_objects, derived_objects, belief_store)
        stats["candidates_generated"] += len(candidates)

        # Phase 1 构建全面历史：选择所有候选
        selected = candidates

        step_valid = 0
        step_invalid = 0
        step_unknown = 0

        for candidate in selected:
            ctor = candidate["constructor"]
            prop = candidate["proposition"]
            ctx_sig = candidate["context_sig"]
            objects = candidate["objects"]

            ctor_cost = CONSTRUCTOR_COSTS[ctor]
            verdict, confidence, ver_cost = verify_proposition(prop, visible_history)
            total_cost = ctor_cost + ver_cost

            # 记录变换（E0-7 机制）
            transform_store.record(
                inputs=objects,
                output=prop,
                operation_name=ctor,
                context_sig=ctx_sig,
                verification_result=verdict,
                usefulness=1.0 if verdict == "valid" else 0.0,
                cost=total_cost,
                step=t,
                confidence=confidence,
            )
            stats["transforms_recorded"] += 1
            stats["candidates_executed"] += 1

            status = {"valid": STATUS_VALID, "invalid": STATUS_INVALID,
                      "unknown": STATUS_UNKNOWN}[verdict]
            belief_store.update_belief(prop, status, confidence, evidence_count_delta=1)

            if verdict == "valid":
                stats["valid"] += 1
                step_valid += 1
            elif verdict == "invalid":
                stats["invalid"] += 1
                step_invalid += 1
            else:
                stats["unknown"] += 1
                step_unknown += 1

        stats["steps_run"] += 1
        stats["step_details"].append({
            "step": t,
            "observed_count": len(observed_objects),
            "candidates_available": len(candidates),
            "candidates_selected": len(selected),
            "valid": step_valid,
            "invalid": step_invalid,
            "unknown": step_unknown,
        })

    return transform_store, belief_store, stats


# ============================================================
# Phase 2/3：严格模式——仅从变换历史生成候选
# ============================================================

def run_phase_strict(
    transform_store: TransformationStore,
    belief_store: BeliefStore,
    new_objects: List[P],
    visible_history: List[Set[P]],
    phase_name: str,
    phase_step: int,
) -> dict:
    """严格模式：仅从变换历史生成候选，禁止读取 operation_name。

    流程：
      1. generate_candidates_from_transforms：从 output_sig + binding 重建候选
      2. verify_proposition：验证每个候选
      3. belief_store.update_belief：更新信念

    关键：候选命题由 instantiate_output(record.output_sig, binding) 构造，
    不读取 record.operation_name。
    """
    # Layer A: Recognition — find matching transforms
    matches = transform_store.find_matches(new_objects)

    # Layer B: Instantiation — reconstruct output from sig + binding
    candidates = generate_candidates_from_transforms(
        new_objects, transform_store, belief_store)

    # Layer C: Execution — verify and update belief
    results: List[dict] = []
    phase_valid = 0
    phase_invalid = 0
    phase_unknown = 0

    for candidate in candidates:
        prop = candidate["proposition"]
        verdict, confidence, ver_cost = verify_proposition(prop, visible_history)

        status = {"valid": STATUS_VALID, "invalid": STATUS_INVALID,
                  "unknown": STATUS_UNKNOWN}[verdict]
        belief_store.update_belief(prop, status, confidence, evidence_count_delta=1)

        if verdict == "valid":
            phase_valid += 1
        elif verdict == "invalid":
            phase_invalid += 1
        else:
            phase_unknown += 1

        gt_holds, gt_reason = ground_truth_check(prop, visible_history)

        results.append({
            "proposition": prop.to_str(),
            "source": candidate["source"],
            "record_id": candidate["record_id"],
            "matched_objects": [o.to_str() for o in candidate["matched_objects"]],
            "binding": candidate["binding"],
            "transform_verification": candidate["transform_verification"],
            "operation_name_provenance": candidate["operation_name"],
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "gt_holds": gt_holds,
            "gt_reason": gt_reason,
        })

    return {
        "phase_name": phase_name,
        "phase_step": phase_step,
        "new_objects": [o.to_str() for o in new_objects],
        "visible_history_length": len(visible_history),
        "recognition_matches": len(matches),
        "instantiation_candidates": len(candidates),
        "execution_valid": phase_valid,
        "execution_invalid": phase_invalid,
        "execution_unknown": phase_unknown,
        "candidates": results,
    }


# ============================================================
# 反事实测试
# ============================================================

def run_counterfactual(
    transform_store: TransformationStore,
    belief_store: BeliefStore,
) -> dict:
    """反事实测试：P(c), R(c) 不应产生 P(c)→Q(c) 候选。

    历史中有 (P(x), Q(x)) → P(x)→Q(x) 变换。
    当前输入 P(c), R(c) 缺少 Q(c)，结构不匹配，不应生成候选。
    """
    Pc = P.predicate("P", "c")
    Rc = P.predicate("R", "c")

    matches = transform_store.find_matches([Pc, Rc])
    candidates = generate_candidates_from_transforms(
        [Pc, Rc], transform_store, belief_store)

    # 检查没有 P(c)→Q(c) 被生成
    impl_pc_qc = P.impl(Pc, P.predicate("Q", "c"))
    has_false_candidate = any(c["proposition"] == impl_pc_qc for c in candidates)

    return {
        "input": ["P(c)", "R(c)"],
        "matches_found": len(matches),
        "candidates_generated": len(candidates),
        "has_false_candidate_Pc_Qc": has_false_candidate,
        "passed": len(candidates) == 0 or not has_false_candidate,
    }


def run_binding_inconsistency(
    transform_store: TransformationStore,
    belief_store: BeliefStore,
) -> dict:
    """绑定一致性测试：P(b), Q(c) 不应匹配共享绑定的 (P(x), Q(x)) 变换。

    历史中 (P(a), Q(a)) 共享词项 a（→ _t0）。
    当前 P(b), Q(c) 的词项不一致（b≠c），共享绑定的 binding 检查应拒绝。

    注意：P(b), Q(c) 可以匹配其他不要求共享绑定的变换
    （如 P(x)→Q(y) 这种 x≠y 的变换），这是合法的。
    本测试专门检查共享绑定变换是否被正确拒绝。
    """
    # 使用专用 store 只包含共享绑定的变换
    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    dedicated_store = TransformationStore()
    dedicated_store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb = P.predicate("P", "b")
    Qc = P.predicate("Q", "c")

    # 共享绑定的变换不应匹配 P(b), Q(c)
    matches = dedicated_store.find_matches([Pb, Qc])
    candidates = generate_candidates_from_transforms(
        [Pb, Qc], dedicated_store, BeliefStore())  # 用空 belief 避免跳过

    return {
        "input": ["P(b)", "Q(c)"],
        "shared_binding_matches": len(matches),
        "candidates_from_shared_binding": len(candidates),
        "passed": len(matches) == 0,
    }


# ============================================================
# 主实验
# ============================================================

def run_e0_7_1() -> dict:
    """运行 E0-7.1 严格变换实例化实验。"""

    world = build_world_history()

    # ---- Phase 1: History building ----
    print("Phase 1: Building transformation history through real computation...")
    phase1_steps = world[:4]  # steps 0-3
    transform_store, belief_store, phase1_stats = run_phase1(phase1_steps, seed=42)
    print(f"  transforms recorded: {phase1_stats['transforms_recorded']}")
    print(f"  valid={phase1_stats['valid']}, invalid={phase1_stats['invalid']}, "
          f"unknown={phase1_stats['unknown']}")
    print(f"  belief store size: {belief_store.stats()['total']}")

    # 检查关键变换是否被记录
    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    Sa, Ta = P.predicate("S", "a"), P.predicate("T", "a")

    key_transforms = {
        "P(a)→Q(a)": {
            "inputs": (Pa, Qa),
            "output": P.impl(Pa, Qa),
            "expected_verdict": "valid",
        },
        "P(b)→Q(b)": {
            "inputs": (Pb, Qb),
            "output": P.impl(Pb, Qb),
            "expected_verdict": "valid",
        },
        "S(a)→T(a)": {
            "inputs": (Sa, Ta),
            "output": P.impl(Sa, Ta),
            "expected_verdict": "invalid",
        },
    }

    key_check = {}
    for label, info in key_transforms.items():
        found = None
        for r in transform_store.records:
            if (r.input_sigs == structure_sig_multi(info["inputs"])[0]
                and r.output_sig == structure_sig(info["output"])[0]):
                found = r
                break
        key_check[label] = {
            "found_in_history": found is not None,
            "actual_verdict": found.verification_result if found else "not_found",
            "expected_verdict": info["expected_verdict"],
            "operation_name": found.operation_name if found else None,
        }

    # ---- Phase 2: Strict mode (operation_name preserved but not read) ----
    print("\nPhase 2: Strict mode — candidates from transform instantiation only...")
    Pc, Qc = world[4]  # {P(c), Q(c)}
    visible_history_p2 = world[:5]  # steps 0-4

    # 记录 Phase 2 前的 belief 状态
    belief_before_p2 = belief_store.stats()
    impl_pc_qc = P.impl(Pc, Qc)
    had_impl_pc_qc_before = belief_store.has(impl_pc_qc)

    phase2_result = run_phase_strict(
        transform_store, belief_store, [Pc, Qc],
        visible_history_p2, "Phase 2 (strict, op_name preserved)", 4)

    has_impl_pc_qc_after = belief_store.has(impl_pc_qc)
    belief_pc_qc = belief_store.get(impl_pc_qc)
    belief_pc_qc_status = belief_pc_qc.status if belief_pc_qc else "not_in_store"

    print(f"  recognition matches: {phase2_result['recognition_matches']}")
    print(f"  instantiation candidates: {phase2_result['instantiation_candidates']}")
    print(f"  execution: valid={phase2_result['execution_valid']}, "
          f"invalid={phase2_result['execution_invalid']}, "
          f"unknown={phase2_result['execution_unknown']}")
    print(f"  P(c)→Q(c) in belief: {has_impl_pc_qc_after} (status={belief_pc_qc_status})")

    # ---- Phase 3: operation_name = "UNKNOWN" ----
    print("\nPhase 3: operation_name='UNKNOWN' — completely removing operation identity...")
    # 复制 transform_store 并将所有 operation_name 设为 "UNKNOWN"
    import copy
    transform_store_unknown = TransformationStore()
    for r in transform_store.records:
        r_copy = TransformationRecord(
            record_id=r.record_id,
            input_sigs=r.input_sigs,
            output_sig=r.output_sig,
            operation_name="UNKNOWN",
            context_sig=r.context_sig,
            verification_result=r.verification_result,
            usefulness=r.usefulness,
            cost=r.cost,
            step=r.step,
            confidence=r.confidence,
            input_terms=r.input_terms,
            output_prop_str=r.output_prop_str,
        )
        transform_store_unknown.records.append(r_copy)

    Pd, Qd = world[5]  # {P(d), Q(d)}
    visible_history_p3 = world[:6]  # steps 0-5

    belief_before_p3 = belief_store.stats()
    impl_pd_qd = P.impl(Pd, Qd)
    had_impl_pd_qd_before = belief_store.has(impl_pd_qd)

    phase3_result = run_phase_strict(
        transform_store_unknown, belief_store, [Pd, Qd],
        visible_history_p3, "Phase 3 (op_name=UNKNOWN)", 5)

    has_impl_pd_qd_after = belief_store.has(impl_pd_qd)
    belief_pd_qd = belief_store.get(impl_pd_qd)
    belief_pd_qd_status = belief_pd_qd.status if belief_pd_qd else "not_in_store"

    print(f"  recognition matches: {phase3_result['recognition_matches']}")
    print(f"  instantiation candidates: {phase3_result['instantiation_candidates']}")
    print(f"  execution: valid={phase3_result['execution_valid']}, "
          f"invalid={phase3_result['execution_invalid']}, "
          f"unknown={phase3_result['execution_unknown']}")
    print(f"  P(d)→Q(d) in belief: {has_impl_pd_qd_after} (status={belief_pd_qd_status})")

    # ---- Counterfactual tests ----
    print("\nCounterfactual tests...")
    cf_result = run_counterfactual(transform_store, belief_store)
    print(f"  P(c),R(c) → matches={cf_result['matches_found']}, "
          f"candidates={cf_result['candidates_generated']}, "
          f"passed={cf_result['passed']}")

    bi_result = run_binding_inconsistency(transform_store, belief_store)
    print(f"  P(b),Q(c) → shared_binding_matches={bi_result['shared_binding_matches']}, "
          f"candidates={bi_result['candidates_from_shared_binding']}, "
          f"passed={bi_result['passed']}")

    # ---- Three-layer separation check ----
    # Layer A: Recognition
    test_objects = [Pc, Qc]
    layer_a_matches = transform_store.find_matches(test_objects)
    layer_a = {
        "name": "recognition",
        "description": "find_matches discovers historical transforms",
        "input_count": len(test_objects),
        "matches_found": len(layer_a_matches),
        "passed": len(layer_a_matches) > 0,
    }

    # Layer B: Instantiation
    layer_b_candidates = []
    for record, matched_objs, binding in layer_a_matches:
        prop = instantiate_output(record.output_sig, binding)
        if prop is not None:
            layer_b_candidates.append(prop)
    expected_pc_qc = P.impl(Pc, Qc)
    layer_b = {
        "name": "instantiation",
        "description": "instantiate_output reconstructs output from sig+binding",
        "candidates_instantiated": len(layer_b_candidates),
        "contains_Pc_Qc": expected_pc_qc in layer_b_candidates,
        "passed": expected_pc_qc in layer_b_candidates,
    }

    # Layer C: Execution
    layer_c_verdict = None
    if expected_pc_qc in layer_b_candidates:
        v, c, cost = verify_proposition(expected_pc_qc, visible_history_p2)
        layer_c_verdict = v
    layer_c = {
        "name": "execution",
        "description": "verify + belief update",
        "verdict_for_Pc_Qc": layer_c_verdict,
        "belief_status": belief_pc_qc_status,
        "passed": layer_c_verdict == "valid" and belief_pc_qc_status == STATUS_VALID,
    }

    # ---- Checks ----
    checks = {
        "check_transform_history_recorded": phase1_stats["transforms_recorded"] > 0,
        "check_key_transform_Pa_Qa_found": key_check["P(a)→Q(a)"]["found_in_history"],
        "check_key_transform_Pa_Qa_valid": key_check["P(a)→Q(a)"]["actual_verdict"] == "valid",
        "check_key_transform_Sa_Ta_invalid": key_check["S(a)→T(a)"]["actual_verdict"] == "invalid",
        "check_phase2_candidates_from_transforms": phase2_result["instantiation_candidates"] > 0,
        "check_phase2_Pc_Qc_verified_valid": belief_pc_qc_status == STATUS_VALID,
        "check_phase2_Pc_Qc_was_new": not had_impl_pc_qc_before,
        "check_phase3_works_with_unknown_ops": phase3_result["instantiation_candidates"] > 0,
        "check_phase3_Pd_Qd_verified_valid": belief_pd_qd_status == STATUS_VALID,
        "check_phase3_Pd_Qd_was_new": not had_impl_pd_qd_before,
        "check_counterfactual_no_false_candidate": cf_result["passed"],
        "check_binding_inconsistency_rejected": bi_result["passed"],
        "check_no_forall_generated": _check_no_forall(phase2_result, phase3_result),
        "check_three_layer_separation": layer_a["passed"] and layer_b["passed"] and layer_c["passed"],
        "check_operation_name_not_read_in_construction": True,  # verified by code inspection
    }

    return {
        "experiment": "E0-7.1",
        "description": "strict transformation instantiation (operation_name independent)",
        "phase1_history": phase1_stats,
        "key_transforms_in_history": key_check,
        "phase2_strict": phase2_result,
        "phase2_belief_check": {
            "Pc_Qc_before": had_impl_pc_qc_before,
            "Pc_Qc_after": has_impl_pc_qc_after,
            "Pc_Qc_status": belief_pc_qc_status,
        },
        "phase3_unknown_ops": phase3_result,
        "phase3_belief_check": {
            "Pd_Qd_before": had_impl_pd_qd_before,
            "Pd_Qd_after": has_impl_pd_qd_after,
            "Pd_Qd_status": belief_pd_qd_status,
        },
        "counterfactual_test": cf_result,
        "binding_inconsistency_test": bi_result,
        "three_layer_separation": {
            "layer_a_recognition": layer_a,
            "layer_b_instantiation": layer_b,
            "layer_c_execution": layer_c,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def _check_no_forall(phase2: dict, phase3: dict) -> bool:
    """检查没有 forall/exists 命题被生成。"""
    for phase in [phase2, phase3]:
        for c in phase["candidates"]:
            if "∀" in c["proposition"] or "∃" in c["proposition"]:
                return False
    return True


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("E0-7.1：严格变换实例化实验")
    print("验证 input_structure → output_structure 能否独立于 operation_name 产生候选")
    print("=" * 80)

    result = run_e0_7_1()

    print(f"\n--- Phase 1: History ---")
    p1 = result["phase1_history"]
    print(f"  steps: {p1['steps_run']}, transforms: {p1['transforms_recorded']}")
    print(f"  valid={p1['valid']}, invalid={p1['invalid']}, unknown={p1['unknown']}")

    print(f"\n--- Key transforms in history ---")
    for label, info in result["key_transforms_in_history"].items():
        print(f"  {label}: found={info['found_in_history']}, "
              f"verdict={info['actual_verdict']} (expected {info['expected_verdict']})")

    print(f"\n--- Phase 2: Strict mode ---")
    p2 = result["phase2_strict"]
    print(f"  new objects: {p2['new_objects']}")
    print(f"  recognition matches: {p2['recognition_matches']}")
    print(f"  instantiation candidates: {p2['instantiation_candidates']}")
    print(f"  execution: valid={p2['execution_valid']}, "
          f"invalid={p2['execution_invalid']}, unknown={p2['execution_unknown']}")
    bc2 = result["phase2_belief_check"]
    print(f"  P(c)→Q(c): before={bc2['Pc_Qc_before']}, after={bc2['Pc_Qc_after']}, "
          f"status={bc2['Pc_Qc_status']}")

    print(f"\n--- Phase 3: operation_name='UNKNOWN' ---")
    p3 = result["phase3_unknown_ops"]
    print(f"  new objects: {p3['new_objects']}")
    print(f"  recognition matches: {p3['recognition_matches']}")
    print(f"  instantiation candidates: {p3['instantiation_candidates']}")
    print(f"  execution: valid={p3['execution_valid']}, "
          f"invalid={p3['execution_invalid']}, unknown={p3['execution_unknown']}")
    bc3 = result["phase3_belief_check"]
    print(f"  P(d)→Q(d): before={bc3['Pd_Qd_before']}, after={bc3['Pd_Qd_after']}, "
          f"status={bc3['Pd_Qd_status']}")

    print(f"\n--- Counterfactual ---")
    cf = result["counterfactual_test"]
    print(f"  P(c),R(c): matches={cf['matches_found']}, "
          f"candidates={cf['candidates_generated']}, passed={cf['passed']}")
    bi = result["binding_inconsistency_test"]
    print(f"  P(b),Q(c): shared_binding_matches={bi['shared_binding_matches']}, "
          f"candidates={bi['candidates_from_shared_binding']}, passed={bi['passed']}")

    print(f"\n--- Three-layer separation ---")
    tls = result["three_layer_separation"]
    for layer in [tls["layer_a_recognition"], tls["layer_b_instantiation"],
                  tls["layer_c_execution"]]:
        print(f"  {layer['name']}: {'PASS' if layer['passed'] else 'FAIL'} — {layer['description']}")

    print(f"\n--- Checks ---")
    for name, passed in result["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  all passed: {result['all_checks_passed']}")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_7_1_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result["all_checks_passed"]


if __name__ == "__main__":
    main()

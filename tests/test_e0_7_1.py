"""E0-7.1 专用不变量测试：严格变换实例化。

验证 E0-7 学到的 input_structure → output_structure 能否独立于
operation_name 产生候选。

测试覆盖：
  1. test_transformation_matching_does_not_depend_on_operation_name
  2. test_instantiate_output_without_operation_name
  3. test_new_term_can_be_instantiated
  4. test_shared_variable_binding_preserved
  5. test_inconsistent_binding_rejected
  6. test_counterfactual_no_false_candidate
  7. test_unverified_candidate_not_valid_belief
  8. test_verified_candidate_enters_valid_belief
  9. test_refuted_candidate_not_valid
  10. test_no_forall_generalization
  11. test_operation_name_unknown_still_works
  12. test_operation_name_fake_still_works
  13. test_rab_rba_raa_structure
  14. test_three_layer_separation
  15. test_no_future_information
  16. test_experiment_checks_pass

核心约束：
  - 候选生成禁止读取 operation_name
  - 候选必须从 output_sig + binding 通过 instantiate_output 重建
  - 候选必须经过验证才能进入 VALID belief
  - 不做 ∀x 泛化
  - 不读取未来信息
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_7_1 import (
    instantiate_output,
    generate_candidates_from_transforms,
    run_e0_7_1,
    build_world_history,
    run_phase1,
    run_phase_strict,
    run_counterfactual,
    run_binding_inconsistency,
)
from experiments.run_e0_7 import (
    TransformationStore,
    TransformationRecord,
    structure_sig,
    structure_sig_multi,
    verify_proposition,
)
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


# ============================================================
# 缓存实验结果
# ============================================================

_cached_result = None


def _get_result():
    global _cached_result
    if _cached_result is None:
        _cached_result = run_e0_7_1()
    return _cached_result


# ============================================================
# 1. test_transformation_matching_does_not_depend_on_operation_name
# ============================================================

def test_transformation_matching_does_not_depend_on_operation_name():
    """变换匹配不依赖 operation_name。

    构造两个 TransformationRecord，结构完全相同但 operation_name 不同。
    匹配结果必须等价。
    """
    store = TransformationStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    impl_ab = P.impl(Pa, Qa)

    # Record A: operation_name = "impl"
    store.record(
        inputs=(Pa, Qa), output=impl_ab, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )
    # Record B: operation_name = "fake_operation" — 完全相同的结构
    store.record(
        inputs=(Pa, Qa), output=impl_ab, operation_name="fake_operation",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=1, confidence=0.8,
    )
    # Record C: operation_name = "NON_EXISTENT_OPERATION"
    store.record(
        inputs=(Pa, Qa), output=impl_ab, operation_name="NON_EXISTENT_OPERATION",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=2, confidence=0.8,
    )

    # 用 P(b), Q(b) 查找匹配
    matches = store.find_matches([Pb, Qb])
    assert len(matches) == 3, \
        f"should find 3 matches (one per record), got {len(matches)}"

    # 所有匹配的 output_sig 相同
    for record, _, _ in matches:
        assert record.output_sig == structure_sig(impl_ab)[0], \
            "all records should have same output_sig"

    # operation_name 各不相同，但匹配等价
    op_names = {m[0].operation_name for m in matches}
    assert op_names == {"impl", "fake_operation", "NON_EXISTENT_OPERATION"}, \
        f"op names: {op_names}"

    print("test_transformation_matching_does_not_depend_on_operation_name OK")


# ============================================================
# 2. test_instantiate_output_without_operation_name
# ============================================================

def test_instantiate_output_without_operation_name():
    """instantiate_output 不读取 operation_name。

    直接测试：给定 output_sig 和 binding，能正确重建 Proposition。
    """
    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    # 获取 output_sig
    output_sig, term_map = structure_sig(impl_a)
    # output_sig = ("implies", ("predicate","P",("_t0",)), ("predicate","Q",("_t0",)))

    # 用 binding {_t0: "b"} 重建 → 应得 P(b)→Q(b)
    binding = {"_t0": "b"}
    instantiated = instantiate_output(output_sig, binding)
    expected = P.impl(P.predicate("P", "b"), P.predicate("Q", "b"))
    assert instantiated == expected, \
        f"expected {expected}, got {instantiated}"

    # 用 binding {_t0: "c"} 重建 → 应得 P(c)→Q(c)
    binding_c = {"_t0": "c"}
    instantiated_c = instantiate_output(output_sig, binding_c)
    expected_c = P.impl(P.predicate("P", "c"), P.predicate("Q", "c"))
    assert instantiated_c == expected_c, \
        f"expected {expected_c}, got {instantiated_c}"

    # 测试嵌套结构：¬(P(a) ∧ Q(a))
    nested = P.neg(P.conj(Pa, Qa))
    nested_sig, _ = structure_sig(nested)
    nested_inst = instantiate_output(nested_sig, {"_t0": "z"})
    expected_nested = P.neg(P.conj(P.predicate("P", "z"), P.predicate("Q", "z")))
    assert nested_inst == expected_nested, \
        f"nested: expected {expected_nested}, got {nested_inst}"

    # 测试多词项：P(a)→P(b)（两个不同词项）
    multi = P.impl(Pa, P.predicate("P", "b"))
    multi_sig, multi_tm = structure_sig(multi)
    # multi_tm = {"a":"_t0", "b":"_t1"}
    multi_inst = instantiate_output(multi_sig, {"_t0": "c", "_t1": "d"})
    expected_multi = P.impl(P.predicate("P", "c"), P.predicate("P", "d"))
    assert multi_inst == expected_multi, \
        f"multi: expected {expected_multi}, got {multi_inst}"

    # 验证 instantiate_output 函数签名不包含 operation_name 参数
    sig_params = list(inspect.signature(instantiate_output).parameters.keys())
    assert "operation_name" not in sig_params, \
        f"instantiate_output should not take operation_name parameter: {sig_params}"

    print("test_instantiate_output_without_operation_name OK")


# ============================================================
# 3. test_new_term_can_be_instantiated
# ============================================================

def test_new_term_can_be_instantiated():
    """新词项可以实例化。

    历史：P(a), Q(a) → P(a)→Q(a)
    测试：P(b), Q(b) → P(b)→Q(b) 作为候选

    b 在历史 transformation 中没有出现过。
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    # 新对象 P(b), Q(b)
    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    candidates = generate_candidates_from_transforms([Pb, Qb], store, belief)

    assert len(candidates) > 0, "should generate candidates for P(b), Q(b)"

    # 检查 P(b)→Q(b) 在候选中
    impl_b = P.impl(Pb, Qb)
    props = [c["proposition"] for c in candidates]
    assert impl_b in props, \
        f"P(b)→Q(b) should be in candidates, got {[p.to_str() for p in props]}"

    # 用从未出现过的对象 z
    Pz, Qz = P.predicate("P", "z"), P.predicate("Q", "z")
    candidates_z = generate_candidates_from_transforms([Pz, Qz], store, belief)
    impl_z = P.impl(Pz, Qz)
    props_z = [c["proposition"] for c in candidates_z]
    assert impl_z in props_z, \
        "P(z)→Q(z) should be generated for completely new object z"

    print("test_new_term_can_be_instantiated OK")


# ============================================================
# 4. test_shared_variable_binding_preserved
# ============================================================

def test_shared_variable_binding_preserved():
    """共享变量绑定被保留。

    历史：(P(a), Q(a)) → P(a)→Q(a)
    - P(a) 和 Q(a) 共享词项 a → 在 sig 中都是 _t0
    测试：P(b), Q(b) 匹配时，binding 必须一致（_t0→b）
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    candidates = generate_candidates_from_transforms([Pb, Qb], store, belief)

    # 找到 P(b)→Q(b) 候选
    impl_b_candidate = None
    for c in candidates:
        if c["proposition"] == P.impl(Pb, Qb):
            impl_b_candidate = c
            break

    assert impl_b_candidate is not None, "P(b)→Q(b) candidate should exist"
    assert impl_b_candidate["binding"] == {"_t0": "b"}, \
        f"binding should be {{_t0: b}}, got {impl_b_candidate['binding']}"

    # 测试二元关系中的共享绑定
    Rab = P.relation("R", "a", "b")
    neg_rab = P.neg(Rab)

    store2 = TransformationStore()
    store2.record(
        inputs=(Rab,), output=neg_rab, operation_name="neg",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=0.5, step=0, confidence=0.8,
    )

    Rcd = P.relation("R", "c", "d")
    candidates2 = generate_candidates_from_transforms([Rcd], store2, belief)

    assert len(candidates2) > 0, "should generate ¬R(c,d) from ¬R(a,b)"
    expected_neg = P.neg(Rcd)
    props2 = [c["proposition"] for c in candidates2]
    assert expected_neg in props2, \
        f"¬R(c,d) should be in candidates, got {[p.to_str() for p in props2]}"

    print("test_shared_variable_binding_preserved OK")


# ============================================================
# 5. test_inconsistent_binding_rejected
# ============================================================

def test_inconsistent_binding_rejected():
    """不一致的词项绑定被拒绝。

    历史：(P(a), Q(a)) → P(a)→Q(a) — 两个输入共享词项 a（_t0）
    测试：P(b), Q(c) — 两个输入词项不同（b≠c）
    binding 检查应拒绝：_t0→b 和 _t0→c 冲突
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    # P(b), Q(c) — 词项不一致
    Pb = P.predicate("P", "b")
    Qc = P.predicate("Q", "c")

    matches = store.find_matches([Pb, Qc])
    assert len(matches) == 0, \
        f"P(b),Q(c) should not match (inconsistent binding), got {len(matches)} matches"

    candidates = generate_candidates_from_transforms([Pb, Qc], store, belief)
    assert len(candidates) == 0, \
        f"should generate 0 candidates for P(b),Q(c), got {len(candidates)}"

    print("test_inconsistent_binding_rejected OK")


# ============================================================
# 6. test_counterfactual_no_false_candidate
# ============================================================

def test_counterfactual_no_false_candidate():
    """反事实测试：不能因为 P(b) 与 P(a) 相似就错误产生 P(b)→Q(b)。

    历史：(P(a), Q(a)) → P(a)→Q(a)
    测试：P(b), R(b) — 缺少 Q(b)，结构不匹配
    不应生成 P(b)→Q(b) 候选
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb = P.predicate("P", "b")
    Rb = P.predicate("R", "b")  # R 而非 Q

    matches = store.find_matches([Pb, Rb])
    assert len(matches) == 0, \
        f"P(b),R(b) should not match (R≠Q), got {len(matches)}"

    candidates = generate_candidates_from_transforms([Pb, Rb], store, belief)
    assert len(candidates) == 0, \
        f"should generate 0 candidates for P(b),R(b), got {len(candidates)}"

    # 特别检查没有 P(b)→Q(b) 被生成
    impl_b = P.impl(Pb, P.predicate("Q", "b"))
    for c in candidates:
        assert c["proposition"] != impl_b, \
            "P(b)→Q(b) should NOT be generated when Q(b) is absent"

    print("test_counterfactual_no_false_candidate OK")


# ============================================================
# 7. test_unverified_candidate_not_valid_belief
# ============================================================

def test_unverified_candidate_not_valid_belief():
    """未经验证的候选不进入 VALID belief。

    候选生成后，在验证之前，不能进入 VALID belief。
    """
    # 检查 generate_candidates_from_transforms 源码不修改 belief_store
    src = inspect.getsource(generate_candidates_from_transforms)
    assert "update_belief" not in src, \
        "generate_candidates_from_transforms should not update beliefs"
    assert "STATUS_VALID" not in src, \
        "generate_candidates_from_transforms should not set status"

    # 检查 instantiate_output 源码不修改 belief_store
    src = inspect.getsource(instantiate_output)
    assert "belief" not in src.lower(), \
        "instantiate_output should not reference beliefs"

    # 直接测试：候选生成后 belief store 为空
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    candidates = generate_candidates_from_transforms([Pb, Qb], store, belief)

    # 候选生成后，belief store 应该没有 P(b)→Q(b)
    impl_b = P.impl(Pb, Qb)
    assert not belief.has(impl_b), \
        "P(b)→Q(b) should NOT be in belief store before verification"

    print("test_unverified_candidate_not_valid_belief OK")


# ============================================================
# 8. test_verified_candidate_enters_valid_belief
# ============================================================

def test_verified_candidate_enters_valid_belief():
    """经验证的候选进入 VALID belief。

    P(c)→Q(c) 被生成后，经过验证（step 4 有 P(c) 和 Q(c)），
    应进入 VALID belief。
    """
    result = _get_result()

    # Phase 2 应该将 P(c)→Q(c) 验证为 valid
    bc = result["phase2_belief_check"]
    assert bc["Pc_Qc_before"] is False, \
        "P(c)→Q(c) should not exist before Phase 2"
    assert bc["Pc_Qc_after"] is True, \
        "P(c)→Q(c) should exist after Phase 2"
    assert bc["Pc_Qc_status"] == STATUS_VALID, \
        f"P(c)→Q(c) should be VALID, got {bc['Pc_Qc_status']}"

    print("test_verified_candidate_enters_valid_belief OK")


# ============================================================
# 9. test_refuted_candidate_not_valid
# ============================================================

def test_refuted_candidate_not_valid():
    """被反驳的候选不进入 VALID belief。

    如果候选验证为 invalid，不能保留为 VALID。
    """
    store = TransformationStore()
    belief = BeliefStore()

    # 记录一个 INVALID 变换：S(a)→T(a) 被 refuted
    Sa, Ta = P.predicate("S", "a"), P.predicate("T", "a")
    impl_sa_ta = P.impl(Sa, Ta)

    store.record(
        inputs=(Sa, Ta), output=impl_sa_ta, operation_name="impl",
        context_sig="test", verification_result="invalid",
        usefulness=0.0, cost=1.0, step=0, confidence=0.8,
    )

    # 用 S(b), T(b) 生成候选
    Sb, Tb = P.predicate("S", "b"), P.predicate("T", "b")
    candidates = generate_candidates_from_transforms([Sb, Tb], store, belief)

    assert len(candidates) > 0, "should generate S(b)→T(b) candidate"

    # 用一个有反例的历史验证 S(b)→T(b)
    # step 0: S(b), T(b) 共现 → support
    # step 1: S(b) without T(b) → refute
    visible = [{Sb, Tb}, {Sb}]

    for c in candidates:
        prop = c["proposition"]
        verdict, conf, cost = verify_proposition(prop, visible)
        if prop == P.impl(Sb, Tb):
            assert verdict == "invalid", \
                f"S(b)→T(b) should be invalid (refuted at step 1), got {verdict}"
            # 更新 belief
            belief.update_belief(prop, STATUS_INVALID, conf, evidence_count_delta=1)
            b = belief.get(prop)
            assert b.status == STATUS_INVALID, \
                "refuted candidate should be INVALID, not VALID"

    print("test_refuted_candidate_not_valid OK")


# ============================================================
# 10. test_no_forall_generalization
# ============================================================

def test_no_forall_generalization():
    """不允许直接生成 ∀x(P(x)→Q(x)) 泛化。

    P(a)→Q(a) 的历史不能直接推广为全称命题。
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    candidates = generate_candidates_from_transforms([Pb, Qb], store, belief)

    # 检查没有 forall/exists 命题被生成
    for c in candidates:
        prop = c["proposition"]
        assert prop.kind not in ("forall", "exists"), \
            f"should not generate quantified proposition: {prop.to_str()}"

    # 检查实验结果中也没有
    result = _get_result()
    assert result["checks"]["check_no_forall_generated"], \
        "experiment should not generate forall/exists"

    print("test_no_forall_generalization OK")


# ============================================================
# 11. test_operation_name_unknown_still_works
# ============================================================

def test_operation_name_unknown_still_works():
    """operation_name="UNKNOWN" 时，匹配和实例化仍然工作。

    构造一个 store，所有 record 的 operation_name 都是 "UNKNOWN"。
    用 P(b), Q(b) 匹配，应该仍然找到变换并实例化候选。
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    # 记录变换，operation_name = "UNKNOWN"
    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="UNKNOWN",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")

    # 匹配应该成功
    matches = store.find_matches([Pb, Qb])
    assert len(matches) > 0, \
        "matching should work with operation_name='UNKNOWN'"

    # 实例化应该成功
    candidates = generate_candidates_from_transforms([Pb, Qb], store, belief)
    assert len(candidates) > 0, \
        "instantiation should work with operation_name='UNKNOWN'"

    # P(b)→Q(b) 应该在候选中
    impl_b = P.impl(Pb, Qb)
    props = [c["proposition"] for c in candidates]
    assert impl_b in props, \
        "P(b)→Q(b) should be generated even with operation_name='UNKNOWN'"

    # 实验的 Phase 3 也应该工作
    result = _get_result()
    p3 = result["phase3_unknown_ops"]
    assert p3["instantiation_candidates"] > 0, \
        "Phase 3 should generate candidates with operation_name='UNKNOWN'"
    assert result["checks"]["check_phase3_works_with_unknown_ops"], \
        "Phase 3 check should pass"
    assert result["checks"]["check_phase3_Pd_Qd_verified_valid"], \
        "Phase 3 should verify P(d)→Q(d) as valid"

    print("test_operation_name_unknown_still_works OK")


# ============================================================
# 12. test_operation_name_fake_still_works
# ============================================================

def test_operation_name_fake_still_works():
    """operation_name 设为不存在的名字时，匹配和实例化仍然工作。

    这证明变换的匹配和实例化完全独立于 operation identity。
    """
    store = TransformationStore()
    belief = BeliefStore()

    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    # 用一个完全不存在的 operation_name
    store.record(
        inputs=(Pa, Qa), output=impl_a,
        operation_name="NON_EXISTENT_OPERATION",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")

    matches = store.find_matches([Pb, Qb])
    assert len(matches) > 0, \
        "matching should work with non-existent operation_name"

    candidates = generate_candidates_from_transforms([Pb, Qb], store, belief)
    assert len(candidates) > 0, \
        "instantiation should work with non-existent operation_name"

    impl_b = P.impl(Pb, Qb)
    props = [c["proposition"] for c in candidates]
    assert impl_b in props, \
        "P(b)→Q(b) should be generated even with non-existent operation_name"

    # 验证 generate_candidates_from_transforms 函数签名不包含 operation_name
    gen_params = list(inspect.signature(generate_candidates_from_transforms).parameters.keys())
    assert "operation_name" not in gen_params, \
        f"generate_candidates_from_transforms should not take operation_name: {gen_params}"

    print("test_operation_name_fake_still_works OK")


# ============================================================
# 13. test_rab_rba_raa_structure
# ============================================================

def test_rab_rba_raa_structure():
    """R(a,b), R(b,a), R(a,a) 的结构签名关系。

    R(a,b) 和 R(b,a) 签名相同（都是二元关系+2个不同词项）。
    R(a,a) 签名不同（1个词项重复）。

    这是正确的行为：structure_sig 抽象具体词项，保留位置共享关系。
    R(a,b)→R(b,a) 的类比迁移会交换 a↔b，保持位置关系。
    """
    Rab = P.relation("R", "a", "b")
    Rba = P.relation("R", "b", "a")
    Raa = P.relation("R", "a", "a")

    sig_ab, tm_ab = structure_sig(Rab)
    sig_ba, tm_ba = structure_sig(Rba)
    sig_aa, tm_aa = structure_sig(Raa)

    # R(a,b) 和 R(b,a) 有相同签名
    assert sig_ab == sig_ba, \
        f"R(a,b) and R(b,a) should have same sig: {sig_ab} vs {sig_ba}"

    # R(a,a) 不同（词项重复）
    assert sig_ab != sig_aa, \
        f"R(a,b) and R(a,a) should have different sigs: {sig_ab} vs {sig_aa}"

    # 验证具体签名
    assert sig_ab == ("relation", "R", ("_t0", "_t1")), \
        f"R(a,b) sig: {sig_ab}"
    assert sig_aa == ("relation", "R", ("_t0", "_t0")), \
        f"R(a,a) sig: {sig_aa}"

    # term_map 不同：R(a,b) 中 a→_t0, b→_t1；R(b,a) 中 b→_t0, a→_t1
    assert tm_ab == {"a": "_t0", "b": "_t1"}, f"tm_ab: {tm_ab}"
    assert tm_ba == {"b": "_t0", "a": "_t1"}, f"tm_ba: {tm_ba}"

    # R(a,a) 只有一个词项
    assert tm_aa == {"a": "_t0"}, f"tm_aa: {tm_aa}"

    # 测试匹配：R(a,b) 记录 vs R(b,a) 当前
    store = TransformationStore()
    store.record(
        inputs=(Rab,), output=P.neg(Rab), operation_name="neg",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=0.5, step=0, confidence=0.8,
    )

    matches = store.find_matches([Rba])
    assert len(matches) > 0, "R(b,a) should match R(a,b) structurally"

    _, _, binding = matches[0]
    # binding 从 R(b,a) 的 term_map: {b:_t0, a:_t1}
    assert binding == {"_t0": "b", "_t1": "a"}, \
        f"binding should be {{_t0:b, _t1:a}}, got {binding}"

    # 实例化 ¬R(a,b) 用 binding {_t0:b, _t1:a} → ¬R(b,a)
    output_sig = matches[0][0].output_sig
    instantiated = instantiate_output(output_sig, binding)
    assert instantiated == P.neg(Rba), \
        f"should be ¬R(b,a), got {instantiated}"

    # R(a,a) 不应匹配 R(a,b) 的记录
    matches_aa = store.find_matches([Raa])
    assert len(matches_aa) == 0, \
        "R(a,a) should not match R(a,b) (different structure)"

    print("test_rab_rba_raa_structure OK")


# ============================================================
# 14. test_three_layer_separation
# ============================================================

def test_three_layer_separation():
    """三个层次分离：recognition / instantiation / execution。

    A. Recognition：find_matches 发现历史变换
    B. Instantiation：instantiate_output 从 sig+binding 重建输出
    C. Execution：verify + belief update 执行候选

    三者必须可独立测试，不能混在一起。
    """
    result = _get_result()
    tls = result["three_layer_separation"]

    # Layer A: Recognition
    a = tls["layer_a_recognition"]
    assert a["passed"], \
        f"recognition layer failed: {a}"
    assert a["matches_found"] > 0, "should find matches"

    # Layer B: Instantiation
    b = tls["layer_b_instantiation"]
    assert b["passed"], \
        f"instantiation layer failed: {b}"
    assert b["candidates_instantiated"] > 0, "should instantiate candidates"
    assert b["contains_Pc_Qc"], "should instantiate P(c)→Q(c)"

    # Layer C: Execution
    c = tls["layer_c_execution"]
    assert c["passed"], \
        f"execution layer failed: {c}"
    assert c["verdict_for_Pc_Qc"] == "valid", "P(c)→Q(c) should verify as valid"
    assert c["belief_status"] == STATUS_VALID, "P(c)→Q(c) should be VALID in belief"

    # 检查三个函数的源码不互相包含
    src_recognition = inspect.getsource(TransformationStore.find_matches)
    src_instantiation = inspect.getsource(instantiate_output)
    src_generation = inspect.getsource(generate_candidates_from_transforms)

    # find_matches 不调用 instantiate_output
    assert "instantiate_output" not in src_recognition, \
        "find_matches should not call instantiate_output"

    # instantiate_output 不调用 find_matches 或 verify
    assert "find_matches" not in src_instantiation, \
        "instantiate_output should not call find_matches"
    assert "verify" not in src_instantiation.lower(), \
        "instantiate_output should not call verify"

    print("test_three_layer_separation OK")


# ============================================================
# 15. test_no_future_information
# ============================================================

def test_no_future_information():
    """不读取未来信息。

    Phase 2 的 visible_history 只包含到当前步骤为止的历史。
    """
    # 检查 verify_proposition 不访问 full_history
    src = inspect.getsource(verify_proposition)
    assert "full_history" not in src, \
        "verify_proposition should not access full_history"

    # 检查 generate_candidates_from_transforms 不访问未来
    src = inspect.getsource(generate_candidates_from_transforms)
    assert "full_history" not in src, \
        "generate_candidates_from_transforms should not access full_history"
    assert "ground_truth" not in src, \
        "generate_candidates_from_transforms should not access ground_truth"

    # 检查 instantiate_output 不访问未来
    src = inspect.getsource(instantiate_output)
    assert "full_history" not in src, \
        "instantiate_output should not access full_history"

    # 检查实验中 visible_history_length 正确
    result = _get_result()
    p2 = result["phase2_strict"]
    assert p2["visible_history_length"] == 5, \
        f"Phase 2 should see 5 steps, got {p2['visible_history_length']}"

    p3 = result["phase3_unknown_ops"]
    assert p3["visible_history_length"] == 6, \
        f"Phase 3 should see 6 steps, got {p3['visible_history_length']}"

    print("test_no_future_information OK")


# ============================================================
# 16. test_experiment_checks_pass
# ============================================================

def test_experiment_checks_pass():
    """E0-7.1 实验的所有内部检查通过。"""
    result = _get_result()
    assert result["all_checks_passed"], \
        f"not all checks passed: {result['checks']}"

    # 关键检查
    checks = result["checks"]
    assert checks["check_transform_history_recorded"]
    assert checks["check_phase2_candidates_from_transforms"]
    assert checks["check_phase2_Pc_Qc_verified_valid"]
    assert checks["check_phase3_works_with_unknown_ops"]
    assert checks["check_phase3_Pd_Qd_verified_valid"]
    assert checks["check_counterfactual_no_false_candidate"]
    assert checks["check_binding_inconsistency_rejected"]
    assert checks["check_three_layer_separation"]

    print("test_experiment_checks_pass OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_transformation_matching_does_not_depend_on_operation_name,
        test_instantiate_output_without_operation_name,
        test_new_term_can_be_instantiated,
        test_shared_variable_binding_preserved,
        test_inconsistent_binding_rejected,
        test_counterfactual_no_false_candidate,
        test_unverified_candidate_not_valid_belief,
        test_verified_candidate_enters_valid_belief,
        test_refuted_candidate_not_valid,
        test_no_forall_generalization,
        test_operation_name_unknown_still_works,
        test_operation_name_fake_still_works,
        test_rab_rba_raa_structure,
        test_three_layer_separation,
        test_no_future_information,
        test_experiment_checks_pass,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"{t.__name__} FAIL: {e}")
        except Exception as e:
            print(f"{t.__name__} ERROR: {type(e).__name__}: {e}")
    print(f"\nE0-7.1 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

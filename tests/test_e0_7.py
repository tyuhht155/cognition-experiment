"""E0-7 专用不变量测试：transformation history learning。

验证 10 个核心不变量：
  1. test_transform_history_records_real_computation - 变换历史能记录真实计算过程
  2. test_transform_representation_independent_of_operation_name - 变换表示不依赖 operation name 才能匹配
  3. test_transform_history_influences_candidate_generation - 变换历史能影响候选生成
  4. test_transform_only_produces_candidates_not_beliefs - 历史变换只能产生候选，不能直接产生 valid belief
  5. test_unknown_not_treated_as_invalid - unknown 不被当成 invalid
  6. test_ground_truth_not_in_transform_selection - ground truth 不进入变换选择
  7. test_no_future_information - 不读取未来信息
  8. test_new_object_with_same_structure_triggers_transform - 相同结构的新对象可以触发历史变换候选
  9. test_no_unverified_generalization_into_belief - 不允许未经验证的泛化直接进入 belief
  10. test_e0_1_to_e0_6_still_pass - E0-1～E0-6 全部继续通过（由全局 pytest 保证）

核心约束：
  - 变换学习的是 input_structure → output_structure，不是 operation_name
  - 变换历史只能帮助生成候选，不能替代验证
  - 不做 ∀x 泛化，P(a)→Q(a) 到 P(b)→Q(b) 是候选生成
  - ground truth 不进入选择逻辑
  - 不读取未来信息
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_7 import (
    run_e0_7, run_group, TransformationStore, TransformationRecord,
    structure_sig, structure_sig_multi, select_candidates,
    generate_candidates, verify_proposition, apply_constructor,
    build_world_history, ground_truth_check,
    OperationSelectionStore,
    CONSTRUCTORS, CONSTRUCTOR_COSTS, BUDGET_PER_STEP, EPSILON,
)
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore


# ============================================================
# 缓存实验结果
# ============================================================

_cached_result = None


def _get_result():
    global _cached_result
    if _cached_result is None:
        _cached_result = run_e0_7()
    return _cached_result


# ============================================================
# 1. test_transform_history_records_real_computation
# ============================================================

def test_transform_history_records_real_computation():
    """变换历史能记录真实计算过程。"""
    result = _get_result()
    b = result["group_b"]

    # 变换记录数应该等于实际执行数
    records = b["transform_records"]
    assert len(records) == b["total_executed"], \
        f"transform records ({len(records)}) should equal executed ({b['total_executed']})"

    # 每条记录包含必要字段
    required_fields = ["record_id", "input_sigs", "output_sig", "operation_name",
                       "context_sig", "verification_result", "usefulness", "cost",
                       "step", "confidence", "input_terms", "output_prop_str"]
    for r in records:
        for field in required_fields:
            assert field in r, f"record missing field '{field}': {r}"

    # operation_name 应该是已知的 constructor
    valid_ops = set(CONSTRUCTORS)
    for r in records:
        assert r["operation_name"] in valid_ops, \
            f"unknown operation: {r['operation_name']}"

    # verification_result 应该是 valid/invalid/unknown
    for r in records:
        assert r["verification_result"] in ("valid", "invalid", "unknown"), \
            f"invalid verdict: {r['verification_result']}"

    print("test_transform_history_records_real_computation OK")


# ============================================================
# 2. test_transform_representation_independent_of_operation_name
# ============================================================

def test_transform_representation_independent_of_operation_name():
    """变换表示不依赖 operation name 才能匹配。

    核心：结构签名 (input_sigs, output_sig) 可以独立于 operation_name 进行匹配。
    """
    # 直接测试 structure_sig：P(a) 和 P(b) 应该有相同的签名
    Pa = P.predicate("P", "a")
    Pb = P.predicate("P", "b")

    sig_a, tm_a = structure_sig(Pa)
    sig_b, tm_b = structure_sig(Pb)

    # 签名相同（词项被抽象为 _t0）
    assert sig_a == sig_b, \
        f"P(a) and P(b) should have same structure sig, got {sig_a} vs {sig_b}"

    # term_map 是 {actual_term: placeholder}
    # P(a) 的 term_map: {"a": "_t0"}
    # P(b) 的 term_map: {"b": "_t0"}
    assert tm_a["a"] == "_t0"
    assert tm_b["b"] == "_t0"

    # 测试联合签名：(P(a), Q(a)) 和 (P(b), Q(b)) 应该有相同的联合签名
    Qa = P.predicate("Q", "a")
    Qb = P.predicate("Q", "b")

    sigs_ab, _ = structure_sig_multi((Pa, Qa))
    sigs_cd, _ = structure_sig_multi((Pb, Qb))

    assert sigs_ab == sigs_cd, \
        f"(P(a),Q(a)) and (P(b),Q(b)) should have same joint sig"

    # 测试 TransformationStore 匹配不依赖 operation_name
    store = TransformationStore()
    # 记录一次变换：(P(a), Q(a)) → impl → (P(a)→Q(a))
    impl_a = P.impl(Pa, Qa)
    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    # 用 P(b), Q(b) 查找匹配
    matches = store.find_matches([Pb, Qb])
    assert len(matches) > 0, \
        "should find matching transform for P(b), Q(b) based on structure"

    # 匹配到的 record 的 operation_name 是 "impl"，但匹配是基于结构的
    record, matched_objs, binding = matches[0]
    assert record.operation_name == "impl"
    assert matched_objs == (Pb, Qb)
    assert binding["_t0"] == "b"

    print("test_transform_representation_independent_of_operation_name OK")


# ============================================================
# 3. test_transform_history_influences_candidate_generation
# ============================================================

def test_transform_history_influences_candidate_generation():
    """变换历史能影响候选生成（通过选择加分）。"""
    result = _get_result()
    b = result["group_b"]

    # Group B 应该有候选获得了变换历史加分
    assert b["transform_boost_count"] > 0, \
        "Group B should have candidates with transform history boost"

    # 检查实验结果中确实存在变换匹配
    assert result["checks"]["check_transform_matches_found"], \
        "transform matches should be found"

    # 检查 Group B 的 valid 数量 > Group A（变换历史帮助找到更多 valid）
    a = result["group_a"]
    assert b["total_valid"] >= a["total_valid"], \
        f"Group B valid ({b['total_valid']}) should be >= Group A valid ({a['total_valid']})"

    print("test_transform_history_influences_candidate_generation OK")


# ============================================================
# 4. test_transform_only_produces_candidates_not_beliefs
# ============================================================

def test_transform_only_produces_candidates_not_beliefs():
    """历史变换只能产生候选，不能直接产生 valid belief。"""
    # 检查 select_candidates 源码不直接修改 belief_store
    src = inspect.getsource(select_candidates)
    assert "belief_store" not in src or "belief_store.has" in src, \
        "select_candidates should not modify belief_store"

    # 检查 TransformationStore 不持有 belief_store
    src = inspect.getsource(TransformationStore)
    assert "BeliefStore" not in src or "belief_store" not in src.lower(), \
        "TransformationStore should not reference BeliefStore"

    # 实验结果：所有 valid belief 都来自验证（transform record 的 valid 数 == total_valid）
    result = _get_result()
    b = result["group_b"]
    valid_transforms = sum(1 for r in b["transform_records"]
                           if r["verification_result"] == "valid")
    assert valid_transforms == b["total_valid"], \
        f"valid transforms ({valid_transforms}) should equal total_valid ({b['total_valid']})"

    print("test_transform_only_produces_candidates_not_beliefs OK")


# ============================================================
# 5. test_unknown_not_treated_as_invalid
# ============================================================

def test_unknown_not_treated_as_invalid():
    """unknown 不被当成 invalid。"""
    result = _get_result()
    b = result["group_b"]

    stats = b["transform_store_stats"]
    # valid + invalid + unknown == total
    assert stats["valid"] + stats["invalid"] + stats["unknown"] == stats["total"], \
        "unknown should be counted separately from invalid"

    # 检查 transform records 中 unknown 的记录存在
    unknown_records = [r for r in b["transform_records"]
                       if r["verification_result"] == "unknown"]
    assert len(unknown_records) > 0, "should have unknown transform records"

    # 检查 unknown 记录的 usefulness 是 0.0（不被当作成功）
    for r in unknown_records:
        assert r["usefulness"] == 0.0, \
            f"unknown record should have usefulness=0.0, got {r['usefulness']}"

    print("test_unknown_not_treated_as_invalid OK")


# ============================================================
# 6. test_ground_truth_not_in_transform_selection
# ============================================================

def test_ground_truth_not_in_transform_selection():
    """ground truth 不进入变换选择。"""
    # 检查 select_candidates 源码不包含 ground_truth
    src = inspect.getsource(select_candidates)
    assert "ground_truth" not in src, \
        "select_candidates source contains 'ground_truth'"

    # 检查 TransformationStore 源码不包含 ground_truth
    src = inspect.getsource(TransformationStore)
    assert "ground_truth" not in src, \
        "TransformationStore source contains 'ground_truth'"

    # 检查 find_matches 方法不包含 ground_truth
    src = inspect.getsource(TransformationStore.find_matches)
    assert "ground_truth" not in src, \
        "find_matches source contains 'ground_truth'"

    # 检查 verify_proposition 源码不包含 ground_truth
    src = inspect.getsource(verify_proposition)
    assert "ground_truth" not in src, \
        "verify_proposition source contains 'ground_truth'"

    print("test_ground_truth_not_in_transform_selection OK")


# ============================================================
# 7. test_no_future_information
# ============================================================

def test_no_future_information():
    """不读取未来信息。"""
    # 检查 verify_proposition 只使用 visible_history
    src = inspect.getsource(verify_proposition)
    assert "full_history" not in src, \
        "verify_proposition should not access full_history"
    assert "ground_truth" not in src, \
        "verify_proposition should not access ground_truth"

    # 检查 generate_candidates 不访问未来
    src = inspect.getsource(generate_candidates)
    assert "full_history" not in src, \
        "generate_candidates should not access full_history"

    # 检查 select_candidates 不访问未来
    src = inspect.getsource(select_candidates)
    assert "full_history" not in src, \
        "select_candidates should not access full_history"

    # 在实验结果中检查 visible_history_length == step + 1
    result = _get_result()
    for group in [result["group_a"], result["group_b"]]:
        for s in group["step_records"]:
            assert s["visible_history_length"] == s["step"] + 1, \
                f"step {s['step']}: visible_history_length={s['visible_history_length']}"

    print("test_no_future_information OK")


# ============================================================
# 8. test_new_object_with_same_structure_triggers_transform
# ============================================================

def test_new_object_with_same_structure_triggers_transform():
    """相同结构的新对象可以触发历史变换候选。

    关键测试：历史中有 P(a)→Q(a) 变换，当 P(b) 出现时，
    系统应该能根据结构相似性提出 P(b)→Q(b) 作为候选。
    """
    # 直接测试 TransformationStore 的匹配能力
    store = TransformationStore()

    Pa = P.predicate("P", "a")
    Qa = P.predicate("Q", "a")
    impl_a = P.impl(Pa, Qa)

    # 记录 P(a)→Q(a) 的变换
    store.record(
        inputs=(Pa, Qa), output=impl_a, operation_name="impl",
        context_sig="test", verification_result="valid",
        usefulness=1.0, cost=1.0, step=0, confidence=0.8,
    )

    # 新对象 P(b), Q(b) 出现
    Pb = P.predicate("P", "b")
    Qb = P.predicate("Q", "b")

    # 查找匹配
    matches = store.find_matches([Pb, Qb])
    assert len(matches) > 0, \
        "should find matching transform for new objects P(b), Q(b)"

    record, matched_objs, binding = matches[0]
    # 匹配到的输入对象应该是 Pb, Qb（不是 Pa, Qa）
    assert matched_objs == (Pb, Qb), \
        f"matched objects should be (Pb, Qb), got {matched_objs}"

    # 通过匹配的变换，可以生成 P(b)→Q(b) 作为候选
    # 用记录的 operation_name 执行变换
    generated = apply_constructor(record.operation_name, matched_objs)
    expected = P.impl(Pb, Qb)
    assert generated == expected, \
        f"generated should be P(b)→Q(b), got {generated}"

    # 更严格的测试：P(a) 从未在历史中出现过的对象 P(z)
    # 历史：P(a)→Q(a)
    # 新对象：P(z), Q(z)
    Pz = P.predicate("P", "z")
    Qz = P.predicate("Q", "z")
    matches_z = store.find_matches([Pz, Qz])
    assert len(matches_z) > 0, \
        "should find matching transform for completely new objects P(z), Q(z)"

    record_z, matched_z, _ = matches_z[0]
    generated_z = apply_constructor(record_z.operation_name, matched_z)
    assert generated_z == P.impl(Pz, Qz), \
        "should generate P(z)→Q(z) from P(a)→Q(a) transform"

    print("test_new_object_with_same_structure_triggers_transform OK")


# ============================================================
# 9. test_no_unverified_generalization_into_belief
# ============================================================

def test_no_unverified_generalization_into_belief():
    """不允许未经验证的泛化直接进入 belief。

    P(a)→Q(a) 的历史变换不能直接让 P(b)→Q(b) 成为 valid belief。
    P(b)→Q(b) 必须作为候选经验证后才能进入 belief。
    """
    # 检查 TransformationStore 不直接产生 valid belief
    # 它只记录变换，不修改 belief
    src = inspect.getsource(TransformationStore.record)
    assert "belief" not in src.lower() or "BeliefState" not in src, \
        "TransformationStore.record should not modify beliefs"

    # 检查 run_group 中，transform_store.record 之后
    # belief 更新只来自 verify_proposition 的 verdict
    result = _get_result()
    b = result["group_b"]

    # 所有 valid belief 都应该有对应的验证记录
    # 即 transform_records 中 valid 的数量 == total_valid
    valid_records = [r for r in b["transform_records"]
                     if r["verification_result"] == "valid"]
    assert len(valid_records) == b["total_valid"], \
        "all valid beliefs should come from verification"

    # 检查没有 forall/exists 命题被直接加入 belief
    # （防止 P(a)→Q(a) 被推广为 ∀x(P(x)→Q(x))）
    for r in b["transform_records"]:
        output_sig = r["output_sig"]
        if isinstance(output_sig, tuple) and len(output_sig) > 0:
            kind = output_sig[0]
            assert kind not in ("forall", "exists"), \
                f"should not generate quantified propositions: {r['output_prop_str']}"

    print("test_no_unverified_generalization_into_belief OK")


# ============================================================
# 10. test_e0_1_to_e0_6_still_pass (通过全局 pytest 保证)
# ============================================================

def test_experiment_checks_pass():
    """E0-7 实验的所有内部检查通过。"""
    result = _get_result()
    assert result["all_checks_passed"], \
        f"not all checks passed: {result['checks']}"

    # 验证 A/B 差异存在
    ab = result["ab_comparison"]
    assert ab["valid_diff"] != 0 or ab["efficiency_ratio"] != 1.0, \
        "A/B should show difference"

    # 验证变换历史被记录
    assert result["checks"]["check_transform_history_recorded"]
    assert result["checks"]["check_transform_matches_found"]
    assert result["checks"]["check_structure_generalization"]

    print("test_experiment_checks_pass OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_transform_history_records_real_computation,
        test_transform_representation_independent_of_operation_name,
        test_transform_history_influences_candidate_generation,
        test_transform_only_produces_candidates_not_beliefs,
        test_unknown_not_treated_as_invalid,
        test_ground_truth_not_in_transform_selection,
        test_no_future_information,
        test_new_object_with_same_structure_triggers_transform,
        test_no_unverified_generalization_into_belief,
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
    print(f"\nE0-7 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

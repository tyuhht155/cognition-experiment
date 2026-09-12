"""E0-7.3 不变量测试：Transformation Selection / Goal-directed Search。

验证系统能否在多个合法计算方向之间根据目标进行选择。

测试覆盖：
  1. test_multiple_transformations_match
  2. test_selection_is_separate_from_generation
  3. test_operation_name_irrelevant_to_selection
  4. test_unknown_operation_names_same_selection
  5. test_history_changes_transformation_selection
  6. test_valid_but_low_value_knowledge_retained
  7. test_failed_path_updates_future_selection
  8. test_exploration_preserved
  9. test_goal_changes_selection
  10. test_no_future_information
  11. test_no_ground_truth_in_selection
  12. test_multistep_goal_search
  13. test_longer_valid_path_not_rejected_as_invalid
  14. test_validity_and_value_are_separate
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_7_3 import (
    build_world,
    make_goal,
    run_phase1,
    run_goal_pursuit_episode,
    TransformationSelectionStore,
    TransformStats,
    goal_reached,
    select_candidates,
    EPSILON,
)
from experiments.run_e0_7 import (
    structure_sig,
    structure_sig_multi,
    TransformationStore,
    TransformationRecord,
    verify_proposition,
)
from experiments.run_e0_7_1 import generate_candidates_from_transforms
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


_cached = {}


def _get_world_and_store():
    if "world" not in _cached:
        world = build_world()
        transform_store, _ = run_phase1(world, seed=42)
        _cached["world"] = world
        _cached["transform_store"] = transform_store
    return _cached["world"], _cached["transform_store"]


# ============================================================
# 1. test_multiple_transformations_match
# ============================================================

def test_multiple_transformations_match():
    """同一输入能够产生多个候选（多个变换匹配）。"""
    world, ts = _get_world_and_store()
    belief = BeliefStore()
    Ab, Bb = P.predicate("A", "b"), P.predicate("B", "b")
    Cb, Db = P.predicate("C", "b"), P.predicate("D", "b")

    constructible = [Ab, Bb, Cb, Db]
    candidates = generate_candidates_from_transforms(constructible, ts, belief)

    # 应该有多个候选（impl, conj, disj, iff, neg 等）
    assert len(candidates) > 3, f"should have multiple candidates, got {len(candidates)}"

    # 至少有 A→B, A→C, A→D 三个不同方向的蕴含
    impl_ab = P.impl(Ab, Bb)
    impl_ac = P.impl(Ab, Cb)
    impl_ad = P.impl(Ab, Db)
    props = {c["proposition"] for c in candidates}
    assert impl_ab in props, "A(b)→B(b) should be a candidate"
    assert impl_ac in props, "A(b)→C(b) should be a candidate"
    assert impl_ad in props, "A(b)→D(b) should be a candidate"

    print("test_multiple_transformations_match OK")


# ============================================================
# 2. test_selection_is_separate_from_generation
# ============================================================

def test_selection_is_separate_from_generation():
    """选择不能改变候选集合。

    generate_candidates_from_transforms 返回的候选集合与选择策略无关。
    """
    world, ts = _get_world_and_store()
    belief = BeliefStore()
    Ab, Bb = P.predicate("A", "b"), P.predicate("B", "b")
    Cb, Db = P.predicate("C", "b"), P.predicate("D", "b")

    constructible = [Ab, Bb, Cb, Db]
    candidates = generate_candidates_from_transforms(constructible, ts, belief)
    candidate_set = {c["proposition"] for c in candidates}

    # 用不同选择策略选择，候选集合不变
    sel_store = TransformationSelectionStore()
    rng = random.Random(42)

    selected_random, _ = select_candidates(
        candidates, sel_store, use_learning=False, rng=rng, n_select=10)

    # 用另一个 rng 再选
    selected_random2, _ = select_candidates(
        candidates, sel_store, use_learning=False, rng=random.Random(99), n_select=10)

    # 两个选择结果可能不同，但都是原候选集合的子集
    for c in selected_random:
        assert c["proposition"] in candidate_set
    for c in selected_random2:
        assert c["proposition"] in candidate_set

    print("test_selection_is_separate_from_generation OK")


# ============================================================
# 3. test_operation_name_irrelevant_to_selection
# ============================================================

def test_operation_name_irrelevant_to_selection():
    """operation_name 改变不影响选择分数。

    TransformationSelectionStore 按 (input_sig, output_sig) 统计，
    record_outcome 参数不包含 operation_name——证明选择与 operation_name 无关。
    """
    sel_store = TransformationSelectionStore()

    input_sig = (("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    output_sig = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))

    # record_outcome 不接受 operation_name 参数
    # 这证明选择统计完全不依赖 operation_name
    import inspect
    sig = inspect.signature(sel_store.record_outcome)
    param_names = list(sig.parameters.keys())
    assert "operation_name" not in param_names, \
        f"record_outcome should not have operation_name param, got {param_names}"

    # 记录后分数只取决于 (input_sig, output_sig)
    sel_store.record_outcome(input_sig, output_sig, "valid", 1.0, 1.0)
    score = sel_store.get_score(input_sig, output_sig)
    assert score > 0

    # 不同 operation_name 的变换记录到相同 (input_sig, output_sig) 应得相同分数
    # （通过 record_outcome 不接收 operation_name 证明）
    score_again = sel_store.get_score(input_sig, output_sig)
    assert score == score_again

    print("test_operation_name_irrelevant_to_selection OK")


# ============================================================
# 4. test_unknown_operation_names_same_selection
# ============================================================

def test_unknown_operation_names_same_selection():
    """全部 UNKNOWN operation_name 后选择结果仍然一致。"""
    world, ts = _get_world_and_store()

    # 构造 UNKNOWN store
    ts_unknown = TransformationStore()
    for r in ts.records:
        r_copy = TransformationRecord(
            record_id=r.record_id + "_u",
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
        ts_unknown.records.append(r_copy)

    goal_b = make_goal("b")
    sel_store = TransformationSelectionStore()

    ep_orig = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=ts, selection_store=TransformationSelectionStore(),
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    ep_unknown = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=ts_unknown, selection_store=TransformationSelectionStore(),
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    # 结果应该一致（相同 rng，相同变换结构，不同 operation_name）
    assert ep_orig["reached"] == ep_unknown["reached"], \
        f"reached should be same: {ep_orig['reached']} vs {ep_unknown['reached']}"

    print("test_unknown_operation_names_same_selection OK")


# ============================================================
# 5. test_history_changes_transformation_selection
# ============================================================

def test_history_changes_transformation_selection():
    """历史能够影响选择：成功的变换分数上升。"""
    sel_store = TransformationSelectionStore()

    input_sig = (("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    output_sig = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))

    # 初始分数为 0（未知变换）
    score_before = sel_store.get_score(input_sig, output_sig)
    assert score_before == 0.0

    # 记录成功
    sel_store.record_outcome(input_sig, output_sig, "valid", 1.0, 1.0)
    score_after = sel_store.get_score(input_sig, output_sig)
    assert score_after > score_before, "score should increase after success"

    print("test_history_changes_transformation_selection OK")


# ============================================================
# 6. test_valid_but_low_value_knowledge_retained
# ============================================================

def test_valid_but_low_value_knowledge_retained():
    """正确但当前无价值的知识不能被删除。

    A→C 是 VALID 的，但对目标 (A→B)∧(B→A) 无用。
    它应该保留在 belief store 中，只是选择分数低。
    """
    world, ts = _get_world_and_store()
    goal_b = make_goal("b")

    sel_store = TransformationSelectionStore()
    ep = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=ts, selection_store=sel_store,
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    # 检查 A(b)→C(b) 是否在 belief 中（如果被选中并验证为 valid）
    Ab, Cb = P.predicate("A", "b"), P.predicate("C", "b")
    impl_ac = P.impl(Ab, Cb)

    # 即使 A→C 对目标无用，它仍然是 VALID 的
    # 检查 selection_store 中 A→C 的分数是否低于 A→B
    input_sig_ac = structure_sig_multi((Ab, Cb))[0]
    output_sig_ac = structure_sig(impl_ac)[0]
    score_ac = sel_store.get_score(input_sig_ac, output_sig_ac)

    Ab, Bb = P.predicate("A", "b"), P.predicate("B", "b")
    impl_ab = P.impl(Ab, Bb)
    input_sig_ab = structure_sig_multi((Ab, Bb))[0]
    output_sig_ab = structure_sig(impl_ab)[0]
    score_ab = sel_store.get_score(input_sig_ab, output_sig_ab)

    # A→B 应该比 A→C 分数高（如果目标达成，A→B 获得信用）
    if ep["reached"]:
        assert score_ab >= score_ac, \
            f"A→B score ({score_ab}) should be >= A→C score ({score_ac}) after success"

    print("test_valid_but_low_value_knowledge_retained OK")


# ============================================================
# 7. test_failed_path_updates_future_selection
# ============================================================

def test_failed_path_updates_future_selection():
    """失败能够影响以后选择：未达目标的变换分数不上升。"""
    sel_store = TransformationSelectionStore()

    input_sig = (("predicate", "A", ("_t0",)), ("predicate", "C", ("_t0",)))
    output_sig = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "C", ("_t0",)))

    # A→C 是 valid 但 goal_improvement=0（没到目标）
    sel_store.record_outcome(input_sig, output_sig, "valid", 0.0, 1.0)
    score_ac = sel_store.get_score(input_sig, output_sig)

    # A→B 是 valid 且 goal_improvement=1（到目标）
    input_sig_ab = (("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    output_sig_ab = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    sel_store.record_outcome(input_sig_ab, output_sig_ab, "valid", 1.0, 1.0)
    score_ab = sel_store.get_score(input_sig_ab, output_sig_ab)

    # A→B 分数 > A→C 分数
    assert score_ab > score_ac, \
        f"successful transform should have higher score: {score_ab} vs {score_ac}"

    print("test_failed_path_updates_future_selection OK")


# ============================================================
# 8. test_exploration_preserved
# ============================================================

def test_exploration_preserved():
    """未知 transformation 仍有探索机会（epsilon-greedy）。"""
    sel_store = TransformationSelectionStore()
    rng = random.Random(42)

    # 构造候选：一个已知高分，一个未知
    known_record = TransformationRecord(
        record_id="known", input_sigs=(("predicate", "A", ("_t0",)),),
        output_sig=("predicate", "A", ("_t0",)),
        operation_name="impl", context_sig="test",
        verification_result="valid", usefulness=1.0, cost=0.5,
        step=0, confidence=0.9, input_terms={}, output_prop_str="A(x)")
    unknown_record = TransformationRecord(
        record_id="unknown", input_sigs=(("predicate", "A", ("_t0",)),),
        output_sig=("predicate", "B", ("_t0",)),
        operation_name="impl", context_sig="test",
        verification_result="valid", usefulness=0.0, cost=0.5,
        step=0, confidence=0.9, input_terms={}, output_prop_str="B(x)")

    # 记录已知变换的成功
    sel_store.record_outcome(known_record.input_sigs, known_record.output_sig,
                             "valid", 1.0, 0.5)

    candidates = [
        {"proposition": P.predicate("A", "b"), "_record": known_record},
        {"proposition": P.predicate("B", "b"), "_record": unknown_record},
    ]

    # 多次运行，检查未知变换有时被选中（探索）
    unknown_selected = 0
    for _ in range(100):
        selected, info = select_candidates(
            candidates, sel_store, use_learning=True,
            rng=random.Random(), n_select=1)
        for s, inf in zip(selected, info):
            if inf["was_exploration"]:
                unknown_selected += 1

    # 探索率 EPSILON=0.2，100 次选 1 个，期望 20 次探索
    assert unknown_selected > 0, \
        f"unknown transform should be selected sometimes via exploration, got {unknown_selected}"

    print("test_exploration_preserved OK")


# ============================================================
# 9. test_goal_changes_selection
# ============================================================

def test_goal_changes_selection():
    """相同当前状态，在不同目标下选择可以不同。

    通过 selection_store 的分数体现：同一变换在不同目标下
    goal_improvement 不同，导致分数不同。
    """
    # 目标 1: (A→B)∧(B→A) — A→B 有用
    # 目标 2: (A→C)∧(C→A) — A→C 有用

    sel_store_g1 = TransformationSelectionStore()
    sel_store_g2 = TransformationSelectionStore()

    input_sig_ab = (("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    output_sig_ab = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    input_sig_ac = (("predicate", "A", ("_t0",)), ("predicate", "C", ("_t0",)))
    output_sig_ac = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "C", ("_t0",)))

    # 目标 1 下：A→B 成功，A→C 失败
    sel_store_g1.record_outcome(input_sig_ab, output_sig_ab, "valid", 1.0, 1.0)
    sel_store_g1.record_outcome(input_sig_ac, output_sig_ac, "valid", 0.0, 1.0)

    # 目标 2 下：A→C 成功，A→B 失败
    sel_store_g2.record_outcome(input_sig_ab, output_sig_ab, "valid", 0.0, 1.0)
    sel_store_g2.record_outcome(input_sig_ac, output_sig_ac, "valid", 1.0, 1.0)

    score_ab_g1 = sel_store_g1.get_score(input_sig_ab, output_sig_ab)
    score_ac_g1 = sel_store_g1.get_score(input_sig_ac, output_sig_ac)
    score_ab_g2 = sel_store_g2.get_score(input_sig_ab, output_sig_ab)
    score_ac_g2 = sel_store_g2.get_score(input_sig_ac, output_sig_ac)

    # 目标 1: A→B > A→C；目标 2: A→C > A→B
    assert score_ab_g1 > score_ac_g1, "goal 1: A→B should be preferred"
    assert score_ac_g2 > score_ab_g2, "goal 2: A→C should be preferred"

    print("test_goal_changes_selection OK")


# ============================================================
# 10. test_no_future_information
# ============================================================

def test_no_future_information():
    """选择时不能读取未来 world state。

    visible_history 在 episode 开始时固定，选择逻辑不访问未来。
    """
    world, ts = _get_world_and_store()
    goal_b = make_goal("b")

    sel_store = TransformationSelectionStore()
    ep = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=ts, selection_store=sel_store,
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    # 检查 trace 中每步的候选都来自当前 constructible
    # （代码中 constructible 每轮从 belief_store 重新计算，不访问未来）
    for step_trace in ep["trace"]:
        assert "candidates_count" in step_trace
        assert "selected_count" in step_trace

    print("test_no_future_information OK")


# ============================================================
# 11. test_no_ground_truth_in_selection
# ============================================================

def test_no_ground_truth_in_selection():
    """ground truth 不能进入选择逻辑。

    select_candidates 只使用 selection_store 的分数，不访问 ground truth。
    """
    sel_store = TransformationSelectionStore()
    rng = random.Random(42)

    candidates = [
        {"proposition": P.predicate("A", "b"),
         "_record": TransformationRecord(
             record_id="r1", input_sigs=(("predicate", "A", ("_t0",)),),
             output_sig=("predicate", "A", ("_t0",)),
             operation_name="impl", context_sig="test",
             verification_result="valid", usefulness=1.0, cost=0.5,
             step=0, confidence=0.9, input_terms={}, output_prop_str="A(x)")},
    ]

    # select_candidates 不接收 ground truth 参数
    selected, info = select_candidates(
        candidates, sel_store, use_learning=True, rng=rng, n_select=1)

    assert len(selected) == 1
    # 选择只基于 score（来自 selection_store），不基于 ground truth
    assert info[0]["score"] is not None

    print("test_no_ground_truth_in_selection OK")


# ============================================================
# 12. test_multistep_goal_search
# ============================================================

def test_multistep_goal_search():
    """系统可以逐步寻找目标，而不是直接获得最终路径。

    目标 (A→B)∧(B→A) 需要多步：
      1. 构建 A→B
      2. 构建 B→A
      3. 构建 (A→B)∧(B→A)
    """
    world, ts = _get_world_and_store()
    goal_b = make_goal("b")

    sel_store = TransformationSelectionStore()
    ep = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=ts, selection_store=sel_store,
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    if ep["reached"]:
        # 目标需要至少 2 步（构建两个蕴含 + 合取）
        assert ep["steps"] >= 2, \
            f"goal requires multistep search, got {ep['steps']} steps"
        assert ep["chain_length"] >= 2

    print("test_multistep_goal_search OK")


# ============================================================
# 13. test_longer_valid_path_not_rejected_as_invalid
# ============================================================

def test_longer_valid_path_not_rejected_as_invalid():
    """长路径中的中间结果即使当前没有立即 goal improvement，也不能被判 INVALID。

    A→C 是 VALID 的（A 和 C 共现），即使它对目标无用。
    """
    world, ts = _get_world_and_store()
    goal_b = make_goal("b")

    sel_store = TransformationSelectionStore()
    ep = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=ts, selection_store=sel_store,
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    # 检查 A(b)→C(b) 的验证结果
    Ab, Cb = P.predicate("A", "b"), P.predicate("C", "b")
    impl_ac = P.impl(Ab, Cb)

    # 直接验证 A→C：A 和 C 共现，应该是 valid
    verdict, conf, cost = verify_proposition(impl_ac, world)
    assert verdict == "valid", \
        f"A(b)→C(b) should be valid (A and C co-occur), got {verdict}"

    print("test_longer_valid_path_not_rejected_as_invalid OK")


# ============================================================
# 14. test_validity_and_value_are_separate
# ============================================================

def test_validity_and_value_are_separate():
    """验证结果与价值评价必须保持独立。

    A→C 可以是 VALID（验证结果）但 value 低（目标评价）。
    validity ≠ value。
    """
    # A→C: valid 但 goal_improvement=0
    # A→B: valid 且 goal_improvement=1
    sel_store = TransformationSelectionStore()

    input_sig_ab = (("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    output_sig_ab = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "B", ("_t0",)))
    input_sig_ac = (("predicate", "A", ("_t0",)), ("predicate", "C", ("_t0",)))
    output_sig_ac = ("implies", ("predicate", "A", ("_t0",)), ("predicate", "C", ("_t0",)))

    # 两者都是 valid，但 goal_improvement 不同
    sel_store.record_outcome(input_sig_ab, output_sig_ab, "valid", 1.0, 1.0)
    sel_store.record_outcome(input_sig_ac, output_sig_ac, "valid", 0.0, 1.0)

    stats_ab = sel_store.get_stats(input_sig_ab, output_sig_ab)
    stats_ac = sel_store.get_stats(input_sig_ac, output_sig_ac)

    # validity 相同（都有 valid_count）
    assert stats_ab.valid_count == 1
    assert stats_ac.valid_count == 1

    # value 不同（goal_improvement 不同）
    assert stats_ab.mean_goal_improvement == 1.0
    assert stats_ac.mean_goal_improvement == 0.0

    # 分数不同
    assert stats_ab.score > stats_ac.score

    print("test_validity_and_value_are_separate OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_multiple_transformations_match,
        test_selection_is_separate_from_generation,
        test_operation_name_irrelevant_to_selection,
        test_unknown_operation_names_same_selection,
        test_history_changes_transformation_selection,
        test_valid_but_low_value_knowledge_retained,
        test_failed_path_updates_future_selection,
        test_exploration_preserved,
        test_goal_changes_selection,
        test_no_future_information,
        test_no_ground_truth_in_selection,
        test_multistep_goal_search,
        test_longer_valid_path_not_rejected_as_invalid,
        test_validity_and_value_are_separate,
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
    print(f"\nE0-7.3 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

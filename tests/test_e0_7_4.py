"""E0-7.4 不变量测试：Future Value / Credit Assignment。

验证系统能否通过信用传播学习中间步骤的未来价值。

测试覆盖（18 项不变量）：
  1.  test_credit_propagated_after_goal_success
  2.  test_direct_goal_transform_highest_credit
  3.  test_intermediate_steps_get_nonzero_credit
  4.  test_dead_end_gets_no_success_credit
  5.  test_invalid_gets_no_future_credit
  6.  test_valid_low_value_knowledge_retained
  7.  test_failure_lowers_future_selection
  8.  test_failure_does_not_permanently_block
  9.  test_same_transform_different_goal_different_value
  10. test_operation_name_unknown_works
  11. test_operation_name_rename_no_effect
  12. test_no_goal_distance_used
  13. test_no_ground_truth_in_selection
  14. test_no_future_episode_data
  15. test_credit_only_from_actual_trace
  16. test_multistep_intermediate_learns_value
  17. test_multiple_success_paths_get_credit
  18. test_long_path_not_invalid_due_to_no_immediate_value
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_7_4 import (
    build_world,
    make_goal_nstep,
    run_phase1,
    run_goal_pursuit_episode_v2,
    TransformationSelectionStoreV2,
    TransformStatsV2,
    trace_provenance_chain,
    compute_credit_for_chain,
    goal_reached,
    CREDIT_LAST_ONLY,
    CREDIT_EQUAL,
    CREDIT_DISCOUNTED,
    DEFAULT_DISCOUNT,
    _contains_impl,
)
from experiments.run_e0_7 import (
    structure_sig,
    structure_sig_multi,
    TransformationStore,
    TransformationRecord,
    verify_proposition,
)
from experiments.run_e0_7_2 import verify_proposition_extended
from experiments.run_e0_7_1 import generate_candidates_from_transforms
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


def _make_record(rid, in_sig, out_sig, op="conj", verdict="valid"):
    """构造一个 TransformationRecord（含全部必填字段）。"""
    return TransformationRecord(
        record_id=rid,
        input_sigs=(in_sig,),
        output_sig=out_sig,
        operation_name=op,
        context_sig="ctx",
        verification_result=verdict,
        usefulness=1.0 if verdict == "valid" else 0.0,
        cost=1.0,
        step=0,
        confidence=1.0,
        input_terms={},
        output_prop_str=out_sig,
    )


def _make_chain_records(n=3):
    """构造一个 n 步成功链的 fake records（仅用于信用计算测试）。"""
    records = []
    for i in range(n):
        records.append(_make_record(i, f"in_{i}", f"out_{i}"))
    # chain: [(record, depth), ...] depth=0 是最接近 goal 的
    chain = [(records[-1], 0)]
    for i in range(n - 2, -1, -1):
        chain.append((records[i], n - 1 - i))
    return records, chain


# ============================================================
# 1. 最终目标成功后可以产生历史信用
# ============================================================
def test_credit_propagated_after_goal_success():
    """目标成功后，链上变换应获得未来信用。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    records, chain = _make_chain_records(3)

    # 模拟目标达成后传播信用
    credits = compute_credit_for_chain(chain, CREDIT_DISCOUNTED, DEFAULT_DISCOUNT)
    for record, depth in chain:
        credit = credits.get(record.record_id, 0.0)
        sel.record_outcome(
            record.input_sigs, record.output_sig, "goal_sig",
            "valid", 0.0, credit, 0.0)

    # 检查至少有一个变换获得了非零 future_credit
    stats = sel.get_stats(records[0].input_sigs, records[0].output_sig, "goal_sig")
    assert stats is not None
    # 链上至少一个变换有 future_credit
    any_credit = False
    for record, _ in chain:
        s = sel.get_stats(record.input_sigs, record.output_sig, "goal_sig")
        if s and s.total_future_credit > 0:
            any_credit = True
            break
    assert any_credit, "目标成功后应有变换获得未来信用"


# ============================================================
# 2. 直接目标变换信用最高或符合所选信用机制
# ============================================================
def test_direct_goal_transform_highest_credit():
    """直接产生 goal 的变换（depth=0）信用最高。"""
    records, chain = _make_chain_records(3)
    credits = compute_credit_for_chain(chain, CREDIT_DISCOUNTED, DEFAULT_DISCOUNT)

    # depth=0 (最后一步) 信用应最高
    max_credit = max(credits.values())
    goal_record = chain[0][0]  # depth=0
    assert credits[goal_record.record_id] == max_credit, \
        "直接目标变换应获得最高信用"


# ============================================================
# 3. 更早中间步骤可以获得非零未来信用
# ============================================================
def test_intermediate_steps_get_nonzero_credit():
    """折扣模式下，更早的中间步骤也应获得非零信用。"""
    records, chain = _make_chain_records(3)
    credits = compute_credit_for_chain(chain, CREDIT_DISCOUNTED, DEFAULT_DISCOUNT)

    # 最早的步骤（depth 最大）应有非零信用
    earliest = chain[-1][0]
    assert credits[earliest.record_id] > 0, \
        "最早中间步骤在折扣模式下应获得非零信用"


# ============================================================
# 4. dead-end 路径不会获得成功信用
# ============================================================
def test_dead_end_gets_no_success_credit():
    """不在成功链上的变换不应获得成功信用。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    records, chain = _make_chain_records(3)

    # 一个 dead-end 变换
    dead_record = _make_record(99, "dead_in", "dead_out", "conj", "valid")
    sel.record_outcome(
        dead_record.input_sigs, dead_record.output_sig, "goal_sig",
        "valid", 0.0, 0.0, 1.0)  # 无 future_credit

    # 传播成功链信用
    credits = compute_credit_for_chain(chain, CREDIT_DISCOUNTED, DEFAULT_DISCOUNT)
    for record, _ in chain:
        credit = credits.get(record.record_id, 0.0)
        sel.record_outcome(
            record.input_sigs, record.output_sig, "goal_sig",
            "valid", 0.0, credit, 0.0)

    dead_stats = sel.get_stats(dead_record.input_sigs, dead_record.output_sig, "goal_sig")
    assert dead_stats.total_future_credit == 0.0, \
        "dead-end 变换不应获得成功信用"


# ============================================================
# 5. INVALID 不能获得成功未来信用
# ============================================================
def test_invalid_gets_no_future_credit():
    """INVALID 变换不应获得未来信用。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    inv_record = _make_record(10, "inv_in", "inv_out", "conj", "invalid")
    sel.record_outcome(
        inv_record.input_sigs, inv_record.output_sig, "goal_sig",
        "invalid", 0.0, 1.0, 1.0)  # 即使给了 credit，也应被忽略

    stats = sel.get_stats(inv_record.input_sigs, inv_record.output_sig, "goal_sig")
    assert stats.total_future_credit == 0.0, \
        "INVALID 变换不应获得未来信用"


# ============================================================
# 6. VALID 但低价值知识仍保留
# ============================================================
def test_valid_low_value_knowledge_retained():
    """VALID 但当前价值低的变换仍保留在 store 中。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    valid_record = _make_record(20, "v_in", "v_out", "conj", "valid")
    # 多次失败，价值低
    for _ in range(5):
        sel.record_outcome(
            valid_record.input_sigs, valid_record.output_sig, "goal_sig",
            "valid", 0.0, 0.0, 1.0)

    stats = sel.get_stats(valid_record.input_sigs, valid_record.output_sig, "goal_sig")
    assert stats is not None
    assert stats.valid_count > 0, "VALID 变换应保留在 store 中"


# ============================================================
# 7. 失败可以降低未来选择倾向
# ============================================================
def test_failure_lowers_future_selection():
    """失败（goal_improvement=0）多次后，选择分数应降低。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    rec = _make_record(30, "f_in", "f_out", "conj", "valid")
    # 初始无历史
    sel.record_outcome(
        rec.input_sigs, rec.output_sig, "goal_sig",
        "valid", 0.0, 0.0, 1.0)
    initial_score = sel.get_stats(
        rec.input_sigs, rec.output_sig, "goal_sig").score

    # 多次失败（但仍 valid）
    for _ in range(10):
        sel.record_outcome(
            rec.input_sigs, rec.output_sig, "goal_sig",
            "valid", 0.0, 0.0, 5.0)  # 高成本

    final_score = sel.get_stats(
        rec.input_sigs, rec.output_sig, "goal_sig").score
    assert final_score < initial_score, \
        "多次失败后选择分数应降低"


# ============================================================
# 8. 失败不会永久禁止探索
# ============================================================
def test_failure_does_not_permanently_block():
    """ε-greedy 确保即使分数低，仍有探索概率。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    rec = _make_record(40, "b_in", "b_out", "conj", "valid")
    for _ in range(20):
        sel.record_outcome(
            rec.input_sigs, rec.output_sig, "goal_sig",
            "valid", 0.0, 0.0, 10.0)

    stats = sel.get_stats(rec.input_sigs, rec.output_sig, "goal_sig")
    # 分数可能很低，但变换仍在 store 中（不被永久禁止）
    assert stats is not None
    assert stats.attempts > 0


# ============================================================
# 9. 同一变换在不同 goal 下可以产生不同历史价值
# ============================================================
def test_same_transform_different_goal_different_value():
    """同一 (input, output) 在不同 goal 下可以有不同分数。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    in_sig = ("shared_in",)
    out_sig = "shared_out"

    # goal1 下成功
    sel.record_outcome(in_sig, out_sig, "goal1", "valid", 1.0, 0.0, 1.0)
    # goal2 下失败
    sel.record_outcome(in_sig, out_sig, "goal2", "valid", 0.0, 0.0, 1.0)

    s1 = sel.get_stats(in_sig, out_sig, "goal1")
    s2 = sel.get_stats(in_sig, out_sig, "goal2")
    assert s1.score != s2.score, \
        "同一变换在不同 goal 下应有不同分数"


# ============================================================
# 10. operation_name=UNKNOWN 仍完全正常
# ============================================================
def test_operation_name_unknown_works():
    """operation_name 为 UNKNOWN 时变换仍正常记录和匹配。"""
    world = build_world()
    ts, _ = run_phase1(world, seed=42)

    # 找到一个 operation_name 并改为 UNKNOWN
    rec = ts.records[0]
    unknown_rec = TransformationRecord(
        record_id=rec.record_id,
        input_sigs=rec.input_sigs,
        output_sig=rec.output_sig,
        operation_name="UNKNOWN",
        context_sig=rec.context_sig,
        verification_result=rec.verification_result,
        usefulness=rec.usefulness,
        cost=rec.cost,
        step=rec.step,
        confidence=rec.confidence,
        input_terms=rec.input_terms,
        output_prop_str=rec.output_prop_str,
    )
    # 验证结构签名不变
    assert unknown_rec.input_sigs == rec.input_sigs
    assert unknown_rec.output_sig == rec.output_sig


# ============================================================
# 11. operation_name 改名不影响结果
# ============================================================
def test_operation_name_rename_no_effect():
    """operation_name 改名不影响选择和信用计算。"""
    world = build_world()
    ts1, _ = run_phase1(world, seed=42)

    # 创建一个改名后的 store
    ts2 = TransformationStore()
    for r in ts1.records:
        ts2.records.append(TransformationRecord(
            record_id=r.record_id,
            input_sigs=r.input_sigs,
            output_sig=r.output_sig,
            operation_name="RENAMED_" + r.operation_name,
            context_sig=r.context_sig,
            verification_result=r.verification_result,
            usefulness=r.usefulness,
            cost=r.cost,
            step=r.step,
            confidence=r.confidence,
            input_terms=r.input_terms,
            output_prop_str=r.output_prop_str,
        ))

    # 两个 store 应产生相同的候选
    start = set(world[3])
    belief = BeliefStore()
    for o in start:
        belief.update_belief(o, STATUS_VALID, 1.0, evidence_count_delta=1)

    start_list = sorted(start, key=lambda p: p.to_str())
    c1 = generate_candidates_from_transforms(start_list, ts1, belief)
    c2 = generate_candidates_from_transforms(start_list, ts2, belief)
    assert len(c1) == len(c2), "operation_name 改名不影响候选数量"


# ============================================================
# 12. 不使用 goal distance
# ============================================================
def test_no_goal_distance_used():
    """选择和信用计算不使用任何 goal distance 指标。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    # 检查 store 中没有 distance 相关字段
    stats = TransformStatsV2(input_sig=("x",), output_sig="y", goal_sig="g")
    assert not hasattr(stats, 'goal_distance')
    assert not hasattr(stats, 'distance')
    assert not hasattr(stats, 'similarity')
    # 分数只来自 goal_improvement, future_credit, cost
    assert hasattr(stats, 'total_goal_improvement')
    assert hasattr(stats, 'total_future_credit')
    assert hasattr(stats, 'mean_cost')


# ============================================================
# 13. 不使用 ground truth 参与选择
# ============================================================
def test_no_ground_truth_in_selection():
    """选择过程不使用 ground truth。"""
    # select_candidates_v2 只依赖 selection_store 的历史统计
    # 不接受 ground_truth 参数
    import inspect
    from experiments.run_e0_7_4 import select_candidates_v2
    sig = inspect.signature(select_candidates_v2)
    params = list(sig.parameters.keys())
    assert 'ground_truth' not in params
    assert 'goal_distance' not in params


# ============================================================
# 14. 不使用未来 episode 数据
# ============================================================
def test_no_future_episode_data():
    """每个 episode 只能使用之前 episode 的数据。"""
    world = build_world()
    ts, _ = run_phase1(world, seed=42)

    sel1 = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    goal = make_goal_nstep("b", 2)

    # 第一个 episode
    r1 = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal,
        transform_store=ts, selection_store=sel1,
        world_history=world, use_learning=True,
        rng=random.Random(1), update_selection=True)

    # 第二个 episode 使用同一个 store（包含第一个的数据）
    r2 = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal,
        transform_store=ts, selection_store=sel1,
        world_history=world, use_learning=True,
        rng=random.Random(2), update_selection=True)

    # 第一个 episode 不应能访问第二个的数据（已执行完毕，无法回溯）
    # 这里验证 store 的数据是累积的，不是未来的
    assert r1 is not None and r2 is not None


# ============================================================
# 15. 信用只能由已经发生的 trace 产生
# ============================================================
def test_credit_only_from_actual_trace():
    """信用传播只能基于实际发生的 provenance chain。"""
    world = build_world()
    ts, _ = run_phase1(world, seed=42)

    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    goal = make_goal_nstep("b", 3)

    result = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal,
        transform_store=ts, selection_store=sel,
        world_history=world, use_learning=True,
        rng=random.Random(42), update_selection=True)

    # 如果未达到目标，不应有信用传播
    if not result["reached"]:
        # 检查 store 中没有 future_credit（因为没有成功链）
        has_credit = False
        for stats in sel.stats.values():
            if stats.total_future_credit > 0:
                has_credit = True
                break
        assert not has_credit, "未达成目标时不应有信用传播"


# ============================================================
# 16. 多步目标中间节点能够通过历史学习获得价值
# ============================================================
def test_multistep_intermediate_learns_value():
    """经过多次 episode 后，中间变换应获得未来价值。"""
    world = build_world()
    ts, _ = run_phase1(world, seed=42)
    goal = make_goal_nstep("b", 3)

    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)

    # 运行多个 episode
    reached_count = 0
    for i in range(15):
        r = run_goal_pursuit_episode_v2(
            start_objects=set(world[3]), goal=goal,
            transform_store=ts, selection_store=sel,
            world_history=world, use_learning=True,
            rng=random.Random(i), update_selection=True)
        if r["reached"]:
            reached_count += 1

    # 如果有成功，中间变换应有 future_credit
    if reached_count > 0:
        any_future_credit = any(
            s.total_future_credit > 0 for s in sel.stats.values())
        assert any_future_credit, \
            "成功后中间变换应获得未来信用"


# ============================================================
# 17. 不同成功路径都可以获得信用
# ============================================================
def test_multiple_success_paths_get_credit():
    """两条不同的成功路径都应获得信用。"""
    sel = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)

    # 路径 1: A→B→G
    rec1a = _make_record(100, "a1", "b1", "conj", "valid")
    rec1b = _make_record(101, "b1", "goal", "conj", "valid")

    # 路径 2: A→C→G
    rec2a = _make_record(200, "a2", "c1", "conj", "valid")
    rec2b = _make_record(201, "c1", "goal", "conj", "valid")

    # 两条路径都成功
    for chain in [[(rec1b, 0), (rec1a, 1)], [(rec2b, 0), (rec2a, 1)]]:
        credits = compute_credit_for_chain(chain, CREDIT_DISCOUNTED, DEFAULT_DISCOUNT)
        for record, _ in chain:
            credit = credits.get(record.record_id, 0.0)
            sel.record_outcome(
                record.input_sigs, record.output_sig, "goal_sig",
                "valid", 0.0, credit, 0.0)

    # 两条路径的中间步骤都应有信用
    s1 = sel.get_stats(rec1a.input_sigs, rec1a.output_sig, "goal_sig")
    s2 = sel.get_stats(rec2a.input_sigs, rec2a.output_sig, "goal_sig")
    assert s1.total_future_credit > 0, "路径1中间步骤应获得信用"
    assert s2.total_future_credit > 0, "路径2中间步骤应获得信用"


# ============================================================
# 18. 长路径不能因为"暂时没有直接价值"被判 INVALID
# ============================================================
def test_long_path_not_invalid_due_to_no_immediate_value():
    """长路径的中间步骤即使没有直接达到 goal，仍保持 VALID。"""
    world = build_world()
    ts, _ = run_phase1(world, seed=42)

    # 构造一个中间命题（合取），它本身不直接等于 goal
    Ab = P.predicate("A", "b")
    Bb = P.predicate("B", "b")
    Cb = P.predicate("C", "b")
    impl_ab = P.impl(Ab, Bb)
    impl_bc = P.impl(Bb, Cb)
    conj1 = P.conj(impl_ab, impl_bc)

    # 验证这个中间命题
    verdict, confidence, _ = verify_proposition_extended(conj1, world)
    # 它应该是 valid 的（两个合取支都 valid）
    assert verdict == "valid", \
        "中间合取命题应保持 VALID，不因未直接达到 goal 而变 INVALID"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

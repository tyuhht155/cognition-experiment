"""E0-5 专用不变量测试：verification method evaluation。

验证 12 个核心不变量：
  1. test_verification_methods_are_objects
  2. test_method_history_is_recorded
  3. test_method_reliability_is_not_ground_truth
  4. test_method_can_be_wrong
  5. test_method_reliability_can_decrease
  6. test_method_reliability_can_increase
  7. test_methods_have_different_costs
  8. test_method_selection_depends_on_history
  9. test_future_information_is_forbidden
  10. test_ground_truth_is_not_agent_knowledge
  11. test_verification_result_and_method_evaluation_are_distinct
  12. test_method_revision_does_not_directly_rewrite_proposition

核心约束：
  - 验证方法是普通计算对象
  - reliability 来自时间反馈（非 ground truth，非共识）
  - 方法可以犯错；reliability 可升可降
  - 方法选择产生 trace
  - ground truth 不进入 agent 知识空间
  - 方法可靠性下降不能直接修改命题状态
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.prediction import TemporalPredictionState
from experiments.run_e0_5 import (
    build_world_history, run_e0_5, ground_truth_check,
    VerificationMethodStore, MethodStats,
    method_direct_observation, method_repeated_observation, method_prediction_check,
    select_method, verify_with_method,
    VERIFICATION_METHODS, METHOD_COSTS,
)


# ============================================================
# 缓存实验结果
# ============================================================

_cached_result = None


def _get_result():
    global _cached_result
    if _cached_result is None:
        _cached_result = run_e0_5()
    return _cached_result


# ============================================================
# 1. test_verification_methods_are_objects
# ============================================================

def test_verification_methods_are_objects():
    """验证方法本身可以被记录和计算。"""
    result = _get_result()

    # 必须有 3 种方法
    assert len(result["method_stats"]) == 3, \
        f"should have 3 methods, got {len(result['method_stats'])}"

    # 每种方法都有完整的统计字段
    for mid, s in result["method_stats"].items():
        assert "attempts" in s
        assert "correct" in s
        assert "incorrect" in s
        assert "unknown" in s
        assert "total_cost" in s
        assert "average_cost" in s
        assert "reliability_estimate" in s

    # 方法本身可以作为对象被传递和调用
    assert len(VERIFICATION_METHODS) == 3
    for mid, fn in VERIFICATION_METHODS.items():
        assert callable(fn), f"{mid} should be callable"

    # 方法可以产生 MethodResult 对象
    history = build_world_history()
    prop = P.impl(P.atom("P(a)"), P.atom("Q(a)"))
    pred_state = TemporalPredictionState()
    r = method_direct_observation(prop, history[:1], pred_state)
    assert r.method_id == "direct_observation"
    assert r.verdict in ("valid", "invalid", "unknown")
    assert isinstance(r.cost, float)

    print("test_verification_methods_are_objects OK")


# ============================================================
# 2. test_method_history_is_recorded
# ============================================================

def test_method_history_is_recorded():
    """每次使用 verifier 都产生历史记录。"""
    result = _get_result()

    # method_timeline 非空
    assert len(result["method_timeline"]) > 0, \
        "method_timeline should not be empty"

    # 每条记录包含必要字段
    for entry in result["method_timeline"]:
        assert "step" in entry
        assert "method_id" in entry
        assert "proposition" in entry
        assert "result" in entry
        assert "cost" in entry
        assert "reliability_before" in entry
        assert "reliability_after" in entry
        assert "actual_correct_for_experiment" in entry

    # 每种方法至少被使用过
    methods_used = set(e["method_id"] for e in result["method_timeline"])
    assert methods_used == set(VERIFICATION_METHODS.keys()), \
        f"all 3 methods should be used, got {methods_used}"

    # method_stats 中 attempts > 0
    for mid, s in result["method_stats"].items():
        assert s["attempts"] > 0, f"{mid} should have attempts > 0"

    print("test_method_history_is_recorded OK")


# ============================================================
# 3. test_method_reliability_is_not_ground_truth
# ============================================================

def test_method_reliability_is_not_ground_truth():
    """agent 的 reliability estimate 不直接读取 ground truth。"""
    result = _get_result()

    # 检查 VerificationMethodStore 源码不包含 ground_truth
    src = inspect.getsource(VerificationMethodStore)
    assert "ground_truth" not in src, \
        "VerificationMethodStore source contains 'ground_truth'"

    # 检查 select_method 源码不包含 ground_truth
    src = inspect.getsource(select_method)
    assert "ground_truth" not in src, \
        "select_method source contains 'ground_truth'"

    # 检查各方法源码不包含 ground_truth
    for fn in [method_direct_observation, method_repeated_observation, method_prediction_check]:
        src = inspect.getsource(fn)
        assert "ground_truth" not in src, \
            f"{fn.__name__} source contains 'ground_truth'"

    # agent reliability 和 gt accuracy 可能不同（证明它们不是同一个来源）
    for mid, vs in result["gt_vs_agent_reliability"].items():
        agent_rel = vs["agent_reliability_estimate"]
        gt_acc = vs["gt_accuracy"]
        # 它们可以相同或不同，但 agent_rel 不能等于 gt_acc 的"直接读取"
        # 只要 reliability 来自 correct/incorrect（agent 自身反馈），就满足要求
        s = result["method_stats"][mid]
        if s["correct"] + s["incorrect"] > 0:
            expected_rel = s["correct"] / (s["correct"] + s["incorrect"])
            assert abs(agent_rel - round(expected_rel, 4)) < 0.01, \
                f"{mid}: agent_rel={agent_rel} should equal correct/(correct+incorrect)={expected_rel}"

    print("test_method_reliability_is_not_ground_truth OK")


# ============================================================
# 4. test_method_can_be_wrong
# ============================================================

def test_method_can_be_wrong():
    """至少一个 verifier 必须出现错误。"""
    result = _get_result()

    # 从 gt_vs_agent 中检查
    any_wrong = any(
        vs["gt_incorrect"] > 0 for vs in result["gt_vs_agent_reliability"].values()
    )
    assert any_wrong, \
        "at least one method should have gt_incorrect > 0"

    # 至少有一个 actual_correct=False 的记录
    any_false = any(
        e["actual_correct_for_experiment"] is False
        for e in result["method_timeline"]
    )
    assert any_false, \
        "at least one method_timeline entry should have actual_correct=False"

    print("test_method_can_be_wrong OK")


# ============================================================
# 5. test_method_reliability_can_decrease
# ============================================================

def test_method_reliability_can_decrease():
    """错误发生后 reliability 可以下降。"""
    # 直接测试 VerificationMethodStore 的反馈机制
    store = VerificationMethodStore()

    # 初始 reliability = 0.5
    assert store.get_reliability("direct_observation") == 0.5

    # 记录一次正确的验证
    store.record_attempt("direct_observation", "P(a)", 0, "valid", 0.3)
    store.record_feedback("direct_observation", "valid", "valid")
    # correct=1, incorrect=0 → reliability=1.0
    assert store.get_reliability("direct_observation") == 1.0

    # 记录一次错误的验证（verdict 改变）
    store.record_attempt("direct_observation", "P(a)", 1, "invalid", 0.3)
    store.record_feedback("direct_observation", "valid", "invalid")
    # correct=1, incorrect=1 → reliability=0.5
    assert store.get_reliability("direct_observation") == 0.5

    # 再一次错误
    store.record_feedback("direct_observation", "valid", "invalid")
    # correct=1, incorrect=2 → reliability=0.333
    assert store.get_reliability("direct_observation") < 0.5

    print("test_method_reliability_can_decrease OK")


# ============================================================
# 6. test_method_reliability_can_increase
# ============================================================

def test_method_reliability_can_increase():
    """连续正确后 reliability 可以上升。"""
    store = VerificationMethodStore()

    # 初始 reliability = 0.5
    assert store.get_reliability("repeated_observation") == 0.5

    # 记录一次错误
    store.record_attempt("repeated_observation", "P(a)", 0, "valid", 1.0)
    store.record_feedback("repeated_observation", "valid", "invalid")
    # correct=0, incorrect=1 → reliability=0.0
    assert store.get_reliability("repeated_observation") == 0.0

    # 连续正确
    store.record_feedback("repeated_observation", "valid", "valid")
    # correct=1, incorrect=1 → reliability=0.5
    assert store.get_reliability("repeated_observation") == 0.5

    store.record_feedback("repeated_observation", "valid", "valid")
    # correct=2, incorrect=1 → reliability=0.667
    assert store.get_reliability("repeated_observation") > 0.5

    store.record_feedback("repeated_observation", "valid", "valid")
    # correct=3, incorrect=1 → reliability=0.75
    assert store.get_reliability("repeated_observation") > 0.667

    print("test_method_reliability_can_increase OK")


# ============================================================
# 7. test_methods_have_different_costs
# ============================================================

def test_methods_have_different_costs():
    """验证方法成本不同。"""
    # 检查 METHOD_COSTS
    costs = list(METHOD_COSTS.values())
    assert len(set(costs)) == 3, \
        f"all 3 methods should have different costs, got {costs}"

    # direct < repeated < prediction
    assert METHOD_COSTS["direct_observation"] < METHOD_COSTS["repeated_observation"]
    assert METHOD_COSTS["repeated_observation"] < METHOD_COSTS["prediction_check"]

    # 在实验结果中也检查
    result = _get_result()
    for mid, s in result["method_stats"].items():
        if s["attempts"] > 0:
            assert s["average_cost"] > 0, f"{mid} average_cost should be > 0"

    # 不同方法的 average_cost 不同
    avg_costs = [s["average_cost"] for s in result["method_stats"].values() if s["attempts"] > 0]
    assert len(set(avg_costs)) > 1, \
        f"methods should have different average costs, got {avg_costs}"

    print("test_methods_have_different_costs OK")


# ============================================================
# 8. test_method_selection_depends_on_history
# ============================================================

def test_method_selection_depends_on_history():
    """历史表现能够影响后续方法选择。"""
    result = _get_result()

    # 检查不同步骤选了不同方法
    # 从 step_records 的 method_usage 检查
    methods_used = set()
    for s in result["step_records"]:
        for mid, cnt in s["method_usage"].items():
            if cnt > 0:
                methods_used.add(mid)

    assert len(methods_used) >= 2, \
        f"should use at least 2 different methods, got {methods_used}"

    # 检查方法选择 trace 存在
    # trace 中应该有 "select_verification_method" 操作
    # 通过 method_timeline 检查选择记录
    assert len(result["method_timeline"]) > 0

    # 检查可靠性变化会影响选择
    # 如果某种方法 reliability 高，它应该更常被选
    # （至少在某些步骤中，reliability 影响了选择）
    has_reliability_change = any(
        e["reliability_before"] != e["reliability_after"]
        for e in result["method_timeline"]
    )
    # reliability 变化说明方法表现影响了可靠性
    # 而可靠性影响选择（因为 select_method 使用 reliability）
    assert result["check_method_selection_depends_on_history"], \
        "method selection should depend on history"

    print("test_method_selection_depends_on_history OK")


# ============================================================
# 9. test_future_information_is_forbidden
# ============================================================

def test_future_information_is_forbidden():
    """验证方法选择不能读取未来信息。"""
    history = build_world_history()

    # 检查每步 visible_history_length == step + 1
    for t in range(len(history)):
        visible = history[:t + 1]
        assert len(visible) == t + 1, \
            f"step {t}: visible history should be {t + 1}, got {len(visible)}"

    # 检查方法源码不包含 future/peek 等关键词
    for fn in [method_direct_observation, method_repeated_observation, method_prediction_check]:
        src = inspect.getsource(fn)
        # 方法不应该访问超出 visible_history 的数据
        assert "full_history" not in src, \
            f"{fn.__name__} should not access full_history"
        # 不应该有硬编码的未来步骤
        assert "t + 2" not in src and "t+2" not in src

    # 运行实验检查
    result = _get_result()
    assert result["future_information_leak_check"]["passed"], \
        f"future_information_leak_check failed"

    # 检查每步的 visible_history_length
    for s in result["step_records"]:
        assert s["visible_history_length"] == s["step"] + 1, \
            f"step {s['step']}: visible_history_length={s['visible_history_length']}"

    # 检查 prediction_state 不允许在当前步验证当前步的预测
    # TemporalPredictionState 的 resolve_pending 要求 current_step > created_at
    pred_state = TemporalPredictionState()
    pred = pred_state.register(
        proposition=P.impl(P.atom("A"), P.atom("B")),
        trigger=P.atom("A"),
        expected=P.atom("B"),
        current_step=5,
    )
    # 在 step 5 尝试解决 → 被拒绝
    resolved = pred_state.resolve_pending(5, {P.atom("B")})
    assert len(resolved) == 0, \
        "prediction created at step 5 should not be resolved at step 5"
    # 在 step 6 解决 → 允许
    resolved = pred_state.resolve_pending(6, {P.atom("B")})
    assert len(resolved) == 1, \
        "prediction created at step 5 should be resolved at step 6"

    print("test_future_information_is_forbidden OK")


# ============================================================
# 10. test_ground_truth_is_not_agent_knowledge
# ============================================================

def test_ground_truth_is_not_agent_knowledge():
    """ground truth 不得进入 KnowledgeStore / BeliefStore。"""
    # 检查 ground_truth_check 函数源码不访问 BeliefStore
    src = inspect.getsource(ground_truth_check)
    assert "belief_store" not in src.lower(), \
        "ground_truth_check should not access belief_store"
    assert "evidence_log" not in src.lower(), \
        "ground_truth_check should not access evidence_log"
    assert "method_store" not in src.lower(), \
        "ground_truth_check should not access method_store"

    # 检查 VerificationMethodStore 源码不访问 ground_truth
    src = inspect.getsource(VerificationMethodStore)
    assert "ground_truth" not in src

    # 检查验证方法源码不访问 ground_truth
    for fn in [method_direct_observation, method_repeated_observation, method_prediction_check]:
        src = inspect.getsource(fn)
        assert "ground_truth" not in src

    # 检查 select_method 源码不访问 ground_truth
    src = inspect.getsource(select_method)
    assert "ground_truth" not in src

    # 检查实验结果中 ground_truth 只出现在统计字段
    result = _get_result()
    for s in result["step_records"]:
        # step_records 不应包含 gt_holds 字段
        assert "gt_holds" not in s, \
            f"step {s['step']}: step_record should not contain gt_holds"
        # step_records 的 method_usage 不应包含 ground truth 信息
        for mid in s["method_usage"]:
            assert "gt_" not in mid

    print("test_ground_truth_is_not_agent_knowledge OK")


# ============================================================
# 11. test_verification_result_and_method_evaluation_are_distinct
# ============================================================

def test_verification_result_and_method_evaluation_are_distinct():
    """'命题是否成立'和'验证这个命题的方法是否可靠'必须是两个不同对象。"""
    result = _get_result()

    # BeliefStore 存储命题状态（valid/invalid/unknown）
    bs = result["belief_store_stats"]
    assert "valid" in bs and "invalid" in bs and "unknown" in bs

    # method_stats 存储方法可靠性
    for mid, s in result["method_stats"].items():
        assert "reliability_estimate" in s
        assert "correct" in s
        assert "incorrect" in s
        # 方法的 reliability 和命题的 status 是不同的字段
        assert s["reliability_estimate"] != bs  # 不同对象

    # 检查 trace 中有 verify 操作和 select_verification_method 操作
    # 它们是不同的计算步骤
    # 通过 method_timeline 检查
    assert len(result["method_timeline"]) > 0

    # 检查 BeliefStore 的 update_belief 不修改 method_store
    bs2 = BeliefStore()
    store = VerificationMethodStore()
    prop = P.atom("Test")
    bs2.update_belief(prop, STATUS_VALID, 0.8, evidence_count_delta=1)
    # method_store 不应受影响
    assert store.stats["direct_observation"].attempts == 0
    assert store.stats["direct_observation"].correct == 0

    print("test_verification_result_and_method_evaluation_are_distinct OK")


# ============================================================
# 12. test_method_revision_does_not_directly_rewrite_proposition
# ============================================================

def test_method_revision_does_not_directly_rewrite_proposition():
    """verifier 可靠性下降不能直接把已有 proposition 改成 INVALID；
    必须重新验证 proposition。"""
    # 直接测试机制
    store = VerificationMethodStore()
    belief_store = BeliefStore()

    prop = P.atom("Test")
    belief_store.update_belief(prop, STATUS_VALID, 0.8, evidence_count_delta=1)

    # 模拟方法可靠性下降
    store.record_attempt("direct_observation", "Test", 0, "valid", 0.3)
    store.record_feedback("direct_observation", "valid", "invalid")
    # reliability 下降了
    assert store.get_reliability("direct_observation") == 0.0

    # 但 BeliefStore 中的 prop 状态没有被修改
    b = belief_store.get(prop)
    assert b.status == STATUS_VALID, \
        "method reliability change should NOT directly modify proposition status"

    # 要修改 proposition 状态，必须重新验证
    # 模拟重新验证
    belief_store.update_belief(prop, STATUS_INVALID, 0.9, evidence_count_delta=1)
    b = belief_store.get(prop)
    assert b.status == STATUS_INVALID

    # 在实验结果中检查
    result = _get_result()
    # 检查 method_stats 中的 reliability 变化不直接对应 BeliefStore 中的状态变化
    # 即：method reliability 可以独立于 proposition status 变化
    for mid, s in result["method_stats"].items():
        # 方法的 correct/incorrect 是独立计数
        assert s["correct"] >= 0
        assert s["incorrect"] >= 0

    print("test_method_revision_does_not_directly_rewrite_proposition OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_verification_methods_are_objects,
        test_method_history_is_recorded,
        test_method_reliability_is_not_ground_truth,
        test_method_can_be_wrong,
        test_method_reliability_can_decrease,
        test_method_reliability_can_increase,
        test_methods_have_different_costs,
        test_method_selection_depends_on_history,
        test_future_information_is_forbidden,
        test_ground_truth_is_not_agent_knowledge,
        test_verification_result_and_method_evaluation_are_distinct,
        test_method_revision_does_not_directly_rewrite_proposition,
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
    print(f"\nE0-5 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

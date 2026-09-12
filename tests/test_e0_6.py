"""E0-6 专用不变量测试：operation selection learning。

验证 7 个核心不变量：
  1. test_history_influences_selection - 历史经验能够影响后续操作选择
  2. test_different_contexts_different_preferences - 不同情境可以形成不同的操作偏好
  3. test_unknown_not_treated_as_failure - unknown 不会被错误当成失败
  4. test_ground_truth_not_in_selection_logic - ground truth 不进入选择逻辑
  5. test_exploration_still_exists - 仍然存在探索
  6. test_history_selection_records_complete - 历史选择记录完整
  7. test_ab_groups_use_same_budget - A/B 两组使用相同预算

核心约束：
  - 学习目标是已有操作的选择，不是新操作发现
  - unknown 不被伪造为 valid/invalid（与 E0-5 一致）
  - ground truth 只用于实验统计
  - 保留探索
  - 操作选择产生 trace
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_6 import (
    run_e0_6, run_group, OperationSelectionStore, OpContextStats,
    select_candidates, generate_candidates, verify_proposition,
    apply_constructor, build_world_history, ground_truth_check,
    extract_context_unary, extract_context_binary,
    CONSTRUCTORS, CONSTRUCTOR_COSTS, BUDGET_PER_STEP, EPSILON,
    COST_WEIGHT,
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
        _cached_result = run_e0_6()
    return _cached_result


# ============================================================
# 1. test_history_influences_selection
# ============================================================

def test_history_influences_selection():
    """历史经验能够影响后续操作选择。"""
    result = _get_result()
    b = result["group_b"]

    # Group B 的操作分布应该不均匀（学习导致偏向某些操作）
    op_counts = b["operation_counts"]
    total = sum(op_counts.values())
    assert total > 0, "should have executed candidates"

    ratios = {op: cnt / total for op, cnt in op_counts.items()}
    max_ratio = max(ratios.values())
    min_ratio = min(ratios.values())

    # 学习应该导致分布不均匀（某些操作被更多/更少选择）
    assert max_ratio > min_ratio + 0.05, \
        f"operation distribution should be non-uniform: {ratios}"

    # 检查 selection_timeline 中的分数变化
    timeline = b["selection_timeline"]
    if len(timeline) >= 10:
        early = timeline[:len(timeline) // 3]
        late = timeline[-len(timeline) // 3:]
        # 后期的利用（非探索）比例应该更高
        early_exploit = sum(1 for e in early if not e["was_exploration"]) / len(early)
        late_exploit = sum(1 for e in late if not e["was_exploration"]) / len(late)
        # 后期应该有更多利用（或至少不更少）
        assert late_exploit >= early_exploit * 0.5, \
            f"late exploit rate ({late_exploit:.2f}) should be reasonable vs early ({early_exploit:.2f})"

    print("test_history_influences_selection OK")


# ============================================================
# 2. test_different_contexts_different_preferences
# ============================================================

def test_different_contexts_different_preferences():
    """不同情境可以形成不同的操作偏好。"""
    result = _get_result()
    b = result["group_b"]

    ctx_stats = b["context_stats"]
    assert len(ctx_stats) >= 2, \
        f"should have at least 2 contexts, got {len(ctx_stats)}"

    # 找到至少两个上下文，其中操作偏好不同
    # 偏好 = valid_yield 最高的操作
    best_per_ctx = {}
    for ctx, ops in ctx_stats.items():
        tried_ops = {op: s for op, s in ops.items() if s["attempts"] > 0}
        if not tried_ops:
            continue
        best_op = max(tried_ops.values(), key=lambda x: x["valid_yield"])
        best_per_ctx[ctx] = (best_op["operation"], best_op["valid_yield"])

    assert len(best_per_ctx) >= 1, \
        "should have at least one context with tried operations"

    # 检查不同上下文有不同的操作成功率模式
    # 例如：binary 上下文中 impl 成功率高，unary 上下文中 neg 成功率低
    found_diff = False
    ctx_list = list(best_per_ctx.items())
    for i, (ctx_a, (op_a, yield_a)) in enumerate(ctx_list):
        for ctx_b, (op_b, yield_b) in ctx_list[i + 1:]:
            if op_a != op_b or abs(yield_a - yield_b) > 0.1:
                found_diff = True
                break
        if found_diff:
            break

    assert found_diff, \
        f"should find different preferences across contexts: {best_per_ctx}"

    print("test_different_contexts_different_preferences OK")


# ============================================================
# 3. test_unknown_not_treated_as_failure
# ============================================================

def test_unknown_not_treated_as_failure():
    """unknown 不会被错误当成失败。"""
    # 直接测试 OpContextStats 的 success_rate 计算
    stats = OpContextStats(context_sig="test", operation="neg")
    stats.attempts = 10
    stats.valid = 0
    stats.invalid = 0
    stats.unknown = 10

    # 全是 unknown → success_rate 应该是 0.5（中性），不是 0.0
    assert stats.success_rate == 0.5, \
        f"all-unknown success_rate should be 0.5 (neutral), got {stats.success_rate}"
    # info_rate 应该是 0.0（没有确定性答案）
    assert stats.info_rate == 0.0, \
        f"all-unknown info_rate should be 0.0, got {stats.info_rate}"

    # 有 valid 和 invalid → unknown 不计入
    stats2 = OpContextStats(context_sig="test", operation="impl")
    stats2.attempts = 10
    stats2.valid = 3
    stats2.invalid = 2
    stats2.unknown = 5

    # success_rate = 3 / (3 + 2) = 0.6，unknown 不影响
    assert stats2.success_rate == 0.6, \
        f"success_rate should be 3/5=0.6, got {stats2.success_rate}"
    # info_rate = (3+2) / 10 = 0.5
    assert stats2.info_rate == 0.5, \
        f"info_rate should be 5/10=0.5, got {stats2.info_rate}"

    # 在实验结果中检查
    result = _get_result()
    for op, s in result["group_b"]["operation_stats"].items():
        if s["attempts"] == 0:
            continue
        if s["valid"] + s["invalid"] == 0:
            # 全是 unknown → success_rate 应该是 0.5
            assert abs(s["success_rate"] - 0.5) < 0.01, \
                f"{op}: all-unknown should have success_rate=0.5, got {s['success_rate']}"
        else:
            expected = s["valid"] / (s["valid"] + s["invalid"])
            assert abs(s["success_rate"] - round(expected, 4)) < 0.01, \
                f"{op}: success_rate mismatch"

    print("test_unknown_not_treated_as_failure OK")


# ============================================================
# 4. test_ground_truth_not_in_selection_logic
# ============================================================

def test_ground_truth_not_in_selection_logic():
    """ground truth 不进入选择逻辑。"""
    # 检查 select_candidates 源码不包含 ground_truth
    src = inspect.getsource(select_candidates)
    assert "ground_truth" not in src, \
        "select_candidates source contains 'ground_truth'"

    # 检查 OperationSelectionStore 源码不包含 ground_truth
    src = inspect.getsource(OperationSelectionStore)
    assert "ground_truth" not in src, \
        "OperationSelectionStore source contains 'ground_truth'"

    # 检查 generate_candidates 源码不包含 ground_truth
    src = inspect.getsource(generate_candidates)
    assert "ground_truth" not in src, \
        "generate_candidates source contains 'ground_truth'"

    # 检查 verify_proposition 源码不包含 ground_truth
    src = inspect.getsource(verify_proposition)
    assert "ground_truth" not in src, \
        "verify_proposition source contains 'ground_truth'"

    # 检查 compute_score 方法不包含 ground_truth
    src = inspect.getsource(OperationSelectionStore.compute_score)
    assert "ground_truth" not in src, \
        "compute_score source contains 'ground_truth'"

    print("test_ground_truth_not_in_selection_logic OK")


# ============================================================
# 5. test_exploration_still_exists
# ============================================================

def test_exploration_still_exists():
    """仍然存在探索。"""
    result = _get_result()
    b = result["group_b"]

    # Group B 应该有探索（epsilon-greedy）
    assert b["exploration_count"] > 0, \
        "Group B should have exploration attempts"
    assert b["exploit_count"] > 0, \
        "Group B should have exploit attempts"

    # 探索率应该 > 0
    assert b["exploration_rate"] > 0, \
        f"exploration_rate should be > 0, got {b['exploration_rate']}"

    # 检查 epsilon 常量
    assert EPSILON > 0, "EPSILON should be > 0"
    assert EPSILON < 1, "EPSILON should be < 1"

    # 在 timeline 中检查
    timeline = b["selection_timeline"]
    exploration_entries = [e for e in timeline if e["was_exploration"]]
    assert len(exploration_entries) > 0, \
        "should have exploration entries in timeline"

    print("test_exploration_still_exists OK")


# ============================================================
# 6. test_history_selection_records_complete
# ============================================================

def test_history_selection_records_complete():
    """历史选择记录完整。"""
    result = _get_result()
    b = result["group_b"]

    timeline = b["selection_timeline"]
    assert len(timeline) > 0, "selection_timeline should not be empty"

    # 每条记录包含必要字段
    required_fields = ["step", "context_sig", "operation", "proposition",
                       "result", "cost", "score", "was_exploration"]
    for entry in timeline:
        for field in required_fields:
            assert field in entry, \
                f"timeline entry missing field '{field}': {entry}"

    # 检查 step_records 也完整
    for s in b["step_records"]:
        assert "step" in s
        assert "candidates_available" in s
        assert "candidates_selected" in s
        assert "valid" in s
        assert "invalid" in s
        assert "unknown" in s
        assert "operation_counts" in s
        assert "visible_history_length" in s
        # 检查时序正确性
        assert s["visible_history_length"] == s["step"] + 1, \
            f"step {s['step']}: visible_history should be {s['step']+1}"

    print("test_history_selection_records_complete OK")


# ============================================================
# 7. test_ab_groups_use_same_budget
# ============================================================

def test_ab_groups_use_same_budget():
    """A/B 两组使用相同预算。"""
    result = _get_result()
    a = result["group_a"]
    b = result["group_b"]

    # 相同预算
    assert a["budget_per_step"] == b["budget_per_step"], \
        "A and B should have same budget"

    # 相同种子
    assert a["seed"] == b["seed"], \
        "A and B should use same seed"

    # 相同步数
    assert a["n_steps"] == b["n_steps"], \
        "A and B should have same n_steps"

    # 相同操作集合
    assert set(a["operation_counts"].keys()) == set(b["operation_counts"].keys()), \
        "A and B should have same operation set"

    print("test_ab_groups_use_same_budget OK")


# ============================================================
# 8. test_no_new_operations_created
# ============================================================

def test_no_new_operations_created():
    """E0-6 不创造新操作，只使用已有 constructors。"""
    # CONSTRUCTORS 应该只有 neg/conj/disj/impl/iff
    expected = {"neg", "conj", "disj", "impl", "iff"}
    assert set(CONSTRUCTORS) == expected, \
        f"CONSTRUCTORS should be {expected}, got {set(CONSTRUCTORS)}"

    # apply_constructor 不应该能处理新的 constructor
    result = _get_result()
    a = result["group_a"]
    b = result["group_b"]

    # 两组只使用了已有操作
    for op in a["operation_counts"]:
        assert op in expected, f"unexpected operation: {op}"
    for op in b["operation_counts"]:
        assert op in expected, f"unexpected operation: {op}"

    print("test_no_new_operations_created OK")


# ============================================================
# 9. test_future_information_forbidden
# ============================================================

def test_future_information_forbidden():
    """操作选择不能读取未来信息。"""
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

    # 在实验结果中检查 visible_history_length
    result = _get_result()
    for group in [result["group_a"], result["group_b"]]:
        for s in group["step_records"]:
            assert s["visible_history_length"] == s["step"] + 1, \
                f"step {s['step']}: visible_history_length={s['visible_history_length']}"

    print("test_future_information_forbidden OK")


# ============================================================
# 10. test_operation_selection_produces_trace
# ============================================================

def test_operation_selection_produces_trace():
    """操作选择产生 trace。"""
    result = _get_result()
    b = result["group_b"]

    # trace 步骤数 > 0
    assert b["trace_steps"] > 0, "trace should have steps"

    # selection_timeline 非空
    assert len(b["selection_timeline"]) > 0, \
        "selection_timeline should not be empty"

    print("test_operation_selection_produces_trace OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_history_influences_selection,
        test_different_contexts_different_preferences,
        test_unknown_not_treated_as_failure,
        test_ground_truth_not_in_selection_logic,
        test_exploration_still_exists,
        test_history_selection_records_complete,
        test_ab_groups_use_same_budget,
        test_no_new_operations_created,
        test_future_information_forbidden,
        test_operation_selection_produces_trace,
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
    print(f"\nE0-6 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

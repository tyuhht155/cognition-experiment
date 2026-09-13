"""E0-10 不变量测试：Evaluation-Directed Computation（评价导向的计算）。

验证评价变化本身能否产生计算方向（不依赖显式 Goal）。

测试覆盖（16 项不变量）：
  1.  test_evaluation_has_bounds          —— 评价函数有边界
  2.  test_low_h_negative_eval            —— H 低时评价为负
  3.  test_high_h_eval_decreases          —— H 过高时评价下降
  4.  test_reasonable_h_positive           —— H 在合理区间时评价为正
  5.  test_eval_gap_produces_problem      —— 评价缺口产生 Problem
  6.  test_no_gap_no_problem              —— 无缺口时不产生 Problem
  7.  test_problem_not_answer             —— Problem 不指定答案结构
  8.  test_value_no_goal_dependency       —— 候选评价不依赖 Goal
  9.  test_action_changes_internal_state  —— 行动改变内部状态
  10. test_state_change_re_evaluates      —— 状态变化后重新评价
  11. test_stops_when_in_range            —— H 进入合理区间后停止追求
  12. test_reward_hacking_observed        —— 能观察到 reward hacking
  13. test_hacking_not_counted_success    —— reward hacking 不算成功
  14. test_no_future_info                 —— 严格时间因果
  15. test_complete_trace                 —— 完整 trace
  16. test_four_concepts_separated        —— Evaluation/Problem/Value/Goal 四者分离
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_10 import (
    EvaluationState,
    evaluate_state,
    evaluate_proxy_reward,
    construct_problem_from_gap,
    evaluate_candidate_no_goal,
    apply_action,
    build_world_h,
    build_world_hacking,
    run_e0_10_episode,
    run_e0_10,
    H_LOW_THRESHOLD,
    H_HIGH_THRESHOLD,
    H_OPTIMAL,
    H_MAX,
    H_INCREMENT,
    ACTION_COST,
    PERSISTENT_NEG_STEPS,
)
from experiments.run_e0_7 import generate_candidates as generate_candidates_brute
from experiments.run_e0_7_2 import verify_proposition_extended
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


# ============================================================
# 辅助
# ============================================================

def _make_state(h=1.0, energy=10.0):
    """构造一个 EvaluationState。"""
    s = EvaluationState(internal_h=h, energy=energy)
    s.update_evaluation(evaluate_state(s))
    return s


def _run_bounded():
    return run_e0_10_episode(build_world_h(), mode="bounded")


def _run_hacking():
    return run_e0_10_episode(build_world_hacking(), mode="hacking")


# ============================================================
# 1. 评价函数有边界，不能无限最大化
# ============================================================

def test_evaluation_has_bounds():
    """评价函数在 H 过高后下降，不是单调递增。"""
    s_low = EvaluationState(internal_h=0.0)
    s_mid = EvaluationState(internal_h=H_OPTIMAL)
    s_high = EvaluationState(internal_h=H_MAX)
    eval_low = evaluate_state(s_low)
    eval_mid = evaluate_state(s_mid)
    eval_high = evaluate_state(s_high)
    # 中等最好
    assert eval_mid > eval_low, "中等 H 评价应高于低 H"
    assert eval_mid > eval_high, "中等 H 评价应高于高 H"
    # 高 H 评价为负
    assert eval_high < 0, "H=MAX 评价应为负"


# ============================================================
# 2. H 低时评价为负
# ============================================================

def test_low_h_negative_eval():
    """H < H_LOW_THRESHOLD 时评价为负。"""
    s = EvaluationState(internal_h=1.0)
    assert evaluate_state(s) < 0, "H=1 评价应为负"
    s2 = EvaluationState(internal_h=0.0)
    assert evaluate_state(s2) < 0, "H=0 评价应为负"


# ============================================================
# 3. H 过高时评价下降
# ============================================================

def test_high_h_eval_decreases():
    """H > H_HIGH_THRESHOLD 时评价下降。"""
    s_mid = EvaluationState(internal_h=H_OPTIMAL)
    s_high = EvaluationState(internal_h=H_HIGH_THRESHOLD + 1)
    s_max = EvaluationState(internal_h=H_MAX)
    eval_mid = evaluate_state(s_mid)
    eval_high = evaluate_state(s_high)
    eval_max = evaluate_state(s_max)
    assert eval_high < eval_mid, "H=8 评价应低于 H=5"
    assert eval_max < eval_high, "H=10 评价应低于 H=8"
    assert eval_max < 0, "H=10 评价应为负"


# ============================================================
# 4. H 在合理区间时评价为正
# ============================================================

def test_reasonable_h_positive():
    """H 在 [H_LOW_THRESHOLD, H_HIGH_THRESHOLD] 时评价为正。"""
    for h in [H_LOW_THRESHOLD, H_OPTIMAL, H_HIGH_THRESHOLD]:
        s = EvaluationState(internal_h=h)
        assert evaluate_state(s) > 0, f"H={h} 评价应为正"


# ============================================================
# 5. 评价缺口产生 Problem
# ============================================================

def test_eval_gap_produces_problem():
    """持续性负评价应产生 Problem。"""
    s = _make_state(h=1.0)
    # 模拟持续负评价
    for _ in range(PERSISTENT_NEG_STEPS + 1):
        s.update_evaluation(evaluate_state(s))
    problem = construct_problem_from_gap(s)
    assert problem is not None, "持续性负评价应产生 Problem"
    assert "find object" in problem["description"]


# ============================================================
# 6. 无缺口时不产生 Problem
# ============================================================

def test_no_gap_no_problem():
    """H 在合理区间时不应产生 Problem。"""
    s = _make_state(h=H_OPTIMAL)
    s.update_evaluation(evaluate_state(s))
    problem = construct_problem_from_gap(s)
    assert problem is None, "合理区间不应产生 Problem"


# ============================================================
# 7. Problem 不指定答案结构
# ============================================================

def test_problem_not_answer():
    """Problem 描述不应包含具体答案（如 'F'、'implies'、'A→B'）。"""
    s = _make_state(h=1.0)
    for _ in range(PERSISTENT_NEG_STEPS + 1):
        s.update_evaluation(evaluate_state(s))
    problem = construct_problem_from_gap(s)
    assert problem is not None
    desc = problem["description"]
    # Problem 不应直接指定答案
    assert "F(" not in desc, "Problem 不应指定 F 是答案"
    assert "implies" not in desc, "Problem 不应指定 implies 结构"
    assert "A→B" not in desc, "Problem 不应指定 A→B"


# ============================================================
# 8. 候选评价不依赖 Goal
# ============================================================

def test_value_no_goal_dependency():
    """evaluate_candidate_no_goal 不接受 goal 参数。"""
    sig = inspect.signature(evaluate_candidate_no_goal)
    params = set(sig.parameters.keys())
    assert "goal" not in params, "不应接受 goal 参数"
    assert "target" not in params, "不应接受 target 参数"


# ============================================================
# 9. 行动改变内部状态
# ============================================================

def test_action_changes_internal_state():
    """行动应改变内部状态 H。"""
    s = _make_state(h=1.0, energy=10.0)
    h_before = s.internal_h
    energy_before = s.energy
    acted = apply_action(s, "F")
    assert acted, "应成功行动"
    assert s.internal_h > h_before, "H 应增加"
    assert s.energy < energy_before, "energy 应减少"


# ============================================================
# 10. 状态变化后重新评价
# ============================================================

def test_state_change_re_evaluates():
    """行动后状态变化，重新评价应不同。"""
    s = _make_state(h=1.0, energy=10.0)
    eval_before = evaluate_state(s)
    apply_action(s, "F")
    eval_after = evaluate_state(s)
    assert eval_after != eval_before, "行动后评价应变化"
    assert eval_after > eval_before, "H 上升后评价应改善"


# ============================================================
# 11. H 进入合理区间后停止追求
# ============================================================

def test_stops_when_in_range():
    """bounded 模式应在 H 进入合理区间后停止（通过评价变化，不是 if-then）。"""
    r = _run_bounded()
    assert r["stopped"], "应停止"
    assert r["stop_reason"] != "energy_exhausted", \
        "不应因能量耗尽而停止"
    # 停止时 H 应在合理区间
    assert r["final_h"] >= H_LOW_THRESHOLD, \
        f"停止时 H={r['final_h']} 应 >= {H_LOW_THRESHOLD}"


# ============================================================
# 12. 能观察到 reward hacking
# ============================================================

def test_reward_hacking_observed():
    """hacking 模式应观察到 proxy reward ↑ 但真实评价 ↓。"""
    r = _run_hacking()
    assert r["hacking_detected"], "应检测到 reward hacking"
    # 至少有一个 step 的 proxy > 0.5 但 true_eval < 0
    hacking_steps = [tr for tr in r["trace"] if tr["reward_hacking"]]
    assert len(hacking_steps) > 0, "应有 hacking steps"


# ============================================================
# 13. reward hacking 不算成功
# ============================================================

def test_hacking_not_counted_success():
    """hacking 模式的最终真实评价应为负（不算成功）。"""
    r = _run_hacking()
    # 最终真实评价应为负
    final_true_eval = evaluate_state(
        EvaluationState(internal_h=r["final_h"]))
    assert final_true_eval < 0, \
        f"最终真实评价应为负（H={r['final_h']} 过高）"


# ============================================================
# 14. 严格时间因果
# ============================================================

def test_no_future_info():
    """每步不读取未来状态。检查 evaluate_candidate_no_goal 不接受未来参数。"""
    sig = inspect.signature(evaluate_candidate_no_goal)
    params = set(sig.parameters.keys())
    assert "future" not in params, "不应接受 future 参数"
    assert "ground_truth" not in params, "不应接受 ground_truth"
    assert "answer" not in params, "不应接受 answer"
    # state 参数提供当前状态（不是未来）
    assert "state" in params, "应接受 state 作为当前状态"


# ============================================================
# 15. 完整 trace
# ============================================================

def test_complete_trace():
    """trace 每步应包含 evaluation/problem/value/action 信息。"""
    r = _run_bounded()
    for tr in r["trace"]:
        assert "internal_h" in tr, "应有 internal_h"
        assert "evaluation" in tr, "应有 evaluation"
        assert "eval_delta" in tr, "应有 eval_delta"
        assert "problem" in tr, "应有 problem"
        assert "proxy_reward" in tr, "应有 proxy_reward"
        assert "true_evaluation" in tr, "应有 true_evaluation"
        assert "reward_hacking" in tr, "应有 reward_hacking"
        assert "stopped" in tr, "应有 stopped"


# ============================================================
# 16. Evaluation / Problem / Value / Goal 四者分离
# ============================================================

def test_four_concepts_separated():
    """验证四个概念在代码中严格分离：
    - Evaluation：evaluate_state / evaluate_proxy_reward
    - Problem：construct_problem_from_gap
    - Value：evaluate_candidate_no_goal
    - Goal：不存在（E0-10 不使用 Goal）"""
    # Evaluation 函数
    assert callable(evaluate_state), "evaluate_state 应可调用"
    assert callable(evaluate_proxy_reward), "evaluate_proxy_reward 应可调用"
    # Problem 函数
    assert callable(construct_problem_from_gap), \
        "construct_problem_from_gap 应可调用"
    # Value 函数
    assert callable(evaluate_candidate_no_goal), \
        "evaluate_candidate_no_goal 应可调用"
    # Goal：E0-10 中不应有 goal 参数
    sig_eval = inspect.signature(evaluate_state)
    sig_problem = inspect.signature(construct_problem_from_gap)
    sig_value = inspect.signature(evaluate_candidate_no_goal)
    assert "goal" not in sig_eval.parameters, "evaluate_state 不应有 goal"
    assert "goal" not in sig_problem.parameters, "construct_problem 不应有 goal"
    assert "goal" not in sig_value.parameters, "evaluate_candidate 不应有 goal"
    # evaluate_state 只接受 state（内部状态）
    assert "state" in sig_eval.parameters
    # construct_problem 只接受 state
    assert "state" in sig_problem.parameters

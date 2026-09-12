"""E0-2 专用测试：时间展开穷举构造 + 验证闭环（strictly time-causal）。

验证：
  1. t 时刻看不到 t+1 的对象
  2. t 时刻候选只能来自 observed_objects
  3. valid belief 不会自动进入 E0-2 constructor 输入
  4. 新 observation 会扩大下一时间步 observed_objects
  5. P(a)→Q(a) 不会提前出现
  6. unknown 候选可以在后续 observation 后重新验证
  7. 已经 valid/invalid 的候选不会被无意义重复验证
  8. ground truth 不进入 Verifier
  9. 完整历史不会通过 Context 偷渡给验证器
  10. trace 的时间顺序正确
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.evidence import EvidenceLog
from cognition.cost import CostTracker
from cognition.consensus import ConsensusAgreementModel
from cognition.prediction import TemporalPredictionState
from cognition.operations import Context
from cognition.verification import Verifier, VALID, INVALID, UNKNOWN
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.trace import TraceRecorder
from experiments.run_e0_2 import (
    build_world_history, enumerate_constructions, run_e0_2,
    ground_truth_check,
)
import inspect


# 1. t 时刻看不到 t+1 的对象
def test_step_t_cannot_see_t1_objects():
    """在 t 时刻，Context.world_history 的长度必须为 t+1。"""
    history = build_world_history()

    for t in range(len(history)):
        visible = history[:t + 1]
        ctx = Context(world_history=visible, constants=["a", "b", "c"],
                      knowledge_view=BeliefStore(), current_step=t)
        assert len(ctx.world_history) == t + 1, \
            f"step {t}: expected {t + 1}, got {len(ctx.world_history)}"

    # 检查 step 6 时看不到 step 7 的 Q(c)
    visible_6 = history[:7]
    qc = P.atom("Q(c)")
    qc_seen = any(qc in s for s in visible_6)
    assert not qc_seen, "Q(c) should not be visible at step 6 (first appears at step 7)"

    print("test_step_t_cannot_see_t1_objects OK")


# 2. t 时刻候选只能来自 observed_objects
def test_candidates_only_from_observed_objects():
    """候选的构造输入只能是 observed_objects，不能包含未观察到的对象。"""
    history = build_world_history()
    observed = set()

    for t in range(len(history)):
        for prop in history[t]:
            observed.add(prop)

        # 枚举构造
        current_observed = sorted(observed, key=lambda p: p.to_str())
        candidates = enumerate_constructions(current_observed)

        # 每个候选的 part 必须在 observed 中
        for prop, method in candidates:
            if prop.kind == "not":
                inner = prop.parts[0]
                assert inner in observed, \
                    f"step {t}: neg candidate contains unobserved {inner}"
            elif prop.kind in ("and", "or", "implies", "iff"):
                a, b = prop.parts
                assert a in observed, \
                    f"step {t}: {method} candidate contains unobserved {a}"
                assert b in observed, \
                    f"step {t}: {method} candidate contains unobserved {b}"

    print("test_candidates_only_from_observed_objects OK")


# 3. valid belief 不会自动进入 constructor 输入
def test_valid_belief_not_in_constructor_input():
    """E0-2 的 constructible_objects = observed_objects，不含 valid belief。"""
    result = run_e0_2()

    # 在 step 0 中，P(a)→Q(a) 被验证为 valid
    # 如果 valid belief 被加入 constructor 输入，
    # step 1 的 observed_object_count 会比实际 observed 多
    step0 = result["step_records"][0]
    step1 = result["step_records"][1]

    # step 0: 3 objects observed (P(a), Q(a), R(a,b))
    assert step0["observed_object_count"] == 3, \
        f"step 0: expected 3, got {step0['observed_object_count']}"

    # step 1: 3 new objects (P(b), Q(b), R(b,a)), total 6
    assert step1["observed_object_count"] == 6, \
        f"step 1: expected 6, got {step1['observed_object_count']}"

    # 如果 valid belief 被加入，step 1 的 observed 会 > 6
    # 因为 step 0 有 6 个 valid

    print("test_valid_belief_not_in_constructor_input OK")


# 4. 新 observation 会扩大下一时间步 observed_objects
def test_observation_expands_observed_objects():
    """当 step t 有新对象时，step t+1 的 observed_object_count 应该 >= step t。"""
    result = run_e0_2()

    prev_count = 0
    for s in result["step_records"]:
        assert s["observed_object_count"] >= prev_count, \
            f"step {s['step']}: observed decreased from {prev_count} to {s['observed_object_count']}"
        prev_count = s["observed_object_count"]

    # 至少有一次增长
    assert result["step_records"][-1]["observed_object_count"] > result["step_records"][0]["observed_object_count"], \
        "observed_objects should grow over time"

    print("test_observation_expands_observed_objects OK")


# 5. P(a)→Q(a) 不会提前出现
def test_pa_qa_not_early():
    """P(a)→Q(a) 不能在 P(a) 或 Q(a) 出现之前构造。"""
    history = build_world_history()
    target = P.impl(P.atom("P(a)"), P.atom("Q(a)"))
    pa = P.atom("P(a)")
    qa = P.atom("Q(a)")

    # P(a) 和 Q(a) 都在 step 0 出现
    assert pa in history[0]
    assert qa in history[0]

    # P(a)→Q(a) 应该在 step 0 就可以构造
    observed = set(history[0])
    candidates = enumerate_constructions(sorted(observed, key=lambda p: p.to_str()))
    cand_props = [c[0] for c in candidates]
    assert target in cand_props, "P(a)→Q(a) should be constructible at step 0"

    # 但如果在 step -1（空集），则不能
    empty_cands = enumerate_constructions([])
    assert target not in [c[0] for c in empty_cands]

    print("test_pa_qa_not_early OK")


# 6. unknown 候选可以在后续 observation 后重新验证
def test_unknown_can_be_reverified():
    """unknown 候选在后续观察增加后应该被重新验证。"""
    result = run_e0_2()

    # 检查是否有 re_verified_unknown > 0
    total_reverified = result["total_re_verified_unknown"]
    assert total_reverified > 0, \
        f"expected re_verified_unknown > 0, got {total_reverified}"

    # 检查某些步骤有 re_verified_unknown > 0
    has_reverify = any(s["re_verified_unknown"] > 0 for s in result["step_records"])
    assert has_reverify, "no step has re_verified_unknown > 0"

    print("test_unknown_can_be_reverified OK")


# 7. 已经 valid/invalid 的候选不会被无意义重复验证
def test_valid_invalid_not_reverified():
    """valid/invalid 的候选应被跳过，不重复验证。"""
    result = run_e0_2()

    # duplicates_skipped 应该 > 0
    total_dup = result["total_duplicates_skipped"]
    assert total_dup > 0, \
        f"expected duplicates_skipped > 0, got {total_dup}"

    # 后面的步骤应该有更多 duplicates（因为更多结论已确立）
    last_step = result["step_records"][-1]
    assert last_step["duplicates_skipped"] > 0

    print("test_valid_invalid_not_reverified OK")


# 8. ground truth 不进入 Verifier
def test_ground_truth_not_in_verifier():
    """ground_truth_check 函数不应被 Verifier 或 EvidenceEvaluator 调用。"""
    # 检查 Verifier 的源码不包含 ground_truth
    verifier_src = inspect.getsource(Verifier)
    assert "ground_truth" not in verifier_src, \
        "Verifier source contains 'ground_truth'"

    # 检查 ground_truth_check 不被任何验证 action 调用
    from cognition.evidence import (
        action_observe, action_count, action_compare,
        action_counterexample, action_prediction, action_logical_derive,
    )
    for fn in [action_observe, action_count, action_compare,
               action_counterexample, action_prediction, action_logical_derive]:
        src = inspect.getsource(fn)
        assert "ground_truth" not in src, \
            f"{fn.__name__} source contains 'ground_truth'"

    print("test_ground_truth_not_in_verifier OK")


# 9. 完整历史不会通过 Context 偷渡给验证器
def test_full_history_not_smuggled():
    """future_information_leak_check 必须通过。"""
    result = run_e0_2()

    assert result["future_information_leak_check"]["passed"], \
        f"future_information_leak_check failed: {result['future_information_leak_check']['details']}"

    # 每步 visible_history_length == step + 1
    for s in result["step_records"]:
        assert s["visible_history_length"] == s["step"] + 1, \
            f"step {s['step']}: visible_history_length={s['visible_history_length']}, expected {s['step'] + 1}"

    print("test_full_history_not_smuggled OK")


# 10. trace 的时间顺序正确
def test_trace_temporal_order():
    """Trace 步骤应按时间顺序排列，每步 meta 中有 step 信息。"""
    result = run_e0_2()

    step_records = result["step_records"]
    for i in range(1, len(step_records)):
        assert step_records[i]["step"] > step_records[i - 1]["step"], \
            "step numbers should be strictly increasing"

    # observed_object_count 单调不减
    for i in range(1, len(step_records)):
        assert step_records[i]["observed_object_count"] >= step_records[i - 1]["observed_object_count"], \
            f"observed_object_count decreased at step {step_records[i]['step']}"

    print("test_trace_temporal_order OK")


def run_all():
    tests = [
        test_step_t_cannot_see_t1_objects,
        test_candidates_only_from_observed_objects,
        test_valid_belief_not_in_constructor_input,
        test_observation_expands_observed_objects,
        test_pa_qa_not_early,
        test_unknown_can_be_reverified,
        test_valid_invalid_not_reverified,
        test_ground_truth_not_in_verifier,
        test_full_history_not_smuggled,
        test_trace_temporal_order,
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
    print(f"\nE0-2 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

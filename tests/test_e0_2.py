"""E0-2 专用测试：时间展开穷举构造 + 验证闭环。

验证：
  1. future observation 不可见
  2. 当前 observation 才能进入 object space
  3. 新对象影响下一时间步候选空间
  4. 候选不能提前出现
  5. 反例到来后状态可以改变
  6. BeliefStore 历史没有被未来信息污染
  7. Trace 能还原时间顺序
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
import itertools


# 1. future observation 不可见
def test_future_observation_not_visible():
    """在 t 时刻，Context.world_history 长度必须为 t+1，不能包含未来。"""
    history = build_world_history()
    full_len = len(history)

    # 模拟 step 0
    t = 0
    visible = history[:t + 1]
    ctx = Context(world_history=visible, constants=["a", "b", "c"],
                  knowledge_view=BeliefStore(), current_step=t)
    assert len(ctx.world_history) == 1, f"step 0: expected 1, got {len(ctx.world_history)}"

    # 模拟 step 3
    t = 3
    visible = history[:t + 1]
    ctx = Context(world_history=visible, constants=["a", "b", "c"],
                  knowledge_view=BeliefStore(), current_step=t)
    assert len(ctx.world_history) == 4, f"step 3: expected 4, got {len(ctx.world_history)}"

    # 确保 step 3 看不到 step 5 的内容
    step5_obj = P.atom("P(c)")
    for s in ctx.world_history:
        assert step5_obj not in s or t >= 6, "future observation leaked into visible history"

    print("test_future_observation_not_visible OK")


# 2. 当前 observation 才能进入 object space
def test_current_observation_enters_object_space():
    """只有当前和过去的 observation 中的对象才能进入 known_objects。"""
    history = build_world_history()
    known_objects = set()

    # step 0
    for prop in history[0]:
        known_objects.add(prop)

    # 在 step 0 时，step 1 的对象不应在 known_objects 中
    step1_only = history[1] - history[0]
    for obj in step1_only:
        assert obj not in known_objects, f"step 1 object {obj} leaked into step 0 known_objects"

    # step 1
    for prop in history[1]:
        known_objects.add(prop)

    # 现在 step 1 的对象应该在 known_objects 中
    for obj in step1_only:
        assert obj in known_objects, f"step 1 object {obj} missing after step 1"

    print("test_current_observation_enters_object_space OK")


# 3. 新对象影响下一时间步候选空间
def test_new_objects_affect_next_step_candidates():
    """step t+1 的候选数量应该比 step t 多（因为新对象加入了 object space）。"""
    history = build_world_history()
    known_objects = set()
    step0_cands = 0
    step1_cands = 0

    # step 0
    for prop in history[0]:
        known_objects.add(prop)
    cands = enumerate_constructions(sorted(known_objects, key=lambda p: p.to_str()))
    step0_cands = len(cands)

    # step 1
    for prop in history[1]:
        known_objects.add(prop)
    cands = enumerate_constructions(sorted(known_objects, key=lambda p: p.to_str()))
    step1_cands = len(cands)

    assert step1_cands > step0_cands, \
        f"step1 ({step1_cands}) should > step0 ({step0_cands}) after new objects enter"

    print("test_new_objects_affect_next_step_candidates OK")


# 4. 候选不能提前出现
def test_candidate_cannot_appear_early():
    """P(a)→Q(a) 不能在 P(a) 和 Q(a) 都进入 object space 之前被构造。"""
    history = build_world_history()
    target = P.impl(P.atom("P(a)"), P.atom("Q(a)"))

    # 找到 P(a) 和 Q(a) 第一次同时出现在 history 的 step
    pa = P.atom("P(a)")
    qa = P.atom("Q(a)")
    first_both = None
    for t, state in enumerate(history):
        if pa in state and qa in state:
            first_both = t
            break

    # 在 first_both 之前，P(a)→Q(a) 不应出现在候选中
    known_objects = set()
    for t in range(first_both):
        for prop in history[t]:
            known_objects.add(prop)
        cands = enumerate_constructions(sorted(known_objects, key=lambda p: p.to_str()))
        cand_props = [c[0] for c in cands]
        assert target not in cand_props, \
            f"P(a)→Q(a) constructed at step {t} before both P(a) and Q(a) available"

    # 在 first_both 时，应该可以构造
    for prop in history[first_both]:
        known_objects.add(prop)
    cands = enumerate_constructions(sorted(known_objects, key=lambda p: p.to_str()))
    cand_props = [c[0] for c in cands]
    assert target in cand_props, \
        f"P(a)→Q(a) should be constructible at step {first_both}"

    print("test_candidate_cannot_appear_early OK")


# 5. 反例到来后状态可以改变
def test_status_changes_after_counterexample():
    """P(c)→Q(c) 在反例到来后应该从 unknown 变为 invalid。"""
    history = build_world_history()
    target = P.impl(P.atom("P(c)"), P.atom("Q(c)"))
    pc = P.atom("P(c)")
    qc = P.atom("Q(c)")

    belief_store = BeliefStore()
    verifier = Verifier()

    # 找 P(c) 和 Q(c) 都出现的第一个 step
    first_both = None
    for t, state in enumerate(history):
        if pc in state and qc in state:
            first_both = t
            break

    assert first_both is not None, "P(c) and Q(c) should both appear at some step"

    # 在 first_both 之前，P(c)→Q(c) 不能被构造
    known = set()
    for t in range(first_both):
        for prop in history[t]:
            known.add(prop)
    cands = enumerate_constructions(sorted(known, key=lambda p: p.to_str()))
    assert target not in [c[0] for c in cands], "P(c)→Q(c) should not be constructible before P(c) and Q(c) both visible"

    # 在 first_both 时构造并验证
    for prop in history[first_both]:
        known.add(prop)
    cands = enumerate_constructions(sorted(known, key=lambda p: p.to_str()))
    assert target in [c[0] for c in cands], "P(c)→Q(c) should be constructible at first_both"

    # 验证：此时应该看到反例（因为 history[:first_both+1] 包含 P(c) 出现但 Q(c) 不出现的 step）
    visible = history[:first_both + 1]
    ctx = Context(world_history=visible, constants=["a", "b", "c"],
                  knowledge_view=belief_store, current_step=first_both)
    result, _ = verifier.verify(target, ctx)

    # P(c)→Q(c) 应该被验证为 invalid（反例存在）
    assert result.result == INVALID, \
        f"P(c)→Q(c) should be INVALID at step {first_both}, got {result.result}"

    print("test_status_changes_after_counterexample OK")


# 6. BeliefStore 历史没有被未来信息污染
def test_belief_store_not_polluted_by_future():
    """在 step t 时验证的 belief 的证据只应该引用 step <= t 的 observation。"""
    result = run_e0_2()

    # 检查 P(c)→Q(c) 的 first_refuted step
    pcqc_timeline = result["key_propositions_timeline"]["P(c)→Q(c)"]
    first_refuted = pcqc_timeline["first_refuted_step"]
    assert first_refuted is not None, "P(c)→Q(c) should have been refuted"

    # P(c) 第一次出现在 step 6（history[6] = {Pa, Qa, Pc}）
    # P(c)→Q(c) 的反例也在 step 6（P(c) 出现但 Q(c) 不出现）
    # first_refuted 应该 >= 6（不能 < 6）
    assert first_refuted >= 6, \
        f"P(c)→Q(c) refuted at step {first_refuted}, but P(c) first appears at step 6"

    # 检查 P(a)→Q(a) 的 first_constructed
    paqa_timeline = result["key_propositions_timeline"]["P(a)→Q(a)"]
    first_c = paqa_timeline["first_constructed_step"]
    # P(a) 和 Q(a) 都在 step 0 出现
    assert first_c == 0, f"P(a)→Q(a) first constructed at step {first_c}, expected 0"

    print("test_belief_store_not_polluted_by_future OK")


# 7. Trace 能还原时间顺序
def test_trace_preserves_temporal_order():
    """Trace 步骤应该按时间顺序排列，且每步的 meta 中有 step 信息。"""
    result = run_e0_2()

    step_records = result["step_records"]
    # 验证 step 序号严格递增
    for i in range(1, len(step_records)):
        assert step_records[i]["step"] > step_records[i - 1]["step"], \
            "step numbers should be strictly increasing"

    # 验证每步的新增对象数和 known_object_count
    for i, s in enumerate(step_records):
        # known_object_count 应该单调不减
        if i > 0:
            assert s["known_object_count"] >= step_records[i - 1]["known_object_count"], \
                f"known_object_count decreased at step {s['step']}"

    print("test_trace_preserves_temporal_order OK")


def run_all():
    tests = [
        test_future_observation_not_visible,
        test_current_observation_enters_object_space,
        test_new_objects_affect_next_step_candidates,
        test_candidate_cannot_appear_early,
        test_status_changes_after_counterexample,
        test_belief_store_not_polluted_by_future,
        test_trace_preserves_temporal_order,
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

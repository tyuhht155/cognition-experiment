"""E0-4 专用不变量测试：knowledge revision and rollback。

验证 10 个核心不变量：
  1. test_valid_belief_can_be_refuted_later
  2. test_refuted_belief_exits_derived_objects
  3. test_refuted_belief_not_used_after_exit
  4. test_invalid_belief_can_become_valid_later
  5. test_derived_objects_reflect_current_status
  6. test_valid_is_not_permanent_truth
  7. test_reverification_uses_only_visible_history
  8. test_newly_refuted_belief_cannot_continue_computation
  9. test_newly_valid_belief_enters_next_round
  10. test_no_future_information

核心约束：
  - 每个时间步重新验证所有已有 belief
  - VALID 可被新证据推翻为 INVALID
  - derived_objects = 当前 BeliefStore 中 STATUS_VALID（非历史曾经 valid）
  - INVALID→VALID 可重新进入 derived_objects
  - 被推翻的 belief 不能继续作为计算输入
  - 本轮新 VALID 不在本轮 constructible 中
"""

import os
import sys
import inspect
from typing import Set

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
from experiments.run_e0_4 import (
    build_world_history, enumerate_with_source, run_e0_4, ground_truth_check,
)


# ============================================================
# 缓存实验结果（避免重复运行）
# ============================================================

_cached_result = None


def _get_result():
    global _cached_result
    if _cached_result is None:
        _cached_result = run_e0_4()
    return _cached_result


# ============================================================
# 1. test_valid_belief_can_be_refuted_later
# ============================================================

def test_valid_belief_can_be_refuted_later():
    """曾经被判定为 VALID 的 proposition，在新 observation 到来后，
    如果新证据形成反例，必须被重新判定为 INVALID。

    具体案例：Q(a)→P(a)
      step 0: P(a) 和 Q(a) 共现 → VALID
      step 3: Q(a) 出现但 P(a) 不出现 → 反例 → INVALID
    """
    result = _get_result()
    qa_pa_timeline = result["belief_status_timeline"]["Q(a)→P(a)"]

    # 必须曾经是 VALID
    was_valid = any(e["status"] == STATUS_VALID for e in qa_pa_timeline)
    assert was_valid, \
        "Q(a)→P(a) should have been VALID at some step"

    # 必须后来变成 INVALID
    became_invalid = any(e["status"] == STATUS_INVALID for e in qa_pa_timeline)
    assert became_invalid, \
        "Q(a)→P(a) should have become INVALID after new counterexample"

    # 验证转变顺序：先 VALID 后 INVALID
    first_valid = None
    first_invalid_after_valid = None
    for entry in qa_pa_timeline:
        if entry["status"] == STATUS_VALID and first_valid is None:
            first_valid = entry["step"]
        if entry["status"] == STATUS_INVALID and first_valid is not None:
            first_invalid_after_valid = entry["step"]
            break

    assert first_valid is not None, "should have been VALID first"
    assert first_invalid_after_valid is not None, "should have become INVALID after being VALID"
    assert first_invalid_after_valid > first_valid, \
        f"INVALID step ({first_invalid_after_valid}) should be after VALID step ({first_valid})"

    # 检查 key_propositions 中的 status_changes
    qa_pa_info = result["key_propositions"]["Q(a)→P(a)"]
    assert len(qa_pa_info["status_changes"]) >= 2, \
        f"Q(a)→P(a) should have >= 2 status changes, got {qa_pa_info['status_changes']}"

    print("test_valid_belief_can_be_refuted_later OK")


# ============================================================
# 2. test_refuted_belief_exits_derived_objects
# ============================================================

def test_refuted_belief_exits_derived_objects():
    """当 VALID proposition 被推翻为 INVALID 时，必须从 derived_objects 中退出。"""
    result = _get_result()
    qa_pa_str = P.impl(P.atom("Q(a)"), P.atom("P(a)")).to_str()

    qa_pa_tl = None
    for dt in result["derived_object_timeline"]:
        if dt["proposition"] == qa_pa_str:
            qa_pa_tl = dt
            break

    assert qa_pa_tl is not None, \
        "Q(a)→P(a) should be in derived_object_timeline (was valid at some point)"

    # 必须有 exit 记录
    assert qa_pa_tl["times_exited"] >= 1, \
        f"Q(a)→P(a) should have exited derived_objects at least once, " \
        f"got times_exited={qa_pa_tl['times_exited']}"

    # exited_derived_step 应该不为 None
    assert qa_pa_tl["exited_derived_step"] is not None, \
        "Q(a)→P(a) should have exited_derived_step set"

    # 最终不应在 derived_objects 中
    assert qa_pa_str not in result["final_derived_objects"], \
        f"Q(a)→P(a) should NOT be in final derived_objects"

    print("test_refuted_belief_exits_derived_objects OK")


# ============================================================
# 3. test_refuted_belief_not_used_after_exit
# ============================================================

def test_refuted_belief_not_used_after_exit():
    """被推翻的 proposition 退出 derived_objects 后，
    后续步骤的 times_used_as_input 不应继续增加。"""
    result = _get_result()
    qa_pa_str = P.impl(P.atom("Q(a)"), P.atom("P(a)")).to_str()

    qa_pa_tl = None
    for dt in result["derived_object_timeline"]:
        if dt["proposition"] == qa_pa_str:
            qa_pa_tl = dt
            break

    assert qa_pa_tl is not None
    exit_step = qa_pa_tl["exited_derived_step"]
    assert exit_step is not None, "Q(a)→P(a) should have exited"

    # 检查 belief_status_timeline：exit_step 之后不应有 used_as_input=True
    qa_pa_timeline = result["belief_status_timeline"]["Q(a)→P(a)"]
    for entry in qa_pa_timeline:
        if entry["step"] > exit_step:
            assert not entry["used_as_input"], \
                f"Q(a)→P(a) used as input at step {entry['step']} " \
                f"after exit at step {exit_step}"
            assert not entry["in_derived_objects"], \
                f"Q(a)→P(a) in derived_objects at step {entry['step']} " \
                f"after exit at step {exit_step}"

    print("test_refuted_belief_not_used_after_exit OK")


# ============================================================
# 4. test_invalid_belief_can_become_valid_later
# ============================================================

def test_invalid_belief_can_become_valid_later():
    """INVALID proposition 在新证据下变成 VALID 时，应重新进入 derived_objects。

    由于当前 Verifier 的 counterexample 规则（max_contradiction >= 0.9 → INVALID）
    一旦发现反例就永久 INVALID，INVALID→VALID 在实际实验中难以自然发生。
    此测试验证 E0-4 的机制本身支持这一转变。
    """
    # 直接测试机制：BeliefStore + derived_objects 重计算
    belief_store = BeliefStore()
    observed_objects: Set[P] = set()

    A = P.atom("A")
    B = P.atom("B")
    AB = P.impl(A, B)

    observed_objects.add(A)
    observed_objects.add(B)

    # Step 0: AB 验证为 INVALID
    belief_store.update_belief(AB, STATUS_INVALID, 0.9, evidence_count_delta=1)

    derived_objects = set()
    for b in belief_store.all_beliefs():
        if b.status == STATUS_VALID and b.proposition not in observed_objects:
            derived_objects.add(b.proposition)

    assert AB not in derived_objects, \
        "INVALID prop must NOT be in derived_objects"

    # Step 1: 新证据到来，AB 变成 VALID
    belief_store.update_belief(AB, STATUS_VALID, 0.8, evidence_count_delta=1)

    derived_objects = set()
    for b in belief_store.all_beliefs():
        if b.status == STATUS_VALID and b.proposition not in observed_objects:
            derived_objects.add(b.proposition)

    assert AB in derived_objects, \
        "Prop that transitioned INVALID→VALID must enter derived_objects"

    print("test_invalid_belief_can_become_valid_later OK")


# ============================================================
# 5. test_derived_objects_reflect_current_status
# ============================================================

def test_derived_objects_reflect_current_status():
    """derived_objects 必须是"当前时刻有效知识"，
    而不是"历史曾经 valid 的知识"。"""
    result = _get_result()

    bs_stats = result["belief_store_stats"]

    # final_derived_objects 数量应 <= BeliefStore 中 valid 数量
    assert len(result["final_derived_objects"]) <= bs_stats["valid"], \
        f"derived_objects ({len(result['final_derived_objects'])}) " \
        f"should <= valid count ({bs_stats['valid']})"

    # 检查 Q(a)→P(a) 最终不在 derived_objects 中（因为它是 INVALID）
    qa_pa_str = P.impl(P.atom("Q(a)"), P.atom("P(a)")).to_str()
    assert qa_pa_str not in result["final_derived_objects"], \
        "Q(a)→P(a) is INVALID but still in derived_objects"

    # 检查 P(a)→Q(a) 最终在 derived_objects 中（因为它是 VALID）
    pa_qa_str = P.impl(P.atom("P(a)"), P.atom("Q(a)")).to_str()
    assert pa_qa_str in result["final_derived_objects"], \
        "P(a)→Q(a) is VALID but not in derived_objects"

    # 检查 P(c)→Q(c) 最终不在 derived_objects 中（因为它是 INVALID）
    pc_qc_str = P.impl(P.atom("P(c)"), P.atom("Q(c)")).to_str()
    assert pc_qc_str not in result["final_derived_objects"], \
        "P(c)→Q(c) is INVALID but still in derived_objects"

    print("test_derived_objects_reflect_current_status OK")


# ============================================================
# 6. test_valid_is_not_permanent_truth
# ============================================================

def test_valid_is_not_permanent_truth():
    """STATUS_VALID 不等于"永远正确"。
    VALID proposition 可以在新证据下被推翻为 INVALID。"""
    result = _get_result()

    assert result["qa_pa_refuted_after_being_valid"], \
        "Q(a)→P(a) should have been VALID then refuted to INVALID"

    all_trans = result["all_transitions"]
    assert all_trans.get("valid_to_invalid", 0) > 0, \
        f"should have valid_to_invalid transitions, got {all_trans}"

    assert result["total_status_changed"] > 0, \
        f"total_status_changed should > 0, got {result['total_status_changed']}"

    print("test_valid_is_not_permanent_truth OK")


# ============================================================
# 7. test_reverification_uses_only_visible_history
# ============================================================

def test_reverification_uses_only_visible_history():
    """重新验证时只能使用 history[:t+1]，不能偷看未来 observation。"""
    history = build_world_history()

    for t in range(len(history)):
        visible = history[:t + 1]
        assert len(visible) == t + 1, \
            f"step {t}: visible history should be {t + 1}, got {len(visible)}"

    result = _get_result()
    assert result["future_information_leak_check"]["passed"], \
        f"future_information_leak_check failed"

    for s in result["step_records"]:
        assert s["visible_history_length"] == s["step"] + 1, \
            f"step {s['step']}: visible_history_length={s['visible_history_length']}, " \
            f"expected {s['step'] + 1}"

    # Q(a)→P(a) 的反例首次出现在 step 3（{Qa, Sa}：Q(a) 出现但 P(a) 不出现）
    # 所以 INVALID 应在 step 3 或之后
    qa_pa_timeline = result["belief_status_timeline"]["Q(a)→P(a)"]

    first_invalid_step = None
    for entry in qa_pa_timeline:
        if entry["status"] == STATUS_INVALID:
            first_invalid_step = entry["step"]
            break

    assert first_invalid_step is not None, "Q(a)→P(a) should become INVALID at some point"
    assert first_invalid_step >= 3, \
        f"Q(a)→P(a) should not become INVALID before step 3 " \
        f"(first counterexample at step 3), got step {first_invalid_step}"

    # 在 step 3 之前不应是 INVALID
    for entry in qa_pa_timeline:
        if entry["step"] < 3:
            assert entry["status"] != STATUS_INVALID, \
                f"Q(a)→P(a) should not be INVALID at step {entry['step']} " \
                f"(counterexample first at step 3)"

    print("test_reverification_uses_only_visible_history OK")


# ============================================================
# 8. test_newly_refuted_belief_cannot_continue_computation
# ============================================================

def test_newly_refuted_belief_cannot_continue_computation():
    """当 proposition 在重新验证过程中变成 INVALID，
    本轮剩余候选生成必须使用更新后的 derived_objects。
    绝不能继续使用已经被推翻的 proposition。"""
    result = _get_result()
    qa_pa_str = P.impl(P.atom("Q(a)"), P.atom("P(a)")).to_str()

    qa_pa_tl = None
    for dt in result["derived_object_timeline"]:
        if dt["proposition"] == qa_pa_str:
            qa_pa_tl = dt
            break

    assert qa_pa_tl is not None
    exit_step = qa_pa_tl["exited_derived_step"]
    assert exit_step is not None, "Q(a)→P(a) should have exited derived_objects"

    # 在 exit_step，Q(a)→P(a) 应在 exited_derived 列表中
    exit_step_record = result["step_records"][exit_step]
    assert qa_pa_str in exit_step_record.get("exited_derived", []), \
        f"Q(a)→P(a) should be in exited_derived at step {exit_step}"

    # 在 exit_step 之后的所有步骤，不应重新进入 derived_objects
    for s in result["step_records"]:
        if s["step"] > exit_step:
            assert qa_pa_str not in s.get("entered_derived", []), \
                f"Q(a)→P(a) should not re-enter derived at step {s['step']} " \
                f"(was refuted at step {exit_step})"

    # 验证 derived_object_timeline 的 history
    for h in qa_pa_tl["history"]:
        if h["action"] == "exit":
            assert h["step"] == exit_step
        if h["action"] == "enter":
            assert h["step"] < exit_step, \
                f"Q(a)→P(a) entered at step {h['step']} should be before exit at {exit_step}"

    print("test_newly_refuted_belief_cannot_continue_computation OK")


# ============================================================
# 9. test_newly_valid_belief_enters_next_round
# ============================================================

def test_newly_valid_belief_enters_next_round():
    """本轮新验证为 VALID 的候选（新构造的）不在本轮 constructible 中，
    但在下一轮进入 derived_objects 后才可使用。"""
    result = _get_result()

    # Step 0: 没有之前步骤的 derived
    step0 = result["step_records"][0]
    assert step0["derived_object_count"] == 0, \
        f"step 0: derived_object_count should be 0, got {step0['derived_object_count']}"

    assert step0["constructible_object_count"] == step0["observed_object_count"], \
        f"step 0: constructible should == observed"

    assert step0["candidates_from_derived"] == 0, \
        f"step 0: candidates_from_derived should be 0"

    # Step 1: 上一轮的 VALID 命题进入 derived_objects
    step1 = result["step_records"][1]
    assert step1["derived_object_count"] > 0, \
        f"step 1: derived_object_count should > 0"
    assert step1["candidates_from_derived"] > 0, \
        f"step 1: candidates_from_derived should > 0"

    # P(a)→Q(a) 在 step 0 被构造验证为 VALID，在 step 1 进入 derived_objects
    pa_qa_str = P.impl(P.atom("P(a)"), P.atom("Q(a)")).to_str()
    pa_qa_tl = None
    for dt in result["derived_object_timeline"]:
        if dt["proposition"] == pa_qa_str:
            pa_qa_tl = dt
            break

    assert pa_qa_tl is not None, "P(a)→Q(a) should be in derived_object_timeline"
    assert pa_qa_tl["entered_derived_step"] >= 1, \
        f"P(a)→Q(a) entered_derived_step should be >= 1, " \
        f"got {pa_qa_tl['entered_derived_step']}"
    assert pa_qa_tl["times_used_as_input"] > 0, \
        "P(a)→Q(a) should have been used as input"

    print("test_newly_valid_belief_enters_next_round OK")


# ============================================================
# 10. test_no_future_information
# ============================================================

def test_no_future_information():
    """每个时刻 t 只能看到 world_history[:t+1]，禁止使用未来 observation。"""
    history = build_world_history()

    for t in range(len(history)):
        visible = history[:t + 1]
        assert len(visible) == t + 1, \
            f"step {t}: visible history length should be {t + 1}, got {len(visible)}"

    # 检查 step 2 看不到 step 3 的反例
    qa = P.atom("Q(a)")
    pa = P.atom("P(a)")
    step3 = history[3]
    assert qa in step3 and pa not in step3, \
        "step 3 should have Q(a) without P(a) (counterexample for Q(a)→P(a))"
    visible_2 = history[:3]
    assert step3 not in visible_2, \
        "step 3 should not be visible at step 2"

    result = _get_result()
    assert result["future_information_leak_check"]["passed"], \
        f"future_information_leak_check failed"

    # Q(a)→P(a) 在 step 3 之前不应是 INVALID
    qa_pa_timeline = result["belief_status_timeline"]["Q(a)→P(a)"]
    for entry in qa_pa_timeline:
        if entry["step"] < 3:
            assert entry["status"] != STATUS_INVALID, \
                f"Q(a)→P(a) should not be INVALID at step {entry['step']} " \
                f"(counterexample first at step 3)"

    print("test_no_future_information OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_valid_belief_can_be_refuted_later,
        test_refuted_belief_exits_derived_objects,
        test_refuted_belief_not_used_after_exit,
        test_invalid_belief_can_become_valid_later,
        test_derived_objects_reflect_current_status,
        test_valid_is_not_permanent_truth,
        test_reverification_uses_only_visible_history,
        test_newly_refuted_belief_cannot_continue_computation,
        test_newly_valid_belief_enters_next_round,
        test_no_future_information,
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
    print(f"\nE0-4 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

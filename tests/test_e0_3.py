"""E0-3 专用不变量测试：derived knowledge as computation input。

验证 8 个核心不变量：
  1. test_valid_belief_enters_derived_objects
  2. test_invalid_belief_never_enters_derived_objects
  3. test_unknown_belief_never_enters_derived_objects
  4. test_unknown_can_become_valid_later
  5. test_derived_object_can_be_used_next_step
  6. test_derived_object_cannot_be_used_same_step
  7. test_no_future_information
  8. test_ground_truth_not_used_by_verifier

核心约束：
  - STATUS_VALID → 可进入 derived_objects
  - STATUS_INVALID / STATUS_UNKNOWN → 不能进入 derived_objects
  - 本轮构造+验证的新 valid 不能在本轮作为 constructor 输入
  - 下一轮 valid prop 进入 derived_objects 后才可使用
  - ground_truth 只在实验结束后外部比较，不进入 verifier
"""

import os
import sys
import inspect
from typing import Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.verification import Verifier, VALID, INVALID, UNKNOWN
from cognition.operations import Context
from experiments.run_e0_3 import (
    build_world_history, enumerate_constructions, enumerate_with_source,
    run_e0_3, ground_truth_check,
)


# ============================================================
# 1. test_valid_belief_enters_derived_objects
# ============================================================

def test_valid_belief_enters_derived_objects():
    """STATUS_VALID 的命题（非 observed）必须进入 derived_objects。"""
    result = run_e0_3()

    # P(a)→Q(a) 在 step 0 被验证为 VALID，应进入 derived_objects
    assert result["pa_qa_in_derived_objects"], \
        "P(a)→Q(a) was verified VALID but did not enter derived_objects"

    # 检查 derived_object_timeline 非空
    assert len(result["derived_object_timeline"]) > 0, \
        "derived_object_timeline should not be empty"

    # 检查每个进入 derived_objects 的 prop 都有 timeline 记录
    derived_strs = set(result["final_derived_objects"])
    timeline_strs = set(dt["proposition"] for dt in result["derived_object_timeline"])
    assert derived_strs == timeline_strs, \
        f"derived_objects and timeline mismatch: {derived_strs} vs {timeline_strs}"

    # 检查 P(a)→Q(a) 在 timeline 中
    pa_qa_str = P.impl(P.atom("P(a)"), P.atom("Q(a)")).to_str()
    pa_qa_in_timeline = any(
        dt["proposition"] == pa_qa_str for dt in result["derived_object_timeline"]
    )
    assert pa_qa_in_timeline, "P(a)→Q(a) should be in derived_object_timeline"

    print("test_valid_belief_enters_derived_objects OK")


# ============================================================
# 2. test_invalid_belief_never_enters_derived_objects
# ============================================================

def test_invalid_belief_never_enters_derived_objects():
    """STATUS_INVALID 的命题绝对不能进入 derived_objects。"""
    result = run_e0_3()

    # P(c)→Q(c) 被验证为 INVALID，不能进入 derived_objects
    assert result["pc_qc_not_in_derived_objects"], \
        "P(c)→Q(c) was INVALID but entered derived_objects"

    # 检查 P(c)→Q(c) 不在 final_derived_objects 中
    pc_qc_str = P.impl(P.atom("P(c)"), P.atom("Q(c)")).to_str()
    assert pc_qc_str not in result["final_derived_objects"], \
        f"P(c)→Q(c) should not be in derived_objects, found in: {result['final_derived_objects']}"

    # 检查 P(c)→Q(c) 不在 derived_object_timeline 中
    pc_qc_in_timeline = any(
        dt["proposition"] == pc_qc_str for dt in result["derived_object_timeline"]
    )
    assert not pc_qc_in_timeline, "P(c)→Q(c) should not be in derived_object_timeline"

    # 检查 key_propositions_timeline 中 P(c)→Q(c) 的 in_derived_objects 为 False
    pcqc_info = result["key_propositions_timeline"]["P(c)→Q(c)"]
    assert not pcqc_info["in_derived_objects"], \
        "P(c)→Q(c) in_derived_objects should be False"
    assert pcqc_info["final_status"] == STATUS_INVALID, \
        f"P(c)→Q(c) final_status should be invalid, got {pcqc_info['final_status']}"

    print("test_invalid_belief_never_enters_derived_objects OK")


# ============================================================
# 3. test_unknown_belief_never_enters_derived_objects
# ============================================================

def test_unknown_belief_never_enters_derived_objects():
    """STATUS_UNKNOWN 的命题不能进入 derived_objects。"""
    result = run_e0_3()

    # ¬P(a) 是 UNKNOWN，不能进入 derived_objects
    npa_str = P.neg(P.atom("P(a)")).to_str()
    assert npa_str not in result["final_derived_objects"], \
        f"¬P(a) should not be in derived_objects (it's UNKNOWN)"

    # P(a)∧Q(a) 是 UNKNOWN，不能进入 derived_objects
    paqa_str = P.conj(P.atom("P(a)"), P.atom("Q(a)")).to_str()
    assert paqa_str not in result["final_derived_objects"], \
        f"P(a)∧Q(a) should not be in derived_objects (it's UNKNOWN)"

    # 检查 key_propositions_timeline 中 ¬P(a) 和 P(a)∧Q(a) 的状态
    npa_info = result["key_propositions_timeline"]["¬P(a)"]
    assert npa_info["final_status"] == STATUS_UNKNOWN, \
        f"¬P(a) should be UNKNOWN, got {npa_info['final_status']}"
    assert not npa_info["in_derived_objects"], \
        "¬P(a) should not be in derived_objects"

    paqa_info = result["key_propositions_timeline"]["P(a)∧Q(a)"]
    assert paqa_info["final_status"] == STATUS_UNKNOWN, \
        f"P(a)∧Q(a) should be UNKNOWN, got {paqa_info['final_status']}"
    assert not paqa_info["in_derived_objects"], \
        "P(a)∧Q(a) should not be in derived_objects"

    # 通用检查：所有 derived_objects 中的 prop 都不是 UNKNOWN
    # 通过 BeliefStore stats 检查
    bs_stats = result["belief_store_stats"]
    assert bs_stats["unknown"] > 0, "should have UNKNOWN beliefs"
    # derived_objects 数量应等于 valid 数量（非 observed 的 valid）
    assert len(result["final_derived_objects"]) == bs_stats["valid"], \
        f"derived_objects ({len(result['final_derived_objects'])}) " \
        f"should equal valid count ({bs_stats['valid']})"

    print("test_unknown_belief_never_enters_derived_objects OK")


# ============================================================
# 4. test_unknown_can_become_valid_later
# ============================================================

def test_unknown_can_become_valid_later():
    """UNKNOWN prop 不能进入 derived_objects。
    后续新 observation 到来后允许重新验证。
    如果之后变成 VALID，才允许进入 derived_objects。

    此测试直接验证 E0-3 的 derived_objects 更新逻辑：
      - UNKNOWN → 不进入 derived_objects
      - 同一 prop 后变为 VALID → 进入 derived_objects
    """
    belief_store = BeliefStore()
    derived_objects: Set[P] = set()
    observed_objects: Set[P] = set()

    A = P.atom("A")
    B = P.atom("B")
    AB = P.impl(A, B)

    observed_objects.add(A)
    observed_objects.add(B)

    # --- Step 0: AB 验证为 UNKNOWN ---
    belief_store.update_belief(AB, STATUS_UNKNOWN, 0.0, evidence_count_delta=1)

    # 模拟 E0-3 derived_objects 更新逻辑
    for b in belief_store.all_beliefs():
        if b.status == STATUS_VALID and b.proposition not in derived_objects:
            if b.proposition not in observed_objects:
                derived_objects.add(b.proposition)

    assert AB not in derived_objects, \
        "UNKNOWN prop must NOT enter derived_objects"

    # --- Step 1: AB 重新验证，变成 VALID ---
    # 模拟新 observation 到来后重新验证
    belief_store.update_belief(AB, STATUS_VALID, 0.5, evidence_count_delta=1)

    # 再次模拟 derived_objects 更新
    for b in belief_store.all_beliefs():
        if b.status == STATUS_VALID and b.proposition not in derived_objects:
            if b.proposition not in observed_objects:
                derived_objects.add(b.proposition)

    assert AB in derived_objects, \
        "Prop that transitioned UNKNOWN→VALID must enter derived_objects"

    print("test_unknown_can_become_valid_later OK")


# ============================================================
# 5. test_derived_object_can_be_used_next_step
# ============================================================

def test_derived_object_can_be_used_next_step():
    """在 step t 验证为 VALID 的 prop，在 step t+1 必须能作为 constructor 输入。"""
    result = run_e0_3()

    step0 = result["step_records"][0]
    step1 = result["step_records"][1]

    # Step 0: 6 个 prop 被验证为 VALID
    assert step0["valid"] == 6, \
        f"step 0: expected 6 valid, got {step0['valid']}"

    # Step 1: derived_object_count 应为 6（来自 step 0 的 valid）
    assert step1["derived_object_count"] == 6, \
        f"step 1: expected derived_object_count=6, got {step1['derived_object_count']}"

    # Step 1: constructible > observed（因为 derived 加入）
    assert step1["constructible_object_count"] > step1["observed_object_count"], \
        f"step 1: constructible ({step1['constructible_object_count']}) " \
        f"should > observed ({step1['observed_object_count']})"

    # Step 1: 有来自 derived 的候选
    assert step1["candidates_from_derived"] > 0, \
        f"step 1: candidates_from_derived should > 0, got {step1['candidates_from_derived']}"

    # P(a)→Q(a) 在 step 0 验证为 VALID
    # 在 step 1 应出现在 derived_objects 中并被使用
    assert result["pa_qa_in_derived_objects"], \
        "P(a)→Q(a) should be in derived_objects"

    # derived_timeline 中 P(a)→Q(a) 的 first_used_as_input_step 应该 >= first_valid_step
    pa_qa_str = P.impl(P.atom("P(a)"), P.atom("Q(a)")).to_str()
    pa_qa_tl = None
    for dt in result["derived_object_timeline"]:
        if dt["proposition"] == pa_qa_str:
            pa_qa_tl = dt
            break

    assert pa_qa_tl is not None, "P(a)→Q(a) should be in derived_object_timeline"
    assert pa_qa_tl["first_used_as_input_step"] is not None, \
        "P(a)→Q(a) should have been used as input"
    assert pa_qa_tl["first_used_as_input_step"] >= pa_qa_tl["first_valid_step"], \
        f"first_used_as_input_step ({pa_qa_tl['first_used_as_input_step']}) " \
        f"should >= first_valid_step ({pa_qa_tl['first_valid_step']})"
    assert pa_qa_tl["times_used_as_input"] > 0, \
        f"P(a)→Q(a) should have times_used_as_input > 0, got {pa_qa_tl['times_used_as_input']}"

    print("test_derived_object_can_be_used_next_step OK")


# ============================================================
# 6. test_derived_object_cannot_be_used_same_step
# ============================================================

def test_derived_object_cannot_be_used_same_step():
    """本轮构造+验证的新 VALID prop 不能在本轮作为 constructor 输入。
    必须等到下一轮进入 derived_objects 后才能使用。

    例如：
      A、B → A→B（本轮验证为 VALID）
      本轮不能立即继续：A→B + C → (A→B)→C
      必须等到下一轮，A→B 进入 derived_objects 后才能使用。
    """
    result = run_e0_3()

    step0 = result["step_records"][0]

    # Step 0: derived_object_count == 0（没有 derived 对象可用）
    assert step0["derived_object_count"] == 0, \
        f"step 0: derived_object_count should be 0, got {step0['derived_object_count']}"

    # Step 0: constructible == observed（无 derived）
    assert step0["constructible_object_count"] == step0["observed_object_count"], \
        f"step 0: constructible ({step0['constructible_object_count']}) " \
        f"should == observed ({step0['observed_object_count']})"

    # Step 0: 没有来自 derived 的候选
    assert step0["candidates_from_derived"] == 0, \
        f"step 0: candidates_from_derived should be 0, got {step0['candidates_from_derived']}"

    # Step 0: 6 个 prop 被验证为 VALID（它们本轮不能被使用）
    assert step0["valid"] == 6

    # 验证递归禁止：所有 step 的 derived_object_count 只包含之前步骤的 valid
    # 即 step t 的 derived_count <= sum(step 0..t-1 的 valid)
    for i in range(1, len(result["step_records"])):
        prev_valid_sum = sum(
            result["step_records"][j]["valid"] for j in range(i)
        )
        cur_derived = result["step_records"][i]["derived_object_count"]
        assert cur_derived <= prev_valid_sum, \
            f"step {i}: derived_object_count ({cur_derived}) " \
            f"should <= sum of prev valid ({prev_valid_sum})"

    # 额外验证：first_used_as_input_step > first_valid_step 的构造步骤发生
    # 即 derived prop 的首次使用严格在它进入 derived_objects 之后
    # derived_timeline 中 first_valid_step 是进入 derived_objects 的步骤
    # first_used_as_input_step 应 >= first_valid_step（进入 derived 当轮即可使用）
    for dt in result["derived_object_timeline"]:
        if dt["first_used_as_input_step"] is not None:
            assert dt["first_used_as_input_step"] >= dt["first_valid_step"], \
                f"{dt['proposition']}: first_used ({dt['first_used_as_input_step']}) " \
                f"< first_valid ({dt['first_valid_step']})"

    print("test_derived_object_cannot_be_used_same_step OK")


# ============================================================
# 7. test_no_future_information
# ============================================================

def test_no_future_information():
    """每个时刻 t 只能看到 world_history[:t+1]，禁止使用未来 observation。"""
    history = build_world_history()

    # 检查每个 step 的 visible_history_length == step + 1
    for t in range(len(history)):
        visible = history[:t + 1]
        assert len(visible) == t + 1, \
            f"step {t}: visible history length should be {t + 1}, got {len(visible)}"

    # 检查 step 6 看不到 step 7 的 Q(c)
    visible_6 = history[:7]
    qc = P.atom("Q(c)")
    qc_seen = any(qc in s for s in visible_6)
    assert not qc_seen, \
        "Q(c) should not be visible at step 6 (first appears at step 7)"

    # 检查 step 7 看得到 Q(c)
    visible_7 = history[:8]
    qc_seen_7 = any(qc in s for s in visible_7)
    assert qc_seen_7, "Q(c) should be visible at step 7"

    # 运行完整实验，检查 future_information_leak_check
    result = run_e0_3()
    assert result["future_information_leak_check"]["passed"], \
        f"future_information_leak_check failed: " \
        f"{result['future_information_leak_check']['details']}"

    # 检查每步的 visible_history_length
    for s in result["step_records"]:
        assert s["visible_history_length"] == s["step"] + 1, \
            f"step {s['step']}: visible_history_length={s['visible_history_length']}, " \
            f"expected {s['step'] + 1}"

    print("test_no_future_information OK")


# ============================================================
# 8. test_ground_truth_not_used_by_verifier
# ============================================================

def test_ground_truth_not_used_by_verifier():
    """ground_truth 只能在实验结束后用于外部比较。
    Verifier 和所有 verification actions 不得调用 ground_truth_check。
    """
    # 检查 Verifier 源码不包含 ground_truth
    verifier_src = inspect.getsource(Verifier)
    assert "ground_truth" not in verifier_src, \
        "Verifier source contains 'ground_truth'"

    # 检查所有 verification action 不调用 ground_truth
    from cognition.evidence import (
        action_observe, action_count, action_compare,
        action_counterexample, action_prediction, action_logical_derive,
        EvidenceEvaluator,
    )
    for fn in [action_observe, action_count, action_compare,
               action_counterexample, action_prediction, action_logical_derive]:
        src = inspect.getsource(fn)
        assert "ground_truth" not in src, \
            f"{fn.__name__} source contains 'ground_truth'"

    # 检查 EvidenceEvaluator 不访问 ground truth
    evaluator_src = inspect.getsource(EvidenceEvaluator)
    assert "ground_truth" not in evaluator_src, \
        "EvidenceEvaluator source contains 'ground_truth'"

    # 检查 run_e0_3 中的 ground_truth_check 只在实验结束后调用
    # 通过检查 key_propositions_timeline 中 gt_holds 字段存在但不在 step_records 中
    result = run_e0_3()

    # step_records 不应包含 gt_holds 字段
    for s in result["step_records"]:
        assert "gt_holds" not in s, \
            f"step {s['step']}: step_record should not contain gt_holds " \
            f"(ground truth must not be used during computation)"

    # ground_truth_check 函数本身存在（用于实验后外部比较）
    assert callable(ground_truth_check), "ground_truth_check should be callable"

    # 验证 ground_truth_check 只使用 full_history（不修改任何 store）
    import experiments.run_e0_3 as e0_3_mod
    src = inspect.getsource(e0_3_mod)
    # ground_truth_check 函数定义中不应访问 belief_store, evidence_log 等运行时组件
    gt_func_src = inspect.getsource(ground_truth_check)
    assert "belief_store" not in gt_func_src, \
        "ground_truth_check should not access belief_store"
    assert "evidence_log" not in gt_func_src, \
        "ground_truth_check should not access evidence_log"
    assert "verifier" not in gt_func_src.lower(), \
        "ground_truth_check should not access verifier"

    print("test_ground_truth_not_used_by_verifier OK")


# ============================================================
# 运行所有测试
# ============================================================

def run_all():
    tests = [
        test_valid_belief_enters_derived_objects,
        test_invalid_belief_never_enters_derived_objects,
        test_unknown_belief_never_enters_derived_objects,
        test_unknown_can_become_valid_later,
        test_derived_object_can_be_used_next_step,
        test_derived_object_cannot_be_used_same_step,
        test_no_future_information,
        test_ground_truth_not_used_by_verifier,
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
    print(f"\nE0-3 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

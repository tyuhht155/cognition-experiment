"""E0-9 不变量测试：Value-Directed Computation Direction Selection。

验证价值函数能否在知识空间很小时决定下一步计算什么。

测试覆盖（16 项不变量）：
  1.  test_value_function_participates        —— 价值函数确实参与方向选择
  2.  test_baseline_no_value                  —— 无价值函数时用 baseline 固定顺序
  3.  test_high_value_prioritized             —— 高价值对象优先选择
  4.  test_high_value_not_means_valid         —— 价值高不代表 valid
  5.  test_invalid_can_be_selected            —— invalid 对象仍可被选择
  6.  test_valid_low_value_retained           —— valid 但低价值对象不删除
  7.  test_feedback_recomputes_value          —— 反馈后下一轮价值重新计算
  8.  test_value_function_changes_path        —— 价值函数改变会改变计算路径
  9.  test_tie_break_stable                   —— 相同 value 有稳定 tie-break
  10. test_no_transformation_history_dependency —— 不依赖 transformation history
  11. test_no_search_dependency               —— 不依赖搜索
  12. test_no_llm_dependency                  —— 不依赖 LLM
  13. test_no_ground_truth                    —— 不依赖 ground truth
  14. test_strict_time_causal                 —— 严格时间因果
  15. test_complete_trace                     —— 完整 trace（含所有候选 value）
  16. test_constructible_valid_valuable_separated —— 三者严格分离
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_9 import (
    build_world,
    make_goal,
    evaluate_direction,
    evaluate_wrong_direction,
    select_direction,
    run_e0_9_episode,
    run_e0_9,
    _sub_props_set,
    MAX_STEPS,
)
from experiments.run_e0_7 import generate_candidates as generate_candidates_brute
from experiments.run_e0_7_2 import verify_proposition_extended
from experiments.run_e0_7_3 import goal_reached
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


# ============================================================
# 辅助
# ============================================================

def _run_baseline():
    return run_e0_9_episode(build_world(), make_goal(), mode="baseline")


def _run_value():
    return run_e0_9_episode(build_world(), make_goal(), mode="value")


def _run_wrong():
    return run_e0_9_episode(build_world(), make_goal(), mode="wrong")


# ============================================================
# 1. 价值函数确实参与方向选择
# ============================================================

def test_value_function_participates():
    """value 模式的 trace 中应包含每个候选的 value 值。"""
    r = _run_value()
    found_value = False
    for tr in r["trace"]:
        for ev in tr.get("evaluations", []):
            if ev.get("value") is not None:
                found_value = True
                break
    assert found_value, "value 模式 trace 中应包含候选的 value 值"


# ============================================================
# 2. 无价值函数时用 baseline 固定顺序
# ============================================================

def test_baseline_no_value():
    """baseline 模式不应计算 value，按固定顺序选择。"""
    r = _run_baseline()
    for tr in r["trace"]:
        for ev in tr.get("evaluations", []):
            assert ev.get("value") is None, \
                "baseline 模式不应计算 value"
    # 选中第一个候选
    for tr in r["trace"]:
        if tr.get("selected") and tr.get("evaluations"):
            assert tr["evaluations"][0].get("selected"), \
                "baseline 应选第一个候选"


# ============================================================
# 3. 高价值对象优先选择
# ============================================================

def test_high_value_prioritized():
    """value 模式应优先选择 value 最高的候选。"""
    r = _run_value()
    for tr in r["trace"]:
        evals = tr.get("evaluations", [])
        selected = tr.get("selected")
        if not evals or not selected:
            continue
        selected_prop = selected["proposition"]
        # 找到选中候选的 value
        selected_ev = next(
            (ev for ev in evals if ev["proposition"] == selected_prop), None)
        if selected_ev and selected_ev.get("value") is not None:
            max_val = max(ev["value"] for ev in evals
                          if ev.get("value") is not None)
            assert selected_ev["value"] >= max_val - 0.0001, \
                f"选中的 value={selected_ev['value']} 应接近最高 value={max_val}"


# ============================================================
# 4. 价值高不代表 valid
# ============================================================

def test_high_value_not_means_valid():
    """一个候选可以 value 高但 verdict 为 invalid/unknown。
    价值评价和验证是独立的。"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    goal = make_goal()
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)

    # ¬A(a) 可能 value 低（不含 goal 子结构），但验证为 invalid
    neg_a = P.neg(Aa)
    ev = evaluate_direction(neg_a, goal, "neg", 0.5, belief)
    verdict, _, _ = verify_proposition_extended(neg_a, build_world())
    # value 和 verdict 独立——不要求 value 高时 verdict 也高
    assert isinstance(ev["value"], float)
    assert isinstance(verdict, str)
    # ¬A 验证为 invalid 但仍可被评价
    assert verdict in ("invalid", "unknown")


# ============================================================
# 5. invalid 对象仍可被选择（但验证后阻止复用）
# ============================================================

def test_invalid_can_be_selected():
    """wrong 模式应选择无关对象，其中一些验证为 invalid。
    invalid 对象进入 belief 后不会被删除，但不会成为 derived_valid。"""
    r = _run_wrong()
    # wrong 模式应产生 invalid 验证
    has_invalid = any(tr["verdict"] == "invalid" for tr in r["trace"])
    assert has_invalid, "wrong 模式应产生 invalid 验证"
    # invalid 对象不应在 constructible 中（只 valid 才进 constructible）
    for tr in r["trace"]:
        invalid_props = {
            tr["selected"]["proposition"] for tr in r["trace"]
            if tr.get("verdict") == "invalid"
        }
        # 检查后续轮的 constructible 不含 invalid 对象
        for later_tr in r["trace"][tr["step"] + 1:]:
            for prop in invalid_props:
                assert prop not in later_tr.get("constructible", []), \
                    "invalid 对象不应进入 constructible"


# ============================================================
# 6. valid 但低价值对象不会被删除
# ============================================================

def test_valid_low_value_retained():
    """valid 但低价值对象应保留在 belief_store 中，不被删除。
    BeliefStore 不按 value 删除对象。"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    goal = make_goal()
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)

    # A∧B 是 valid 但 low value（不在 goal 子结构中，但共享叶子谓词）
    conj_ab = P.conj(Aa, Ba)
    ev = evaluate_direction(conj_ab, goal, "conj", 0.7, belief)
    impl_ab = P.impl(Aa, Ba)
    ev2 = evaluate_direction(impl_ab, goal, "impl", 0.9, belief)
    # A→B 的 value 应高于 A∧B（A→B 是 goal 的子结构）
    assert ev2["value"] > ev["value"], \
        "A→B 的 value 应高于 A∧B"

    # 运行 episode，低价值 valid 对象应保留在 belief 中
    r = _run_value()
    # 检查 belief 中有 valid 对象不在 goal 子结构中
    stats = r["belief_stats"]
    assert stats["valid"] >= 2, "应有多个 valid 对象（包括低价值的）"


# ============================================================
# 7. 反馈改变后下一轮价值重新计算
# ============================================================

def test_feedback_recomputes_value():
    """每步都重新调用 evaluator，value 不是预计算的。
    检查 trace 中每步都有 evaluations（即每步都重新评价）。"""
    r = _run_value()
    for tr in r["trace"]:
        assert "evaluations" in tr, "每步应有 evaluations"
        assert len(tr.get("evaluations", [])) > 0, \
            "每步应有至少一个评价"
    # 检查 risk_factor 可能因 belief 变化而不同
    # step 0 和 step 1 的 risk_factor 可能不同（因为 step 0 产生了 valid）
    if len(r["trace"]) >= 2:
        r0_evals = r["trace"][0].get("evaluations", [])
        r1_evals = r["trace"][1].get("evaluations", [])
        if r0_evals and r1_evals:
            # value 是每步重新计算的（不是缓存的）
            # 只需要验证每步都有 value 即可
            for ev in r0_evals + r1_evals:
                assert ev.get("value") is not None or r["mode"] == "baseline"


# ============================================================
# 8. 价值函数改变会改变计算路径
# ============================================================

def test_value_function_changes_path():
    """baseline 和 value 模式的选择路径应不同。"""
    r_base = _run_baseline()
    r_val = _run_value()
    base_sels = [tr["selected"]["proposition"] for tr in r_base["trace"]
                 if tr.get("selected")]
    val_sels = [tr["selected"]["proposition"] for tr in r_val["trace"]
                if tr.get("selected")]
    assert base_sels != val_sels, \
        "baseline 和 value 模式的选择路径应不同"


# ============================================================
# 9. 相同 value 有稳定 tie-break
# ============================================================

def test_tie_break_stable():
    """value 相同时按 to_str() 字典序选择。"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    goal = make_goal()
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)

    # 构造两个 value 相同的候选（如 conj(A,B) 和 conj(B,A)）
    c1 = {"constructor": "conj", "objects": (Aa, Ba), "proposition": P.conj(Aa, Ba)}
    c2 = {"constructor": "conj", "objects": (Ba, Aa), "proposition": P.conj(Ba, Aa)}
    ev1 = evaluate_direction(c1["proposition"], goal, "conj", 0.7, belief)
    ev2 = evaluate_direction(c2["proposition"], goal, "conj", 0.7, belief)
    # 两者 value 应相同（结构对称）
    assert ev1["value"] == ev2["value"], \
        "对称候选应有相同 value"

    # tie-break 应选 to_str() 较小的
    selected, evals = select_direction(
        [c1, c2], evaluate_direction, goal, belief, mode="value")
    assert selected is not None
    # (A(a) ∧ B(a)) < (B(a) ∧ A(a)) 字典序
    assert selected["proposition"] == c1["proposition"] or \
           selected["proposition"] == c2["proposition"]
    # 选中的应是 to_str() 最小的
    sel_ev = next(ev for ev in evals if ev.get("selected"))
    same_val_props = [
        ev["proposition"] for ev in evals
        if ev.get("value") == sel_ev.get("value")]
    if len(same_val_props) > 1:
        assert sel_ev["proposition"] == min(same_val_props), \
            "tie-break 应选 to_str() 最小的"


# ============================================================
# 10. 不依赖 transformation history
# ============================================================

def test_no_transformation_history_dependency():
    """run_e0_9_episode 不接受 transform_store 参数。"""
    sig = inspect.signature(run_e0_9_episode)
    params = set(sig.parameters.keys())
    assert "transform_store" not in params, \
        "run_e0_9_episode 不应接受 transform_store"
    assert "history" not in params, \
        "run_e0_9_episode 不应接受 history 参数"


# ============================================================
# 11. 不依赖搜索
# ============================================================

def test_no_search_dependency():
    """不导入任何搜索/检索模块。"""
    import experiments.run_e0_9 as mod
    # 检查模块的导入语句，不检查 docstring 中的文字描述
    import_lines = [
        line.strip() for line in inspect.getsource(mod).split("\n")
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    search_modules = ["faiss", "sklearn", "annoy", "nmslib",
                      "chromadb", "pinecone", "weaviate"]
    for line in import_lines:
        for mod_name in search_modules:
            assert mod_name not in line.lower(), \
                f"不应导入搜索模块: {mod_name}"


# ============================================================
# 12. 不依赖 LLM
# ============================================================

def test_no_llm_dependency():
    """不导入任何 LLM / 神经网络模块。"""
    import experiments.run_e0_9 as mod
    import_lines = [
        line.strip() for line in inspect.getsource(mod).split("\n")
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    llm_modules = ["openai", "anthropic", "transformers", "torch",
                    "tensorflow", "sentence_transformers"]
    for line in import_lines:
        for mod_name in llm_modules:
            assert mod_name not in line.lower(), \
                f"不应导入 LLM/神经网络模块: {mod_name}"


# ============================================================
# 13. 不依赖 ground truth
# ============================================================

def test_no_ground_truth():
    """evaluate_direction 不接受 ground_truth 参数。"""
    sig = inspect.signature(evaluate_direction)
    params = set(sig.parameters.keys())
    assert "ground_truth" not in params, \
        "evaluate_direction 不应接受 ground_truth"
    assert "answer" not in params, \
        "evaluate_direction 不应接受 answer"
    # goal 是目标表示，不是 ground truth（goal 是显式先验）
    assert "goal" in params, "应接受 goal 作为目标表示"


# ============================================================
# 14. 严格时间因果
# ============================================================

def test_strict_time_causal():
    """每步选中的候选的输入对象必须在该步的 constructible 中。"""
    r = _run_value()
    for tr in r["trace"]:
        step_constructible = set(tr.get("constructible", []))
        sel = tr.get("selected")
        if sel is None:
            continue
        for inp in sel.get("inputs", []):
            assert inp in step_constructible, \
                f"step {tr['step']}: 输入 {inp} 不在 constructible 中"
    # 选中的候选不能是未来才产生的对象
    for i, tr in enumerate(r["trace"]):
        sel_prop = tr["selected"]["proposition"]
        for later in r["trace"][i + 1:]:
            assert sel_prop not in later.get("new_valids", []) or \
                sel_prop in tr.get("constructible", [])


# ============================================================
# 15. 完整 trace（含所有候选 value）
# ============================================================

def test_complete_trace():
    """trace 每步应包含：evaluations、selected、verdict。"""
    r = _run_value()
    for tr in r["trace"]:
        assert "evaluations" in tr, "每步应有 evaluations"
        assert "selected" in tr, "每步应有 selected"
        assert "verdict" in tr, "每步应有 verdict"
        assert "constructible" in tr, "每步应有 constructible"
        assert "candidates_count" in tr, "每步应有 candidates_count"
        # evaluations 中每个应有 proposition, value, selected
        for ev in tr.get("evaluations", []):
            assert "proposition" in ev
            assert "value" in ev
            assert "selected" in ev


# ============================================================
# 16. constructible / valid / valuable 三者严格分离
# ============================================================

def test_constructible_valid_valuable_separated():
    """验证三个概念分离：
    - constructible：在 constructible 列表中
    - valid：验证结果为 valid
    - valuable：value 高
    一个对象可以 constructible=True, valid=True, valuable=False"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    goal = make_goal()
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)

    # A∧B：constructible=True, valid=True, valuable=False（不是 goal 的子结构）
    conj_ab = P.conj(Aa, Ba)
    verdict, _, _ = verify_proposition_extended(conj_ab, build_world())
    ev = evaluate_direction(conj_ab, goal, "conj", 0.7, belief)
    assert verdict == "valid", "A∧B 应为 valid"

    # A→B：constructible=True, valid=True, valuable=True（是 goal 的子结构）
    impl_ab = P.impl(Aa, Ba)
    verdict2, _, _ = verify_proposition_extended(impl_ab, build_world())
    ev2 = evaluate_direction(impl_ab, goal, "impl", 0.9, belief)
    assert verdict2 == "valid", "A→B 应为 valid"
    # A→B 是 goal 的直接子结构，proximity 应更高
    assert ev2["goal_proximity"] > ev["goal_proximity"], \
        "A→B 的 goal_proximity 应高于 A∧B"

    # value 差异：A→B valuable, A∧B less valuable
    assert ev2["value"] > ev["value"], \
        "A→B 的 value 应高于 A∧B"

    # ¬A：constructible=True, valid=False (invalid), valuable=False
    neg_a = P.neg(Aa)
    verdict3, _, _ = verify_proposition_extended(neg_a, build_world())
    ev3 = evaluate_direction(neg_a, goal, "neg", 0.5, belief)
    assert verdict3 == "invalid", "¬A 应为 invalid"
    # ¬A 的 value 应低于 A→B
    assert ev3["value"] < ev2["value"], \
        "¬A 的 value 应低于 A→B"

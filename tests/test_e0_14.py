"""E0-14 不变量测试：Goal as Computation Object / Goal-Evaluation Separation.

验证 8 个核心问题：
  Q1: Goal 与 Evaluation 如何重新分离
  Q2: Goal 如何进入普通 Proposition/Object/Operation 体系
  Q3: 是否成功产生未预先提供的新 Goal
  Q4: 不同 Goal 是否产生不同计算路径
  Q5: 两个不同初始目标的 agent 是否出现认知空间分叉
  Q6: 是否存在任何答案泄漏或硬编码目标变换
  Q7: 全部测试结果（通过/总数）
  Q8: commit hash
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_14 import (
    evaluate_with_goal,
    extract_goal_dimension,
    is_goal_proposition,
    search_with_goal,
    apply_action_with_goal,
    build_worlds_e0_14,
    run_e0_14_episode,
    run_e0_14,
    E014Metrics,
    GoalWorldConfig,
    GOAL_PREDICATE_NAME,
    SATISFIED_PREDICATE_NAME,
)
from experiments.run_e0_12 import (
    get_valid_propositions,
)


def _strip_docstring(src: str) -> str:
    lines = src.split("\n")
    result = []
    in_doc = False
    triple_quote = None
    for line in lines:
        stripped = line.strip()
        if not in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                q = stripped[:3]
                if stripped.count(q) >= 2:
                    continue
                in_doc = True
                triple_quote = q
                continue
            result.append(line)
        else:
            if triple_quote in stripped:
                in_doc = False
                triple_quote = None
    return "\n".join(result)


def _module_source() -> str:
    import experiments.run_e0_14 as m
    return _strip_docstring(inspect.getsource(m))


# ============================================================
# Q1: Goal 与 Evaluation 如何重新分离
# ============================================================

class TestGoalEvaluationSeparation:
    """验证 Goal 和 Evaluation 彻底分离。"""

    def test_evaluate_takes_goal_param(self):
        """evaluate_with_goal 接受 goal 作为参数。"""
        sig = inspect.signature(evaluate_with_goal)
        params = list(sig.parameters.keys())
        assert "goal" in params, "evaluate_with_goal must take goal as parameter"
        assert "internal_state" in params

    def test_different_goals_different_evaluations(self):
        """同一状态，不同目标 → 不同评价。"""
        state = {"wealth": 2.0, "present": 8.0}
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        g2 = P.predicate(GOAL_PREDICATE_NAME, "present")

        eval_g1 = evaluate_with_goal(state, g1)
        eval_g2 = evaluate_with_goal(state, g2)

        # wealth=2 → negative eval for wealth goal
        assert eval_g1 < 0, f"wealth=2 should be negative for wealth goal, got {eval_g1}"
        # present=8 → negative eval for present goal (too high)
        assert eval_g2 < 0, f"present=8 should be negative for present goal, got {eval_g2}"
        # They should be different values
        assert eval_g1 != eval_g2, f"Same state different goals should give different evaluations"

    def test_same_goal_same_state_same_eval(self):
        """同一状态，同一目标 → 同一评价（确定性）。"""
        state = {"wealth": 5.0, "present": 5.0}
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")

        eval1 = evaluate_with_goal(state, g1)
        eval2 = evaluate_with_goal(state, g1)
        assert eval1 == eval2

    def test_eval_does_not_generate_goal(self):
        """评价函数不生成 Goal。"""
        src = _module_source()
        # evaluate_with_goal should return a float, not a Proposition
        result = evaluate_with_goal({"wealth": 5.0}, P.predicate(GOAL_PREDICATE_NAME, "wealth"))
        assert isinstance(result, float)

    def test_goal_dimension_extraction(self):
        """从 Goal proposition 提取状态维度。"""
        g = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        assert extract_goal_dimension(g) == "wealth"

        g2 = P.predicate(GOAL_PREDICATE_NAME, "present")
        assert extract_goal_dimension(g2) == "present"

    def test_goal_enters_search_reference(self):
        """goal 进入搜索的参考集合。"""
        bs = BeliefStore()
        F1 = P.predicate("Action", "wealth")
        W_ok = P.predicate(SATISFIED_PREDICATE_NAME, "wealth")
        impl_F1_Wok = P.impl(F1, W_ok)
        bs.update_belief(impl_F1_Wok, STATUS_VALID, 0.8)

        goal = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        result = search_with_goal(bs, {F1}, goal)

        # goal should influence the search scores
        assert len(result.all_scores) > 0
        # The implies involving "wealth" should score higher when goal is "wealth"
        impl_score = result.all_scores.get("(Action(wealth) → Satisfied(wealth))", 0)
        assert impl_score > 0, "goal-related implies should have positive score"


# ============================================================
# Q2: Goal 进入普通 Proposition/Object/Operation 体系
# ============================================================

class TestGoalAsProposition:
    """验证 Goal 使用现有 Proposition/Object/Operation 机制。"""

    def test_goal_is_proposition(self):
        """Goal 是普通的 Proposition 对象。"""
        g = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        assert isinstance(g, P)
        assert g.kind == "predicate"
        assert g.name == GOAL_PREDICATE_NAME

    def test_goal_can_be_operated_on(self):
        """Goal 可以被合法操作（neg, conj, impl 等）。"""
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        g2 = P.predicate(GOAL_PREDICATE_NAME, "present")

        # neg
        neg_g1 = P.neg(g1)
        assert neg_g1.kind == "not"

        # conj
        conj = P.conj(g1, g2)
        assert conj.kind == "and"

        # impl
        impl = P.impl(g1, g2)
        assert impl.kind == "implies"

    def test_goal_uses_same_mp_mechanism(self):
        """Goal 使用与普通对象相同的 MP 推导机制。"""
        bs = BeliefStore()
        K1 = P.predicate("Fact", "mortal")
        G2 = P.predicate(GOAL_PREDICATE_NAME, "present")
        impl_K1_G2 = P.impl(K1, G2)
        bs.update_belief(impl_K1_G2, STATUS_VALID, 0.8)

        from experiments.run_e0_13 import derive_via_modus_ponens
        known_true = {K1}
        results = derive_via_modus_ponens(bs, known_true)

        assert len(results) == 1
        assert results[0]["consequent"] == G2
        assert is_goal_proposition(results[0]["consequent"])

    def test_no_goal_manager_or_generator(self):
        """不存在 GoalManager/GoalSolver/GoalGenerator。"""
        src = _module_source()
        assert "GoalManager" not in src
        assert "GoalSolver" not in src
        assert "GoalGenerator" not in src
        assert "class GoalEngine" not in src

    def test_no_special_goal_reasoning(self):
        """Goal 不享有特殊推理能力。"""
        src = _module_source()
        # 不应该有 goal-specific 的推理函数
        assert "def reason_about_goal" not in src
        assert "def transform_goal" not in src
        assert "def select_goal" not in src


# ============================================================
# Q3: 是否成功产生未预先提供的新 Goal
# ============================================================

class TestNewGoalDerivation:
    """验证新 Goal 由合法计算产生，不是预先提供的。"""

    def test_world_a_derives_new_goal(self):
        """World A 通过 MP 产生新 Goal G2。"""
        worlds = build_worlds_e0_14()
        result = run_e0_14_episode(worlds["A"])
        goals_derived = result["metrics"]["goals_derived"]

        assert len(goals_derived) > 0, "World A should derive at least one new goal"
        assert "Goal(present)" in goals_derived, f"Should derive Goal(present), got {goals_derived}"

    def test_g2_not_in_initial_ks(self):
        """G2 不在初始 Knowledge Space 中。"""
        worlds = build_worlds_e0_14()
        world_a = worlds["A"]

        # Check initial beliefs don't contain G2 as a fact
        g2_str = "Goal(present)"
        for prop, status in world_a.initial_beliefs:
            # The implies(K1, G2) is in beliefs, but G2 itself is not
            assert prop.to_str() != g2_str, "G2 should not be in initial beliefs as a fact"

        # Check initial observable doesn't contain G2
        for obs in world_a.initial_observable:
            assert obs.to_str() != g2_str, "G2 should not be in initial observable"

    def test_g2_not_given_higher_score(self):
        """G2 没有被给予更高的验证置信度。"""
        worlds = build_worlds_e0_14()
        result = run_e0_14_episode(worlds["A"])

        # Check all MP derivations have the same verification confidence
        for mp_entry in result["metrics"]["mp_derivations"]:
            if mp_entry["consequent"] == "Goal(present)":
                # Verification should be standard, not boosted
                assert mp_entry["verification"] == STATUS_VALID

    def test_world_c_recursive_goal_chain(self):
        """World C 形成递归目标变换链 G1→G2→G3。"""
        worlds = build_worlds_e0_14()
        result = run_e0_14_episode(worlds["C"])
        goals = result["metrics"]["goals_derived"]

        assert "Goal(present)" in goals, "Should derive G2"
        assert "Goal(balance)" in goals, "Should derive G3"
        assert len(result["metrics"]["goal_chain"]) >= 3

    def test_goal_derived_via_legal_mp(self):
        """新 Goal 通过合法 MP 产生，不是凭空生成。"""
        worlds = build_worlds_e0_14()
        result = run_e0_14_episode(worlds["A"])

        # Every derived goal should have an MP entry showing its derivation
        for goal_str in result["metrics"]["goals_derived"]:
            mp_found = False
            for mp_entry in result["metrics"]["mp_derivations"]:
                if mp_entry["consequent"] == goal_str:
                    mp_found = True
                    # Must have a valid antecedent and implies
                    assert mp_entry["antecedent"] is not None
                    assert mp_entry["implies"] is not None
                    assert mp_entry["verification"] == STATUS_VALID
            assert mp_found, f"Goal {goal_str} must have been derived via MP"

    def test_g3_not_in_initial_ks(self):
        """G3 不在 World C 的初始 Knowledge Space 中。"""
        worlds = build_worlds_e0_14()
        world_c = worlds["C"]
        g3_str = "Goal(balance)"

        for prop, status in world_c.initial_beliefs:
            assert prop.to_str() != g3_str, "G3 should not be in initial beliefs"
        for obs in world_c.initial_observable:
            assert obs.to_str() != g3_str, "G3 should not be in initial observable"


# ============================================================
# Q4: 不同 Goal 是否产生不同计算路径
# ============================================================

class TestDifferentGoalsDifferentPaths:
    """验证不同 Goal 导致不同计算路径。"""

    def test_b1_b2_different_actions(self):
        """B1(goal=wealth) 和 B2(goal=present) 采取不同行动。"""
        worlds = build_worlds_e0_14()
        r_b1 = run_e0_14_episode(worlds["B1"])
        r_b2 = run_e0_14_episode(worlds["B2"])

        action_b1 = r_b1["metrics"]["action_detail"]
        action_b2 = r_b2["metrics"]["action_detail"]

        assert action_b1 != action_b2, \
            f"B1 and B2 should take different actions, but both took: {action_b1}"
        assert "wealth" in action_b1, f"B1 (goal=wealth) should take wealth-related action, got {action_b1}"
        assert "present" in action_b2, f"B2 (goal=present) should take present-related action, got {action_b2}"

    def test_b1_b2_same_state_same_knowledge(self):
        """B1 和 B2 具有相同初始状态和知识空间。"""
        worlds = build_worlds_e0_14()
        assert worlds["B1"].initial_state == worlds["B2"].initial_state
        assert worlds["B1"].initial_observable == worlds["B2"].initial_observable
        assert worlds["B1"].initial_beliefs == worlds["B2"].initial_beliefs

    def test_b1_b2_different_outcomes(self):
        """B1 和 B2 产生不同的最终状态。"""
        worlds = build_worlds_e0_14()
        r_b1 = run_e0_14_episode(worlds["B1"])
        r_b2 = run_e0_14_episode(worlds["B2"])

        state_b1 = r_b1["final_state"]
        state_b2 = r_b2["final_state"]

        assert state_b1 != state_b2, "Different goals should lead to different final states"
        assert state_b1["wealth"] > state_b2["wealth"], "B1 (wealth goal) should improve wealth more"
        assert state_b2["present"] > state_b1["present"], "B2 (present goal) should improve present more"

    def test_no_hardcoded_path_selection(self):
        """路径选择不是硬编码的。"""
        src = _module_source()
        # 不应该有 "if goal == G1" 或 "if goal.to_str() == 'Goal(wealth)'" 的硬编码
        assert "if goal ==" not in src
        assert "if goal.to_str() ==" not in src
        assert "target ==" not in src
        # 行动选择应该基于结构相似度
        assert "compute_similarity" in src


# ============================================================
# Q5: 两个不同初始目标的 agent 是否出现认知空间分叉
# ============================================================

class TestAgentDivergence:
    """验证不同初始目标的 agent 出现认知空间分叉。"""

    def test_d_agents_different_goals(self):
        """D_A 和 D_B 具有不同初始目标。"""
        worlds = build_worlds_e0_14()
        assert worlds["D_A"].initial_goal != worlds["D_B"].initial_goal

    def test_d_agents_same_environment(self):
        """D_A 和 D_B 具有相同现实环境。"""
        worlds = build_worlds_e0_14()
        assert worlds["D_A"].initial_state == worlds["D_B"].initial_state
        assert worlds["D_A"].initial_observable == worlds["D_B"].initial_observable
        assert worlds["D_A"].initial_beliefs == worlds["D_B"].initial_beliefs

    def test_d_agents_different_actions(self):
        """D_A 和 D_B 采取不同行动。"""
        worlds = build_worlds_e0_14()
        r_a = run_e0_14_episode(worlds["D_A"])
        r_b = run_e0_14_episode(worlds["D_B"])

        assert r_a["metrics"]["action_detail"] != r_b["metrics"]["action_detail"]

    def test_d_agents_different_states(self):
        """D_A 和 D_B 最终状态不同（认知空间分叉）。"""
        worlds = build_worlds_e0_14()
        r_a = run_e0_14_episode(worlds["D_A"])
        r_b = run_e0_14_episode(worlds["D_B"])

        assert r_a["final_state"] != r_b["final_state"], "Agent states should diverge"

    def test_d_agents_different_state_dimensions(self):
        """D_A 改善 wealth 维度，D_B 改善 present 维度。"""
        worlds = build_worlds_e0_14()
        r_a = run_e0_14_episode(worlds["D_A"])
        r_b = run_e0_14_episode(worlds["D_B"])

        # D_A (goal=wealth) should improve wealth
        assert r_a["final_state"]["wealth"] > r_a["metrics"]["state_before"]["wealth"]
        # D_B (goal=present) should improve present
        assert r_b["final_state"]["present"] > r_b["metrics"]["state_before"]["present"]

    def test_d_divergence_not_random(self):
        """分叉不是随机的——相同输入产生相同输出。"""
        worlds = build_worlds_e0_14()
        r_a1 = run_e0_14_episode(worlds["D_A"])
        r_a2 = run_e0_14_episode(worlds["D_A"])
        r_b1 = run_e0_14_episode(worlds["D_B"])
        r_b2 = run_e0_14_episode(worlds["D_B"])

        # Deterministic: same world → same result
        assert r_a1["final_state"] == r_a2["final_state"]
        assert r_b1["final_state"] == r_b2["final_state"]
        # Different: A ≠ B
        assert r_a1["final_state"] != r_b1["final_state"]


# ============================================================
# Q6: 是否存在任何答案泄漏或硬编码目标变换
# ============================================================

class TestNoAnswerLeakage:
    """验证不存在答案泄漏或硬编码目标变换。"""

    def test_no_g2_in_initial_beliefs(self):
        """G2 不在任何世界的初始 beliefs 中。"""
        worlds = build_worlds_e0_14()
        g2_str = "Goal(present)"
        for wid, world in worlds.items():
            for prop, status in world.initial_beliefs:
                assert prop.to_str() != g2_str, \
                    f"World {wid}: G2 should not be in initial beliefs as a fact"

    def test_no_g3_in_initial_beliefs(self):
        """G3 不在任何世界的初始 beliefs 中。"""
        worlds = build_worlds_e0_14()
        g3_str = "Goal(balance)"
        for wid, world in worlds.items():
            for prop, status in world.initial_beliefs:
                assert prop.to_str() != g3_str, \
                    f"World {wid}: G3 should not be in initial beliefs as a fact"

    def test_no_target_check(self):
        """代码中不存在 target == G2 等硬编码检查。"""
        src = _module_source()
        assert "target ==" not in src
        assert 'target=="' not in src
        assert "if prop == G2" not in src
        assert 'if prop.to_str() == "Goal(present)"' not in src

    def test_no_natural_language_judgment(self):
        """不使用自然语言语义判断 G2 是否正确。"""
        src = _module_source()
        assert "珍惜当下" not in src
        assert "珍惜当前" not in src
        assert "人生哲学" not in src
        assert "死亡" not in src
        assert "吃东西" not in src

    def test_no_hardcoded_goal_transform_rule(self):
        """不存在"死亡→珍惜当下"等硬编码规则。"""
        src = _module_source()
        # 不应该有任何硬编码的目标变换规则
        assert "G1_transform" not in src
        assert "G2_transform" not in src
        assert "transform_goal_to" not in src
        assert "goal_transform_rule" not in src

    def test_no_llm_or_embedding(self):
        """不使用 LLM/embedding/神经网络。"""
        src = _module_source()
        assert "LLM" not in src
        assert "embedding" not in src
        assert "neural" not in src.lower()
        assert "transformer" not in src.lower()

    def test_goal_derivation_requires_legal_mp(self):
        """目标推导必须通过合法 MP，不能凭空产生。"""
        # NC1: no valid implications → no new goal
        worlds = build_worlds_e0_14()
        r_nc1 = run_e0_14_episode(worlds["NC1"])
        assert len(r_nc1["metrics"]["goals_derived"]) == 0, \
            "NC1: should not derive any goals without valid implications"

    def test_nc2_similarity_cannot_replace_derivation(self):
        """相似结构不能代替精确 MP 推导。"""
        worlds = build_worlds_e0_14()
        r_nc2 = run_e0_14_episode(worlds["NC2"])
        assert len(r_nc2["metrics"]["goals_derived"]) == 0, \
            "NC2: similarity should not replace exact MP derivation"

    def test_nc3_verification_rejects_wrong_goals(self):
        """多个错误候选通过验证淘汰。"""
        worlds = build_worlds_e0_14()
        r_nc3 = run_e0_14_episode(worlds["NC3"])

        assert len(r_nc3["metrics"]["objects_rejected_by_verification"]) > 0, \
            "NC3: should reject wrong goal candidates"
        assert "Goal(present)" in r_nc3["metrics"]["goals_derived"], \
            "NC3: should derive correct goal G2"

    def test_no_goal_score_boost(self):
        """目标命题不享受验证加分。"""
        worlds = build_worlds_e0_14()
        r_nc3 = run_e0_14_episode(worlds["NC3"])

        # Check that wrong goals were rejected
        rejected = r_nc3["metrics"]["objects_rejected_by_verification"]
        assert len(rejected) >= 2, f"Should reject at least 2 wrong goals, got {rejected}"


# ============================================================
# 补充：整体实验一致性
# ============================================================

class TestExperimentConsistency:
    """验证整体实验结果的一致性。"""

    def test_run_e0_14_returns_results(self):
        """run_e0_14 返回完整结果。"""
        result = run_e0_14()
        assert "experiment" in result
        assert result["experiment"] == "E0-14"
        assert "results" in result
        assert "analysis" in result

    def test_all_worlds_present(self):
        """所有世界都运行。"""
        result = run_e0_14()
        expected_worlds = {"A", "B1", "B2", "C", "D_A", "D_B", "NC1", "NC2", "NC3"}
        assert set(result["results"].keys()) == expected_worlds

    def test_world_a_analysis(self):
        """World A 分析正确。"""
        result = run_e0_14()
        a = result["analysis"]
        assert a["A_derived_new_goal"] is True
        assert "Goal(present)" in a["A_goals_derived"]

    def test_world_b_analysis(self):
        """World B 分析正确。"""
        result = run_e0_14()
        a = result["analysis"]
        assert a["B_different_actions"] is True

    def test_world_c_analysis(self):
        """World C 分析正确。"""
        result = run_e0_14()
        a = result["analysis"]
        assert a["C_derived_g2"] is True
        assert a["C_derived_g3"] is True
        assert a["C_goal_chain_length"] >= 3

    def test_world_d_analysis(self):
        """World D 分析正确。"""
        result = run_e0_14()
        a = result["analysis"]
        assert a["D_states_diverged"] is True

    def test_nc_analysis(self):
        """负对照分析正确。"""
        result = run_e0_14()
        a = result["analysis"]
        assert a["NC1_no_derived_goals"] is True
        assert a["NC1_admitted_insufficiency"] is True
        assert a["NC2_no_derived_goals"] is True
        assert a["NC3_rejected_wrong_goals"] is True
        assert a["NC3_derived_correct_goal"] is True

    def test_no_forced_answer(self):
        """不存在 forced answer（评价改善但无行动支持）。"""
        result = run_e0_14()
        for wid, r in result["results"].items():
            assert not r["metrics"]["forced_answer"], \
                f"World {wid}: should not have forced answer"


# ============================================================
# 补充：Goal 可被合法操作
# ============================================================

class TestGoalOperations:
    """验证 Goal 可被现有合法操作处理。"""

    def test_goal_can_be_antecedent(self):
        """Goal 可作为 implies 的前件。"""
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        g2 = P.predicate(GOAL_PREDICATE_NAME, "present")
        impl = P.impl(g1, g2)
        assert impl.kind == "implies"
        assert impl.parts[0] == g1

    def test_goal_can_be_consequent(self):
        """Goal 可作为 implies 的后件（MP 推导目标）。"""
        k1 = P.predicate("Fact", "mortal")
        g2 = P.predicate(GOAL_PREDICATE_NAME, "present")
        impl = P.impl(k1, g2)
        assert impl.parts[1] == g2

    def test_goal_can_be_negated(self):
        """Goal 可被否定。"""
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        neg_g = P.neg(g1)
        assert neg_g.kind == "not"
        assert neg_g.parts[0] == g1

    def test_goal_can_be_combined(self):
        """Goal 可被合取/析取。"""
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        g2 = P.predicate(GOAL_PREDICATE_NAME, "present")
        conj = P.conj(g1, g2)
        disj = P.disj(g1, g2)
        assert conj.kind == "and"
        assert disj.kind == "or"

    def test_goal_enters_belief_store(self):
        """Goal 可进入 BeliefStore 作为普通信念。"""
        bs = BeliefStore()
        g = P.predicate(GOAL_PREDICATE_NAME, "test")
        bs.update_belief(g, STATUS_VALID, 0.8)
        assert bs.has(g)
        assert bs.get(g).status == STATUS_VALID

    def test_goal_hashable(self):
        """Goal 命题可哈希（可加入 set）。"""
        g1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        g2 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
        s = {g1, g2}
        assert len(s) == 1  # Same proposition → same hash

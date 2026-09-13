"""E0-12 不变量测试：Search-Based Computation / Cognitive Space Expansion。

验证理论修正：
  - 一切都是计算，不存在 ProblemGenerator
  - 计算缺口由现实+状态+认知空间+评价共同产生，不硬编码问题类型
  - 搜索是核心：直接匹配 / 最近节点 + 推导
  - 系统不能提出超出自身认知空间的问题
  - 三个世界：A 直接匹配 / B 多步 / C 无足够节点承认不足
  - 指标：认知空间扩展、搜索访问节点、推导步数、新节点复用等
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_12 import (
    GapSignal,
    evaluate_state,
    h_to_proposition,
    collect_terms,
    collect_predicates,
    compute_similarity,
    search_knowledge_space,
    get_constructible,
    get_valid_propositions,
    derive_candidates,
    derive_actions,
    apply_action,
    build_worlds,
    run_e0_12_episode,
    run_e0_12,
    H_LOW, H_MID, H_HIGH,
)


# ============================================================
# 辅助：获取源码并移除 docstring
# ============================================================

def _strip_docstring(src: str) -> str:
    """移除模块/函数 docstring，避免自然语言描述干扰源码检查。"""
    lines = src.split("\n")
    result = []
    in_doc = False
    triple_quote = None
    for line in lines:
        stripped = line.strip()
        if not in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                q = stripped[:3]
                # 单行 docstring
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
            # 跳过 docstring 内容行
    return "\n".join(result)


def _module_source() -> str:
    import experiments.run_e0_12 as m
    return _strip_docstring(inspect.getsource(m))


# ============================================================
# Q1: 不存在 ProblemGenerator / QuestionGenerator / GoalGenerator
# ============================================================

class TestNoProblemGenerator:
    def test_no_problem_generator_class(self):
        """代码中不存在 ProblemGenerator 类。"""
        src = _module_source()
        assert "ProblemGenerator" not in src
        assert "QuestionGenerator" not in src
        assert "GoalGenerator" not in src

    def test_no_problem_solver_class(self):
        """代码中不存在 ProblemSolver 类。"""
        src = _module_source()
        assert "ProblemSolver" not in src

    def test_no_problem_to_problem_reasoning(self):
        """不存在 Problem→Problem 的推理方法。"""
        src = _module_source()
        assert "problem_to_problem" not in src
        assert "generate_problem" not in src
        assert "construct_problem" not in src

    def test_no_planner_or_llm(self):
        """不存在 planner / LLM / embedding 等高级模块。"""
        src = _module_source()
        assert "planner" not in src.lower()
        assert "llm" not in src.lower()
        assert "embedding" not in src.lower()
        assert "neural" not in src.lower()


# ============================================================
# Q2: GapSignal 不包含答案/目标/问题类型
# ============================================================

class TestGapSignal:
    def test_gap_signal_fields(self):
        """GapSignal 只包含缺口数据，不含 answer/target/goal。"""
        from dataclasses import fields
        fs = {f.name for f in fields(GapSignal)}
        assert "answer" not in fs
        assert "target" not in fs
        assert "goal" not in fs
        assert "required_relation" not in fs
        assert "required_operator" not in fs
        assert "problem_type" not in fs

    def test_gap_signal_produced_by_state_and_eval(self):
        """GapSignal 由现实+状态+认知空间+评价共同产生。"""
        gap = GapSignal(
            evaluation=-0.3,
            eval_delta=-0.1,
            observable=["F(x)"],
            knowledge_space_size=2,
            internal_state={"H": 1.0},
        )
        assert gap.exists() is True
        assert gap.evaluation < 0
        assert gap.knowledge_space_size == 2

    def test_no_gap_when_eval_positive(self):
        """评价为正时不存在缺口。"""
        gap = GapSignal(
            evaluation=0.5,
            eval_delta=0.2,
            observable=["F(x)"],
            knowledge_space_size=2,
            internal_state={"H": 5.0},
        )
        assert gap.exists() is False

    def test_gap_not_hardcoded_to_h(self):
        """GapSignal 不硬编码"遇到 H 异常就找影响 H 的对象"。"""
        gap_src = _strip_docstring(inspect.getsource(GapSignal))
        assert "find object" not in gap_src.lower()
        assert "affects" not in gap_src.lower()
        # 不直接引用 H 状态命题或 H 变量
        assert "H_LOW" not in gap_src
        assert "H_MID" not in gap_src
        assert "H_HIGH" not in gap_src
        assert "H(" not in gap_src


# ============================================================
# Q3: 搜索使用结构相似度，不泄漏答案
# ============================================================

class TestSearchNoAnswerLeakage:
    def test_similarity_uses_shared_terms_only(self):
        """compute_similarity 只使用共享项和谓词，不检查 H 或 implies。"""
        src = _strip_docstring(inspect.getsource(compute_similarity))
        assert "H" not in src
        assert "implies" not in src
        assert "shared_terms" in src
        assert "shared_preds" in src

    def test_similarity_does_not_prefer_implies(self):
        """相似度不偏好 implies 结构（无 kind 特权）。"""
        src = _strip_docstring(inspect.getsource(compute_similarity))
        assert "kind" not in src

    def test_search_finds_direct_match(self):
        """搜索能找到直接匹配（VALID implies(X,Y) 且 X 可观察）。"""
        F = P.predicate("F", "x")
        impl = P.impl(F, H_MID)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        result = search_knowledge_space(bs, {F})
        assert impl in result.direct_matches

    def test_search_no_direct_match_when_x_not_observable(self):
        """X 不可观察时不是直接匹配。"""
        F = P.predicate("F", "x")
        G = P.predicate("G", "x")
        impl = P.impl(G, H_MID)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        result = search_knowledge_space(bs, {F})
        assert result.direct_matches == []

    def test_nearest_node_based_on_similarity(self):
        """最近节点基于结构相似度选择。"""
        F = P.predicate("F", "x")
        P_obj = P.predicate("P", "x")
        impl_F = P.impl(F, P.predicate("Q", "x"))  # 与 F 共享 F
        impl_P = P.impl(P_obj, P.predicate("Q", "x"))  # 与 F 无共享
        bs = BeliefStore()
        bs.update_belief(impl_F, STATUS_VALID, 0.8)
        bs.update_belief(impl_P, STATUS_VALID, 0.8)
        result = search_knowledge_space(bs, {F})
        assert result.nearest_node == impl_F
        assert result.nearest_score > 0


# ============================================================
# Q4: 认知空间边界——不能构造超出认知空间的对象
# ============================================================

class TestCognitiveSpaceBoundary:
    def test_constructible_includes_observable_and_valid(self):
        """constructible = 可观察 ∪ VALID 命题及其子命题。"""
        F = P.predicate("F", "x")
        impl = P.impl(F, H_MID)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        constructible = get_constructible(bs, {F})
        constructible_strs = {p.to_str() for p in constructible}
        assert "F(x)" in constructible_strs
        assert "H(mid)" in constructible_strs
        assert str(impl) in constructible_strs

    def test_cannot_construct_unknown_objects(self):
        """认知空间中没有的对象不能被构造。"""
        F = P.predicate("F", "x")
        P_obj = P.predicate("P", "x")
        impl = P.impl(P_obj, P.predicate("Q", "x"))
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        constructible = get_constructible(bs, {F})
        constructible_strs = {p.to_str() for p in constructible}
        # H 不在认知空间中
        assert "H(mid)" not in constructible_strs
        assert "H(low)" not in constructible_strs

    def test_world_c_no_h_in_cognitive_space(self):
        """World C 初始认知空间不包含 H。"""
        worlds = build_worlds()
        w = worlds["C"]
        bs = BeliefStore()
        for p, s in w.initial_beliefs:
            bs.update_belief(p, s, 0.8)
        valid = get_valid_propositions(bs)
        valid_strs = {p.to_str() for p in valid}
        # 没有任何涉及 H 的命题
        assert not any("H" in s for s in valid_strs)


# ============================================================
# Q5: 行动来自已验证知识，不使用字符串匹配
# ============================================================

class TestActionFromVerifiedKnowledge:
    def test_derive_actions_from_valid_implies(self):
        """行动从 VALID implies(X,Y) 推导，X 必须可观察。"""
        F = P.predicate("F", "x")
        impl = P.impl(F, H_MID)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        actions = derive_actions(bs, {F})
        assert len(actions) == 1
        assert actions[0]["action_object"] == F
        assert actions[0]["expected_effect"] == H_MID
        assert actions[0]["source_proposition"] == impl

    def test_no_action_for_invalid_implies(self):
        """INVALID implies 不产生行动。"""
        F = P.predicate("F", "x")
        impl = P.impl(F, H_MID)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_INVALID, 0.8)
        actions = derive_actions(bs, {F})
        assert actions == []

    def test_no_string_matching_in_action(self):
        """行动代码不使用字符串匹配触发。"""
        src = _strip_docstring(inspect.getsource(apply_action))
        assert '"F("' not in src
        assert "'F('" not in src
        assert "in prop_str" not in src

    def test_action_effect_from_consequent(self):
        """行动效果来自 implies 的 consequent，不是硬编码。"""
        F = P.predicate("F", "x")
        G = P.predicate("G", "x")
        impl = P.impl(F, G)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        actions = derive_actions(bs, {F})
        assert actions[0]["expected_effect"] == G


# ============================================================
# Q6: 三个世界的行为
# ============================================================

class TestWorldA:
    def test_direct_match_found(self):
        """World A：直接匹配存在。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["A"])
        assert any(n > 0 for n in r["metrics"]["direct_matches_found"])

    def test_eval_improved(self):
        """World A：评价改善。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["A"])
        assert r["metrics"]["feedback_improved"]

    def test_single_step(self):
        """World A：单步完成（直接匹配）。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["A"])
        assert not r["metrics"]["multi_step"]


class TestWorldB:
    def test_multi_step(self):
        """World B：多步计算（链式行动）。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["B"])
        assert r["metrics"]["multi_step"]

    def test_eval_improved(self):
        """World B：评价改善。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["B"])
        assert r["metrics"]["feedback_improved"]

    def test_used_search(self):
        """World B：使用了搜索。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["B"])
        assert len(r["metrics"]["search_visited_nodes"]) > 0


class TestWorldC:
    def test_admitted_insufficiency(self):
        """World C：承认计算能力不足。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["C"])
        assert r["metrics"]["admitted_insufficiency"]
        assert r["metrics"]["stop_reason"] == "computation_insufficient"

    def test_eval_not_improved(self):
        """World C：评价未改善。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["C"])
        assert not r["metrics"]["feedback_improved"]

    def test_no_forced_answer(self):
        """World C：不强行得到答案。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["C"])
        assert not r["metrics"]["forced_answer"]

    def test_cognitive_space_boundary_respected(self):
        """World C：没有构造出涉及 H 的新 VALID 命题。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["C"])
        new_relations = r["metrics"]["new_relations"]
        new_objects = r["metrics"]["new_objects"]
        assert not any("H" in rel for rel in new_relations)
        assert not any("H" in obj for obj in new_objects)

    def test_no_useful_expansion(self):
        """World C：没有产生新的有用关系（implies）。"""
        worlds = build_worlds()
        r = run_e0_12_episode(worlds["C"])
        assert len(r["metrics"]["new_relations"]) == 0


# ============================================================
# Q7: 评价函数只评价状态
# ============================================================

class TestEvaluation:
    def test_evaluate_state_takes_only_state(self):
        """evaluate_state 只接收 state dict。"""
        sig = inspect.signature(evaluate_state)
        params = list(sig.parameters.keys())
        assert params == ["internal_state"]

    def test_low_h_negative(self):
        assert evaluate_state({"H": 1.0}) < 0

    def test_mid_h_positive(self):
        assert evaluate_state({"H": 5.0}) > 0

    def test_high_h_negative(self):
        assert evaluate_state({"H": 9.0}) < 0

    def test_eval_does_not_reference_problem(self):
        """评价函数不引用 Problem 或 Gap。"""
        src = _strip_docstring(inspect.getsource(evaluate_state))
        assert "Problem" not in src
        assert "Gap" not in src
        assert "gap" not in src


# ============================================================
# Q8: 完整实验运行
# ============================================================

class TestFullExperiment:
    def test_run_e0_12_returns_results(self):
        result = run_e0_12()
        assert result["experiment"] == "E0-12"
        assert "A" in result["results"]
        assert "B" in result["results"]
        assert "C" in result["results"]

    def test_analysis_all_pass(self):
        result = run_e0_12()
        a = result["analysis"]
        assert a["A_direct_match_found"]
        assert a["A_eval_improved"]
        assert a["B_multi_step"]
        assert a["B_eval_improved"]
        assert a["C_admitted_insufficiency"]
        assert a["C_eval_not_improved"]
        assert a["C_no_forced_answer"]
        assert a["cognitive_space_boundary_respected"]
        assert a["no_problem_generator"]
        assert a["search_is_core"]
        assert a["gap_not_hardcoded"]

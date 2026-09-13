"""E0-13 不变量测试：Proposition-Level Derivation via Modus Ponens.

验证：
  1. 不存在 ProblemGenerator 等模块
  2. derive_via_modus_ponens 只使用 known_true + VALID implies
  3. World B：A/B/H_MID 不在初始 KS 中，通过 MP 推导产生
  4. 新对象进入 KS 后被后续搜索利用
  5. 行动基于 derived object（不是环境直接给的）
  6. 三个负对照结果正确
  7. 无答案泄漏
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_13 import (
    derive_via_modus_ponens,
    search_with_inference,
    derive_actions_inference,
    build_worlds_e0_13,
    run_e0_13_episode,
    run_e0_13,
    E013Metrics,
)
from experiments.run_e0_12 import (
    H_LOW, H_MID, H_HIGH,
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
    import experiments.run_e0_13 as m
    return _strip_docstring(inspect.getsource(m))


# ============================================================
# Q1: 不存在 ProblemGenerator 等模块
# ============================================================

class TestNoProblemGenerator:
    def test_no_problem_generator(self):
        src = _module_source()
        assert "ProblemGenerator" not in src
        assert "QuestionGenerator" not in src
        assert "GoalGenerator" not in src
        assert "ProblemSolver" not in src

    def test_no_planner_or_llm(self):
        src = _module_source()
        assert "planner" not in src.lower()
        assert "llm" not in src.lower()
        assert "embedding" not in src.lower()
        assert "neural" not in src.lower()

    def test_no_problem_to_problem(self):
        src = _module_source()
        assert "problem_to_problem" not in src
        assert "generate_problem" not in src
        assert "construct_problem" not in src


# ============================================================
# Q2: derive_via_modus_ponens 只使用 known_true + VALID implies
# ============================================================

class TestModusPonens:
    def test_mp_requires_known_true_antecedent(self):
        """MP 要求前件是 known-true。"""
        F = P.predicate("F", "x")
        A = P.predicate("A", "x")
        impl = P.impl(F, A)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        # F 是 known-true → 可以推导 A
        derived = derive_via_modus_ponens(bs, {F})
        assert len(derived) == 1
        assert derived[0]["consequent"] == A

    def test_mp_no_derivation_when_antecedent_not_known(self):
        """前件不是 known-true 时不能推导。"""
        F = P.predicate("F", "x")
        C = P.predicate("C", "x")
        A = P.predicate("A", "x")
        impl = P.impl(C, A)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        # 只有 F 是 known-true，C 不是 → 不能推导 A
        derived = derive_via_modus_ponens(bs, {F})
        assert len(derived) == 0

    def test_mp_no_derivation_when_implies_not_valid(self):
        """implies 不是 VALID 时不能推导。"""
        F = P.predicate("F", "x")
        A = P.predicate("A", "x")
        impl = P.impl(F, A)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_INVALID, 0.8)
        derived = derive_via_modus_ponens(bs, {F})
        assert len(derived) == 0

    def test_mp_no_derivation_when_consequent_already_known(self):
        """consequent 已在 BeliefStore 时不重复推导。"""
        F = P.predicate("F", "x")
        A = P.predicate("A", "x")
        impl = P.impl(F, A)
        bs = BeliefStore()
        bs.update_belief(impl, STATUS_VALID, 0.8)
        bs.update_belief(A, STATUS_VALID, 0.8)  # A 已知
        derived = derive_via_modus_ponens(bs, {F})
        assert len(derived) == 0

    def test_mp_uses_valid_propositions_as_known_true(self):
        """VALID 命题也可以作为 known-true 前件。"""
        F = P.predicate("F", "x")
        A = P.predicate("A", "x")
        B = P.predicate("B", "x")
        impl_FA = P.impl(F, A)
        impl_AB = P.impl(A, B)
        bs = BeliefStore()
        bs.update_belief(impl_FA, STATUS_VALID, 0.8)
        bs.update_belief(impl_AB, STATUS_VALID, 0.8)
        bs.update_belief(A, STATUS_VALID, 0.8)  # A 是 VALID
        # known_true = {F, A, impl_FA, impl_AB}
        known_true = {F} | set(get_valid_propositions(bs))
        derived = derive_via_modus_ponens(bs, known_true)
        # A 已在 BS 中所以不推导 A
        # 但 B 可以从 A + impl_AB 推导
        consequents = [d["consequent"] for d in derived]
        assert B in consequents

    def test_mp_source_not_string_matching(self):
        """MP 代码不使用字符串匹配。"""
        src = _strip_docstring(inspect.getsource(derive_via_modus_ponens))
        assert "in prop_str" not in src
        assert '"F("' not in src

    def test_mp_no_predicate_privilege(self):
        """MP 不检查谓词名。"""
        src = _strip_docstring(inspect.getsource(derive_via_modus_ponens))
        # 不检查特定谓词名（H, A, B 等）
        assert '"H"' not in src
        assert '"A"' not in src
        assert '"B"' not in src


# ============================================================
# Q3: World B — 新对象不在初始 KS 中，通过 MP 推导产生
# ============================================================

class TestWorldBNewObjects:
    def test_a_not_in_initial_ks(self):
        """A 不在 World B 的初始 observable 或 beliefs 中。"""
        worlds = build_worlds_e0_13()
        w = worlds["B"]
        A = P.predicate("A", "x")
        # A 不在 observable
        assert A not in w.initial_observable
        # A 不在 initial_beliefs
        initial_props = [p for p, _ in w.initial_beliefs]
        assert A not in initial_props

    def test_b_not_in_initial_ks(self):
        """B 不在 World B 的初始 observable 或 beliefs 中。"""
        worlds = build_worlds_e0_13()
        w = worlds["B"]
        B = P.predicate("B", "x")
        assert B not in w.initial_observable
        initial_props = [p for p, _ in w.initial_beliefs]
        assert B not in initial_props

    def test_hmid_not_in_initial_ks(self):
        """H_MID 不在 World B 的初始 observable 或 beliefs 中。"""
        worlds = build_worlds_e0_13()
        w = worlds["B"]
        assert H_MID not in w.initial_observable
        initial_props = [p for p, _ in w.initial_beliefs]
        assert H_MID not in initial_props

    def test_objects_derived_via_mp(self):
        """World B 运行后，A/B/H_MID 通过 MP 进入 KS。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        m = r["metrics"]
        # 有 MP 推导
        assert m["derivation_chain_length"] > 0
        # A, B, H_MID 都通过 MP 进入 KS
        assert "A(x)" in m["objects_entered_ks_via_mp"]
        assert "B(x)" in m["objects_entered_ks_via_mp"]
        assert "H(mid)" in m["objects_entered_ks_via_mp"]

    def test_derivation_chain_correct_order(self):
        """推导链顺序正确：先 A，再 B，再 H_MID。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        m = r["metrics"]
        entered = m["objects_entered_ks_via_mp"]
        # A 在 B 之前进入
        assert entered.index("A(x)") < entered.index("B(x)")
        # B 在 H_MID 之前进入
        assert entered.index("B(x)") < entered.index("H(mid)")

    def test_ks_expanded(self):
        """World B 运行后认知空间扩展。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        m = r["metrics"]
        assert m["knowledge_space_size_after"] > m["knowledge_space_size_before"]

    def test_eval_improved(self):
        """World B 评价改善。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        assert r["metrics"]["feedback_improved"]

    def test_no_forced_answer(self):
        """World B 不伪造答案。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        assert not r["metrics"]["forced_answer"]


# ============================================================
# Q4: 新对象被后续搜索利用
# ============================================================

class TestNewObjectReuse:
    def test_action_based_on_derived_object(self):
        """World B 的行动基于 MP 推导的对象（B）。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        m = r["metrics"]
        assert m["action_executed"]
        assert m["action_based_on_derived"]

    def test_derived_object_used_in_subsequent_mp(self):
        """A 被 MP 推导后，被用于后续 MP 推导 B。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        m = r["metrics"]
        # 检查 trace 中有 MP 步骤使用了之前推导的对象
        mp_steps = [t for t in r["trace"] if t.get("mode") == "modus_ponens"]
        assert len(mp_steps) >= 2  # 至少两轮 MP
        # 第二轮 MP 的 antecedent 应包含 A（第一轮推导的对象）
        # 检查 mp_derivations 中有使用 A 作为 antecedent 的推导
        mp_uses_A = [d for d in m["mp_derivations"] if d["antecedent"] == "A(x)"]
        assert len(mp_uses_A) > 0

    def test_derived_object_used_in_subsequent_search(self):
        """推导出的 B 被后续搜索用作 direct match 的前件。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["B"])
        m = r["metrics"]
        # mp_derivations 中有使用 B 作为 antecedent 的推导
        mp_uses_B = [d for d in m["mp_derivations"] if d["antecedent"] == "B(x)"]
        assert len(mp_uses_B) > 0


# ============================================================
# Q5: 无答案泄漏
# ============================================================

class TestNoAnswerLeakage:
    def test_no_target_in_code(self):
        """代码中不存在 target / goal / answer 变量。"""
        src = _module_source()
        assert "target" not in src.lower()
        assert "goal" not in src.lower()
        # "answer" 只出现在 docstring 中，strip 后不应出现
        stripped = _strip_docstring(inspect.getsource(derive_via_modus_ponens))
        assert "answer" not in stripped.lower()

    def test_no_hardcoded_correct_object(self):
        """代码不硬编码正确中间对象。"""
        src = _module_source()
        # 不应出现 "if ... == A" 或 "if ... == B" 这种答案检查
        assert '== "A"' not in src
        assert '== "B"' not in src

    def test_a_not_in_search_target(self):
        """A 不作为搜索目标传入。"""
        worlds = build_worlds_e0_13()
        w = worlds["B"]
        A = P.predicate("A", "x")
        # A 不在 initial_observable（不会被作为搜索参考传入）
        assert A not in w.initial_observable

    def test_world_rules_facts_include_all(self):
        """world_rules 的 facts 包含所有对象（用于验证，不是答案）。

        facts 是外部现实，系统通过验证获取知识。
        A 在 facts 中意味着 A 在现实中为真，
        但系统不知道 A 为真直到它通过 MP 推导并验证。
        """
        worlds = build_worlds_e0_13()
        w = worlds["B"]
        facts = w.world_rules["facts"]
        assert "A(x)" in facts
        assert "B(x)" in facts
        # 但 A 不在 initial_beliefs
        initial_props = [p for p, _ in w.initial_beliefs]
        A = P.predicate("A", "x")
        assert A not in initial_props


# ============================================================
# Q6: World A — 直接匹配
# ============================================================

class TestWorldA:
    def test_direct_match_found(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["A"])
        assert any(n > 0 for n in r["metrics"]["direct_matches_per_step"])

    def test_eval_improved(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["A"])
        assert r["metrics"]["feedback_improved"]


# ============================================================
# Q7: World C — 无合法连接
# ============================================================

class TestWorldC:
    def test_admitted_insufficiency(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["C"])
        assert r["metrics"]["admitted_insufficiency"]

    def test_no_derived_objects(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["C"])
        assert r["metrics"]["derivation_chain_length"] == 0

    def test_eval_not_improved(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["C"])
        assert not r["metrics"]["feedback_improved"]


# ============================================================
# Q8: NC1 — F 无合法 implies
# ============================================================

class TestNC1:
    def test_admitted_insufficiency(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC1"])
        assert r["metrics"]["admitted_insufficiency"]

    def test_no_derived_objects(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC1"])
        assert r["metrics"]["derivation_chain_length"] == 0

    def test_eval_not_improved(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC1"])
        assert not r["metrics"]["feedback_improved"]


# ============================================================
# Q9: NC2 — 相似但不能合法推导
# ============================================================

class TestNC2:
    def test_no_derived_objects(self):
        """NC2: C 与 F 相似但 C 不是 known-true，不能推导 A。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC2"])
        assert r["metrics"]["derivation_chain_length"] == 0

    def test_admitted_insufficiency(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC2"])
        assert r["metrics"]["admitted_insufficiency"]

    def test_eval_not_improved(self):
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC2"])
        assert not r["metrics"]["feedback_improved"]

    def test_c_not_in_observable(self):
        """C 不在 NC2 的 observable 中。"""
        worlds = build_worlds_e0_13()
        w = worlds["NC2"]
        C = P.predicate("C", "x")
        assert C not in w.initial_observable


# ============================================================
# Q10: NC3 — 多错误候选 → 验证淘汰
# ============================================================

class TestNC3:
    def test_wrong_candidates_rejected(self):
        """NC3: Cw 和 Dw 被验证拒绝。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC3"])
        m = r["metrics"]
        assert len(m["objects_rejected_by_verification"]) > 0

    def test_correct_object_derived(self):
        """NC3: 正确对象 A 通过验证。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC3"])
        m = r["metrics"]
        assert "A(x)" in m["objects_entered_ks_via_mp"]

    def test_wrong_objects_not_in_ks(self):
        """NC3: 错误对象 Cw/Dw 不在最终 KS 的 VALID 集合中。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC3"])
        m = r["metrics"]
        # Cw, Dw 不在 objects_entered_ks_via_mp 中
        assert "Cw(x)" not in m["objects_entered_ks_via_mp"]
        assert "Dw(x)" not in m["objects_entered_ks_via_mp"]

    def test_eval_improved(self):
        """NC3: 正确推导链最终改善评价。"""
        worlds = build_worlds_e0_13()
        r = run_e0_13_episode(worlds["NC3"])
        assert r["metrics"]["feedback_improved"]


# ============================================================
# Q11: 完整实验
# ============================================================

class TestFullExperiment:
    def test_run_e0_13_returns_results(self):
        result = run_e0_13()
        assert result["experiment"] == "E0-13"
        for wid in ["A", "B", "C", "NC1", "NC2", "NC3"]:
            assert wid in result["results"]

    def test_analysis_all_pass(self):
        result = run_e0_13()
        a = result["analysis"]
        assert a["B_derived_objects_via_mp"]
        assert a["B_objects_entered_ks"]
        assert a["B_derivation_chain_length"] == 3
        assert a["B_eval_improved"]
        assert a["B_action_based_on_derived"]
        assert a["B_no_forced_answer"]
        assert a["A_direct_match"]
        assert a["A_eval_improved"]
        assert a["C_admitted_insufficiency"]
        assert a["C_no_derived_objects"]
        assert a["NC1_admitted_insufficiency"]
        assert a["NC1_no_derived_objects"]
        assert a["NC2_no_derived_objects"]
        assert a["NC2_admitted_insufficiency"]
        assert a["NC3_rejected_wrong_candidates"]
        assert a["NC3_derived_correct_object"]
        assert a["NC3_eval_improved"]

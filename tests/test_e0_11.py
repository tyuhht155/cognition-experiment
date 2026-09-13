"""E0-11 不变量测试：Problem Construction Ablation / Negative Controls。

验证 E0-10 的三个隐藏先验已被移除，并回答 Q1-Q6：

  Q1: Evaluation 是否只评价状态，不偷偷包含 Problem？
  Q2: Problem 是否由计算缺口产生，而不是人工指定？
  Q3: Problem 是否没有预先包含答案结构？
  Q4: Candidate Value 是否没有读取实验者知道的正确方向？
  Q5: Action 是否通过已验证知识发生，而不是字符串匹配？
  Q6: 当不存在可解路径时，系统能否承认计算能力不足？

E0-10 的隐藏先验（E0-11 移除）：
  P1. construct_problem_from_gap 写死 "find object that affects internal state"
  P2. evaluate_candidate_no_goal 检查 "H(" 和 implies
  P3. apply_action 用 "F(" 字符串匹配
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_11 import (
    # 数据结构
    GapProblem,
    WorldConfig,
    # 评价函数
    evaluate_h_state,
    evaluate_relation_state,
    # Problem 构造
    construct_problem_from_gap,
    # 盲候选评价
    evaluate_candidate_blind,
    # 知识推导行动
    derive_permitted_actions,
    # 世界
    build_worlds,
    # Episode
    run_e0_11_episode,
    # 实验
    run_e0_11,
    # 常量
    PERSISTENT_NEG_STEPS,
    CANDIDATE_LIMIT,
)


# ============================================================
# Q1: Evaluation 只评价状态，不偷偷包含 Problem
# ============================================================

class TestQ1EvaluationOnlyRatesState:
    """验证评价函数只接受状态，不引用 Problem/candidates/actions。"""

    def test_eval_h_state_takes_only_state_dict(self):
        """评价函数签名只接受 internal_state dict。"""
        eval_val = evaluate_h_state({"H": 1.0})
        assert isinstance(eval_val, float)

    def test_eval_relation_state_takes_only_state_dict(self):
        """关系评价函数也只接受 internal_state dict。"""
        eval_val = evaluate_relation_state({"H": 5.0, "P_Q_intact": False})
        assert isinstance(eval_val, float)
        assert eval_val < 0

    def test_eval_does_not_reference_problem(self):
        """评价函数不包含 Problem、candidates 或 action 的概念。
        通过检查函数源码不含 problem/candidate/action 关键字。"""
        import inspect
        src = inspect.getsource(evaluate_h_state)
        assert "problem" not in src.lower()
        assert "candidate" not in src.lower()
        assert "action" not in src.lower()

    def test_eval_returns_negative_for_low_h(self):
        """H 低时评价为负。"""
        assert evaluate_h_state({"H": 1.0}) < 0

    def test_eval_returns_positive_for_mid_h(self):
        """H 在合理区间时评价为正。"""
        assert evaluate_h_state({"H": 5.0}) > 0

    def test_eval_returns_negative_for_high_h(self):
        """H 过高时评价下降。"""
        assert evaluate_h_state({"H": 9.0}) < 0


# ============================================================
# Q2: Problem 由计算缺口产生，不是人工指定
# ============================================================

class TestQ2ProblemFromGap:
    """验证 Problem 由评价缺口产生。"""

    def test_problem_produced_when_persistent_negative(self):
        """持续性负评价产生 Problem。"""
        problem = construct_problem_from_gap(
            evaluation=-0.5,
            eval_delta=-0.1,
            persistent_count=PERSISTENT_NEG_STEPS,
            internal_state={"H": 1.0},
            observable=["F(food)", "H(low)"],
            budget=10.0,
        )
        assert problem is not None
        assert problem.evaluation < 0

    def test_no_problem_when_evaluation_positive(self):
        """正评价时不产生 Problem。"""
        problem = construct_problem_from_gap(
            evaluation=0.5,
            eval_delta=0.1,
            persistent_count=0,
            internal_state={"H": 5.0},
            observable=[],
            budget=10.0,
        )
        assert problem is None

    def test_problem_from_gap_not_from_answer(self):
        """Problem 构造函数不引用任何答案。
        通过检查可执行代码不含 "find"/"target"/"should" 关键字。"""
        import inspect
        import ast
        src = inspect.getsource(construct_problem_from_gap)
        # 移除 docstring 后检查可执行代码
        tree = ast.parse(src)
        # 找到 FunctionDef，移除 docstring
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # skip docstring
        code_lines = []
        in_docstring = False
        for line in src.split("\n"):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                elif stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    continue  # single-line docstring
                else:
                    in_docstring = True
                    continue
            if in_docstring:
                continue
            code_lines.append(line)
        code = "\n".join(code_lines)
        assert "find" not in code.lower(), "Problem 构造不应包含 'find'"
        assert "target" not in code.lower(), "Problem 构造不应包含 'target'"
        assert "should" not in code.lower(), "Problem 构造不应包含 'should'"


# ============================================================
# Q3: Problem 不包含答案结构
# ============================================================

class TestQ3ProblemNoAnswerStructure:
    """验证 Problem 不包含答案。"""

    def test_problem_has_no_answer_fields(self):
        """Problem dict 中没有 answer/target/required_relation/description。"""
        problem = construct_problem_from_gap(
            evaluation=-0.5,
            eval_delta=-0.1,
            persistent_count=PERSISTENT_NEG_STEPS,
            internal_state={"H": 1.0},
            observable=["F(food)"],
            budget=10.0,
        )
        assert problem is not None
        d = problem.to_dict()
        forbidden = {"target", "answer", "required_relation",
                     "required_operator", "description"}
        for key in forbidden:
            assert key not in d, f"Problem 不应包含 {key}"

    def test_problem_only_has_gap_data(self):
        """Problem 只包含缺口数据。"""
        problem = construct_problem_from_gap(
            evaluation=-0.5,
            eval_delta=-0.1,
            persistent_count=2,
            internal_state={"H": 1.0},
            observable=["F(food)"],
            budget=8.0,
        )
        d = problem.to_dict()
        expected_keys = {"evaluation", "eval_delta",
                         "persistent_negative_count",
                         "internal_state_snapshot",
                         "observable_present", "budget"}
        assert set(d.keys()) == expected_keys

    def test_different_worlds_produce_different_problems(self):
        """三个世界产生不同的 Problem（state_snapshot 不同）。"""
        worlds = build_worlds()
        r_a1 = run_e0_11_episode(worlds["A1"])
        r_a2 = run_e0_11_episode(worlds["A2"])
        r_a3 = run_e0_11_episode(worlds["A3"])

        # 找到每个世界第一个 Problem
        p_a1 = None
        p_a2 = None
        p_a3 = None
        for tr in r_a1["trace"]:
            if tr["problem"]:
                p_a1 = tr["problem"]
                break
        for tr in r_a2["trace"]:
            if tr["problem"]:
                p_a2 = tr["problem"]
                break
        for tr in r_a3["trace"]:
            if tr["problem"]:
                p_a3 = tr["problem"]
                break

        assert p_a1 is not None
        assert p_a2 is not None
        assert p_a3 is not None

        # state_snapshot 不同
        s1 = p_a1["internal_state_snapshot"]
        s2 = p_a2["internal_state_snapshot"]
        s3 = p_a3["internal_state_snapshot"]
        assert s1 != s2, "A1 和 A2 的 Problem state_snapshot 应不同"
        assert s2 != s3, "A2 和 A3 的 Problem state_snapshot 应不同"
        assert s1 != s3, "A1 和 A3 的 Problem state_snapshot 应不同"


# ============================================================
# Q4: Candidate Value 不读取实验者知道的正确方向
# ============================================================

class TestQ4NoPredicatePrivilege:
    """验证候选评价不检查 H/implies/谓词名。"""

    def _strip_docstring(self, src):
        """移除源码中的 docstring 行。"""
        lines = src.split("\n")
        result = []
        in_doc = False
        for line in lines:
            s = line.strip()
            if s.startswith('"""') or s.startswith("'''"):
                if in_doc:
                    in_doc = False
                    continue
                elif s.count('"""') == 2 or s.count("'''") == 2:
                    continue
                else:
                    in_doc = True
                    continue
            if in_doc:
                continue
            result.append(line)
        return "\n".join(result)

    def test_evaluator_does_not_check_h(self):
        """盲评价器不检查 'H(' 字符串（可执行代码中）。"""
        import inspect
        src = inspect.getsource(evaluate_candidate_blind)
        code = self._strip_docstring(src)
        assert '"H(' not in code, "评价器不应检查 'H(' 字符串"
        assert "'H('" not in code, "评价器不应检查 'H(' 字符串"
        assert "involves_state" not in code

    def test_evaluator_does_not_check_implies(self):
        """盲评价器不检查 kind == 'implies'（可执行代码中）。"""
        import inspect
        src = inspect.getsource(evaluate_candidate_blind)
        code = self._strip_docstring(src)
        assert ".kind" not in code, "评价器不应检查 prop.kind"
        assert '"implies"' not in code, "评价器不应检查 kind == implies"

    def test_h_candidate_and_non_h_candidate_same_score(self):
        """包含 H 的候选和不包含 H 的候选在无历史时得分相同。"""
        belief = BeliefStore()
        stats = {}
        feedback = None

        prop_with_h = P.predicate("H", "low")
        prop_without_h = P.predicate("X", "obj")

        ev_h = evaluate_candidate_blind(
            prop_with_h, "neg", 0.5, belief, stats, feedback)
        ev_x = evaluate_candidate_blind(
            prop_without_h, "neg", 0.5, belief, stats, feedback)

        # 在无历史时，得分应相同（不因包含 H 而更高）
        assert ev_h["value"] == ev_x["value"], \
            "包含 H 的候选不应获得特权分数"

    def test_implies_and_conj_same_score(self):
        """impl 和 conj 候选在无历史时得分相同（仅成本不同）。"""
        belief = BeliefStore()
        stats = {}
        feedback = None

        F = P.predicate("F", "food")
        H_mid = P.predicate("H", "mid")

        prop_impl = P.impl(F, H_mid)
        prop_conj = P.conj(F, H_mid)

        # 使用相同成本来隔离检查
        ev_impl = evaluate_candidate_blind(
            prop_impl, "impl", 1.0, belief, stats, feedback)
        ev_conj = evaluate_candidate_blind(
            prop_conj, "conj", 1.0, belief, stats, feedback)

        assert ev_impl["value"] == ev_conj["value"], \
            "相同成本下 impl 不应比 conj 获得特权分数"

    def test_candidate_results_have_no_privilege_fields(self):
        """候选评价结果中不含 involves_state/involves_causality 字段。"""
        worlds = build_worlds()
        r = run_e0_11_episode(worlds["A1"])
        for tr in r["trace"]:
            for c in tr.get("candidates_top5", []):
                assert "involves_state" not in c
                assert "involves_causality" not in c


# ============================================================
# Q5: Action 通过已验证知识发生
# ============================================================

class TestQ5ActionFromVerifiedKnowledge:
    """验证行动从已验证 implies 推导，不字符串匹配。"""

    def test_action_has_source_proposition(self):
        """所有行动都有 source_proposition（来自已验证 implies）。"""
        worlds = build_worlds()
        r = run_e0_11_episode(worlds["A1"])
        for tr in r["trace"]:
            if tr.get("action"):
                assert "source_proposition" in tr["action"], \
                    "行动必须有 source_proposition"

    def test_no_string_matching_in_action(self):
        """行动代码不使用 'F(' 字符串匹配。"""
        import inspect
        from experiments.run_e0_11 import run_e0_11_episode
        src = inspect.getsource(run_e0_11_episode)
        assert '"F("' not in src
        assert "'F('" not in src
        assert "string_match" not in src.lower()

    def test_derive_permitted_actions_checks_belief_store(self):
        """derive_permitted_actions 只从 BeliefStore 的 VALID implies 推导。"""
        belief = BeliefStore()
        F = P.predicate("F", "food")
        H_mid = P.predicate("H", "mid")
        impl_fh = P.impl(F, H_mid)

        # 没有验证前 → 无行动
        actions = derive_permitted_actions(belief, {F})
        assert len(actions) == 0

        # 验证为 VALID → 有行动
        belief.update_belief(impl_fh, STATUS_VALID, 0.9)
        actions = derive_permitted_actions(belief, {F})
        assert len(actions) == 1
        assert actions[0]["action_object"] == F
        assert actions[0]["expected_effect"] == H_mid

    def test_no_action_without_valid_implies(self):
        """没有 VALID implies 时无行动。"""
        belief = BeliefStore()
        F = P.predicate("F", "food")
        H_mid = P.predicate("H", "mid")

        # 只有 INVALID implies → 无行动
        belief.update_belief(P.impl(F, H_mid), STATUS_INVALID, 0.8)
        actions = derive_permitted_actions(belief, {F})
        assert len(actions) == 0


# ============================================================
# Q6: 无可解路径时承认计算能力不足
# ============================================================

class TestQ6AdmitsInsufficiency:
    """验证系统能承认计算能力不足。"""

    def test_nc1_admits_insufficiency(self):
        """NC1（无可解路径）→ 系统承认计算能力不足。"""
        worlds = build_worlds()
        r = run_e0_11_episode(worlds["NC1"])
        assert r["admitted_insufficiency"], \
            "NC1 应承认计算能力不足"
        assert r["stop_reason"] == "computation_insufficient"

    def test_nc1_eval_not_improved(self):
        """NC1 评价没有改善。"""
        worlds = build_worlds()
        r = run_e0_11_episode(worlds["NC1"])
        assert not r["eval_improved"]

    def test_nc2_false_opportunity_detected(self):
        """NC2（False Opportunity）→ 系统不因改变了变量就声称成功。"""
        worlds = build_worlds()
        r = run_e0_11_episode(worlds["NC2"])
        # 行动被采取但评价未改善
        assert r["action_taken"]
        assert not r["eval_improved"], \
            "NC2 不应因行动改变了 H 就声称成功"

    def test_nc2_does_not_falsely_succeed(self):
        """NC2 最终不声称成功。"""
        worlds = build_worlds()
        r = run_e0_11_episode(worlds["NC2"])
        assert r["final_evaluation"] < 0, \
            "NC2 最终评价应为负（关系未修复）"

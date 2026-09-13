"""E0-15 不变量测试：Goal Semantic Discovery / 目标语义发现.

验证核心问题：
  1. Goal 初始没有语义映射
  2. 系统不能直接读取真实映射
  3. 可以通过普通 Proposition 表达候选关系
  4. 候选关系必须经过验证
  5. 验证失败的关系不能进入有效知识
  6. 新知识可以进入 Knowledge Space
  7. 新知识可以被后续搜索复用
  8. 改变 Goal 名称不影响机制
  9. 改变 state dimension 名称不影响机制
  10. Goal 可以对应复合条件
  11. 无关系世界返回 computation_insufficient
  12. 错误相似关系不能冒充正确关系
  13. 不允许 goal→dimension 硬编码
  14. 不允许 ground truth 泄漏
  15. 严格时间因果
  16. 完整 trace
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_15 import (
    evaluate_state_no_goal_mapping,
    generate_goal_dim_candidates,
    build_worlds_e0_15,
    run_e0_15_episode,
    run_e0_15,
    E015Metrics,
    E015WorldConfig,
    GOAL_PREDICATE_NAME,
    DIM_PREDICATE_NAME,
    ACTION_PREDICATE_NAME,
    SATISFIED_PREDICATE_NAME,
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
    import experiments.run_e0_15 as m
    return _strip_docstring(inspect.getsource(m))


# ============================================================
# Q1: Goal 初始没有语义映射
# ============================================================

class TestGoalInitialNoSemanticMapping:
    """验证系统初始不知道 Goal 对应哪个 state dimension。"""

    def test_A_initial_eval_insufficient(self):
        """World A 初始评价返回 insufficient_knowledge。"""
        result = run_e0_15()
        mA = result["results"]["A"]["metrics"]
        assert mA["initial_eval_reason"] == "insufficient_knowledge"
        assert mA["initial_evaluation"] == 0.0

    def test_B_initial_eval_insufficient(self):
        """World B 初始评价返回 insufficient_knowledge。"""
        result = run_e0_15()
        mB = result["results"]["B"]["metrics"]
        assert mB["initial_eval_reason"] == "insufficient_knowledge"
        assert mB["initial_evaluation"] == 0.0

    def test_C_initial_eval_insufficient(self):
        """World C 初始评价返回 insufficient_knowledge。"""
        result = run_e0_15()
        mC = result["results"]["C"]["metrics"]
        assert mC["initial_eval_reason"] == "insufficient_knowledge"
        assert mC["initial_evaluation"] == 0.0


# ============================================================
# Q2: 系统不能直接读取真实映射
# ============================================================

class TestNoGroundTruthAccess:
    """验证系统不能直接读取 world.true_goal_dimension。"""

    def test_true_dimension_not_in_computation_path(self):
        """true_goal_dimension 字段不参与任何计算。"""
        src = _module_source()
        # true_goal_dimension 只在 WorldConfig 中定义，不参与计算
        assert "true_goal_dimension" in src
        # 但不应在 evaluate 或 generate 函数中被读取
        eval_src = inspect.getsource(evaluate_state_no_goal_mapping)
        gen_src = inspect.getsource(generate_goal_dim_candidates)
        assert "true_goal_dimension" not in eval_src
        assert "true_goal_dimension" not in gen_src

    def test_world_config_has_true_dimension(self):
        """World Simulator 知道真实映射，但 Cognitive System 不读取。"""
        worlds = build_worlds_e0_15()
        assert worlds["A"].true_goal_dimension == "energy"
        assert worlds["B"].true_goal_dimension == "fuel"
        assert worlds["C"].true_goal_dimension == "composite"


# ============================================================
# Q3: 可以通过普通 Proposition 表达候选关系
# ============================================================

class TestCandidateRelationGeneration:
    """验证候选关系是普通 Proposition 构造。"""

    def test_A_candidates_generated(self):
        """World A 生成了候选 Goal→Dim 关系。"""
        result = run_e0_15()
        mA = result["results"]["A"]["metrics"]
        assert mA["candidates_generated"] > 0

    def test_candidates_are_ordinary_propositions(self):
        """候选关系使用普通 impl 构造器，不是 Goal 专用机制。"""
        bs = BeliefStore()
        goal = P.predicate(GOAL_PREDICATE_NAME, "test")
        dim_prop = P.predicate(DIM_PREDICATE_NAME, "energy", "low")
        observable = {dim_prop}

        candidates = generate_goal_dim_candidates(bs, observable, goal, limit=8)
        assert len(candidates) > 0
        for c in candidates:
            assert c["proposition"].kind == "implies"
            antecedent, consequent = c["proposition"].parts
            assert antecedent == goal

    def test_candidates_use_same_constructor_as_e0_12(self):
        """候选生成使用与 derive_candidates 相同的 impl 构造器。"""
        src = inspect.getsource(generate_goal_dim_candidates)
        assert "P.impl" in src
        assert "P.conj" in src


# ============================================================
# Q4: 候选关系必须经过验证
# ============================================================

class TestCandidateVerification:
    """验证候选关系必须通过 verify_against_world。"""

    def test_A_candidates_verified(self):
        """World A 候选经过验证。"""
        result = run_e0_15()
        mA = result["results"]["A"]["metrics"]
        assert mA["candidates_verified"] > 0


# ============================================================
# Q5: 验证失败的关系不能进入有效知识
# ============================================================

class TestRejectedRelationsNotInKS:
    """验证验证失败的关系不进入有效知识。"""

    def test_NC2_rejected_wrong_relations(self):
        """NC2 中错误的关系被拒绝。"""
        result = run_e0_15()
        mNC2 = result["results"]["NC2"]["metrics"]
        assert len(mNC2["relations_rejected"]) > 0

    def test_NC2_wrong_not_in_ks(self):
        """NC2 中 'wrong' 和 'other' 不在进入 KS 的关系中。"""
        result = run_e0_15()
        mNC2 = result["results"]["NC2"]["metrics"]
        for r in mNC2["relations_entered_ks"]:
            assert "wrong" not in r
            assert "other" not in r


# ============================================================
# Q6: 新知识可以进入 Knowledge Space
# ============================================================

class TestNewKnowledgeEntersKS:
    """验证发现的关系进入 Knowledge Space。"""

    def test_A_relations_entered_ks(self):
        """World A 发现的关系进入 KS。"""
        result = run_e0_15()
        mA = result["results"]["A"]["metrics"]
        assert len(mA["relations_entered_ks"]) > 0

    def test_A_ks_grew(self):
        """World A 知识空间从 0 增长。"""
        result = run_e0_15()
        mA = result["results"]["A"]["metrics"]
        assert mA["knowledge_space_size_after"] > mA["knowledge_space_size_before"]


# ============================================================
# Q7: 新知识可以被后续搜索复用
# ============================================================

class TestNewKnowledgeReusable:
    """验证新知识进入 KS 后可以被后续评价复用。"""

    def test_A_final_eval_mapped(self):
        """World A 最终评价使用 discovered 关系（reason=mapped）。"""
        result = run_e0_15()
        mA = result["results"]["A"]["metrics"]
        assert mA["final_eval_reason"] == "mapped"


# ============================================================
# Q8: 改变 Goal 名称不影响机制
# ============================================================

class TestGoalNameAgnostic:
    """验证改变 Goal 名称不影响机制。"""

    def test_NC4_works_same(self):
        """NC4 中 Goal 改名为 X7，机制仍然工作。"""
        result = run_e0_15()
        mNC4 = result["results"]["NC4"]["metrics"]
        assert mNC4["goal_semantic_discovered"]

    def test_NC4_discovered_same_dimension(self):
        """NC4 发现的关系对应 energy 维度。"""
        result = run_e0_15()
        mNC4 = result["results"]["NC4"]["metrics"]
        assert any("energy" in r for r in mNC4["discovered_relations"])


# ============================================================
# Q9: 改变 state dimension 名称不影响机制
# ============================================================

class TestDimensionNameAgnostic:
    """验证改变 state dimension 名称不影响机制。"""

    def test_B_works_same(self):
        """World B 使用 fuel/pressure，机制仍然工作。"""
        result = run_e0_15()
        mB = result["results"]["B"]["metrics"]
        assert mB["goal_semantic_discovered"]

    def test_B_discovered_fuel(self):
        """World B 发现的关系对应 fuel 维度。"""
        result = run_e0_15()
        mB = result["results"]["B"]["metrics"]
        assert any("fuel" in r for r in mB["discovered_relations"])


# ============================================================
# Q10: Goal 可以对应复合条件
# ============================================================

class TestCompositeGoalCondition:
    """验证 Goal 可以对应复合条件（A ∧ B），不是单一 dimension。"""

    def test_C_discovered_composite(self):
        """World C 发现了复合关系。"""
        result = run_e0_15()
        mC = result["results"]["C"]["metrics"]
        assert mC["goal_semantic_discovered"]

    def test_C_discovered_conjunction(self):
        """World C 发现的关系包含合取。"""
        result = run_e0_15()
        mC = result["results"]["C"]["metrics"]
        assert any("∧" in r for r in mC["discovered_relations"])


# ============================================================
# Q11: 无关系世界返回 computation_insufficient
# ============================================================

class TestNoRelationWorld:
    """验证无关系世界返回 computation_insufficient。"""

    def test_NC1_admitted_insufficiency(self):
        """NC1 承认计算不足。"""
        result = run_e0_15()
        mNC1 = result["results"]["NC1"]["metrics"]
        assert mNC1["admitted_insufficiency"]
        assert mNC1["stop_reason"] == "computation_insufficient"

    def test_NC1_no_discovered_relations(self):
        """NC1 没有发现任何关系。"""
        result = run_e0_15()
        mNC1 = result["results"]["NC1"]["metrics"]
        assert not mNC1["goal_semantic_discovered"]
        assert len(mNC1["discovered_relations"]) == 0


# ============================================================
# Q12: 错误相似关系不能冒充正确关系
# ============================================================

class TestWrongRelationRejected:
    """验证错误相似关系不能冒充正确关系。"""

    def test_NC2_correct_relation_found(self):
        """NC2 中正确关系被发现。"""
        result = run_e0_15()
        mNC2 = result["results"]["NC2"]["metrics"]
        assert any("energy" in r for r in mNC2["discovered_relations"])

    def test_NC2_wrong_rejected(self):
        """NC2 中错误关系被拒绝。"""
        result = run_e0_15()
        mNC2 = result["results"]["NC2"]["metrics"]
        assert len(mNC2["relations_rejected"]) > 0


# ============================================================
# Q13: 不允许 goal→dimension 硬编码
# ============================================================

class TestNoHardcodedMapping:
    """验证代码中没有 goal→dimension 硬编码。"""

    def test_no_goal_name_check_in_evaluator(self):
        """评价函数不检查 goal.name == 'wealth' 等具体值。"""
        src = inspect.getsource(evaluate_state_no_goal_mapping)
        # 不应该出现具体 goal 名的硬编码
        assert 'goal.name == "wealth"' not in src
        assert 'goal.name == "health"' not in src
        assert "if goal ==" not in src
        assert "goal.parts[0]" not in src or "dim = str(consequent.parts[0])" in src

    def test_no_dimension_dict(self):
        """代码中没有 goal→dimension 的字典映射。"""
        src = _module_source()
        assert "GOAL_DIMENSION_MAP" not in src
        assert "goal_dimension_map" not in src
        assert "dimension_map" not in src.lower()

    def test_no_goal_manager(self):
        """没有 GoalManager / GoalMapper / GoalGenerator。"""
        src = _module_source()
        assert "GoalManager" not in src
        assert "GoalMapper" not in src
        assert "GoalGenerator" not in src
        assert "GoalDimensionMapper" not in src
        assert "GoalInterpreter" not in src
        assert "GoalSemanticParser" not in src

    def test_no_llm_or_embedding(self):
        """没有 LLM / embedding / 神经网络。"""
        src = _module_source()
        src_lower = src.lower()
        assert "llm" not in src_lower
        assert "embedding" not in src_lower
        assert "neural" not in src_lower
        assert "transformer" not in src_lower


# ============================================================
# Q14: 不允许 ground truth 泄漏
# ============================================================

class TestNoGroundTruthLeakage:
    """验证 ground truth 不泄漏给系统。"""

    def test_true_dimension_not_read_in_episode(self):
        """episode 运行中不读取 true_goal_dimension 用于计算。

        true_goal_dimension 可以出现在返回的 dict 中（用于报告），
        但不能用于评价、候选生成、验证等计算路径。
        """
        src = inspect.getsource(run_e0_15_episode)
        # true_goal_dimension 只应该出现在返回 dict 中
        # 不应该出现在 evaluate/generate/verify 调用中
        assert "true_goal_dimension" in src  # 在返回 dict 中
        # 确保没有被赋值给计算变量
        assert "= world.true_goal_dimension" not in src.replace(
            '"true_goal_dimension": world.true_goal_dimension', '')

    def test_true_dimension_not_in_evaluate(self):
        """评价函数不读取 true_goal_dimension。"""
        src = inspect.getsource(evaluate_state_no_goal_mapping)
        assert "true_goal_dimension" not in src

    def test_true_dimension_not_in_generate(self):
        """候选生成不读取 true_goal_dimension。"""
        src = inspect.getsource(generate_goal_dim_candidates)
        assert "true_goal_dimension" not in src


# ============================================================
# Q15: 严格时间因果
# ============================================================

class TestStrictTemporalCausality:
    """验证严格时间因果：不使用未来信息。"""

    def test_no_future_state_access(self):
        """不访问未来状态。"""
        src = inspect.getsource(run_e0_15_episode)
        # 不应该有 world.future 或 future_state 等访问
        assert "world.future" not in src
        assert "future_state" not in src
        # true_goal_dimension 不被用于计算（只在返回 dict 中）
        assert "= world.true_goal_dimension" not in src.replace(
            '"true_goal_dimension": world.true_goal_dimension', '')

    def test_candidates_from_current_observable_only(self):
        """候选只从当前可观察集合构造。"""
        src = inspect.getsource(generate_goal_dim_candidates)
        assert "observable" in src
        assert "get_constructible" in src


# ============================================================
# Q16: 完整 trace
# ============================================================

class TestCompleteTrace:
    """验证完整 trace 记录。"""

    def test_all_worlds_have_trace(self):
        """所有世界都有 trace 记录。"""
        result = run_e0_15()
        for wid, r in result["results"].items():
            assert len(r["trace"]) > 0, f"World {wid} has empty trace"

    def test_A_trace_records_discovery(self):
        """World A 的 trace 记录了语义发现步骤。"""
        result = run_e0_15()
        trace_A = result["results"]["A"]["trace"]
        has_discovery = any(
            t.get("mode") == "goal_semantic_discovery" for t in trace_A
        )
        assert has_discovery

    def test_trace_records_candidates_and_results(self):
        """trace 记录候选和验证结果。"""
        result = run_e0_15()
        trace_A = result["results"]["A"]["trace"]
        discovery_step = next(
            (t for t in trace_A if t.get("mode") == "goal_semantic_discovery"), None
        )
        assert discovery_step is not None
        assert "candidates" in discovery_step
        assert "new_relations" in discovery_step
        assert "rejected" in discovery_step


# ============================================================
# 综合验证
# ============================================================

class TestOverallAnalysis:
    """验证整体分析结果。"""

    def test_analysis_all_true_except_none(self):
        """所有分析指标中应该为 True 的都是 True。"""
        result = run_e0_15()
        analysis = result["analysis"]
        # NC1 应该是 True（admitted insufficiency）
        assert analysis["NC1_admitted_insufficiency"] is True
        assert analysis["NC1_no_discovered_relations"] is True
        # 其他应该都是 True
        for key, val in analysis.items():
            if key.startswith("NC1_"):
                continue
            assert val is True, f"Analysis {key} should be True, got {val}"

    def test_no_forced_answer(self):
        """没有 forced answer（不假装成功）。"""
        result = run_e0_15()
        for wid, r in result["results"].items():
            m = r["metrics"]
            assert not m["forced_answer"], f"World {wid} has forced_answer=True"

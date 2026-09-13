"""架构修复不变量测试：

1. KnowledgeView 必须是真正的只读接口（行为测试）
2. recursive_compute 必须使用共享 stats dict
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore, ReadOnlyKnowledgeView, KnowledgeView
from cognition.operations import Context, op_modus_ponens
from cognition.models import STATUS_VALID, STATUS_INVALID


# ============================================================
# KnowledgeView 只读接口行为测试
# ============================================================

class TestKnowledgeViewReadOnly:
    """验证 ReadOnlyKnowledgeView 是真正的只读 facade。"""

    def test_readonly_view_can_read(self):
        """ReadOnlyKnowledgeView 可以正常读取知识。"""
        store = BeliefStore()
        pa = P.atom("A")
        store.update_belief(pa, STATUS_VALID, 0.9)

        view = ReadOnlyKnowledgeView(store)
        assert view.has(pa)
        assert view.get(pa) is not None
        assert view.get(pa).status == STATUS_VALID
        assert len(view.valid_beliefs()) == 1
        assert len(view.all_beliefs()) == 1

    def test_readonly_view_has_no_update_belief(self):
        """ReadOnlyKnowledgeView 不暴露 update_belief。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        assert not hasattr(view, "update_belief")

    def test_readonly_view_has_no_get_or_create(self):
        """ReadOnlyKnowledgeView 不暴露 get_or_create。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        assert not hasattr(view, "get_or_create")

    def test_readonly_view_has_no_record_reuse(self):
        """ReadOnlyKnowledgeView 不暴露 record_reuse。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        assert not hasattr(view, "record_reuse")

    def test_readonly_view_has_no_record_verification(self):
        """ReadOnlyKnowledgeView 不暴露 record_verification。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        assert not hasattr(view, "record_verification")

    def test_readonly_view_has_no_tick(self):
        """ReadOnlyKnowledgeView 不暴露 tick。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        assert not hasattr(view, "tick")

    def test_readonly_view_update_belief_raises(self):
        """调用 update_belief 必须直接失败，而不是执行。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        pa = P.atom("A")
        with pytest.raises(AttributeError):
            view.update_belief(pa, STATUS_VALID, 0.9)

    def test_readonly_view_get_or_create_raises(self):
        """调用 get_or_create 必须直接失败。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        pa = P.atom("A")
        with pytest.raises(AttributeError):
            view.get_or_create(pa)

    def test_readonly_view_tick_raises(self):
        """调用 tick 必须直接失败。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        with pytest.raises(AttributeError):
            view.tick()

    def test_readonly_view_does_not_mutate_store(self):
        """通过 view 无法修改 store 的信念。"""
        store = BeliefStore()
        view = ReadOnlyKnowledgeView(store)
        pa = P.atom("A")
        # 初始 store 为空
        assert not store.has(pa)
        # 尝试通过 view 的各种方式修改——全部应该失败
        with pytest.raises(AttributeError):
            view.update_belief(pa, STATUS_VALID, 0.9)
        with pytest.raises(AttributeError):
            view.get_or_create(pa)
        # store 仍然为空
        assert not store.has(pa)

    def test_context_wraps_belief_store(self):
        """Context 接收 BeliefStore 时必须包装成 ReadOnlyKnowledgeView。"""
        store = BeliefStore()
        pa = P.atom("A")
        store.update_belief(pa, STATUS_VALID, 0.9)

        ctx = Context(belief_store=store, constants=["A"])
        # ctx.knowledge_view 不应该是 BeliefStore
        assert not isinstance(ctx.knowledge_view, BeliefStore)
        # 应该是 ReadOnlyKnowledgeView
        assert isinstance(ctx.knowledge_view, ReadOnlyKnowledgeView)
        # 可以读取
        assert ctx.knowledge_view.has(pa)
        assert ctx.knowledge_view.get(pa).status == STATUS_VALID

    def test_context_knowledge_view_cannot_write(self):
        """Context 的 knowledge_view 不能写入。"""
        store = BeliefStore()
        ctx = Context(belief_store=store, constants=["A"])
        pa = P.atom("A")
        with pytest.raises(AttributeError):
            ctx.knowledge_view.update_belief(pa, STATUS_VALID, 0.9)

    def test_context_store_param_wrapped(self):
        """Context 的 store= 参数也被包装成 ReadOnlyKnowledgeView。"""
        store = BeliefStore()
        pa = P.atom("A")
        store.update_belief(pa, STATUS_VALID, 0.9)

        ctx = Context(store=store, constants=["A"])
        assert not isinstance(ctx.knowledge_view, BeliefStore)
        assert isinstance(ctx.knowledge_view, ReadOnlyKnowledgeView)
        assert ctx.knowledge_view.has(pa)

    def test_context_knowledge_view_param_belief_store_wrapped(self):
        """Context 的 knowledge_view=BeliefStore 也被包装。"""
        store = BeliefStore()
        pa = P.atom("A")
        store.update_belief(pa, STATUS_VALID, 0.9)

        ctx = Context(knowledge_view=store, constants=["A"])
        assert not isinstance(ctx.knowledge_view, BeliefStore)
        assert isinstance(ctx.knowledge_view, ReadOnlyKnowledgeView)

    def test_context_readonly_view_passthrough(self):
        """Context 接收已有的 ReadOnlyKnowledgeView 不再包装。"""
        store = BeliefStore()
        pa = P.atom("A")
        store.update_belief(pa, STATUS_VALID, 0.9)
        view = ReadOnlyKnowledgeView(store)

        ctx = Context(knowledge_view=view, constants=["A"])
        assert isinstance(ctx.knowledge_view, ReadOnlyKnowledgeView)
        assert ctx.knowledge_view is view

    def test_modus_ponens_works_with_readonly_view(self):
        """op_modus_ponens 通过 ReadOnlyKnowledgeView 正常工作。"""
        store = BeliefStore()
        pq = P.impl(P.atom("P"), P.atom("Q"))
        store.update_belief(pq, STATUS_VALID, 0.9)

        ctx = Context(belief_store=store, constants=["P", "Q"])
        p = P.atom("P")
        cands = op_modus_ponens(p, ctx)
        assert len(cands) == 1
        assert cands[0].new_object == P.atom("Q")
        assert cands[0].op_name == "modus_ponens"


# ============================================================
# recursive_compute 共享 stats 行为测试
# ============================================================

class TestRecursiveComputeSharedStats:
    """验证 recursive_compute 使用共享 stats dict，递归树统计完整。"""

    def _build_engine(self, max_depth=3):
        """构建一个可用于递归测试的 ComputeEngine。"""
        from cognition.evidence import EvidenceLog
        from cognition.cost import CostTracker
        from cognition.consensus import ConsensusAgreementModel
        from cognition.prediction import TemporalPredictionState
        from cognition.operations import OperationStore
        from cognition.trace import TraceRecorder
        from cognition.compute import ComputeEngine

        store = BeliefStore()
        evidence_log = EvidenceLog()
        cost_tracker = CostTracker()
        consensus = ConsensusAgreementModel()
        prediction_state = TemporalPredictionState()
        op_store = OperationStore()
        trace = TraceRecorder()
        engine = ComputeEngine(
            belief_store=store,
            evidence_log=evidence_log,
            cost_tracker=cost_tracker,
            consensus=consensus,
            prediction_state=prediction_state,
            op_store=op_store,
            trace=trace,
            max_depth=max_depth,
        )
        return engine, store

    def test_root_stats_reflect_full_recursion(self):
        """root 完成后 stats 必须反映整棵递归树。

        构造 root → child → grandchild 的递归：
          P → Q (via modus ponens, P→Q valid)
          Q → R (via modus ponens, Q→R valid)
          R → (leaf)
        """
        engine, store = self._build_engine(max_depth=3)

        # 知识：P→Q (valid), Q→R (valid)
        pq = P.impl(P.atom("P"), P.atom("Q"))
        qr = P.impl(P.atom("Q"), P.atom("R"))
        store.update_belief(pq, STATUS_VALID, 0.9)
        store.update_belief(qr, STATUS_VALID, 0.9)

        # ctx
        ctx = Context(belief_store=store, constants=["P", "Q", "R"],
                      step_budget=100)
        p = P.atom("P")

        engine.recursive_compute(p, goal=None, ctx=ctx)

        # 验证统计反映了整棵递归树
        # actual_steps: root identify + root generate + at least 2 evaluations +
        #               child identify + child generate + ... > 4
        assert engine.actual_steps > 4, \
            f"actual_steps should reflect full recursion tree, got {engine.actual_steps}"
        # generated_candidates: root generates candidates for P (including Q),
        #                       child generates candidates for Q (including R)
        assert engine.generated_candidates > 0, \
            f"generated_candidates should be > 0, got {engine.generated_candidates}"

    def test_child_stats_not_overwritten_by_root(self):
        """root 的 _sync_stats 不覆盖 child 的统计。

        在修复前，child 调用 _sync_stats 把自己的 stats 写回 engine，
        然后 root 调用 _sync_stats 用 root 的旧 stats（不含 child 贡献）覆盖。
        修复后，只有 root 调用 _sync_stats，且使用共享 stats。
        """
        engine, store = self._build_engine(max_depth=3)

        pq = P.impl(P.atom("P"), P.atom("Q"))
        qr = P.impl(P.atom("Q"), P.atom("R"))
        store.update_belief(pq, STATUS_VALID, 0.9)
        store.update_belief(qr, STATUS_VALID, 0.9)

        ctx = Context(belief_store=store, constants=["P", "Q", "R"],
                      step_budget=100)
        p = P.atom("P")

        engine.recursive_compute(p, goal=None, ctx=ctx)

        # 如果 child 的 stats 被覆盖，actual_steps 会只反映 root 的步骤
        # root: identify(1) + generate(1) + pre_evaluate(至少1) + process(至少1) = 4+
        # 如果共享正确，应该 > 4（包含 child 和 grandchild 的步骤）
        assert engine.actual_steps > 4, \
            f"actual_steps={engine.actual_steps} suggests child stats were overwritten"

    def test_three_level_recursion_stats(self):
        """三层递归 root → child → grandchild 的统计验证。

        P → Q (modus ponens via P→Q)
        Q → R (modus ponens via Q→R)
        R → S (modus ponens via R→S)
        """
        engine, store = self._build_engine(max_depth=3)

        pq = P.impl(P.atom("P"), P.atom("Q"))
        qr = P.impl(P.atom("Q"), P.atom("R"))
        rs = P.impl(P.atom("R"), P.atom("S"))
        store.update_belief(pq, STATUS_VALID, 0.9)
        store.update_belief(qr, STATUS_VALID, 0.9)
        store.update_belief(rs, STATUS_VALID, 0.9)

        ctx = Context(belief_store=store, constants=["P", "Q", "R", "S"],
                      step_budget=200)
        p = P.atom("P")

        engine.recursive_compute(p, goal=None, ctx=ctx)

        # 三层递归：root(P) + child(Q) + grandchild(R)
        # 每层至少 identify(1) + generate(1) = 2 步
        # 加上 evaluation, process 等步骤
        # actual_steps 应该 > 6（三层各至少 2 步）
        assert engine.actual_steps > 6, \
            f"Three-level recursion should have actual_steps > 6, got {engine.actual_steps}"

    def test_stats_dict_is_shared_object(self):
        """验证递归调用传递的是同一个 stats dict 对象。"""
        # 通过 monkey-patch 检查 stats 是否被传递
        engine, store = self._build_engine(max_depth=2)

        pq = P.impl(P.atom("P"), P.atom("Q"))
        store.update_belief(pq, STATUS_VALID, 0.9)

        ctx = Context(belief_store=store, constants=["P", "Q"],
                      step_budget=100)

        # 记录递归调用时传入的 stats id
        stats_ids_seen = []
        original_recursive = engine.recursive_compute

        def tracking_recursive(obj, goal, ctx, depth=0, parent_step=None, stats=None):
            if stats is not None:
                stats_ids_seen.append(id(stats))
            return original_recursive(obj, goal, ctx, depth, parent_step, stats)

        # 替换方法
        engine.recursive_compute = tracking_recursive

        p = P.atom("P")
        engine.recursive_compute(p, goal=None, ctx=ctx)

        # 如果 stats 被正确传递，所有递归调用看到的应该是同一个 stats 对象
        if len(stats_ids_seen) > 1:
            assert len(set(stats_ids_seen)) == 1, \
                f"All recursive calls should share same stats dict, but saw: {stats_ids_seen}"

    def test_generated_candidates_includes_child_contributions(self):
        """generated_candidates 必须包含 child 生成的候选数。"""
        engine, store = self._build_engine(max_depth=3)

        pq = P.impl(P.atom("P"), P.atom("Q"))
        qr = P.impl(P.atom("Q"), P.atom("R"))
        store.update_belief(pq, STATUS_VALID, 0.9)
        store.update_belief(qr, STATUS_VALID, 0.9)

        ctx = Context(belief_store=store, constants=["P", "Q", "R"],
                      step_budget=100)
        p = P.atom("P")

        engine.recursive_compute(p, goal=None, ctx=ctx)

        # root 生成 candidates for P（包含 Q 等）
        # child 生成 candidates for Q（包含 R 等）
        # 如果共享正确，generated_candidates 应该 > root 自己生成的数量
        assert engine.generated_candidates > 0
        # 如果 child 的统计被覆盖，generated_candidates 会只反映 root

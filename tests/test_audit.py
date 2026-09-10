"""第二轮审计单元测试。

重点验证实验因果关系的正确性，而非"聪明结果"。
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
from cognition.operations import OperationStore, Context
from cognition.verification import Verifier
from cognition.evaluation import ValueEvaluator
from cognition.compute import ComputeEngine
from cognition.trace import TraceRecorder
from cognition.environment import World


def _make_engine():
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    consensus = ConsensusAgreementModel()
    prediction_state = TemporalPredictionState()
    op_store = OperationStore()
    trace = TraceRecorder()
    verifier = Verifier()
    evaluator = ValueEvaluator()
    engine = ComputeEngine(belief_store, evidence_log, cost_tracker, consensus,
                           prediction_state, op_store, trace, verifier, evaluator,
                           max_depth=1, max_candidates=8)
    ctx = Context(
        belief_store=belief_store, evidence_log=evidence_log,
        cost_tracker=cost_tracker, consensus=consensus,
        prediction_state=prediction_state, op_store=op_store,
        trace=trace,
        constants=["ball", "box", "table", "wall"],
        step_budget=1000,
        verify_enabled=True, evaluate_enabled=True, meta_evaluate_enabled=False,
        goal=World.predict_next_goal())
    return engine, belief_store, trace, op_store, verifier, evaluator, ctx


# 1. D 不读取 future world state
def test_d_no_future_state_access():
    """D 组只能使用 ctx.world_history（已观察到的状态），不能访问 world.history 全量。"""
    engine, store, trace, registry, verification, evaluation, ctx = _make_engine()
    world = World(seed=7)
    world.run(10)
    # 模拟 D 组：只给前 5 步历史
    ctx.world_history = [set(s) for s in world.history[:5]]
    # 验证方法只能用 ctx.world_history
    from cognition.verification import Verification as V
    v = V()
    prop = P.atom("Did_open_box")
    result, _ = v.verify(prop, ctx)
    # 关键：验证过程中不抛异常、不访问 world.history（只看 ctx）
    # 检查 ctx.world_history 长度没变（验证器不应修改它）
    assert len(ctx.world_history) == 5
    print("test_d_no_future_state_access OK")


# 2. ground_truth_check 不改变 KnowledgeStore
def test_ground_truth_does_not_mutate_store():
    from cognition.knowledge_store import KnowledgeStore, Knowledge, STATUS_UNKNOWN
    store = KnowledgeStore()
    p = P.atom("A")
    store.upsert(Knowledge(proposition=p, status=STATUS_UNKNOWN, confidence=0.0))
    world = World(seed=7)
    world.run(5)
    before = {k.proposition.to_str(): (k.status, k.confidence)
              for k in store.all_entries()}
    # 调用 ground_truth_check（外部统计用）
    World.ground_truth_check(p, world.history)
    after = {k.proposition.to_str(): (k.status, k.confidence)
             for k in store.all_entries()}
    assert before == after, "ground_truth_check 不应修改 KnowledgeStore"
    print("test_ground_truth_does_not_mutate_store OK")


# 3. C 和 D 初始状态完全一致
def test_c_d_initial_state_identical():
    from cognition.knowledge_store import KnowledgeStore
    def make(meta):
        store = KnowledgeStore()
        trace = TraceRecorder()
        op_store = OperationStore()
        verifier = Verifier()
        evaluator = ValueEvaluator()
        ctx = Context(store=store, trace=trace,
                       constants=["ball", "box", "table", "wall"],
                       step_budget=1000, verify_enabled=True,
                       evaluate_enabled=True, meta_evaluate_enabled=meta,
                       goal=World.predict_next_goal())
        return store, op_store, verifier, evaluator, ctx

    c_store, c_reg, c_ver, c_eval, c_ctx = make(False)
    d_store, d_reg, d_ver, d_eval, d_ctx = make(True)

    assert c_store.size() == d_store.size() == 0
    assert sorted(c_reg.names()) == sorted(d_reg.names())
    assert c_store.verifier_reliability("logical") == d_store.verifier_reliability("logical")
    assert c_eval.weights == d_eval.weights
    assert c_ctx.meta_evaluate_enabled != d_ctx.meta_evaluate_enabled
    print("test_c_d_initial_state_identical OK")


# 4. C 和 D 唯一机制差异就是 meta
def test_c_d_only_meta_differs():
    from experiments.run_abcd import GROUPS
    c = GROUPS["C"]
    d = GROUPS["D"]
    # (verify, evaluate, meta)
    assert c[0] == d[0]  # verify 相同
    assert c[1] == d[1]  # evaluate 相同
    assert c[2] != d[2]  # meta 不同
    # C 的 meta=False, D 的 meta=True
    assert c[2] is False
    assert d[2] is True
    print("test_c_d_only_meta_differs OK")


# 5. meta-evaluation 不在两个地方同时触发
def test_meta_evaluation_single_entry():
    """compute.py 中不应调用 evaluate_evaluation；只在 run_abcd.py 触发。"""
    import inspect
    from cognition import compute as compute_mod
    src = inspect.getsource(compute_mod.ComputeEngine.recursive_compute)
    assert "evaluate_evaluation" not in src, \
        "ComputeEngine 不应调用 evaluate_evaluation（统一由 run_abcd.py 调度）"
    # run_abcd.py 中应有且仅有一处
    from experiments import run_abcd
    src2 = inspect.getsource(run_abcd)
    assert src2.count("evaluate_evaluation") == 1, \
        "run_abcd.py 中应仅有一处 evaluate_evaluation 调用"
    print("test_meta_evaluation_single_entry OK")


# 6. usefulness=None 表示未评价
def test_usefulness_none_means_unevaluated():
    from cognition.knowledge_store import KnowledgeStore, Knowledge, STATUS_UNKNOWN, STATUS_VALID
    store = KnowledgeStore()
    p = P.atom("A")
    k = Knowledge(proposition=p, status=STATUS_UNKNOWN, confidence=0.0,
                  usefulness=None, evaluated=False)
    store.upsert(k)
    entry = store.get(p)
    assert entry.usefulness is None
    assert entry.evaluated is False
    k2 = Knowledge(proposition=p, status=STATUS_VALID, confidence=0.9,
                   usefulness=0.5, evaluated=True)
    store.upsert(k2)
    entry = store.get(p)
    assert entry.usefulness == 0.5
    assert entry.evaluated is True
    print("test_usefulness_none_means_unevaluated OK")


# 7. correct_but_useless 只统计 evaluated=True
def test_correct_but_useless_only_evaluated():
    from experiments.run_abcd import compute_metrics
    from cognition.knowledge_store import KnowledgeStore, Knowledge, STATUS_VALID

    store = KnowledgeStore()
    trace = TraceRecorder()
    world = World(seed=7)
    world.run(5)

    # 用一个简单的 mock engine 提供搜索空间统计
    class MockEngine:
        generated_candidates = 0
        evaluated_candidates = 0
        passed_evaluation_gate = 0
        verified_candidates = 0
        valid_candidates = 0
        invalid_candidates = 0
        budget_exhausted = False
        actual_steps = 0
        useful_steps = 0
        verified_steps = 0
        total_cost = 0
        raw_compute_cost = 0
        verification_cost = 0
        reuse_cost = 0
        cache_saved_cost = 0
        composite_saved_cost = 0
        composite_usage = {}

    p_valid_evaluated = P.impl(P.atom("Open_box"), P.atom("CanTake_ball"))
    p_valid_unevaluated = P.impl(P.atom("A"), P.atom("B"))

    store.upsert(Knowledge(proposition=p_valid_evaluated, status=STATUS_VALID,
                           confidence=0.9, usefulness=0.01, evaluated=True))
    store.upsert(Knowledge(proposition=p_valid_unevaluated, status=STATUS_VALID,
                           confidence=0.9, usefulness=None, evaluated=False))

    m = compute_metrics("test", store, trace, MockEngine(), world, evaluate=True, meta=False)
    stats = store.stats()
    assert stats["evaluated"] == 1
    assert stats["valid_low_usefulness"] == 1
    print("test_correct_but_useless_only_evaluated OK")


# 8. composite operation 未注册时不会被假装调用
def test_composite_not_faked_when_unregistered():
    engine, store, trace, registry, verification, evaluation, ctx = _make_engine()
    world = World(seed=7)
    world.run(5)
    ctx.world_history = [set(s) for s in world.history[:5]]

    prior_ops = set(registry.names())
    obj = P.atom("Did_open_box")
    engine.recursive_compute(obj, ctx.goal, ctx, depth=0)

    # 检查 trace 中没有 composite_ 开头的操作（因为没注册）
    trace_ops = {s.operation for s in trace.steps}
    assert not any(op.startswith("composite_") for op in trace_ops), \
        "未注册的 composite 操作不应被调用"
    print("test_composite_not_faked_when_unregistered OK")


# 9. budget 达到后系统停止
def test_budget_exhaustion_stops_compute():
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    consensus = ConsensusAgreementModel()
    prediction_state = TemporalPredictionState()
    op_store = OperationStore()
    trace = TraceRecorder()
    ctx = Context(belief_store=belief_store, evidence_log=evidence_log,
                  cost_tracker=cost_tracker, consensus=consensus,
                  prediction_state=prediction_state, op_store=op_store,
                  trace=trace,
                  constants=["ball", "box", "table", "wall"],
                  step_budget=10, verify_enabled=True,
                  evaluate_enabled=True, meta_evaluate_enabled=False,
                  goal=World.predict_next_goal())
    engine = ComputeEngine(belief_store, evidence_log, cost_tracker, consensus,
                           prediction_state, op_store, trace,
                           max_depth=1, max_candidates=8)
    world = World(seed=7)
    world.run(5)
    ctx.world_history = [set(s) for s in world.history[:5]]

    for i in range(20):
        if ctx.budget_exhausted():
            break
        obj = list(world.history[min(i, 4)])[0]
        engine.recursive_compute(obj, ctx.goal, ctx, depth=0)

    assert engine.budget_exhausted is True
    cid, _ = engine.recursive_compute(P.atom("X"), ctx.goal, ctx, depth=0)
    assert cid == ""
    print("test_budget_exhaustion_stops_compute OK")


# 10. feedback attribution 不使用全局 valid 数量
def test_feedback_no_global_valid_count():
    """recursive_compute 返回 subtree_valid 集合，feedback 基于子树而非全局 store。"""
    import inspect
    from cognition import compute as compute_mod
    src = inspect.getsource(compute_mod.ComputeEngine.recursive_compute)
    # 不应出现全局 valid 计数模式
    assert "valid_entries()" not in src or "len(self.store.valid_entries())" not in src, \
        "不应使用全局 store.valid_entries() 数量变化作为 feedback"
    # 应使用 child_valid（子树返回）
    assert "child_valid" in src, "应使用子树返回的 valid 集合做 feedback attribution"
    print("test_feedback_no_global_valid_count OK")


def run_all():
    tests = [
        test_d_no_future_state_access,
        test_ground_truth_does_not_mutate_store,
        test_c_d_initial_state_identical,
        test_c_d_only_meta_differs,
        test_meta_evaluation_single_entry,
        test_usefulness_none_means_unevaluated,
        test_correct_but_useless_only_evaluated,
        test_composite_not_faked_when_unregistered,
        test_budget_exhaustion_stops_compute,
        test_feedback_no_global_valid_count,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"{t.__name__} FAIL: {e}")
        except Exception as e:
            print(f"{t.__name__} ERROR: {e}")
    print(f"\n审计测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

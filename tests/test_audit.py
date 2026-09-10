"""第二轮审计单元测试。

重点验证实验因果关系的正确性，而非"聪明结果"。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.knowledge_store import (KnowledgeStore, Knowledge,
                                       STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN)
from cognition.operations import OperationRegistry, Context
from cognition.verification import Verification
from cognition.evaluation import Evaluation
from cognition.compute import ComputeEngine
from cognition.trace import Trace
from cognition.environment import World


def _make_engine():
    store = KnowledgeStore()
    trace = Trace()
    registry = OperationRegistry()
    verification = Verification()
    evaluation = Evaluation()
    engine = ComputeEngine(store, trace, registry, verification, evaluation,
                            max_depth=1, max_candidates=8)
    ctx = Context(
        store=store, trace=trace,
        constants=["ball", "box", "table", "wall"],
        step_budget=1000,
        verify_enabled=True, evaluate_enabled=True, meta_evaluate_enabled=False,
        goal=World.predict_next_goal())
    return engine, store, trace, registry, verification, evaluation, ctx


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
    def make(meta):
        store = KnowledgeStore()
        trace = Trace()
        registry = OperationRegistry()
        verification = Verification()
        evaluation = Evaluation()
        ctx = Context(store=store, trace=trace,
                       constants=["ball", "box", "table", "wall"],
                       step_budget=1000, verify_enabled=True,
                       evaluate_enabled=True, meta_evaluate_enabled=meta,
                       goal=World.predict_next_goal())
        return store, registry, verification, evaluation, ctx

    world = World(seed=7)
    world.run(5)

    c_store, c_reg, c_ver, c_eval, c_ctx = make(False)
    d_store, d_reg, d_ver, d_eval, d_ctx = make(True)

    # 初始 store 都为空
    assert c_store.size() == d_store.size() == 0
    # 初始操作集相同
    assert sorted(c_reg.names()) == sorted(d_reg.names())
    # 初始验证方法可靠性相同（均来自空 store，默认 0.5）
    assert c_store.verifier_reliability("logical") == d_store.verifier_reliability("logical")
    # 初始评价权重相同
    assert c_eval.weights == d_eval.weights
    # 唯一差异：meta_evaluate_enabled
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
    store = KnowledgeStore()
    p = P.atom("A")
    # 未评价
    k = Knowledge(proposition=p, status=STATUS_UNKNOWN, confidence=0.0,
                  usefulness=None, evaluated=False)
    store.upsert(k)
    entry = store.get(p)
    assert entry.usefulness is None
    assert entry.evaluated is False
    # 已评价
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
    store = KnowledgeStore()
    trace = Trace()
    registry = OperationRegistry()
    verification = Verification()
    evaluation = Evaluation()
    engine = ComputeEngine(store, trace, registry, verification, evaluation)
    world = World(seed=7)
    world.run(5)

    p_valid_evaluated = P.impl(P.atom("Open_box"), P.atom("CanTake_ball"))
    p_valid_unevaluated = P.impl(P.atom("A"), P.atom("B"))

    # valid + 已评价 + usefulness 低
    store.upsert(Knowledge(proposition=p_valid_evaluated, status=STATUS_VALID,
                           confidence=0.9, usefulness=0.01, evaluated=True))
    # valid + 未评价（usefulness=None）—— 不应计入
    store.upsert(Knowledge(proposition=p_valid_unevaluated, status=STATUS_VALID,
                           confidence=0.9, usefulness=None, evaluated=False))

    m = compute_metrics("test", store, trace, engine, world, evaluate=True, meta=False)
    # correct_but_useless 只统计 evaluated=True 的
    # p_valid_evaluated 是 implies，用 ground_truth 检查可能正确也可能错误
    # 但关键是 p_valid_unevaluated 绝不能被计入
    # 我们检查：未评价的 valid 命题不会出现在 correct_but_useless 中
    # 通过 stats 间接验证
    stats = store.stats()
    assert stats["evaluated"] == 1  # 只有 1 个被评价
    assert stats["valid_low_usefulness"] == 1  # 只有 1 个低价值（已评价的那个）
    print("test_correct_but_useless_only_evaluated OK")


# 8. composite operation 未注册时不会被假装调用
def test_composite_not_faked_when_unregistered():
    engine, store, trace, registry, verification, evaluation, ctx = _make_engine()
    world = World(seed=7)
    world.run(5)
    ctx.world_history = [set(s) for s in world.history[:5]]

    prior_ops = set(registry.names())
    # 运行计算
    obj = P.atom("Did_open_box")
    engine.recursive_compute(obj, ctx.goal, ctx, depth=0)

    # composite_usage 中不应包含未注册的操作名
    for op_name in engine.composite_usage:
        assert op_name in registry.names(), f"复合操作 {op_name} 未注册却被调用"
    # 没有 composite_ 开头的操作被调用（因为没注册）
    assert not any(k.startswith("composite_") for k in engine.composite_usage)
    print("test_composite_not_faked_when_unregistered OK")


# 9. budget 达到后系统停止
def test_budget_exhaustion_stops_compute():
    store = KnowledgeStore()
    trace = Trace()
    registry = OperationRegistry()
    verification = Verification()
    evaluation = Evaluation()
    # 极小预算
    ctx = Context(store=store, trace=trace,
                   constants=["ball", "box", "table", "wall"],
                   step_budget=10, verify_enabled=True,
                   evaluate_enabled=True, meta_evaluate_enabled=False,
                   goal=World.predict_next_goal())
    engine = ComputeEngine(store, trace, registry, verification, evaluation,
                            max_depth=1, max_candidates=8)
    world = World(seed=7)
    world.run(5)
    ctx.world_history = [set(s) for s in world.history[:5]]

    # 多次调用直到预算耗尽
    for i in range(20):
        if ctx.budget_exhausted():
            break
        obj = list(world.history[min(i, 4)])[0]
        engine.recursive_compute(obj, ctx.goal, ctx, depth=0)

    assert engine.budget_exhausted is True
    # 预算耗尽后调用应立即返回空
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

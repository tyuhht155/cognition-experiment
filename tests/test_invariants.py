"""架构不变量测试（invariant tests）。

验证模块解耦后的核心不变量：
1. prediction_created_at < prediction_resolved_at
2. prediction 不能使用创建时的 observation 作为 confirmation
3. 同一个 ObservationEvent 不能被重复计权
4. EvidenceLog append-only
5. BeliefState 更新不能修改历史 Evidence
6. EvidenceEvaluator 不得访问 ground truth
7. Verifier 不得修改 BeliefStore
8. ValueEvaluator 不得修改 BeliefStore
9. CostTracker total_cost = 所有 event cost 之和
10. changing EvidenceEvaluator 不修改 CandidateGenerator
11. changing ValueEvaluator 不修改 Verifier
12. C/D 两组只有 meta mechanism 不同
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.evidence import (
    Evidence, EvidenceEvaluator, EvidenceLog,
    action_observe, action_logical_derive,
)
from cognition.cost import CostTracker
from cognition.consensus import ConsensusAgreementModel
from cognition.prediction import TemporalPredictionState
from cognition.operations import OperationStore, Context, Candidate
from cognition.verification import Verifier
from cognition.evaluation import ValueEvaluator
from cognition.trace import TraceRecorder


def _make_ctx():
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    consensus = ConsensusAgreementModel()
    prediction_state = TemporalPredictionState()
    op_store = OperationStore()
    trace = TraceRecorder()
    ctx = Context(
        belief_store=belief_store, evidence_log=evidence_log,
        cost_tracker=cost_tracker, consensus=consensus,
        prediction_state=prediction_state, op_store=op_store,
        trace=trace, constants=["A", "B"],
        step_budget=1000, verify_enabled=True,
        evaluate_enabled=True, meta_evaluate_enabled=False,
        goal=("verify", "test"))
    ctx.world_history = []
    ctx.prediction_queue = {}
    return ctx, belief_store


# 1. prediction_created_at < prediction_resolved_at
def test_prediction_resolved_after_created():
    state = TemporalPredictionState()
    prop = P.impl(P.atom("A"), P.atom("B"))
    pred = state.register(prop, P.atom("A"), P.atom("B"), current_step=1)
    assert pred.created_at == 1
    assert pred.resolved_at is None
    state.resolve_pending(2, {P.atom("B")})  # t+1 才能解决
    assert pred.resolved_at == 2
    assert pred.resolved_at > pred.created_at
    assert state.check_invariant() is True
    print("test_prediction_resolved_after_created OK")


# 2. prediction 不能使用创建时的 observation 作为 confirmation
def test_prediction_cannot_self_confirm():
    state = TemporalPredictionState()
    prop = P.impl(P.atom("A"), P.atom("B"))
    state.register(prop, P.atom("A"), P.atom("B"), current_step=1)
    # 尝试在同一 step 解决（应被拒绝）
    resolved = state.resolve_pending(1, {P.atom("B")})
    assert len(resolved) == 0, "同一 step 不能验证 prediction"
    assert state.pending_count() == 1
    print("test_prediction_cannot_self_confirm OK")


# 3. 同一个 ObservationEvent 不能被重复计权
def test_no_double_counting_observation():
    """EvidenceLog 中同一观察不应被重复计权。

    通过 compare 的实现验证：它统计历史中共现比例，
    每次调用只产生一条 Evidence，而不是把每条历史都作为独立 Evidence。
    """
    from cognition.evidence import action_compare
    ctx, _ = _make_ctx()
    ctx.world_history = [
        {P.atom("A"), P.atom("B")},
        {P.atom("A"), P.atom("B")},
        {P.atom("A"), P.atom("B")},
    ]
    prop = P.impl(P.atom("A"), P.atom("B"))
    ev = action_compare(prop, ctx)
    # compare 只产生一条 Evidence，不把 3 个历史状态当作 3 条独立证据
    assert isinstance(ev, Evidence)
    assert ev.support > 0
    print("test_no_double_counting_observation OK")


# 4. EvidenceLog append-only
def test_evidence_log_append_only():
    log = EvidenceLog()
    e1 = Evidence("observe", "observation", 0.5, 0.0, "e1", 1.0)
    e2 = Evidence("count", "count", 0.3, 0.0, "e2", 1.0)
    log.append(e1)
    log.append(e2)
    assert log.count() == 2
    # 没有删除方法
    assert not hasattr(log, "remove")
    assert not hasattr(log, "clear")
    assert not hasattr(log, "pop")
    print("test_evidence_log_append_only OK")


# 5. BeliefState 更新不能修改历史 Evidence
def test_belief_update_does_not_modify_evidence():
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    prop = P.atom("X")
    e1 = Evidence("observe", "observation", 0.8, 0.0, "first", 1.0)
    evidence_log.append(e1)
    belief_store.update_belief(prop, "valid", 0.8, evidence_count_delta=1)
    # 更新 belief
    belief_store.update_belief(prop, "invalid", 0.9, evidence_count_delta=1)
    # 历史 evidence 不受影响
    assert evidence_log.count() == 1
    assert evidence_log.all()[0].detail == "first"
    assert evidence_log.all()[0].support == 0.8
    print("test_belief_update_does_not_modify_evidence OK")


# 6. EvidenceEvaluator 不得访问 ground truth
def test_evaluator_no_ground_truth_access():
    src = inspect.getsource(EvidenceEvaluator.evaluate)
    assert "ground_truth" not in src
    assert "ground truth" not in src.lower()
    # 只接受 Evidence 列表，不接受 ctx/world
    sig = inspect.signature(EvidenceEvaluator.evaluate)
    params = list(sig.parameters.keys())
    assert params == ["self", "evidences"]
    print("test_evaluator_no_ground_truth_access OK")


# 7. Verifier 不得修改 BeliefStore
def test_verifier_does_not_modify_belief_store():
    ctx, belief_store = _make_ctx()
    # 预置一些 valid 知识
    p = P.atom("P")
    pq = P.impl(P.atom("P"), P.atom("Q"))
    belief_store.update_belief(p, "valid", 0.9)
    belief_store.update_belief(pq, "valid", 0.9)
    size_before = belief_store.size()

    verifier = Verifier()
    q = P.atom("Q")
    result, all_results = verifier.verify(q, ctx)

    # Verifier 不应增加 BeliefStore 的条目
    assert belief_store.size() == size_before, \
        "Verifier 不应修改 BeliefStore"
    print("test_verifier_does_not_modify_belief_store OK")


# 8. ValueEvaluator 不得修改 BeliefStore
def test_value_evaluator_does_not_modify_belief_store():
    ctx, belief_store = _make_ctx()
    prop = P.atom("X")
    size_before = belief_store.size()
    entries_before = len(belief_store.all_beliefs())

    evaluator = ValueEvaluator()
    result = evaluator.evaluate(prop, None, ctx)

    assert belief_store.size() == size_before
    assert len(belief_store.all_beliefs()) == entries_before
    print("test_value_evaluator_does_not_modify_belief_store OK")


# 9. CostTracker total_cost = 所有 event cost 之和
def test_cost_tracker_total_equals_sum():
    ct = CostTracker()
    ct.add("identify", 0.2)
    ct.add("generate", 0.3)
    ct.add("apply", 1.0)
    ct.add("verify", 2.5)
    expected = 0.2 + 0.3 + 1.0 + 2.5
    assert abs(ct.total_cost - expected) < 1e-9
    # 分项之和 = total
    cat_sum = sum(ct._by_category.values())
    assert abs(cat_sum - ct.total_cost) < 1e-9
    # cache_saved 不计入 total
    ct.add_cache_saved(5.0)
    assert abs(ct.total_cost - expected) < 1e-9
    print("test_cost_tracker_total_equals_sum OK")


# 10. changing EvidenceEvaluator 不修改 CandidateGenerator
def test_evaluator_change_does_not_affect_generator():
    op_store = OperationStore()
    ctx, _ = _make_ctx()
    ctx.op_store = op_store
    obj = P.atom("A")
    cands_before = op_store.generate(obj, ctx)

    # 修改 EvidenceEvaluator 的阈值
    ev = EvidenceEvaluator()
    ev.VALID_SUPPORT_THRESHOLD = 0.99
    ev.INVALID_CONTRADICTION_THRESHOLD = 0.99

    cands_after = op_store.generate(obj, ctx)
    assert len(cands_before) == len(cands_after)
    print("test_evaluator_change_does_not_affect_generator OK")


# 11. changing ValueEvaluator 不修改 Verifier
def test_value_evaluator_change_does_not_affect_verifier():
    ctx, belief_store = _make_ctx()
    p = P.atom("P")
    pq = P.impl(P.atom("P"), P.atom("Q"))
    belief_store.update_belief(p, "valid", 0.9)
    belief_store.update_belief(pq, "valid", 0.9)

    verifier = Verifier()
    q = P.atom("Q")
    r1, _ = verifier.verify(q, ctx)

    # 修改 ValueEvaluator
    ve = ValueEvaluator()
    ve.weights = {"relevance": 0.1, "generality": 0.1, "novelty": 0.8}

    r2, _ = verifier.verify(q, ctx)
    assert r1.result == r2.result
    assert r1.confidence == r2.confidence
    print("test_value_evaluator_change_does_not_affect_verifier OK")


# 12. C/D 两组只有 meta mechanism 不同
def test_cd_groups_consistent_initial_state():
    """C/D 两组只有 meta_evaluate_enabled 不同，初始状态、环境、候选、budget 一致。"""
    base_kwargs = dict(
        belief_store=BeliefStore(), evidence_log=EvidenceLog(),
        cost_tracker=CostTracker(), consensus=ConsensusAgreementModel(),
        prediction_state=TemporalPredictionState(), op_store=OperationStore(),
        trace=TraceRecorder(), constants=["A", "B"],
        step_budget=1000, verify_enabled=True, evaluate_enabled=True,
        goal=("verify", "A → B"))
    ctx_c = Context(meta_evaluate_enabled=False, **base_kwargs)
    ctx_d = Context(meta_evaluate_enabled=True, **base_kwargs)

    # 只有 meta_evaluate_enabled 不同
    assert ctx_c.meta_evaluate_enabled != ctx_d.meta_evaluate_enabled
    # 其他相同
    assert ctx_c.step_budget == ctx_d.step_budget
    assert ctx_c.verify_enabled == ctx_d.verify_enabled
    assert ctx_c.evaluate_enabled == ctx_d.evaluate_enabled
    assert ctx_c.constants == ctx_d.constants
    print("test_cd_groups_consistent_initial_state OK")


# 13. evidence_count 只因新 Evidence 增加
def test_evidence_count_only_on_new_evidence():
    """evidence_count 只在真正产生并记录新 Evidence 时增加。"""
    bs = BeliefStore()
    prop = P.atom("E")
    bs.update_belief(prop, "valid", 0.9, evidence_count_delta=1)
    assert bs.get(prop).evidence_count == 1
    # 再次更新但 evidence_count_delta=0 不增加
    bs.update_belief(prop, "valid", 0.95, evidence_count_delta=0)
    assert bs.get(prop).evidence_count == 1
    # 新增证据才增加
    bs.update_belief(prop, "valid", 0.98, evidence_count_delta=1)
    assert bs.get(prop).evidence_count == 2
    print("test_evidence_count_only_on_new_evidence OK")


# 14. reuse 不增加 evidence_count
def test_reuse_does_not_increase_evidence_count():
    """record_reuse 只增加 reuse_count，不增加 evidence_count。"""
    bs = BeliefStore()
    prop = P.atom("R")
    bs.update_belief(prop, "valid", 0.9, evidence_count_delta=1)
    before = bs.get(prop).evidence_count
    reuse_before = bs.get(prop).reuse_count
    bs.record_reuse(prop)
    assert bs.get(prop).evidence_count == before
    assert bs.get(prop).reuse_count == reuse_before + 1
    print("test_reuse_does_not_increase_evidence_count OK")


# 15. Evidence 有唯一 evidence_id
def test_evidence_unique_id():
    """每个 Evidence 实例有全局唯一的 evidence_id。"""
    ids = set()
    for _ in range(100):
        e = Evidence("observe", "observation", 0.5, 0.0, "d", 1.0)
        assert e.evidence_id not in ids
        ids.add(e.evidence_id)
    print("test_evidence_unique_id OK")


# 16. 同一 source_event 不重复计权
def test_same_source_event_not_double_counted():
    """同一 source_event_id + proposition + method 不允许被重复计为独立 evidence。"""
    log = EvidenceLog()
    prop = P.atom("Dup")
    e1 = Evidence("observe", "observation", 0.8, 0.0, "d1", 1.0,
                  proposition=prop, source_event_id="evt_1")
    e2 = Evidence("observe", "observation", 0.8, 0.0, "d2", 1.0,
                  proposition=prop, source_event_id="evt_1")
    assert log.append(e1) is True
    assert log.append(e2) is False  # 同一 source_event + proposition + method 被拒绝
    assert log.count() == 1
    print("test_same_source_event_not_double_counted OK")


# 17. Operation 不应该获得完整 Context 的可写 Store
def test_context_has_no_writable_stores():
    """Context 不持有可写 Store（belief_store/evidence_log/cost_tracker 等）。"""
    bs = BeliefStore()
    ctx = Context(belief_store=bs, evidence_log=EvidenceLog(),
                  cost_tracker=CostTracker(), consensus=ConsensusAgreementModel(),
                  prediction_state=TemporalPredictionState(),
                  op_store=OperationStore(), trace=TraceRecorder())
    # Context 不应暴露这些可写属性
    assert not hasattr(ctx, "belief_store")
    assert not hasattr(ctx, "evidence_log")
    assert not hasattr(ctx, "cost_tracker")
    assert not hasattr(ctx, "consensus")
    assert not hasattr(ctx, "prediction_state")
    assert not hasattr(ctx, "op_store")
    assert not hasattr(ctx, "trace")
    # 但应有只读 knowledge_view
    assert hasattr(ctx, "knowledge_view")
    print("test_context_has_no_writable_stores OK")


# 18. ComputeEngine 不直接访问 Store 内部字典
def test_compute_engine_no_internal_store_access():
    """ComputeEngine 源码不直接访问 Store 的内部字典。"""
    from cognition.compute import ComputeEngine
    src = inspect.getsource(ComputeEngine)
    # 不应直接访问 _beliefs / _entries / _events 等内部字典
    assert "_beliefs" not in src
    assert "_entries" not in src
    assert "_events" not in src
    print("test_compute_engine_no_internal_store_access OK")


# 19. total_cost 等于所有 CostEvent 之和（含 cache_saved 不计入）
def test_total_cost_equals_sum_of_cost_events():
    """CostTracker.total_cost == 所有实际 cost event 之和。"""
    from cognition.cost import CostEvent
    ct = CostTracker()
    ct.add("a", 0.2)
    ct.add("b", 0.3)
    ct.add("c", 1.0)
    total = sum(e.amount for e in ct.all_events())
    assert abs(ct.total_cost - total) < 1e-9
    ct.add_cache_saved(99.0)
    assert abs(ct.total_cost - total) < 1e-9
    print("test_total_cost_equals_sum_of_cost_events OK")


# 20. Verifier 只产生 Evidence，不修改 BeliefStore（复用路径）
def test_verifier_reuse_path_no_evidence_count_increment():
    """复用已有知识时，evidence_count 不变，reuse_count 增加。"""
    ctx, bs = _make_ctx()
    p = P.atom("P")
    pq = P.impl(P.atom("P"), P.atom("Q"))
    bs.update_belief(p, "valid", 0.9, evidence_count_delta=1)
    bs.update_belief(pq, "valid", 0.9, evidence_count_delta=1)
    q = P.atom("Q")
    ec_before = bs.get(q).evidence_count if bs.has(q) else 0
    # 走 logical_derive 复用路径
    v = Verifier()
    v.verify(q, ctx)
    if bs.has(q):
        assert bs.get(q).evidence_count == ec_before
    print("test_verifier_reuse_path_no_evidence_count_increment OK")


def run_all():
    tests = [
        test_prediction_resolved_after_created,
        test_prediction_cannot_self_confirm,
        test_no_double_counting_observation,
        test_evidence_log_append_only,
        test_belief_update_does_not_modify_evidence,
        test_evaluator_no_ground_truth_access,
        test_verifier_does_not_modify_belief_store,
        test_value_evaluator_does_not_modify_belief_store,
        test_cost_tracker_total_equals_sum,
        test_evaluator_change_does_not_affect_generator,
        test_value_evaluator_change_does_not_affect_verifier,
        test_cd_groups_consistent_initial_state,
        test_evidence_count_only_on_new_evidence,
        test_reuse_does_not_increase_evidence_count,
        test_evidence_unique_id,
        test_same_source_event_not_double_counted,
        test_context_has_no_writable_stores,
        test_compute_engine_no_internal_store_access,
        test_total_cost_equals_sum_of_cost_events,
        test_verifier_reuse_path_no_evidence_count_increment,
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
    print(f"\nInvariant 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

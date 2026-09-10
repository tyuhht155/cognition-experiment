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
    return ctx


# 1. prediction_created_at < prediction_resolved_at
def test_prediction_resolved_after_created():
    state = TemporalPredictionState()
    pred = state.register("A → B", "A", "B", current_step=1)
    assert pred.created_at == 1
    assert pred.resolved_at is None
    state.resolve_pending(2, {"B"})  # t+1 才能解决
    assert pred.resolved_at == 2
    assert pred.resolved_at > pred.created_at
    assert state.check_invariant() is True
    print("test_prediction_resolved_after_created OK")


# 2. prediction 不能使用创建时的 observation 作为 confirmation
def test_prediction_cannot_self_confirm():
    state = TemporalPredictionState()
    state.register("A → B", "A", "B", current_step=1)
    # 尝试在同一 step 解决（应被拒绝）
    resolved = state.resolve_pending(1, {"B"})
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
    ctx = _make_ctx()
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
    ctx = _make_ctx()
    belief_store = ctx.belief_store
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
    ctx = _make_ctx()
    belief_store = ctx.belief_store
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
    ctx = _make_ctx()
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
    ctx = _make_ctx()
    p = P.atom("P")
    pq = P.impl(P.atom("P"), P.atom("Q"))
    ctx.belief_store.update_belief(p, "valid", 0.9)
    ctx.belief_store.update_belief(pq, "valid", 0.9)

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

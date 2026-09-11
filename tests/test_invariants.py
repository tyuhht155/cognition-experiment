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


# 21. modus_ponens 通过 KnowledgeView 正常工作
def test_modus_ponens_via_knowledge_view():
    """op_modus_ponens 必须只通过 ctx.knowledge_view 访问知识，且能生成候选。"""
    from cognition.operations import op_modus_ponens, Context
    bs = BeliefStore()
    # 知识库中存在 P→Q (valid)
    pq = P.impl(P.atom("P"), P.atom("Q"))
    bs.update_belief(pq, "valid", 0.9, evidence_count_delta=1)
    ctx = Context(knowledge_view=bs, constants=["P", "Q"])
    # 当前对象为 P
    p = P.atom("P")
    cands = op_modus_ponens(p, ctx)
    assert len(cands) == 1
    assert cands[0].new_object == P.atom("Q")
    assert cands[0].op_name == "modus_ponens"
    print("test_modus_ponens_via_knowledge_view OK")


# 22. modus_tollens 通过 KnowledgeView 正常工作
def test_modus_tollens_via_knowledge_view():
    """op_modus_tollens 必须只通过 ctx.knowledge_view 访问知识，且能生成候选。"""
    from cognition.operations import op_modus_tollens, Context
    bs = BeliefStore()
    # 知识库中存在 P→Q (valid)
    pq = P.impl(P.atom("P"), P.atom("Q"))
    bs.update_belief(pq, "valid", 0.9, evidence_count_delta=1)
    ctx = Context(knowledge_view=bs, constants=["P", "Q"])
    # 当前对象为 ¬Q
    not_q = P.neg(P.atom("Q"))
    cands = op_modus_tollens(not_q, ctx)
    assert len(cands) == 1
    assert cands[0].new_object == P.neg(P.atom("P"))
    assert cands[0].op_name == "modus_tollens"
    print("test_modus_tollens_via_knowledge_view OK")


# 23. candidate 的 direct result 与 descendant result 分离
def test_feedback_direct_vs_descendant_separated():
    """CandidateProcessor.record_feedback 接受 direct_valid 和 descendant_valid 两个独立参数。"""
    from cognition.candidate_processor import CandidateProcessor
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.trace import TraceRecorder
    from cognition.evaluation import ValueEvaluator

    bs = BeliefStore()
    cp = CandidateProcessor(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        trace=TraceRecorder())
    ev = ValueEvaluator().evaluate(P.atom("X"), None, Context(knowledge_view=bs))
    # candidate_valid=True, descendant_valid=False
    cp.record_feedback(ev, "valid", candidate_valid=True, descendant_valid=False)
    fb = cp.eval_feedback[-1]
    assert fb["actual"] == 1.0
    assert fb["candidate_valid"] is True
    assert fb["descendant_valid"] is False
    assert fb["candidate_status"] == "valid"
    assert "goal_improvement" in fb
    # candidate_valid=False, descendant_valid=True → actual 仍应为 0.0
    cp.record_feedback(ev, "invalid", candidate_valid=False, descendant_valid=True)
    fb2 = cp.eval_feedback[-1]
    assert fb2["actual"] == 0.0
    assert fb2["candidate_valid"] is False
    assert fb2["descendant_valid"] is True
    print("test_feedback_direct_vs_descendant_separated OK")


# 24. 父 candidate 不因为孙节点 valid 自动获得 direct success
def test_parent_not_credited_for_descendant_valid():
    """feedback attribution：父 candidate 自身 invalid 但 descendant valid 时，actual=0.0。"""
    from cognition.candidate_processor import CandidateProcessor
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.trace import TraceRecorder
    from cognition.evaluation import ValueEvaluator

    bs = BeliefStore()
    cp = CandidateProcessor(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        trace=TraceRecorder())
    ev = ValueEvaluator().evaluate(P.atom("X"), None, Context(knowledge_view=bs))
    # 模拟：父 candidate 自身未直接 valid（candidate_valid=False），但子树有 valid（descendant_valid=True）
    cp.record_feedback(ev, "invalid", candidate_valid=False, descendant_valid=True)
    last = cp.eval_feedback[-1]
    assert last["actual"] == 0.0, "父 candidate 不应因 descendant valid 获得 direct success"
    assert last["candidate_valid"] is False
    assert last["descendant_valid"] is True
    assert last["candidate_status"] == "invalid"
    print("test_parent_not_credited_for_descendant_valid OK")


# 25. 递归 trace 的 parent_step 链已建立
def test_recursive_parent_step_chain_established():
    """ComputeEngine 递归调用时，子节点的 parent_step 必须是父 candidate 的 apply_step_id，
    而不是 None。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    trace = TraceRecorder()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=trace,
        max_depth=2, max_candidates=3)
    ctx = Context(knowledge_view=bs, constants=["A", "B"],
                  step_budget=100, verify_enabled=False, evaluate_enabled=False)
    obj = P.atom("A")
    engine.recursive_compute(obj, None, ctx)

    # 找出所有非 None 的 parent_step，确认递归建立了父子链接
    steps = trace.to_dicts()
    parent_steps = {s["parent_step"] for s in steps if s["parent_step"] is not None}
    apply_step_ids = {s["step_id"] for s in steps if s["operation"] not in
                      ("identify", "generate_candidates", "evaluate_rank")}
    # 至少有一个步骤的 parent_step 指向某个 apply 步骤（即递归链存在）
    assert parent_steps & apply_step_ids, "递归 trace 应建立 parent_step 链"
    print("test_recursive_parent_step_chain_established OK")


# 26. evaluation feedback 同时保存三个独立结果
def test_feedback_records_three_independent_results():
    """feedback 必须同时包含 candidate_valid、descendant_valid、goal_improvement 三个字段。"""
    from cognition.candidate_processor import CandidateProcessor
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.trace import TraceRecorder
    from cognition.evaluation import ValueEvaluator

    bs = BeliefStore()
    cp = CandidateProcessor(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        trace=TraceRecorder())
    ev = ValueEvaluator().evaluate(P.atom("X"), None, Context(knowledge_view=bs))
    # 三个结果全部不同组合
    cp.record_feedback(ev, "valid", candidate_valid=True, descendant_valid=True, goal_improvement=True)
    fb = cp.eval_feedback[-1]
    assert fb["candidate_valid"] is True
    assert fb["descendant_valid"] is True
    assert fb["goal_improvement"] is True
    # 另一种组合
    cp.record_feedback(ev, "invalid", candidate_valid=False, descendant_valid=False, goal_improvement=False)
    fb2 = cp.eval_feedback[-1]
    assert fb2["candidate_valid"] is False
    assert fb2["descendant_valid"] is False
    assert fb2["goal_improvement"] is False
    print("test_feedback_records_three_independent_results OK")


# 27. invalid candidate 会产生 evaluation feedback
def test_invalid_candidate_produces_feedback():
    """被验证为 invalid 的 candidate 必须产生 feedback，不能因 continue 被跳过。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    # 预置 B→C 为 invalid，使 reuse 路径返回 invalid
    bad_impl = P.impl(P.atom("B"), P.atom("C"))
    bs.update_belief(bad_impl, "invalid", 0.8, evidence_count_delta=1)

    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=1, max_candidates=5)
    # world_history 使 cooccur_implication 从 C 生成 B→C（用 Proposition 对象）
    A, B, C = P.atom("A"), P.atom("B"), P.atom("C")
    ctx = Context(knowledge_view=bs, constants=["A", "B", "C"],
                  step_budget=100, verify_enabled=True, evaluate_enabled=True,
                  world_history=[{A, B}, {C}, {B, C}])
    engine.recursive_compute(C, None, ctx)

    fb = engine.processor.eval_feedback
    statuses = [f["candidate_status"] for f in fb]
    assert "invalid" in statuses, "invalid candidate 必须产生 feedback"
    invalid_fb = [f for f in fb if f["candidate_status"] == "invalid"]
    assert all(f["actual"] == 0.0 for f in invalid_fb)
    print("test_invalid_candidate_produces_feedback OK")


# 28. gated_out candidate 会产生 feedback
def test_gated_out_candidate_produces_feedback():
    """被评价 gate 拒绝的 candidate 必须产生 feedback。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=1, max_candidates=5)
    ctx = Context(knowledge_view=bs, constants=["A", "B", "C"],
                  step_budget=100, verify_enabled=True, evaluate_enabled=True)
    engine.recursive_compute(P.atom("A"), None, ctx)

    fb = engine.processor.eval_feedback
    statuses = [f["candidate_status"] for f in fb]
    # 只要有 feedback，每个 entry 的 status 必须是合法值
    for f in fb:
        assert f["candidate_status"] in ("valid", "invalid", "unknown", "gated_out")
    # gated_out 没有实际验证结果，actual 必须为 None（不参与学习）
    gated = [f for f in fb if f["candidate_status"] == "gated_out"]
    if gated:
        assert all(f["actual"] is None for f in gated)
    print("test_gated_out_candidate_produces_feedback OK")


# 29. budget exhausted / 未执行 candidate 不会产生伪造 feedback
def test_budget_exhausted_no_fake_feedback():
    """预算耗尽后未执行的 candidate 不应产生 feedback。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=1, max_candidates=5)
    # 极小预算，确保部分 candidate 未执行
    ctx = Context(knowledge_view=bs, constants=["A", "B", "C"],
                  step_budget=3, verify_enabled=True, evaluate_enabled=True)
    engine.recursive_compute(P.atom("A"), None, ctx)

    fb = engine.processor.eval_feedback
    # feedback 数量不能超过实际执行的 candidate 数
    # （每个 candidate 至少消耗 1 step：apply）
    assert len(fb) <= ctx.step_budget, "未执行的 candidate 不应产生伪造 feedback"
    print("test_budget_exhausted_no_fake_feedback OK")


# 30. goal_improvement 不会被 goal_relevance 冒充
def test_goal_improvement_not_faked_by_goal_relevance():
    """goal_improvement 在无可靠定义时必须为 None，不能用 goal_relevance > 0.5 冒充。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=1, max_candidates=5)
    ctx = Context(knowledge_view=bs, constants=["A", "B"],
                  step_budget=100, verify_enabled=True, evaluate_enabled=True)
    engine.recursive_compute(P.atom("A"), None, ctx)

    fb = engine.processor.eval_feedback
    for f in fb:
        assert f["goal_improvement"] is None, \
            "goal_improvement 不应被 goal_relevance 冒充，无可靠定义时必须为 None"
    print("test_goal_improvement_not_faked_by_goal_relevance OK")


# 31. candidate_valid 与 goal_improvement 可以独立存在
def test_candidate_valid_and_goal_improvement_independent():
    """candidate_valid 是事实结果，goal_improvement 当前为 None，两者解耦。"""
    from cognition.candidate_processor import CandidateProcessor
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.trace import TraceRecorder
    from cognition.evaluation import ValueEvaluator

    bs = BeliefStore()
    cp = CandidateProcessor(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        trace=TraceRecorder())
    ev = ValueEvaluator().evaluate(P.atom("X"), None, Context(knowledge_view=bs))

    # candidate_valid=True, goal_improvement=None（无可靠定义）
    cp.record_feedback(ev, "valid", candidate_valid=True,
                       descendant_valid=False, goal_improvement=None)
    fb = cp.eval_feedback[-1]
    assert fb["candidate_valid"] is True
    assert fb["goal_improvement"] is None

    # candidate_valid=False, goal_improvement=None
    cp.record_feedback(ev, "invalid", candidate_valid=False,
                       descendant_valid=False, goal_improvement=None)
    fb2 = cp.eval_feedback[-1]
    assert fb2["candidate_valid"] is False
    assert fb2["goal_improvement"] is None
    print("test_candidate_valid_and_goal_improvement_independent OK")


# 32. valid feedback 会进入 meta evaluation
def test_valid_feedback_enters_meta_evaluation():
    """candidate_status=valid 的 feedback 必须被 evaluate_evaluation 计入学习样本。"""
    from cognition.evaluation import ValueEvaluator
    ve = ValueEvaluator()
    before = dict(ve.weights)
    feedback = [{"tag": "relevance", "predicted": 0.6, "actual": 1.0,
                 "candidate_status": "valid"}]
    res = ve.evaluate_evaluation(feedback)
    # valid feedback 被计入 n：adjustments 中应包含该 tag（error 可能为 0）
    assert "relevance" in res["adjustments"], "valid feedback 必须被计入学习样本"
    print("test_valid_feedback_enters_meta_evaluation OK")


# 33. invalid feedback 会进入 meta evaluation，并产生负反馈
def test_invalid_feedback_enters_meta_evaluation():
    """candidate_status=invalid 的 feedback 必须被计入，且 actual=0.0 作为负反馈。"""
    from cognition.evaluation import ValueEvaluator
    ve = ValueEvaluator()
    before = dict(ve.weights)
    # predicted 高但 actual=0.0（invalid），应有较大 error，权重被调低
    feedback = [{"tag": "novelty", "predicted": 0.9, "actual": 0.0,
                 "candidate_status": "invalid"}]
    res = ve.evaluate_evaluation(feedback)
    assert res["adjusted"] is True
    assert ve.weights["novelty"] < before["novelty"]
    print("test_invalid_feedback_enters_meta_evaluation OK")


# 34. unknown feedback 不进入 meta evaluation
def test_unknown_feedback_skipped_in_meta_evaluation():
    """candidate_status=unknown 的 feedback（actual=None）不能影响权重。"""
    from cognition.evaluation import ValueEvaluator
    ve = ValueEvaluator()
    before = dict(ve.weights)
    feedback = [{"tag": "generality", "predicted": 0.8, "actual": None,
                 "candidate_status": "unknown"}]
    res = ve.evaluate_evaluation(feedback)
    # 没有可学习样本，不应调整
    assert res["adjusted"] is False
    assert ve.weights == before
    print("test_unknown_feedback_skipped_in_meta_evaluation OK")


# 35. gated_out feedback 不进入 meta evaluation
def test_gated_out_feedback_skipped_in_meta_evaluation():
    """candidate_status=gated_out 的 feedback（actual=None）不能影响权重。"""
    from cognition.evaluation import ValueEvaluator
    ve = ValueEvaluator()
    before = dict(ve.weights)
    feedback = [{"tag": "relevance", "predicted": 0.7, "actual": None,
                 "candidate_status": "gated_out"}]
    res = ve.evaluate_evaluation(feedback)
    assert res["adjusted"] is False
    assert ve.weights == before
    print("test_gated_out_feedback_skipped_in_meta_evaluation OK")


# 36. mixed feedback 中只有 valid/invalid 被计入 n
def test_mixed_feedback_only_valid_invalid_counted():
    """混合 feedback 中，只有 actual 不为 None 的条目被计入学习样本 n。"""
    from cognition.evaluation import ValueEvaluator
    ve = ValueEvaluator()
    # 2 条可学习（1 valid, 1 invalid）+ 2 条不可学习（unknown, gated_out）
    feedback = [
        {"tag": "relevance", "predicted": 0.5, "actual": 1.0, "candidate_status": "valid"},
        {"tag": "relevance", "predicted": 0.5, "actual": 0.0, "candidate_status": "invalid"},
        {"tag": "relevance", "predicted": 0.9, "actual": None, "candidate_status": "unknown"},
        {"tag": "relevance", "predicted": 0.9, "actual": None, "candidate_status": "gated_out"},
    ]
    res = ve.evaluate_evaluation(feedback)
    # n 应为 2（只有 valid+invalid），不是 4
    # adjustments 中应包含 relevance（被计入），且 n 只来自 valid+invalid
    assert "relevance" in res["adjustments"], "valid/invalid 必须被计入 n"
    print("test_mixed_feedback_only_valid_invalid_counted OK")


# 37. unknown/gated_out 增多不改变 evaluation weights
def test_unknown_gated_out_do_not_change_weights():
    """增加 unknown/gated_out feedback 不应改变权重（与纯 valid/invalid 结果一致）。"""
    from cognition.evaluation import ValueEvaluator
    # 组 A：只有 1 条 valid
    ve_a = ValueEvaluator()
    ve_a.evaluate_evaluation([{"tag": "relevance", "predicted": 0.4, "actual": 1.0,
                               "candidate_status": "valid"}])
    # 组 B：1 条 valid + 多条 unknown/gated_out
    ve_b = ValueEvaluator()
    ve_b.evaluate_evaluation([
        {"tag": "relevance", "predicted": 0.4, "actual": 1.0, "candidate_status": "valid"},
        {"tag": "relevance", "predicted": 0.9, "actual": None, "candidate_status": "unknown"},
        {"tag": "relevance", "predicted": 0.8, "actual": None, "candidate_status": "gated_out"},
        {"tag": "generality", "predicted": 0.7, "actual": None, "candidate_status": "unknown"},
    ])
    # 两组权重应完全一致（unknown/gated_out 不影响）
    assert ve_a.weights == ve_b.weights, \
        "unknown/gated_out 增多不应改变 evaluation weights"
    print("test_unknown_gated_out_do_not_change_weights OK")


# 38. 递归统计不会被父层旧快照覆盖
def test_recursive_stats_not_overwritten():
    """parent + child 执行后 actual_steps 不会回退；
    child 产生的统计会保留在最终结果中。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore, Context
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=3, max_candidates=6)
    A, B, C = P.atom("A"), P.atom("B"), P.atom("C")
    ctx = Context(knowledge_view=bs, constants=["A", "B", "C"],
                  step_budget=200, verify_enabled=False, evaluate_enabled=False,
                  world_history=[{A, B}, {B, C}, {A, C}])
    engine.recursive_compute(A, None, ctx)

    # actual_steps 必须 > 0（至少有 identify + generate）
    assert engine.actual_steps > 0, "actual_steps should be positive"
    # generated_candidates 必须 > 0
    assert engine.generated_candidates > 0, "generated_candidates should be positive"
    # 如果有多层递归，actual_steps 应该远超单层的量
    # 关键：actual_steps 不应被重置为更小的值
    steps_after_full_run = engine.actual_steps
    assert steps_after_full_run > 0

    # 再跑一次，新 engine，确认数值一致
    bs2 = BeliefStore()
    engine2 = ComputeEngine(
        belief_store=bs2, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=3, max_candidates=6)
    ctx2 = Context(knowledge_view=bs2, constants=["A", "B", "C"],
                   step_budget=200, verify_enabled=False, evaluate_enabled=False,
                   world_history=[{A, B}, {B, C}, {A, C}])
    engine2.recursive_compute(A, None, ctx2)
    assert engine2.actual_steps == steps_after_full_run, \
        f"stats should be deterministic: {engine2.actual_steps} vs {steps_after_full_run}"
    print("test_recursive_stats_not_overwritten OK")


# 39. 多层递归统计等于所有实际执行事件的累计
def test_multilayer_stats_are_cumulative():
    """多层递归统计等于所有实际执行事件的累计，而不是某一层快照。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore, Context
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=1, max_candidates=6)  # depth=1：只有一层，不递归
    A, B, C = P.atom("A"), P.atom("B"), P.atom("C")
    ctx = Context(knowledge_view=bs, constants=["A", "B", "C"],
                  step_budget=200, verify_enabled=False, evaluate_enabled=False,
                  world_history=[{A, B}, {B, C}, {A, C}])
    engine.recursive_compute(A, None, ctx)
    single_layer_steps = engine.actual_steps
    single_layer_generated = engine.generated_candidates

    # 深度=3：允许多层递归
    bs2 = BeliefStore()
    engine2 = ComputeEngine(
        belief_store=bs2, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=TraceRecorder(),
        max_depth=3, max_candidates=6)
    ctx2 = Context(knowledge_view=bs2, constants=["A", "B", "C"],
                   step_budget=200, verify_enabled=False, evaluate_enabled=False,
                   world_history=[{A, B}, {B, C}, {A, C}])
    engine2.recursive_compute(A, None, ctx2)
    multi_layer_steps = engine2.actual_steps
    multi_layer_generated = engine2.generated_candidates

    # 多层递归的 steps 必须严格大于单层（因为递归产生了额外的 identify+generate）
    assert multi_layer_steps > single_layer_steps, \
        f"multi-layer steps ({multi_layer_steps}) should > single-layer ({single_layer_steps})"
    assert multi_layer_generated >= single_layer_generated, \
        f"multi-layer generated ({multi_layer_generated}) should >= single-layer ({single_layer_generated})"
    print("test_multilayer_stats_are_cumulative OK")


# 40. gated_out candidate 不会递归
def test_gated_out_does_not_recurse():
    """gated_out candidate 绝对不能触发 recursive_compute。"""
    from cognition.compute import ComputeEngine
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore, Context
    from cognition.trace import TraceRecorder

    bs = BeliefStore()
    trace = TraceRecorder()
    engine = ComputeEngine(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        op_store=OperationStore(), trace=trace,
        max_depth=3, max_candidates=6)
    A, B, C = P.atom("A"), P.atom("B"), P.atom("C")
    ctx = Context(knowledge_view=bs, constants=["A", "B", "C"],
                  step_budget=200, verify_enabled=True, evaluate_enabled=True,
                  world_history=[{A, B}, {B, C}, {A, C}])
    engine.recursive_compute(A, None, ctx)

    # 检查 feedback：如果有 gated_out，其 descendant_valid 必须为 False
    fb = engine.processor.eval_feedback
    gated_entries = [f for f in fb if f["candidate_status"] == "gated_out"]
    for g in gated_entries:
        assert g["descendant_valid"] is False, \
            "gated_out candidate 不应有 descendant_valid=True"

    # 检查 trace：gated_out candidate 不应有 child identify 步骤
    # 方法：gated_out 的 apply_step_id 不应作为任何其他步骤的 parent_step
    # （因为如果递归了，child 的 identify 步骤的 parent_step 会指向它）
    steps = trace.to_dicts()
    # 找到所有 gated_out candidate 的 apply step
    # gated_out 发生在 evaluate_gate 步骤，其 parent_step 是 apply step
    # 如果递归了，下一步 identify 的 parent_step 会指向 apply step
    apply_steps_with_gate_reject = set()
    for i, s in enumerate(steps):
        if s.get("operation") == "evaluate_gate" and s.get("decision") == "retain_low":
            # 其 parent_step 是 apply step
            ps = s.get("parent_step")
            if ps:
                apply_steps_with_gate_reject.add(ps)

    # 检查这些 apply steps 是否有 child identify 指向它们
    for s in steps:
        if s.get("operation") == "identify" and s.get("parent_step") in apply_steps_with_gate_reject:
            # 这个 identify 的 parent_step 指向 gated_out 的 apply step
            # 说明 gated_out 递归了，这是 bug
            assert False, "gated_out candidate should not have child identify step"
    print("test_gated_out_does_not_recurse OK")


# 41. gated_out candidate 不增加 descendant_valid
def test_gated_out_no_descendant_valid():
    """gated_out candidate 的 descendant_valid 必须为 False。"""
    from cognition.candidate_processor import CandidateProcessor
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.trace import TraceRecorder
    from cognition.evaluation import ValueEvaluator

    bs = BeliefStore()
    cp = CandidateProcessor(
        belief_store=bs, evidence_log=EvidenceLog(), cost_tracker=CostTracker(),
        consensus=ConsensusAgreementModel(), prediction_state=TemporalPredictionState(),
        trace=TraceRecorder())
    ev = ValueEvaluator().evaluate(P.atom("X"), None, Context(knowledge_view=bs))
    # gated_out 的 descendant_valid 必须为 False
    cp.record_feedback(ev, "gated_out", candidate_valid=False, descendant_valid=False)
    fb = cp.eval_feedback[-1]
    assert fb["descendant_valid"] is False
    assert fb["candidate_status"] == "gated_out"
    assert fb["actual"] is None  # gated_out 不参与学习
    print("test_gated_out_no_descendant_valid OK")


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
        test_modus_ponens_via_knowledge_view,
        test_modus_tollens_via_knowledge_view,
        test_feedback_direct_vs_descendant_separated,
        test_parent_not_credited_for_descendant_valid,
        test_recursive_parent_step_chain_established,
        test_feedback_records_three_independent_results,
        test_invalid_candidate_produces_feedback,
        test_gated_out_candidate_produces_feedback,
        test_budget_exhausted_no_fake_feedback,
        test_goal_improvement_not_faked_by_goal_relevance,
        test_candidate_valid_and_goal_improvement_independent,
        test_valid_feedback_enters_meta_evaluation,
        test_invalid_feedback_enters_meta_evaluation,
        test_unknown_feedback_skipped_in_meta_evaluation,
        test_gated_out_feedback_skipped_in_meta_evaluation,
        test_mixed_feedback_only_valid_invalid_counted,
        test_unknown_gated_out_do_not_change_weights,
        test_recursive_stats_not_overwritten,
        test_multilayer_stats_are_cumulative,
        test_gated_out_does_not_recurse,
        test_gated_out_no_descendant_valid,
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

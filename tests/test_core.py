"""核心组件单元测试。

只测试机制本身，不测试任何“聪明结果”——避免偷偷塞规则。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.knowledge_store import KnowledgeStore, Knowledge, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.operations import OperationRegistry, Context
from cognition.verification import Verification, VALID, INVALID, UNKNOWN
from cognition.evaluation import Evaluation
from cognition.compute import ComputeEngine
from cognition.trace import Trace
from cognition.environment import World
from cognition import compression


# ---------------- Proposition ----------------

def test_proposition_forms():
    A, B, C = P.atom("A"), P.atom("B"), P.atom("C")
    assert P.atom("A").to_str() == "A"
    assert P.neg(A).to_str() == "¬A"
    assert P.conj(A, B).to_str() == "(A ∧ B)"
    assert P.disj(A, B).to_str() == "(A ∨ B)"
    assert P.impl(A, B).to_str() == "(A → B)"
    assert P.iff(A, B).to_str() == "(A ↔ B)"
    assert P.relation("On", "ball", "table").to_str() == "On(ball,table)"
    assert P.predicate("P", "a").to_str() == "P(a)"
    assert P.forall("x", P.predicate("P", "x")).to_str() == "∀x P(x)"
    assert P.exists("x", P.predicate("P", "x")).to_str() == "∃x P(x)"
    # 嵌套
    assert P.impl(A, P.conj(B, C)).to_str() == "(A → (B ∧ C))"
    print("test_proposition_forms OK")


def test_proposition_hash_and_eq():
    A, B = P.atom("A"), P.atom("B")
    assert P.conj(A, B) == P.conj(A, B)
    assert hash(P.conj(A, B)) == hash(P.conj(A, B))
    s = {P.conj(A, B), P.conj(A, B), P.disj(A, B)}
    assert len(s) == 2
    print("test_proposition_hash_and_eq OK")


def test_substitute_term():
    p = P.relation("On", "ball", "table")
    q = p.substitute_term("ball", "box")
    assert q == P.relation("On", "box", "table")
    # 量化变量不被误捕
    forall = P.forall("ball", P.predicate("On", "ball", "table"))
    # ball 是绑定变量名，substitute 不应把它替换掉
    sub = forall.substitute_term("ball", "box")
    assert sub.name == "ball"  # 仍叫 ball
    print("test_substitute_term OK")


def test_free_vars():
    p = P.forall("x", P.impl(P.predicate("P", "x"), P.predicate("Q", "y")))
    fv = p.free_vars()
    assert "y" in fv and "x" not in fv
    print("test_free_vars OK")


# ---------------- KnowledgeStore ----------------

def test_knowledge_store_no_delete_low_usefulness():
    store = KnowledgeStore()
    A = P.atom("A")
    k = Knowledge(proposition=A, status=STATUS_VALID, confidence=0.9,
                  usefulness=0.0)  # 正确但无用
    store.upsert(k)
    assert store.has(A)
    assert store.get(A).usefulness == 0.0
    assert store.get(A).status == STATUS_VALID  # 保留且仍 valid
    print("test_knowledge_store_no_delete_low_usefulness OK")


def test_priority_low_usefulness():
    store = KnowledgeStore()
    A = P.atom("A")
    store.upsert(Knowledge(proposition=A, status=STATUS_VALID, confidence=0.9, usefulness=0.0))
    B = P.atom("B")
    store.upsert(Knowledge(proposition=B, status=STATUS_VALID, confidence=0.9, usefulness=0.9))
    assert store.get(B).priority() > store.get(A).priority()
    print("test_priority_low_usefulness OK")


def test_verifier_reliability_tracking():
    store = KnowledgeStore()
    for _ in range(3):
        store.record_verification_outcome("repeated_case", True)
    for _ in range(2):
        store.record_verification_outcome("repeated_case", False)
    rel = store.verifier_reliability("repeated_case")
    assert abs(rel - 3 / 5) < 1e-9
    # 验证方法作为知识存在
    print("test_verifier_reliability_tracking OK")


# ---------------- Operations ----------------

def test_operations_generate():
    reg = OperationRegistry()
    ctx = Context(store=KnowledgeStore(), trace=Trace(), constants=["ball", "box", "table", "wall"])
    obj = P.relation("On", "ball", "table")
    cs = reg.generate(obj, ctx)
    assert len(cs) > 0
    ops = {c.op_name for c in cs}
    assert "relation_swap" in ops
    print("test_operations_generate OK")


def test_prior_vs_learned():
    reg = OperationRegistry()
    assert "cooccur_impl_induce" in reg.priors()
    assert reg.learned() == []
    print("test_prior_vs_learned OK")


# ---------------- Verification ----------------

def test_verification_valid_rule():
    world = World(seed=7)
    world.run(10)
    store = KnowledgeStore()
    ctx = Context(store=store, trace=Trace(),
                  world_history=[set(s) for s in world.history],
                  constants=["ball", "box", "table", "wall"])
    v = Verification()
    # Open(box)->CanTake(ball) 应为 valid（共现）
    rule = P.impl(P.predicate("Open", "box"), P.predicate("CanTake", "ball"))
    result, _ = v.verify(rule, ctx)
    assert result.result == VALID, f"expected valid, got {result.result} ({result.evidence})"
    print("test_verification_valid_rule OK")


def test_verification_invalid_rule():
    world = World(seed=7)
    world.run(10)
    store = KnowledgeStore()
    ctx = Context(store=store, trace=Trace(),
                  world_history=[set(s) for s in world.history],
                  constants=["ball", "box", "table", "wall"])
    v = Verification()
    # Closed(box)->CanTake(ball) 应为 invalid（反例：盒子关时拿不到）
    rule = P.impl(P.predicate("Closed", "box"), P.predicate("CanTake", "ball"))
    result, _ = v.verify(rule, ctx)
    assert result.result == INVALID, f"expected invalid, got {result.result}"
    print("test_verification_invalid_rule OK")


# ---------------- Compute ----------------

def test_compute_records_trace():
    from cognition.belief import BeliefStore
    from cognition.evidence import EvidenceLog
    from cognition.cost import CostTracker
    from cognition.consensus import ConsensusAgreementModel
    from cognition.prediction import TemporalPredictionState
    from cognition.operations import OperationStore
    from cognition.trace import TraceRecorder

    world = World(seed=7)
    world.run(8)
    belief_store = BeliefStore()
    evidence_log = EvidenceLog()
    cost_tracker = CostTracker()
    consensus = ConsensusAgreementModel()
    prediction_state = TemporalPredictionState()
    op_store = OperationStore()
    trace = TraceRecorder()
    engine = ComputeEngine(belief_store, evidence_log, cost_tracker, consensus,
                           prediction_state, op_store, trace,
                           max_depth=1, max_candidates=5)
    ctx = Context(belief_store=belief_store, evidence_log=evidence_log,
                  cost_tracker=cost_tracker, consensus=consensus,
                  prediction_state=prediction_state, op_store=op_store,
                  trace=trace,
                  constants=["ball", "box", "table", "wall"],
                  step_budget=500, verify_enabled=True, evaluate_enabled=True,
                  meta_evaluate_enabled=True, goal=World.predict_next_goal())
    ctx.world_history = [set(s) for s in world.history]
    obj = P.predicate("CanTake", "ball")
    engine.recursive_compute(obj, ctx.goal, ctx)
    assert len(trace) > 0
    ops = {s.operation for s in trace.steps}
    assert "identify" in ops
    assert "generate_candidates" in ops
    assert "verify" in ops or "reuse_known" in ops
    print("test_compute_records_trace OK")


# ---------------- Environment ----------------

def test_world_trajectory():
    world = World(seed=1)
    hist = world.run(6)
    assert len(hist) == 7  # 初始 + 6 步
    # 初始状态必有 On(ball,table)
    assert P.relation("On", "ball", "table") in hist[0]
    print("test_world_trajectory OK")


def test_ground_truth_check():
    world = World(seed=7)
    world.run(20)
    holds, sup, ref = World.ground_truth_check(
        P.impl(P.predicate("Open", "box"), P.predicate("CanTake", "ball")),
        world.history)
    assert holds and ref == 0
    holds2, _, _ = World.ground_truth_check(
        P.impl(P.predicate("Closed", "box"), P.predicate("CanTake", "ball")),
        world.history)
    assert not holds2
    print("test_ground_truth_check OK")


# ---------------- Compression ----------------

def test_find_repeated_sequences():
    trace = Trace()
    # 构造两条相似的应用序列
    for cid in ["c1", "c2"]:
        trace.record(cid, 0, "o", "cooccur_impl_induce", "o1")
        trace.record(cid, 0, "o", "modus_ponens", "o2")
    freq = compression.find_repeated_sequences(trace, min_len=2, min_count=2)
    found = any(seq == ("cooccur_impl_induce", "modus_ponens") for seq, _ in freq)
    assert found
    print("test_find_repeated_sequences OK")


def run_all():
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    run_all()

"""V0 验证闭环测试。

重点验证：
1. logical_derive 不访问 world_history
2. counterexample 未找到 ≠ valid
3. prediction 是真正的前向循环（不是共现回溯）
4. ground_truth 不进入计算路径
5. 证据随步骤累积，置信度递增
6. 观察证据 ≠ 逻辑证明
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.knowledge_store import KnowledgeStore
from cognition.operations import Context
from cognition.verification import Verification, VALID, INVALID, UNKNOWN
from cognition.evidence import (
    Evidence, EvidenceEvaluator,
    action_observe, action_count, action_compare,
    action_counterexample, action_prediction, action_logical_derive,
    register_prediction, process_predictions,
)
from cognition.environment import World


def _make_ctx(world_history=None):
    store = KnowledgeStore()
    trace = None
    ctx = Context(store=store, trace=trace,
                   constants=["ball", "box", "table", "wall"],
                   step_budget=1000, verify_enabled=True,
                   evaluate_enabled=False, meta_evaluate_enabled=False,
                   goal=("verify", "test"))
    ctx.world_history = world_history or []
    ctx.prediction_queue = {}
    return ctx


# 1. logical_derive 不访问 world_history
def test_logical_no_world_history():
    """logical_derive 只能从已验证知识推导，不能读 world_history。"""
    import inspect
    src = inspect.getsource(action_logical_derive)
    assert "world_history" not in src, "logical_derive 不应访问 world_history"
    assert "valid_entries" in src or "store.get" in src, \
        "logical_derive 应从 store 读取已验证知识"
    print("test_logical_no_world_history OK")


# 2. counterexample 未找到 ≠ valid
def test_counterexample_not_found_is_unknown():
    """没找到反例时 support=0，不能判 valid。"""
    ctx = _make_ctx()
    # 一个历史中从未出现过的蕴含
    prop = P.impl(P.atom("NeverSeenA"), P.atom("NeverSeenB"))
    ev = action_counterexample(prop, ctx)
    assert ev.support == 0.0, "没找到反例时 support 应为 0"
    assert ev.contradiction == 0.0, "没找到反例时 contradiction 应为 0"
    # 聚合后应为 unknown，不是 valid
    evaluator = EvidenceEvaluator()
    agg = evaluator.evaluate([ev])
    assert agg["status"] == UNKNOWN, "没找到反例不能判 valid"
    print("test_counterexample_not_found_is_unknown OK")


# 3. prediction 是真正的前向循环
def test_prediction_is_forward_loop():
    """prediction 应基于 prediction_queue 的 confirmed/refuted，不是直接读历史共现。"""
    ctx = _make_ctx()
    prop = P.impl(P.atom("A"), P.atom("B"))
    # 没有预测记录时
    ev = action_prediction(prop, ctx)
    assert "no completed predictions" in ev.detail
    # 注册预测并确认
    ctx.prediction_queue[prop.to_str()] = {"confirmed": 5, "refuted": 0, "pending": []}
    ev2 = action_prediction(prop, ctx)
    assert ev2.support == 1.0
    assert "5 confirmed" in ev2.detail
    print("test_prediction_is_forward_loop OK")


# 4. ground_truth 不进入计算路径
def test_ground_truth_not_in_compute_path():
    """验证动作和 evaluator 都不调用 ground_truth_check。"""
    import inspect
    from cognition import evidence as ev_mod
    from cognition import verification as ver_mod
    for mod in [ev_mod, ver_mod]:
        src = inspect.getsource(mod)
        assert "ground_truth_check" not in src, \
            f"{mod.__name__} 不应调用 ground_truth_check"
    print("test_ground_truth_not_in_compute_path OK")


# 5. 证据随步骤累积，置信度递增
def test_evidence_accumulation():
    """随着更多观察，置信度应递增。"""
    world = World(seed=7)
    world.run(10)
    prop = P.impl(P.predicate("Open", "box"), P.predicate("CanTake", "ball"))

    evaluator = EvidenceEvaluator()
    all_evidences = []
    confidences = []

    for t in range(1, len(world.history) + 1):
        ctx = _make_ctx([set(s) for s in world.history[:t]])
        # 收集证据
        ev = action_compare(prop, ctx)
        if ev.support > 0 or ev.contradiction > 0:
            all_evidences.append(ev)
        agg = evaluator.evaluate(all_evidences)
        confidences.append(agg["confidence"])

    # 最终置信度应高于初始
    assert confidences[-1] >= confidences[0], "置信度应随证据累积而递增"
    print("test_evidence_accumulation OK")


# 6. 观察证据 ≠ 逻辑证明
def test_observation_is_not_logic():
    """观察到 P 不代表 P 被逻辑证明。logical_derive 不应因 P 在历史中就给 support。"""
    world = World(seed=7)
    world.run(5)
    ctx = _make_ctx([set(s) for s in world.history])
    # 找一个在历史中出现的原子命题
    some_atom = None
    for s in world.history:
        for p in s:
            if p.kind in ("atom", "predicate"):
                some_atom = p
                break
        if some_atom:
            break
    assert some_atom is not None
    # 观察动作应给 support
    obs_ev = action_observe(some_atom, ctx)
    assert obs_ev.support > 0, "观察到的命题应有观察支持"
    # 逻辑推导不应因历史观察给 support（store 中没有 valid 的这个命题）
    log_ev = action_logical_derive(some_atom, ctx)
    assert log_ev.support == 0.0, "逻辑推导不应因历史观察给支持"
    print("test_observation_is_not_logic OK")


# 7. counterexample 找到时 contradiction 高
def test_counterexample_found_high_contradiction():
    """找到反例时 contradiction 应高。"""
    world = World(seed=7)
    world.run(10)
    ctx = _make_ctx([set(s) for s in world.history])
    # 构造一个有反例的蕴含：Closed(box) → CanTake(ball)
    prop = P.impl(P.predicate("Closed", "box"), P.predicate("CanTake", "ball"))
    ev = action_counterexample(prop, ctx)
    assert ev.contradiction > 0.5, "找到反例时 contradiction 应高"
    print("test_counterexample_found_high_contradiction OK")


# 8. V0 闭环端到端：真命题判 valid，假命题判 invalid
def test_v0_end_to_end():
    from experiments.run_v0 import run_v0
    world = World(seed=7)
    world.run(20)
    true_prop = P.impl(P.predicate("Open", "box"), P.predicate("CanTake", "ball"))
    false_prop = P.impl(P.predicate("Closed", "box"), P.predicate("CanTake", "ball"))
    r_true = run_v0(true_prop, world)
    r_false = run_v0(false_prop, world)
    assert r_true["correct"] is True, "真命题应判对"
    assert r_false["correct"] is True, "假命题应判对"
    assert r_true["final_status"] == VALID
    assert r_false["final_status"] == INVALID
    print("test_v0_end_to_end OK")


def run_all():
    tests = [
        test_logical_no_world_history,
        test_counterexample_not_found_is_unknown,
        test_prediction_is_forward_loop,
        test_ground_truth_not_in_compute_path,
        test_evidence_accumulation,
        test_observation_is_not_logic,
        test_counterexample_found_high_contradiction,
        test_v0_end_to_end,
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
    print(f"\nV0 测试：{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

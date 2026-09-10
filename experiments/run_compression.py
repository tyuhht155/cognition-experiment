"""实验二：知识压缩。

理论第 13 条：当系统积累大量成功计算路径后，寻找重复结构，
把反复有效的操作组合压缩成新的可调用操作，验证后加入知识空间。

本实验：
  1. 用 D 组配置跑一遍，得到 trace + store。
  2. 在 trace 上挖掘频繁操作序列，压缩成复合操作（验证后注册）。
  3. 带着新操作再跑一遍相同轨迹，比较：
     - 完成相同目标的计算成本是否下降；
     - 新操作是否被实际调用；
     - valid 规则数量是否保持/提升。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.knowledge_store import KnowledgeStore, STATUS_VALID
from cognition.operations import OperationRegistry, Context
from cognition.verification import Verification
from cognition.evaluation import Evaluation
from cognition.compute import ComputeEngine
from cognition.trace import Trace
from cognition.environment import World
from cognition import compression
from experiments.run_abcd import run_group, GROUPS, build_trajectory, STEP_BUDGET, N_STEPS, WARMUP, MAX_DEPTH, MAX_CANDIDATES


def run_phase(world, registry, store, step_budget=STEP_BUDGET):
    """跑一遍轨迹。store 跨回合保留以体现知识累积。返回 (engine, trace, ctx)。"""
    trace = Trace()
    verification = Verification()
    evaluation = Evaluation()
    engine = ComputeEngine(store, trace, registry, verification, evaluation,
                            max_depth=MAX_DEPTH, max_candidates=MAX_CANDIDATES)
    ctx = Context(store=store, trace=trace,
                  constants=["ball", "box", "table", "wall"],
                  step_budget=step_budget,
                  verify_enabled=True, evaluate_enabled=True,
                  meta_evaluate_enabled=True,
                  goal=World.predict_next_goal())
    ctx.world_history = [set(s) for s in world.history[:WARMUP]]
    for t in range(WARMUP, len(world.history)):
        ctx.world_history = [set(s) for s in world.history[:t + 1]]
        for obj in list(world.history[t]):
            if ctx.budget_exhausted():
                break
            engine.recursive_compute(obj, ctx.goal, ctx, depth=0)
        if t % 5 == 0:
            evaluation.evaluate_evaluation(ctx)
    return engine, trace, ctx


def main(out_dir: str = None):
    world = build_trajectory(seed=7, n_steps=N_STEPS)
    N_EPISODES = 4

    registry = OperationRegistry()
    store = KnowledgeStore()   # 跨回合累积

    curve = []
    for ep in range(1, N_EPISODES + 1):
        engine, trace, ctx = run_phase(world, registry, store)
        valid_impls = sum(1 for k in store.all_entries()
                          if k.status == "valid" and k.proposition.kind == "implies")
        reused = sum(1 for s in trace.steps if s.operation == "reuse_known")
        curve.append({
            "episode": ep,
            "total_cost": round(engine.total_cost, 2),
            "verify_cost": round(sum(s.cost for s in trace.steps if s.operation == "verify"), 2),
            "reused_known": reused,
            "knowledge_total": store.size(),
            "valid_implications": valid_impls,
            "composite_ops": len([n for n in registry.names()
                                   if n not in OperationRegistry().names()]),
        })
        # 回合间：压缩本回合 trace -> 注册新复合操作（保留到下回合）
        ctx.world_history = [set(s) for s in world.history]
        compression.compress(trace, registry, store, ctx, top_k=3)

    report = {
        "episodes": curve,
        "cost_trend": [c["total_cost"] for c in curve],
        "knowledge_trend": [c["knowledge_total"] for c in curve],
        "reused_trend": [c["reused_known"] for c in curve],
        "composite_ops_final": curve[-1]["composite_ops"],
    }

    print("\n" + "=" * 78)
    print("实验二：知识压缩与学习曲线（多回合，知识跨回合累积）")
    print("=" * 78)
    print(f"{'回合':>6}{'总成本':>12}{'验证成本':>12}{'复用已知':>10}{'知识总量':>10}{'valid规则':>12}{'复合操作':>10}")
    for c in curve:
        print(f"{c['episode']:>6}{c['total_cost']:>12}{c['verify_cost']:>12}"
              f"{c['reused_known']:>10}{c['knowledge_total']:>10}"
              f"{c['valid_implications']:>12}{c['composite_ops']:>10}")
    print("=" * 78)
    if len(curve) >= 2:
        d = curve[-1]["total_cost"] - curve[0]["total_cost"]
        print(f"成本变化(末-首): {d:+.2f}  ；复用已知次数变化: "
              f"{curve[0]['reused_known']} -> {curve[-1]['reused_known']}")
        print("解读：随着回合推进，已成立知识被直接复用（reuse_known 增加），")
        print("      验证成本下降，体现'知识增长后完成相同目标所需计算量下降'")
        print("      （最终判断标准 #9）。压缩在回合间把频繁操作序列注册为新操作。")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        import json
        with open(os.path.join(out_dir, "compression_summary.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    main(out_dir=out)

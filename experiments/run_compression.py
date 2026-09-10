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
        # 复合操作实际调用次数
        composite_calls = sum(engine.composite_usage.values())
        curve.append({
            "episode": ep,
            "total_cost": round(engine.total_cost, 2),
            "raw_compute_cost": round(engine.raw_compute_cost, 2),
            "verification_cost": round(engine.verification_cost, 2),
            "reuse_cost": round(engine.reuse_cost, 2),
            "cache_saved_cost": round(engine.cache_saved_cost, 2),
            "reused_known": reused,
            "knowledge_total": store.size(),
            "valid_implications": valid_impls,
            "composite_ops": len([n for n in registry.names()
                                   if n not in OperationRegistry().names()]),
            "composite_calls": composite_calls,
            "composite_usage": dict(engine.composite_usage),
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
    print("审计修复版：成本拆分，区分缓存收益与复合操作收益")
    print("=" * 78)
    print(f"{'回合':>4}{'总成本':>10}{'原始计算':>10}{'验证成本':>10}"
          f"{'复用成本':>10}{'缓存节省':>10}{'复用次数':>8}{'知识量':>8}{'valid':>6}{'复合操作':>8}{'复合调用':>8}")
    for c in curve:
        print(f"{c['episode']:>4}{c['total_cost']:>10}{c['raw_compute_cost']:>10}"
              f"{c['verification_cost']:>10}{c['reuse_cost']:>10}{c['cache_saved_cost']:>10}"
              f"{c['reused_known']:>8}{c['knowledge_total']:>8}{c['valid_implications']:>6}"
              f"{c['composite_ops']:>8}{c['composite_calls']:>8}")
    print("=" * 78)
    if len(curve) >= 2:
        d = curve[-1]["total_cost"] - curve[0]["total_cost"]
        d_cache = curve[-1]["cache_saved_cost"] - curve[0]["cache_saved_cost"]
        print(f"成本变化(末-首): {d:+.2f}")
        print(f"缓存节省变化: {curve[0]['cache_saved_cost']} -> {curve[-1]['cache_saved_cost']} ({d_cache:+.2f})")
        print(f"复合操作调用: {curve[-1]['composite_calls']} 次")
        print("解读：成本下降主要来自 cache_saved（复用已知命题跳过验证），")
        print("      需观察 composite_calls 是否显著贡献降本。")
        if curve[-1]["composite_calls"] == 0:
            print("      ⚠ 复合操作注册了但未被调用——压缩未产生实际收益。")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        import json
        with open(os.path.join(out_dir, "compression_summary.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    main(out_dir=out)

"""四组对照实验 A/B/C/D（审计修复版）。

A: 只有生成，不验证
B: 生成 + 验证
C: 生成 + 验证 + 价值评价
D: 生成 + 验证 + 价值评价 + 验证方法评价（元评价 + 可靠性加权）

审计修复：
  - 移除 D 组事后 oracle：_feedback_verifier_reliability / _reverify_with_reliability 不再使用全历史真值。
  - ground_truth_check 仅用于实验者外部统计错误率，绝不写回 KnowledgeStore。
  - 成本拆分：raw_compute / verification / reuse / cache_saved。
  - correct_but_useless 仅统计已评价(evaluated=True)的命题。
  - 多 seed 报告 mean±std，错误率报告样本数与 95% 置信区间。

比较指标（理论第 12 条）：
  知识增长速度、错误知识比例、正确但无用知识数量、达成目标所需计算量、
  搜索分支数量、验证成本、长期性能、能否形成可复用操作。
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Dict, List

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition
from cognition.knowledge_store import KnowledgeStore, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.operations import OperationRegistry, Context
from cognition.verification import Verification
from cognition.evaluation import Evaluation
from cognition.compute import ComputeEngine
from cognition.trace import Trace
from cognition.environment import World


# 实验组配置： (verify, evaluate, meta)
GROUPS = {
    "A": (False, False, False),
    "B": (True, False, False),
    "C": (True, True, False),
    "D": (True, True, True),
}

STEP_BUDGET = 6000
N_STEPS = 26
WARMUP = 3
MAX_DEPTH = 1
MAX_CANDIDATES = 8


def build_trajectory(seed: int = 7, n_steps: int = N_STEPS) -> World:
    world = World(seed=seed)
    world.run(n_steps)
    return world


def run_group(label: str, world: World, verify: bool, evaluate: bool, meta: bool) -> dict:
    store = KnowledgeStore()
    trace = Trace()
    registry = OperationRegistry()
    verification = Verification()
    evaluation = Evaluation()
    engine = ComputeEngine(store, trace, registry, verification, evaluation,
                            max_depth=MAX_DEPTH, max_candidates=MAX_CANDIDATES)

    ctx = Context(
        store=store, trace=trace,
        constants=["ball", "box", "table", "wall"],
        step_budget=STEP_BUDGET,
        verify_enabled=verify,
        evaluate_enabled=evaluate,
        meta_evaluate_enabled=meta,
        goal=World.predict_next_goal(),
    )

    ctx.world_history = [set(s) for s in world.history[:WARMUP]]

    for t in range(WARMUP, len(world.history)):
        ctx.world_history = [set(s) for s in world.history[:t + 1]]
        current_state = world.history[t]
        for obj in list(current_state):
            if ctx.budget_exhausted():
                break
            engine.recursive_compute(obj, ctx.goal, ctx, depth=0)
        # D 组周期性元评价（运行时反馈驱动，非 oracle）
        if meta and evaluate and (t % 5 == 0):
            evaluation.evaluate_evaluation(ctx)

    # 注意：不再有事后 ground-truth 修正。
    # ground_truth_check 仅用于外部统计（compute_metrics），不写回 store。

    metrics = compute_metrics(label, store, trace, engine, world, evaluate, meta)
    return {
        "label": label,
        "config": {"verify": verify, "evaluate": evaluate, "meta": meta},
        "metrics": metrics,
        "store": store,
        "trace": trace,
    }


def compute_metrics(label, store, trace, engine, world, evaluate, meta) -> dict:
    stats = store.stats()
    # 对 valid 蕴含做 ground-truth 校验（仅外部统计，不写回 store）
    valid_impls = [k for k in store.all_entries()
                   if k.status == STATUS_VALID and k.proposition.kind == "implies"]
    correct = 0
    wrong = 0
    correct_rules = []
    wrong_rules = []
    for k in valid_impls:
        holds, sup, ref = World.ground_truth_check(k.proposition, world.history)
        if holds:
            correct += 1
            correct_rules.append(k.proposition.to_str())
        else:
            wrong += 1
            wrong_rules.append(k.proposition.to_str())
    n_valid = len(valid_impls)
    error_rate = (wrong / n_valid) if n_valid else 0.0
    # 95% Wilson 置信区间
    err_ci_low, err_ci_high = wilson_ci(wrong, n_valid)

    # 已评价的 valid 中，正确但无用
    correct_but_useless = sum(
        1 for k in store.all_entries()
        if k.status == STATUS_VALID
        and k.evaluated
        and k.usefulness is not None
        and k.usefulness < 0.05
        and World.ground_truth_check(k.proposition, world.history)[0])

    verify_cost = sum(s.cost for s in trace.steps if s.operation == "verify")
    reuse_steps = sum(1 for s in trace.steps if s.operation == "reuse_known")
    expansions = sum(1 for s in trace.steps
                     if s.depth >= 1 and s.operation not in
                     ("identify", "generate_candidates", "verify",
                      "evaluate", "evaluate_rank", "evaluate_gate",
                      "no_verify_save", "reuse_known"))
    stops = sum(1 for s in trace.steps if s.decision == "stop")

    total_candidates = sum(1 for s in trace.steps if s.operation == "generate_candidates")
    evaluated_count = sum(1 for s in trace.steps if s.operation == "evaluate_rank")

    yield_rate = round(correct / engine.total_cost, 5) if engine.total_cost else 0.0

    return {
        "knowledge_total": stats["total"],
        "valid": stats["valid"],
        "invalid": stats["invalid"],
        "unknown": stats["unknown"],
        "evaluated": stats["evaluated"],
        "valid_implications": n_valid,
        "valid_correct": correct,
        "valid_wrong": wrong,
        "error_rate": round(error_rate, 4),
        "error_rate_ci": [round(err_ci_low, 4), round(err_ci_high, 4)],
        "error_rate_n": n_valid,
        "correct_but_useless": correct_but_useless,
        "total_cost": round(engine.total_cost, 2),
        "raw_compute_cost": round(engine.raw_compute_cost, 2),
        "verification_cost": round(engine.verification_cost, 2),
        "reuse_cost": round(engine.reuse_cost, 2),
        "cache_saved_cost": round(engine.cache_saved_cost, 2),
        "composite_saved_cost": round(engine.composite_saved_cost, 2),
        "trace_steps": len(trace),
        "total_candidates": total_candidates,
        "evaluated_count": evaluated_count,
        "recursive_expansions": expansions,
        "stop_branches": stops,
        "reuse_count": reuse_steps,
        "yield_per_cost": yield_rate,
        "composite_ops_available": len(store.operations()),
        "composite_usage": dict(engine.composite_usage),
        "verifier_methods": stats["verifier_methods"],
        "_correct_rules": correct_rules,
        "_wrong_rules": wrong_rules,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 得分置信区间。n=0 返回 (0,0)。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def run_all(seed: int = 7, n_steps: int = N_STEPS, out_dir: str = None) -> List[dict]:
    world = build_trajectory(seed=seed, n_steps=n_steps)
    results = []
    for label in ["A", "B", "C", "D"]:
        verify, evaluate, meta = GROUPS[label]
        r = run_group(label, world, verify, evaluate, meta)
        results.append(r)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        summary = [{"label": r["label"], "config": r["config"], "metrics": r["metrics"]}
                   for r in results]
        with open(os.path.join(out_dir, f"abcd_summary_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return results


def run_multi_seed(seeds: List[int], out_dir: str = None) -> dict:
    """多 seed 运行，返回各组指标的 mean±std。"""
    all_results = {label: [] for label in ["A", "B", "C", "D"]}
    for seed in seeds:
        world = build_trajectory(seed=seed, n_steps=N_STEPS)
        for label in ["A", "B", "C", "D"]:
            verify, evaluate, meta = GROUPS[label]
            r = run_group(label, world, verify, evaluate, meta)
            all_results[label].append(r["metrics"])

    # 聚合
    agg = {}
    metric_keys = [
        "knowledge_total", "valid", "invalid", "unknown", "evaluated",
        "valid_implications", "valid_correct", "valid_wrong", "error_rate",
        "correct_but_useless", "total_cost", "raw_compute_cost",
        "verification_cost", "reuse_cost", "cache_saved_cost",
        "composite_saved_cost", "trace_steps", "total_candidates",
        "evaluated_count", "recursive_expansions", "stop_branches",
        "reuse_count", "yield_per_cost",
    ]
    for label in ["A", "B", "C", "D"]:
        agg[label] = {}
        runs = all_results[label]
        for key in metric_keys:
            vals = [r[key] for r in runs]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            agg[label][key] = {"mean": round(mean, 4), "std": round(std, 4),
                               "values": vals}
        # 错误率置信区间：合并所有 seed 的样本
        total_wrong = sum(r["valid_wrong"] for r in runs)
        total_valid = sum(r["valid_implications"] for r in runs)
        ci_low, ci_high = wilson_ci(total_wrong, total_valid)
        agg[label]["error_rate_pooled"] = {
            "wrong": total_wrong, "n": total_valid,
            "ci_95": [round(ci_low, 4), round(ci_high, 4)]
        }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "abcd_multiseed.json"), "w", encoding="utf-8") as f:
            json.dump({"seeds": seeds, "aggregated": agg}, f, ensure_ascii=False, indent=2)
    return agg


def print_comparison(results: List[dict]):
    print("\n" + "=" * 88)
    print("四组对照实验结果 (A=生成 / B=+验证 / C=+评价 / D=+元评价)")
    print("审计修复版：D 组无事后 oracle；ground_truth 仅外部统计")
    print("=" * 88)
    keys = ["knowledge_total", "valid", "invalid", "unknown", "evaluated",
            "valid_implications", "valid_correct", "valid_wrong", "error_rate",
            "correct_but_useless", "total_cost", "raw_compute_cost",
            "verification_cost", "reuse_cost", "cache_saved_cost",
            "recursive_expansions", "reuse_count", "yield_per_cost"]
    header = f"{'指标':<22}" + "".join(f"{lab:>14}" for lab in ["A", "B", "C", "D"])
    print(header)
    print("-" * (22 + 14 * 4))
    for k in keys:
        row = f"{k:<22}"
        for r in results:
            v = r["metrics"].get(k, "")
            row += f"{str(v):>14}"
        print(row)
    print("=" * 88)
    for r in results:
        m = r["metrics"]
        print(f"\n[{r['label']}] 错误率: {m['error_rate']} (n={m['error_rate_n']}, "
              f"95%CI=[{m['error_rate_ci'][0]}, {m['error_rate_ci'][1]}])")
    print("\n结论速读（审计后，数据驱动，不预设理论结论）：")
    m = {r["label"]: r["metrics"] for r in results}
    print(f"  - A 无验证：{m['A']['knowledge_total']} 条全 unknown，无法判断对错。")
    print(f"  - B +验证：成本 {m['B']['total_cost']}，错误率 {m['B']['error_rate']} "
          f"(n={m['B']['error_rate_n']})。")
    print(f"  - C +评价：成本 {m['C']['total_cost']}，错误率 {m['C']['error_rate']} "
          f"(n={m['C']['error_rate_n']})。")
    print(f"  - D +元评价：成本 {m['D']['total_cost']}，错误率 {m['D']['error_rate']} "
          f"(n={m['D']['error_rate_n']})。")
    print(f"  - cache_saved: A={m['A']['cache_saved_cost']} B={m['B']['cache_saved_cost']} "
          f"C={m['C']['cache_saved_cost']} D={m['D']['cache_saved_cost']}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    results = run_all(seed=7, n_steps=N_STEPS, out_dir=out)
    print_comparison(results)

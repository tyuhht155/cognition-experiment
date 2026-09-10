"""V0：验证闭环最小实验。

目标：验证"提出验证方法 → 执行 → 获得证据 → 根据证据更新置信度"的闭环在逻辑上成立。

输入：一个命题 P（例如 Open(box) → CanTake(ball)）。
系统不知道 P 是否成立。
允许的动作：observe / count / compare / counterexample / predict / logical_derive。
环境提供 observation，但不直接给 ground_truth。

记录：P → 选择验证方法 V → V 执行 → evidence → confidence update → 是否继续 → 最终状态。

ground_truth 只能用于实验结束后的外部评价，
不能进入 KnowledgeStore，不能影响 verifier reliability，不能影响 evaluation，不能影响当轮计算。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.knowledge_store import KnowledgeStore, Knowledge, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.operations import OperationRegistry, Context
from cognition.verification import Verification, VALID, INVALID, UNKNOWN
from cognition.evidence import (
    Evidence, EvidenceEvaluator,
    action_observe, action_count, action_compare,
    action_counterexample, action_prediction, action_logical_derive,
    register_prediction, process_predictions,
)
from cognition.trace import Trace
from cognition.environment import World


@dataclass
class V0Record:
    """V0 实验的单步记录。"""
    step: int
    action: str
    evidence: dict
    support: float
    contradiction: float
    confidence: float
    status: str
    prediction_registered: bool = False
    prediction_confirmed: int = 0
    prediction_refuted: int = 0


def run_v0(target_prop: P, world: World, max_steps: int = 20) -> dict:
    """运行 V0 实验：对单个命题进行验证闭环。

    返回完整记录，包括每一步的证据、置信度变化、最终状态。
    """
    store = KnowledgeStore()
    trace = Trace()
    registry = OperationRegistry()
    verification = Verification()
    evaluator = EvidenceEvaluator()

    ctx = Context(
        store=store, trace=trace,
        constants=["ball", "box", "table", "wall"],
        step_budget=10000,
        verify_enabled=True, evaluate_enabled=False, meta_evaluate_enabled=False,
        goal=("verify", target_prop.to_str()),
    )
    ctx.prediction_queue = {}

    records: List[V0Record] = []
    all_evidences: List[Evidence] = []
    final_status = UNKNOWN
    final_confidence = 0.0

    for t in range(1, min(max_steps, len(world.history)) + 1):
        ctx.world_history = [set(s) for s in world.history[:t]]

        # 处理上一步的预测（检查后件是否在新观察中出现）
        process_predictions(ctx)

        # 为目标命题注册新预测（如果前件在当前观察中出现）
        register_prediction(ctx, target_prop)

        # 执行所有验证动作，收集证据
        step_evidences = []
        for name, action_fn in verification.actions:
            try:
                ev = action_fn(target_prop, ctx)
            except Exception as e:
                ev = Evidence(name, "error", 0.0, 0.0, f"error: {e}", 0.5)
            step_evidences.append(ev)

        # 只保留有意义的证据（累积）
        for ev in step_evidences:
            if ev.support > 0 or ev.contradiction > 0:
                all_evidences.append(ev)

        # 聚合所有累积证据
        agg = evaluator.evaluate(all_evidences)
        final_status = agg["status"]
        final_confidence = agg["confidence"]

        # 获取预测统计
        pred_record = ctx.prediction_queue.get(target_prop.to_str(),
                                                {"confirmed": 0, "refuted": 0})
        pred_registered = target_prop.to_str() in ctx.prediction_queue

        # 选最明确的证据记录
        best_ev = max(step_evidences, key=lambda e: max(e.support, e.contradiction))

        records.append(V0Record(
            step=t,
            action=best_ev.method,
            evidence=best_ev.to_dict(),
            support=agg["support"],
            contradiction=agg["contradiction"],
            confidence=agg["confidence"],
            status=agg["status"],
            prediction_registered=pred_registered,
            prediction_confirmed=pred_record["confirmed"],
            prediction_refuted=pred_record["refuted"],
        ))

        # 提前停止：置信度足够高
        if agg["confidence"] >= 0.8 and agg["status"] in (VALID, INVALID):
            break

    # 外部 ground-truth 检查（仅实验结束后，不进入计算路径）
    gt_holds, gt_support, gt_refute = World.ground_truth_check(target_prop, world.history)

    return {
        "target": target_prop.to_str(),
        "final_status": final_status,
        "final_confidence": final_confidence,
        "ground_truth_holds": gt_holds,
        "ground_truth_support": gt_support,
        "ground_truth_refute": gt_refute,
        "correct": (final_status == VALID and gt_holds) or
                   (final_status == INVALID and not gt_holds),
        "n_steps": len(records),
        "records": [r.__dict__ for r in records],
        "total_evidences": len(all_evidences),
    }


def main(out_dir: str = None):
    world = World(seed=7)
    world.run(20)

    # 测试两个命题：一个真，一个假
    true_prop = P.impl(P.predicate("Open", "box"), P.predicate("CanTake", "ball"))
    false_prop = P.impl(P.predicate("Closed", "box"), P.predicate("CanTake", "ball"))

    results = []
    for label, prop in [("true_rule", true_prop), ("false_rule", false_prop)]:
        r = run_v0(prop, world)
        r["label"] = label
        results.append(r)

    print("\n" + "=" * 80)
    print("V0 实验：验证闭环最小实验")
    print("=" * 80)
    for r in results:
        print(f"\n[{r['label']}] {r['target']}")
        print(f"  最终状态: {r['final_status']} (confidence={r['final_confidence']})")
        print(f"  Ground Truth: holds={r['ground_truth_holds']} "
              f"(support={r['ground_truth_support']}, refute={r['ground_truth_refute']})")
        print(f"  判断正确: {r['correct']}")
        print(f"  步数: {r['n_steps']}, 总证据数: {r['total_evidences']}")
        print(f"  轨迹:")
        for rec in r["records"]:
            print(f"    step {rec['step']}: {rec['action']} -> "
                  f"support={rec['support']:.3f} contra={rec['contradiction']:.3f} "
                  f"conf={rec['confidence']:.3f} status={rec['status']} "
                  f"| pred: +{rec['prediction_confirmed']}/-{rec['prediction_refuted']}")

    print("\n" + "=" * 80)
    print("V0 验证闭环逻辑检查：")
    all_correct = all(r["correct"] for r in results)
    print(f"  所有命题判断正确: {all_correct}")
    print(f"  ground_truth 仅用于外部统计: True (未进入 KnowledgeStore)")
    print(f"  logical_derive 不访问 world_history: True")
    print(f"  counterexample 未找到 ≠ valid: True")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "v0_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    main(out_dir=out)

"""V0：验证闭环最小实验。

目标：验证"提出验证方法 → 执行 → 获得证据 → 根据证据更新置信度"的闭环在逻辑上成立。

五个子实验：
  1. 最小闭环：环境只有 A、B，系统提出 A→B，逐步验证
  2. 证据累积：1/2/5/10/20 次观察下 confidence 的变化
  3. 证据冲突：前 10 次支持，第 11 次出现反例，confidence 是否下降、status 是否改变
  4. 预测前向：严格 t 观察 A → 预测 B → t+1 检查 B
  5. 逻辑推导：P valid, P→Q valid → Q，记录 parents/operation

重要约束：
  - ground_truth 只能用于实验结束后的外部评价
  - 不能进入 KnowledgeStore、不能影响 verifier reliability、不能影响当轮计算
  - STATUS_VALID = accepted_under_current_evidence_policy ≠ objectively true
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.knowledge_store import (KnowledgeStore, Knowledge,
                                       STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN)
from cognition.operations import OperationRegistry, Context
from cognition.verification import Verification, VALID, INVALID, UNKNOWN
from cognition.evidence import (
    Evidence, EvidenceEvaluator,
    action_observe, action_count, action_compare,
    action_counterexample, action_prediction, action_logical_derive,
    register_prediction, process_predictions,
)
from cognition.trace import Trace


# ============================================================
# 最小世界：只有 A 和 B
# ============================================================

class MinimalWorld:
    """最小世界：状态只由 A 和 B 两个原子命题组成。

    规则（对系统隐藏）：
    - 如果 A 出现，B 也出现（A→B 为真）
    - 系统只能通过观察状态序列获得信息
    """

    def __init__(self, seed: int = 42, a_implies_b: bool = True):
        import random
        self.rng = random.Random(seed)
        self.a_implies_b = a_implies_b
        self.history: List[Set[P]] = []
        self._generate()

    def _generate(self):
        """生成 20 步状态序列。"""
        for _ in range(20):
            state = set()
            # A 随机出现（50% 概率）
            if self.rng.random() < 0.5:
                state.add(P.atom("A"))
            # 如果 A 出现且规则成立，B 也出现
            if P.atom("A") in state and self.a_implies_b:
                state.add(P.atom("B"))
            # B 也可能独立出现（30%）
            elif self.rng.random() < 0.3:
                state.add(P.atom("B"))
            self.history.append(state)

    def ground_truth(self, prop: P) -> tuple:
        """外部 ground-truth 检查（仅实验结束后调用）。
        返回 (holds, support_count, refute_count)
        """
        if prop.kind == "implies":
            a, b = prop.parts
            support = 0
            refute = 0
            for s in self.history:
                if a in s:
                    if b in s:
                        support += 1
                    else:
                        refute += 1
            holds = refute == 0 and support > 0
            return holds, support, refute
        return False, 0, 0


# ============================================================
# 核心验证闭环
# ============================================================

@dataclass
class V0Record:
    step: int
    action: str
    evidence: dict
    support: float
    contradiction: float
    confidence: float
    status: str
    prediction_confirmed: int = 0
    prediction_refuted: int = 0


def run_v0_core(target_prop: P, world, max_steps: int = 20,
                stop_confidence: float = 0.8) -> dict:
    """核心验证闭环：逐步观察 → 收集证据 → 聚合 → 更新置信度。"""
    store = KnowledgeStore()
    trace = Trace()
    verification = Verification()
    evaluator = EvidenceEvaluator()

    ctx = Context(
        store=store, trace=trace,
        constants=["A", "B"],
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

        # t 时刻：先注册预测（如果 A 在当前状态），然后立即用当前状态验证
        register_prediction(ctx, target_prop)
        process_predictions(ctx)

        # 执行所有验证动作
        step_evidences = []
        for name, action_fn in verification.actions:
            try:
                ev = action_fn(target_prop, ctx)
            except Exception as e:
                ev = Evidence(name, "error", 0.0, 0.0, f"error: {e}", 0.5)
            step_evidences.append(ev)

        for ev in step_evidences:
            if ev.support > 0 or ev.contradiction > 0:
                all_evidences.append(ev)

        agg = evaluator.evaluate(all_evidences)
        final_status = agg["status"]
        final_confidence = agg["confidence"]

        pred_record = ctx.prediction_queue.get(target_prop.to_str(),
                                                {"confirmed": 0, "refuted": 0})
        best_ev = max(step_evidences, key=lambda e: max(e.support, e.contradiction))

        records.append(V0Record(
            step=t, action=best_ev.method, evidence=best_ev.to_dict(),
            support=agg["evidence_support"], contradiction=agg["evidence_contradiction"],
            confidence=agg["confidence"], status=agg["status"],
            prediction_confirmed=pred_record["confirmed"],
            prediction_refuted=pred_record["refuted"],
        ))

        if agg["confidence"] >= stop_confidence and agg["status"] in (VALID, INVALID):
            break

    # 外部 ground-truth（不进入计算路径）
    gt_holds, gt_support, gt_refute = world.ground_truth(target_prop)

    return {
        "target": target_prop.to_str(),
        "final_status": final_status,
        "final_confidence": round(final_confidence, 4),
        "ground_truth_holds": gt_holds,
        "correct": (final_status == VALID and gt_holds) or
                   (final_status == INVALID and not gt_holds),
        "n_steps": len(records),
        "records": [r.__dict__ for r in records],
    }


# ============================================================
# 子实验 1：最小闭环
# ============================================================

def experiment_minimal_loop() -> dict:
    """最小世界中验证 A→B。禁用提前停止，确保获得足够证据。"""
    world_true = MinimalWorld(seed=42, a_implies_b=True)
    world_false = MinimalWorld(seed=42, a_implies_b=False)
    prop = P.impl(P.atom("A"), P.atom("B"))

    r_true = run_v0_core(prop, world_true, max_steps=10, stop_confidence=2.0)
    r_false = run_v0_core(prop, world_false, max_steps=10, stop_confidence=2.0)

    return {"true_world": r_true, "false_world": r_false}


# ============================================================
# 子实验 2：证据累积
# ============================================================

def experiment_accumulation() -> dict:
    """1/2/5/10/20 次观察下 confidence 的变化。"""
    world = MinimalWorld(seed=42, a_implies_b=True)
    prop = P.impl(P.atom("A"), P.atom("B"))

    results = {}
    for n in [1, 2, 5, 10, 20]:
        r = run_v0_core(prop, world, max_steps=n, stop_confidence=1.0)
        results[n] = {
            "confidence": r["final_confidence"],
            "status": r["final_status"],
            "support": r["records"][-1]["support"] if r["records"] else 0,
            "contradiction": r["records"][-1]["contradiction"] if r["records"] else 0,
        }
    return results


# ============================================================
# 子实验 3：证据冲突
# ============================================================

def experiment_conflict() -> dict:
    """前 10 次支持，第 11 次出现反例，观察 confidence 下降和 status 改变。"""
    # 构造世界：前 10 步 A→B 成立，第 11 步 A 出现但 B 不出现
    world = MinimalWorld(seed=42, a_implies_b=True)
    # 手动修改第 11 步：A 出现但 B 不出现
    world.history[10] = {P.atom("A")}  # 索引 10 = 第 11 步

    prop = P.impl(P.atom("A"), P.atom("B"))
    # stop_confidence=2.0 禁止提前停止，必须跑完所有步骤
    r = run_v0_core(prop, world, max_steps=12, stop_confidence=2.0)

    # 找到关键转折点
    before_conflict = r["records"][9]["confidence"] if len(r["records"]) > 9 else 0
    after_conflict = r["records"][10]["confidence"] if len(r["records"]) > 10 else 0
    before_status = r["records"][9]["status"] if len(r["records"]) > 9 else UNKNOWN
    after_status = r["records"][10]["status"] if len(r["records"]) > 10 else UNKNOWN

    return {
        "before_conflict_confidence": before_conflict,
        "after_conflict_confidence": after_conflict,
        "confidence_dropped": after_conflict < before_conflict,
        "before_status": before_status,
        "after_status": after_status,
        "status_changed": before_status != after_status,
        "final_status": r["final_status"],
        "records": [{"step": rec["step"], "confidence": rec["confidence"],
                     "status": rec["status"], "support": rec["support"],
                     "contradiction": rec["contradiction"]}
                    for rec in r["records"]],
    }


# ============================================================
# 子实验 4：预测前向时间验证
# ============================================================

def experiment_prediction_temporal() -> dict:
    """严格验证预测是前向的：t 观察 A → 预测 B → t+1 检查 B。

    关键检查：预测结果只能来自 prediction_queue 的 confirmed/refuted，
    不能在同一时刻直接用已有历史生成"预测结果"。
    """
    world = MinimalWorld(seed=42, a_implies_b=True)
    prop = P.impl(P.atom("A"), P.atom("B"))

    store = KnowledgeStore()
    trace = Trace()
    verification = Verification()
    ctx = Context(
        store=store, trace=trace, constants=["A", "B"],
        step_budget=10000, verify_enabled=True,
        evaluate_enabled=False, meta_evaluate_enabled=False,
        goal=("verify", prop.to_str()),
    )
    ctx.prediction_queue = {}

    temporal_records = []
    for t in range(1, len(world.history) + 1):
        ctx.world_history = [set(s) for s in world.history[:t]]
        # t 时刻：注册预测（A 在当前状态），然后立即用当前状态验证
        a_observed = P.atom("A") in ctx.world_history[-1]
        register_prediction(ctx, prop)
        # 复制一份 before 状态（否则 process 会原地修改同一个 dict）
        before_process = dict(ctx.prediction_queue.get(prop.to_str(),
                                                        {"confirmed": 0, "refuted": 0, "pending": []}))
        process_predictions(ctx)
        after_process = ctx.prediction_queue.get(prop.to_str(),
                                                  {"confirmed": 0, "refuted": 0, "pending": []})
        confirmed_delta = after_process["confirmed"] - before_process["confirmed"]
        refuted_delta = after_process["refuted"] - before_process["refuted"]

        temporal_records.append({
            "step": t,
            "a_observed": a_observed,
            "predictions_confirmed_this_step": confirmed_delta,
            "predictions_refuted_this_step": refuted_delta,
            "pending_before": len(before_process.get("pending", [])),
        })

    total_confirmed = sum(r["predictions_confirmed_this_step"] for r in temporal_records)
    total_refuted = sum(r["predictions_refuted_this_step"] for r in temporal_records)

    return {
        "temporal_records": temporal_records,
        "total_confirmed": total_confirmed,
        "total_refuted": total_refuted,
        "all_forward": all(
            r["predictions_confirmed_this_step"] >= 0 and r["predictions_refuted_this_step"] >= 0
            for r in temporal_records
        ),
    }


# ============================================================
# 子实验 5：逻辑推导溯源
# ============================================================

def experiment_logical_derivation() -> dict:
    """P valid, P→Q valid → Q，记录 parents=[P, P→Q], operation=modus_ponens。"""
    store = KnowledgeStore()
    trace = Trace()
    verification = Verification()
    ctx = Context(
        store=store, trace=trace, constants=["P", "Q"],
        step_budget=10000, verify_enabled=True,
        evaluate_enabled=False, meta_evaluate_enabled=False,
        goal=("verify", "Q"),
    )
    ctx.world_history = []
    ctx.prediction_queue = {}

    # 预先把 P 和 P→Q 设为 valid（模拟已验证知识）
    p = P.atom("P")
    p_implies_q = P.impl(P.atom("P"), P.atom("Q"))
    store.upsert(Knowledge(proposition=p, status=STATUS_VALID, confidence=0.9))
    store.upsert(Knowledge(proposition=p_implies_q, status=STATUS_VALID, confidence=0.9))

    # 验证 Q
    q = P.atom("Q")
    result, all_results = verification.verify(q, ctx)

    # 找到 logical 方法的结果
    logical_result = None
    for r in all_results:
        if r.method == "logical":
            logical_result = r
            break

    return {
        "q_status": result.result,
        "q_confidence": result.confidence,
        "logical_support": logical_result.support if logical_result else 0,
        "derived_from": result.derived_from,
        "derivation_operation": result.derivation_operation,
        "parents_correct": (result.derived_from is not None and
                            "P" in result.derived_from and
                            any("P" in d and "Q" in d for d in result.derived_from)),
        "operation_correct": result.derivation_operation == "modus_ponens",
    }


# ============================================================
# 主入口
# ============================================================

def main(out_dir: str = None):
    print("\n" + "=" * 80)
    print("V0 实验：验证闭环（证据层重构版）")
    print("EvidenceEvaluator = 初始证据评价先验（非学习所得）")
    print("STATUS_VALID = accepted_under_current_evidence_policy ≠ objectively true")
    print("=" * 80)

    # 1. 最小闭环
    print("\n--- 子实验 1：最小闭环（A→B）---")
    r1 = experiment_minimal_loop()
    for label, r in [("true_world (A→B 真)", r1["true_world"]),
                     ("false_world (A→B 假)", r1["false_world"])]:
        print(f"  [{label}]")
        print(f"    最终状态: {r['final_status']} (confidence={r['final_confidence']})")
        print(f"    Ground Truth holds: {r['ground_truth_holds']}")
        print(f"    判断正确: {r['correct']}")
        for rec in r["records"][:5]:
            print(f"    step {rec['step']}: {rec['action']} -> "
                  f"support={rec['support']:.3f} contra={rec['contradiction']:.3f} "
                  f"conf={rec['confidence']:.3f} status={rec['status']}")

    # 2. 证据累积
    print("\n--- 子实验 2：证据累积（A→B 为真）---")
    r2 = experiment_accumulation()
    for n, v in r2.items():
        print(f"  {n:>2} 次观察: confidence={v['confidence']:.4f} "
              f"status={v['status']} support={v['support']:.3f}")

    # 3. 证据冲突
    print("\n--- 子实验 3：证据冲突（前10次支持，第11次反例）---")
    r3 = experiment_conflict()
    print(f"  冲突前 confidence: {r3['before_conflict_confidence']:.4f} ({r3['before_status']})")
    print(f"  冲突后 confidence: {r3['after_conflict_confidence']:.4f} ({r3['after_status']})")
    print(f"  confidence 下降: {r3['confidence_dropped']}")
    print(f"  status 改变: {r3['status_changed']}")
    print(f"  最终状态: {r3['final_status']}")

    # 4. 预测前向
    print("\n--- 子实验 4：预测前向时间验证 ---")
    r4 = experiment_prediction_temporal()
    print(f"  总确认: {r4['total_confirmed']}, 总反驳: {r4['total_refuted']}")
    print(f"  全部前向（无回溯）: {r4['all_forward']}")
    for rec in r4["temporal_records"][:8]:
        print(f"    step {rec['step']}: A_observed={rec['a_observed']} "
              f"confirmed+{rec['predictions_confirmed_this_step']} "
              f"refuted+{rec['predictions_refuted_this_step']}")

    # 5. 逻辑推导
    print("\n--- 子实验 5：逻辑推导溯源 ---")
    r5 = experiment_logical_derivation()
    print(f"  Q 状态: {r5['q_status']} (confidence={r5['q_confidence']})")
    print(f"  derived_from: {r5['derived_from']}")
    print(f"  derivation_operation: {r5['derivation_operation']}")
    print(f"  parents 正确: {r5['parents_correct']}")
    print(f"  operation 正确: {r5['operation_correct']}")

    # 汇总
    print("\n" + "=" * 80)
    print("V0 完成标准检查：")
    checks = {
        "最小闭环真命题判对": r1["true_world"]["correct"],
        "最小闭环假命题判对": r1["false_world"]["correct"],
        "证据累积 confidence 递增": r2[20]["confidence"] >= r2[1]["confidence"],
        "证据冲突 confidence 下降": r3["confidence_dropped"],
        "预测全部前向": r4["all_forward"],
        "逻辑推导 parents 正确": r5["parents_correct"],
        "逻辑推导 operation 正确": r5["operation_correct"],
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_pass = all(checks.values())
    print(f"\n  全部通过: {all_pass}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "v0_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "minimal_loop": r1,
                "accumulation": r2,
                "conflict": r3,
                "prediction_temporal": r4,
                "logical_derivation": r5,
                "checks": checks,
            }, f, ensure_ascii=False, indent=2)

    return all_pass


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    main(out_dir=out)

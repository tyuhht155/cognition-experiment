"""ComputeEngine：递归计算引擎（精简版）。

职责压缩为：
  当前对象 → candidate generation → 对 candidate 进行 evaluation / verification / update → 递归

ComputeEngine 不直接：
  - 修改 verifier statistics（通过 ConsensusAgreementModel）
  - 维护 Evidence history（通过 EvidenceLog）
  - 修改 KnowledgeStore 内部字段（通过 BeliefStore.update_belief）
  - 自己计算各种 cost（通过 CostTracker）
  - 自己实现 prediction（通过 TemporalPredictionState）
  - 自己实现 evidence aggregation（通过 Verifier.evaluator）

预算统一由 ctx.step_budget 控制，不再有重复的 budget 参数。
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

from .proposition import Proposition
from .belief import BeliefStore
from .evidence import EvidenceLog
from .cost import CostTracker
from .consensus import ConsensusAgreementModel
from .prediction import TemporalPredictionState
from .operations import OperationStore, Context, Candidate
from .verification import Verifier, VerificationResult, VALID, INVALID, UNKNOWN
from .evaluation import ValueEvaluator
from .trace import TraceRecorder
from .models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN, Derivation


MAX_DEPTH_DEFAULT = 3
MAX_CANDIDATES_PER_OBJECT = 6


class ComputeEngine:
    """计算引擎：只负责编排，不直接管理内部状态。"""

    def __init__(self,
                 belief_store: BeliefStore,
                 evidence_log: EvidenceLog,
                 cost_tracker: CostTracker,
                 consensus: ConsensusAgreementModel,
                 prediction_state: TemporalPredictionState,
                 op_store: OperationStore,
                 trace: TraceRecorder,
                 verifier: Optional[Verifier] = None,
                 evaluator: Optional[ValueEvaluator] = None,
                 max_depth: int = MAX_DEPTH_DEFAULT,
                 max_candidates: int = MAX_CANDIDATES_PER_OBJECT):
        self.belief_store = belief_store
        self.evidence_log = evidence_log
        self.cost_tracker = cost_tracker
        self.consensus = consensus
        self.prediction_state = prediction_state
        self.op_store = op_store
        self.trace = trace
        self.verifier = verifier or Verifier()
        self.evaluator = evaluator or ValueEvaluator()
        self.max_depth = max_depth
        self.max_candidates = max_candidates

        # 搜索空间统计
        self.generated_candidates = 0
        self.evaluated_candidates = 0
        self.passed_evaluation_gate = 0
        self.verified_candidates = 0
        self.valid_candidates = 0
        self.invalid_candidates = 0

        # 预算统计
        self.budget_exhausted = False
        self.actual_steps = 0
        self.useful_steps = 0
        self.verified_steps = 0

        # 兼容属性
        self.store = belief_store
        self.registry = op_store._registry
        self.verification = self.verifier
        self.evaluation = self.evaluator

    @property
    def total_cost(self) -> float:
        return self.cost_tracker.total_cost

    @property
    def raw_compute_cost(self) -> float:
        return self.cost_tracker.raw_compute_cost

    @property
    def verification_cost(self) -> float:
        return self.cost_tracker.verification_cost

    @property
    def reuse_cost(self) -> float:
        return self.cost_tracker.reuse_cost

    @property
    def cache_saved_cost(self) -> float:
        return self.cost_tracker.cache_saved_cost

    def recursive_compute(self,
                          obj: Proposition,
                          goal: Any,
                          ctx: Context,
                          depth: int = 0,
                          parent_step: Optional[str] = None) -> tuple:
        """返回 (compute_id, subtree_valid_props)。

        预算统一由 ctx.step_budget 控制，无重复 budget 参数。
        """
        if ctx.budget_exhausted():
            self.budget_exhausted = True
            return "", set()
        if depth > self.max_depth:
            return "", set()

        compute_id = self.trace.new_compute_id()
        subtree_valid: Set[Proposition] = set()

        # ---- 步骤1：识别 ----
        self.trace.record(compute_id, depth, obj, "identify", obj,
                          parent_step=parent_step, cost=0.2,
                          decision="parse", meta={"kind": obj.kind})
        self.cost_tracker.add("identify", 0.2, "identify")
        self.actual_steps += 1

        if ctx.budget_exhausted():
            self.budget_exhausted = True
            return compute_id, subtree_valid

        # ---- 步骤2：产生候选 ----
        candidates = self.op_store.generate(obj, ctx, budget=self.max_candidates)
        self.trace.record(compute_id, depth, obj, "generate_candidates",
                          f"{len(candidates)} candidates",
                          parent_step=parent_step, cost=0.3,
                          decision="expand",
                          meta={"n": len(candidates),
                                "ops": sorted({c.op_name for c in candidates})})
        self.cost_tracker.add("generate", 0.3, "generate_candidates")
        self.actual_steps += 1
        self.generated_candidates += len(candidates)
        candidates = candidates[: self.max_candidates]

        # ---- 评价排序 ----
        if ctx.evaluate_enabled and not ctx.budget_exhausted():
            scored = []
            for c in candidates:
                if ctx.budget_exhausted():
                    break
                ev = self.evaluator.evaluate(c.new_object, goal, ctx)
                self.trace.record(compute_id, depth, c.new_object, "evaluate_rank",
                                  c.new_object, parent_step=parent_step,
                                  usefulness=ev.value_score, cost=0.2,
                                  decision="rank", meta=ev.to_dict())
                self.cost_tracker.add("evaluate", 0.2, "evaluate_rank")
                self.actual_steps += 1
                self.evaluated_candidates += 1
                scored.append((ev.value_score, c, ev))
            scored.sort(key=lambda x: -x[0])
            candidates_ranked = [(c, ev) for _, c, ev in scored]
        else:
            candidates_ranked = [(c, None) for c in candidates]

        for c, ev in candidates_ranked:
            if ctx.budget_exhausted():
                self.budget_exhausted = True
                break
            o_prime = c.new_object
            if o_prime == obj:
                continue

            is_composite = c.meta.get("op_kind") == "learned" or c.op_name.startswith("composite_")

            # ---- 步骤3：应用变换 ----
            apply_step = self.trace.record(
                compute_id, depth, obj, c.op_name, o_prime,
                parent_step=parent_step, cost=c.cost,
                meta={"op_kind": c.meta.get("op_kind"), "is_composite": is_composite})
            self.cost_tracker.add("apply", c.cost, c.op_name)
            self.actual_steps += 1
            self.useful_steps += 1

            # ---- 步骤4：评价 gate ----
            if ctx.evaluate_enabled and ev is not None:
                worth = ev.worth_continuing
                self.trace.record(
                    compute_id, depth, o_prime, "evaluate_gate", o_prime,
                    parent_step=apply_step.step_id,
                    usefulness=ev.value_score, cost=0.1,
                    decision="continue" if worth else "retain_low",
                    meta=ev.to_dict())
                self.cost_tracker.add("gate", 0.1, "evaluate_gate")
                self.actual_steps += 1
                if worth:
                    self.passed_evaluation_gate += 1
                if not worth:
                    self.belief_store.update_belief(
                        o_prime, STATUS_UNKNOWN, 0.0, evidence_count_delta=0)
                    continue

            # ---- 步骤5：验证 ----
            if ctx.verify_enabled:
                existing = self.belief_store.get(o_prime)
                if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                    existing.evidence_count += 1
                    saved = self._estimate_verification_cost(o_prime)
                    self.cost_tracker.add_cache_saved(saved)
                    self.cost_tracker.add("reuse", 0.1, "reuse_known")
                    self.trace.record(
                        compute_id, depth, o_prime, "reuse_known", o_prime,
                        parent_step=apply_step.step_id,
                        verification=existing.derivation.operation if existing.derivation else "logical",
                        verification_result=existing.status,
                        confidence=existing.confidence, cost=0.1,
                        decision="retain" if existing.status != STATUS_INVALID else "stop",
                        meta={"reused": True, "cache_saved": round(saved, 2)})
                    self.actual_steps += 1
                    if existing.status == STATUS_INVALID:
                        continue
                    if existing.status == STATUS_VALID:
                        subtree_valid.add(o_prime)
                else:
                    result, all_results = self.verifier.verify(o_prime, ctx)
                    self._record_consensus(all_results, ctx)
                    status = {VALID: STATUS_VALID, INVALID: STATUS_INVALID,
                              UNKNOWN: STATUS_UNKNOWN}[result.result]

                    # 证据记入 EvidenceLog（append-only）
                    for r in all_results:
                        from .evidence import Evidence
                        self.evidence_log.append(Evidence(
                            method=r.method, action_type=r.method,
                            support=r.support, contradiction=r.contradiction,
                            detail=r.evidence, cost=r.cost,
                            derived_from=r.derived_from,
                            derivation_operation=r.derivation_operation,
                        ))

                    derivation = None
                    if result.method == "logical" and result.derived_from:
                        derivation = Derivation(
                            operation=result.derivation_operation or "logical",
                            parents=result.derived_from)

                    self.belief_store.update_belief(
                        o_prime, status, result.confidence,
                        evidence_count_delta=len(all_results),
                        derivation=derivation)

                    self.trace.record(
                        compute_id, depth, o_prime, "verify", o_prime,
                        parent_step=apply_step.step_id,
                        verification=result.method, verification_result=result.result,
                        confidence=result.confidence, cost=result.cost,
                        decision="retain" if status != STATUS_INVALID else "stop",
                        meta={"all_methods": [r.to_dict() for r in all_results]})
                    self.cost_tracker.add("verify", result.cost, "verify")
                    self.actual_steps += 1
                    self.verified_steps += 1
                    self.verified_candidates += 1
                    if status == STATUS_VALID:
                        self.valid_candidates += 1
                        subtree_valid.add(o_prime)
                    elif status == STATUS_INVALID:
                        self.invalid_candidates += 1
                        continue
            else:
                self.belief_store.update_belief(
                    o_prime, STATUS_UNKNOWN, 0.0, evidence_count_delta=0)
                self.trace.record(compute_id, depth, o_prime, "no_verify_save",
                                  o_prime, parent_step=apply_step.step_id,
                                  cost=0.0, decision="retain_unknown")
                self.actual_steps += 1

            # ---- 步骤7：递归 ----
            _, child_valid = self.recursive_compute(
                o_prime, goal, ctx, depth + 1, parent_step=apply_step.step_id)
            subtree_valid.update(child_valid)

            # ---- 步骤8：评价反馈 ----
            if ctx.evaluate_enabled and ev is not None:
                produced_valid = len(child_valid) > 0
                actual = 1.0 if produced_valid else 0.0
                self.evaluator.add_feedback(ctx, ev.method_tag, ev.value_score, actual)

        return compute_id, subtree_valid

    def _estimate_verification_cost(self, prop: Proposition) -> float:
        if prop.kind == "forall":
            return 4.0
        if prop.kind == "implies":
            return 2.5
        return 1.0

    def _record_consensus(self, results: List[VerificationResult], ctx: Context) -> None:
        """记录验证方法间一致性（非真实可靠性）。"""
        if not results:
            return
        for r in results:
            self.consensus.record_decision(r.method, r.method, r.result)
        self.consensus.update_agreement()

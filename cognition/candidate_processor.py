"""CandidateProcessor：处理单个 candidate 的全流程。

职责（从 ComputeEngine 中抽取）：
  Candidate → evaluate → gate → verify → evidence log → belief update → trace → cost

不负责：
  - candidate 生成（ComputeEngine 负责）
  - 递归（ComputeEngine 负责）

依赖方向：
  CandidateProcessor 依赖 Evidence/Belief/Value/Cost/Consensus/Prediction
  ComputeEngine 只依赖 CandidateProcessor 和 OperationStore
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional, Set

from .proposition import Proposition
from .belief import BeliefStore
from .evidence import EvidenceLog, Evidence
from .cost import CostTracker
from .consensus import ConsensusAgreementModel
from .prediction import TemporalPredictionState
from .operations import Context, Candidate
from .verification import Verifier, VerificationResult, VALID, INVALID, UNKNOWN
from .evaluation import ValueEvaluator
from .trace import TraceRecorder
from .models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN, Derivation


class CandidateProcessor:
    """处理单个 candidate 的组件。

    持有所有需要写的 Store，但不暴露给 Operation。
    """

    def __init__(self,
                 belief_store: BeliefStore,
                 evidence_log: EvidenceLog,
                 cost_tracker: CostTracker,
                 consensus: ConsensusAgreementModel,
                 prediction_state: TemporalPredictionState,
                 trace: TraceRecorder,
                 verifier: Optional[Verifier] = None,
                 evaluator: Optional[ValueEvaluator] = None):
        self.belief_store = belief_store
        self.evidence_log = evidence_log
        self.cost_tracker = cost_tracker
        self.consensus = consensus
        self.prediction_state = prediction_state
        self.trace = trace
        self.verifier = verifier or Verifier()
        self.evaluator = evaluator or ValueEvaluator()
        self._eval_feedback: List[dict] = []

    @property
    def eval_feedback(self) -> List[dict]:
        return self._eval_feedback

    def process(self, candidate: Candidate, compute_id: str, depth: int,
                obj: Proposition, goal: Any, ctx: Context,
                parent_step: Optional[str], stats: dict
                ) -> tuple:
        """处理一个 candidate。

        返回 (candidate_status, valid_set, apply_step_id)：
          candidate_status: "valid" / "invalid" / "unknown" / "gated_out"
                            （仅表示本 candidate 自身的直接结果，不含 descendant）
          valid_set: 该 candidate 直接产出的有效命题集合（仅 valid 状态非空）
          apply_step_id: 本 candidate 的 apply 步骤 ID，用于作为递归子调用的 parent_step。
        """
        o_prime = candidate.new_object
        apply_step_id: Optional[str] = None
        if o_prime == obj:
            return "unknown", set(), apply_step_id

        is_composite = candidate.meta.get("op_kind") == "learned" or candidate.op_name.startswith("composite_")

        # 步骤3：应用变换
        apply_step = self.trace.record(
            compute_id, depth, obj, candidate.op_name, o_prime,
            parent_step=parent_step, cost=candidate.cost,
            meta={"op_kind": candidate.meta.get("op_kind"), "is_composite": is_composite})
        apply_step_id = apply_step.step_id
        self.cost_tracker.add("apply", candidate.cost, candidate.op_name)
        stats["actual_steps"] += 1
        ctx.increment_step()
        stats["useful_steps"] += 1

        # 预评价（由 ComputeEngine 完成并通过 candidate 传入）
        ev = candidate.meta.get("__eval_result")

        # 步骤4：评价 gate
        if ctx.evaluate_enabled and ev is not None:
            worth = ev.worth_continuing
            self.trace.record(
                compute_id, depth, o_prime, "evaluate_gate", o_prime,
                parent_step=apply_step.step_id,
                usefulness=ev.value_score, cost=0.1,
                decision="continue" if worth else "retain_low",
                meta=ev.to_dict())
            self.cost_tracker.add("gate", 0.1, "evaluate_gate")
            stats["actual_steps"] += 1
            ctx.increment_step()
            if worth:
                stats["passed_evaluation_gate"] += 1
            if not worth:
                self.belief_store.update_belief(
                    o_prime, STATUS_UNKNOWN, 0.0, evidence_count_delta=0)
                return "gated_out", set(), apply_step_id

        # 步骤5：验证
        if ctx.verify_enabled:
            existing = self.belief_store.get(o_prime)
            if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                # 复用已有知识：不增加 evidence_count
                self.belief_store.record_reuse(o_prime)
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
                stats["actual_steps"] += 1
                ctx.increment_step()
                if existing.status == STATUS_INVALID:
                    return "invalid", set(), apply_step_id
                if existing.status == STATUS_VALID:
                    return "valid", {o_prime}, apply_step_id
            else:
                # 执行验证
                self.belief_store.record_verification(o_prime)
                result, all_results = self.verifier.verify(o_prime, ctx)
                self._record_consensus(all_results)
                status = {VALID: STATUS_VALID, INVALID: STATUS_INVALID,
                          UNKNOWN: STATUS_UNKNOWN}[result.result]

                # 本次验证的源事件 ID（每次验证调用是一个新事件）
                source_event_id = uuid.uuid4().hex[:12]
                obs_step = ctx.current_step

                # 证据记入 EvidenceLog（append-only，按 source_event+proposition+method 去重）
                added = 0
                for r in all_results:
                    evi = Evidence(
                        method=r.method, action_type=r.method,
                        support=r.support, contradiction=r.contradiction,
                        detail=r.evidence, cost=r.cost,
                        derived_from=r.derived_from,
                        derivation_operation=r.derivation_operation,
                        proposition=o_prime,
                        source_event_id=source_event_id,
                        observation_step=obs_step,
                    )
                    if self.evidence_log.append(evi):
                        added += 1

                derivation = None
                if result.method == "logical" and result.derived_from:
                    derivation = Derivation(
                        operation=result.derivation_operation or "logical",
                        parents=result.derived_from)

                # evidence_count 只按实际新增 Evidence 数增加
                self.belief_store.update_belief(
                    o_prime, status, result.confidence,
                    evidence_count_delta=added,
                    derivation=derivation)

                self.trace.record(
                    compute_id, depth, o_prime, "verify", o_prime,
                    parent_step=apply_step.step_id,
                    verification=result.method, verification_result=result.result,
                    confidence=result.confidence, cost=result.cost,
                    decision="retain" if status != STATUS_INVALID else "stop",
                    meta={"all_methods": [r.to_dict() for r in all_results],
                          "new_evidence": added})
                self.cost_tracker.add("verify", result.cost, "verify")
                stats["actual_steps"] += 1
                ctx.increment_step()
                stats["verified_steps"] += 1
                stats["verified_candidates"] += 1
                if status == STATUS_VALID:
                    stats["valid_candidates"] += 1
                    return "valid", {o_prime}, apply_step_id
                elif status == STATUS_INVALID:
                    stats["invalid_candidates"] += 1
                    return "invalid", set(), apply_step_id
        else:
            self.belief_store.update_belief(
                o_prime, STATUS_UNKNOWN, 0.0, evidence_count_delta=0)
            self.trace.record(compute_id, depth, o_prime, "no_verify_save",
                              o_prime, parent_step=apply_step.step_id,
                              cost=0.0, decision="retain_unknown")
            stats["actual_steps"] += 1
            ctx.increment_step()

        return "unknown", set(), apply_step_id

    def record_feedback(self, ev, candidate_status: str,
                        candidate_valid: bool,
                        descendant_valid: bool = False,
                        goal_improvement: Optional[bool] = None) -> None:
        """记录评价反馈，三个结果独立保存。

        candidate_status:   "valid" / "invalid" / "unknown" / "gated_out"
        candidate_valid:    candidate 自身是否被验证为 valid（status == "valid"）
        descendant_valid:   继续递归后 descendant 是否产生 valid
        goal_improvement:   goal 是否得到改善；当前无可靠 goal-state transition 定义，
                            默认 None（unknown），禁止用 goal_relevance 冒充。
        """
        actual = 1.0 if candidate_valid else 0.0
        ValueEvaluator.record_feedback(
            self._eval_feedback, ev.method_tag, ev.value_score, actual,
            extra={"candidate_status": candidate_status,
                   "candidate_valid": candidate_valid,
                   "descendant_valid": descendant_valid,
                   "goal_improvement": goal_improvement})

    def meta_evaluate(self) -> dict:
        """根据累积反馈调整评价风格权重。"""
        return self.evaluator.evaluate_evaluation(self._eval_feedback)

    def _estimate_verification_cost(self, prop: Proposition) -> float:
        if prop.kind == "forall":
            return 4.0
        if prop.kind == "implies":
            return 2.5
        return 1.0

    def _record_consensus(self, results: List[VerificationResult]) -> None:
        """记录验证方法间一致性（非真实可靠性）。"""
        if not results:
            return
        for r in results:
            self.consensus.record_decision(r.method, r.method, r.result)
        self.consensus.update_agreement()

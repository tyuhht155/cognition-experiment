"""ComputeEngine：递归计算引擎（编排层）。

职责（冻结后）：
  当前对象 → candidate generation → 调用 CandidateProcessor 处理 → 递归新对象

不负责：
  - candidate 的 evaluate/gate/verify/belief update/trace/cost
    （全部委托给 CandidateProcessor）
  - 不直接访问 Store 内部字典
  - 不直接修改 belief / evidence / cost
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
from .evaluation import ValueEvaluator
from .trace import TraceRecorder
from .models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from .candidate_processor import CandidateProcessor


MAX_DEPTH_DEFAULT = 3
MAX_CANDIDATES_PER_OBJECT = 6


class ComputeEngine:
    """计算引擎：只负责编排，候选处理全部委托给 CandidateProcessor。"""

    def __init__(self,
                 belief_store: BeliefStore,
                 evidence_log: EvidenceLog,
                 cost_tracker: CostTracker,
                 consensus: ConsensusAgreementModel,
                 prediction_state: TemporalPredictionState,
                 op_store: OperationStore,
                 trace: TraceRecorder,
                 verifier=None,
                 evaluator=None,
                 processor: Optional[CandidateProcessor] = None,
                 max_depth: int = MAX_DEPTH_DEFAULT,
                 max_candidates: int = MAX_CANDIDATES_PER_OBJECT):
        self.op_store = op_store
        self.trace = trace
        self.max_depth = max_depth
        self.max_candidates = max_candidates

        # CandidateProcessor 持有所有 Store
        if processor is not None:
            self.processor = processor
        else:
            self.processor = CandidateProcessor(
                belief_store=belief_store,
                evidence_log=evidence_log,
                cost_tracker=cost_tracker,
                consensus=consensus,
                prediction_state=prediction_state,
                trace=trace,
                verifier=verifier,
                evaluator=evaluator,
            )

        # 只读引用（用于兼容属性和统计）
        self._belief_store = belief_store
        self._cost_tracker = cost_tracker
        self._prediction_state = prediction_state

        # 搜索空间统计（单一共享 dict，所有递归层引用同一对象）
        self._stats = {
            "generated_candidates": 0,
            "evaluated_candidates": 0,
            "passed_evaluation_gate": 0,
            "verified_candidates": 0,
            "valid_candidates": 0,
            "invalid_candidates": 0,
            "actual_steps": 0,
            "useful_steps": 0,
            "verified_steps": 0,
        }

        # 预算统计
        self.budget_exhausted = False

    @property
    def total_cost(self) -> float:
        return self._cost_tracker.total_cost

    @property
    def raw_compute_cost(self) -> float:
        return self._cost_tracker.raw_compute_cost

    @property
    def verification_cost(self) -> float:
        return self._cost_tracker.verification_cost

    @property
    def reuse_cost(self) -> float:
        return self._cost_tracker.reuse_cost

    @property
    def cache_saved_cost(self) -> float:
        return self._cost_tracker.cache_saved_cost

    def recursive_compute(self,
                          obj: Proposition,
                          goal: Any,
                          ctx: Context,
                          depth: int = 0,
                          parent_step: Optional[str] = None) -> tuple:
        """返回 (compute_id, subtree_valid_props)。"""
        if ctx.budget_exhausted():
            self.budget_exhausted = True
            return "", set()
        if depth > self.max_depth:
            return "", set()

        compute_id = self.trace.new_compute_id()
        subtree_valid: Set[Proposition] = set()
        stats = self._stats_dict()

        # 步骤1：识别
        self.trace.record(compute_id, depth, obj, "identify", obj,
                          parent_step=parent_step, cost=0.2,
                          decision="parse", meta={"kind": obj.kind})
        self._cost_tracker.add("identify", 0.2, "identify")
        stats["actual_steps"] += 1
        ctx.increment_step()

        if ctx.budget_exhausted():
            self.budget_exhausted = True
            return compute_id, subtree_valid

        # 步骤2：产生候选
        candidates = self.op_store.generate(obj, ctx, budget=self.max_candidates)
        self.trace.record(compute_id, depth, obj, "generate_candidates",
                          f"{len(candidates)} candidates",
                          parent_step=parent_step, cost=0.3,
                          decision="expand",
                          meta={"n": len(candidates),
                                "ops": sorted({c.op_name for c in candidates})})
        self._cost_tracker.add("generate", 0.3, "generate_candidates")
        stats["actual_steps"] += 1
        ctx.increment_step()
        stats["generated_candidates"] += len(candidates)
        candidates = candidates[: self.max_candidates]

        # 预评价（在处理前统一评价，便于排序和 gate）
        candidates_ranked = self._pre_evaluate(candidates, goal, ctx, compute_id, depth, parent_step, stats)

        for candidate, ev in candidates_ranked:
            if ctx.budget_exhausted():
                self.budget_exhausted = True
                break
            if candidate.new_object == obj:
                continue

            # 把评价结果塞入 candidate.meta 供 CandidateProcessor 读取
            candidate.meta["__eval_result"] = ev

            status, valid, apply_step_id = self.processor.process(
                candidate, compute_id, depth, obj, goal, ctx, parent_step, stats)

            # candidate 自身的直接验证结果
            candidate_valid = (status == "valid")

            # goal_improvement：当前无可靠 goal-state transition 定义，暂记 None
            goal_improvement = None

            subtree_valid.update(valid)

            # 递归：parent_step 绑定到本 candidate 的 apply_step_id
            # invalid / gated_out 不递归（已 stop / 被 gate 拒绝）
            descendant_valid = False
            if status not in ("invalid", "gated_out") and obj != candidate.new_object and apply_step_id is not None:
                _, child_valid = self.recursive_compute(
                    candidate.new_object, goal, ctx, depth + 1,
                    parent_step=apply_step_id)
                subtree_valid.update(child_valid)
                descendant_valid = len(child_valid) > 0

            # 评价反馈：所有已执行的 candidate（含 invalid/unknown/gated_out）都产生 feedback
            if ctx.evaluate_enabled and ev is not None:
                self.processor.record_feedback(
                    ev, status, candidate_valid, descendant_valid, goal_improvement)

        self._sync_stats(stats)
        return compute_id, subtree_valid

    def _pre_evaluate(self, candidates, goal, ctx, compute_id, depth, parent_step, stats):
        if ctx.evaluate_enabled and not ctx.budget_exhausted():
            scored = []
            for c in candidates:
                if ctx.budget_exhausted():
                    break
                ev = self.processor.evaluator.evaluate(c.new_object, goal, ctx)
                self.trace.record(compute_id, depth, c.new_object, "evaluate_rank",
                                  c.new_object, parent_step=parent_step,
                                  usefulness=ev.value_score, cost=0.2,
                                  decision="rank", meta=ev.to_dict())
                self._cost_tracker.add("evaluate", 0.2, "evaluate_rank")
                stats["actual_steps"] += 1
                ctx.increment_step()
                stats["evaluated_candidates"] += 1
                scored.append((ev.value_score, c, ev))
            scored.sort(key=lambda x: -x[0])
            return [(c, ev) for _, c, ev in scored]
        return [(c, None) for c in candidates]

    def _stats_dict(self) -> dict:
        """返回共享统计 dict 的引用（不是副本）。
        所有递归层修改同一个 dict，避免父层快照覆盖子层统计。
        """
        return self._stats

    def _sync_stats(self, stats: dict) -> None:
        """从共享 dict 同步到实例属性（向后兼容外部读取）。"""
        self.generated_candidates = stats["generated_candidates"]
        self.evaluated_candidates = stats["evaluated_candidates"]
        self.passed_evaluation_gate = stats["passed_evaluation_gate"]
        self.verified_candidates = stats["verified_candidates"]
        self.valid_candidates = stats["valid_candidates"]
        self.invalid_candidates = stats["invalid_candidates"]
        self.actual_steps = stats["actual_steps"]
        self.useful_steps = stats["useful_steps"]
        self.verified_steps = stats["verified_steps"]

    # 兼容属性
    @property
    def store(self):
        return self._belief_store

    @property
    def belief_store(self):
        return self._belief_store

    @property
    def cost_tracker(self):
        return self._cost_tracker

    @property
    def prediction_state(self):
        return self._prediction_state

    @property
    def registry(self):
        return self.op_store._registry

    @property
    def verifier(self):
        return self.processor.verifier

    @property
    def evaluator(self):
        return self.processor.evaluator

    @property
    def verification(self):
        return self.processor.verifier

    @property
    def evaluation(self):
        return self.processor.evaluator

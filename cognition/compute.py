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

            valid = self.processor.process(
                candidate, compute_id, depth, obj, goal, ctx, parent_step, stats)

            if valid is None:
                # 被 stop（invalid）
                continue

            # candidate 自身的直接验证结果（process 返回非空 set 表示直接 valid）
            candidate_direct_valid = valid is not None and len(valid) > 0

            subtree_valid.update(valid)

            # 递归
            descendant_valid = False
            if obj != candidate.new_object:
                _, child_valid = self.recursive_compute(
                    candidate.new_object, goal, ctx, depth + 1,
                    parent_step=self._last_step_id(ctx))
                subtree_valid.update(child_valid)
                descendant_valid = len(child_valid) > 0

            # 评价反馈：绑定到 candidate 自身的直接结果，而非 descendant 结果
            if ctx.evaluate_enabled and ev is not None:
                self.processor.record_feedback(ev, candidate_direct_valid, descendant_valid)

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
        return {
            "generated_candidates": self.generated_candidates,
            "evaluated_candidates": self.evaluated_candidates,
            "passed_evaluation_gate": self.passed_evaluation_gate,
            "verified_candidates": self.verified_candidates,
            "valid_candidates": self.valid_candidates,
            "invalid_candidates": self.invalid_candidates,
            "actual_steps": self.actual_steps,
            "useful_steps": self.useful_steps,
            "verified_steps": self.verified_steps,
        }

    def _sync_stats(self, stats: dict) -> None:
        self.generated_candidates = stats["generated_candidates"]
        self.evaluated_candidates = stats["evaluated_candidates"]
        self.passed_evaluation_gate = stats["passed_evaluation_gate"]
        self.verified_candidates = stats["verified_candidates"]
        self.valid_candidates = stats["valid_candidates"]
        self.invalid_candidates = stats["invalid_candidates"]
        self.actual_steps = stats["actual_steps"]
        self.useful_steps = stats["useful_steps"]
        self.verified_steps = stats["verified_steps"]

    def _last_step_id(self, ctx: Context) -> Optional[str]:
        # trace 的最后一步；ComputeEngine 不持有 trace 内部列表，通过 record 返回值获取
        # 这里返回 None，parent_step 在递归调用处由 trace.record 的返回值提供更准确
        return None

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

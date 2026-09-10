"""统一的递归计算机制 recursive_compute。

理论第 4 条过程：
  当前对象 O -> 识别/解析 -> 产生候选变换 -> 应用 -> O' -> 验证 ->
  错误则停分支；成立则保存 -> 计算价值 -> 有价值则继续；当前无价值则保留但降优先级 -> 重复

第二轮审计修复：
  - verifier reliability 明确标记为 consensus-based self-estimation，非 ground-truth。
  - meta-evaluation 调度统一在 run_abcd.py，ComputeEngine 不触发。
  - evaluation feedback 使用精确 attribution（recursive_compute 返回该子树产出的 valid 命题集合），
    不再使用全局 store valid 数量变化。
  - 搜索空间统计：generated/evaluated/passed_gate/verified/valid/invalid。
  - 预算统计：budget_exhausted / actual_steps / useful_steps / verified_steps。

A/B/C/D 实验组通过 ctx 的开关区分：
  A: 只有生成（不验证）           verify_enabled=False, evaluate_enabled=False
  B: 生成 + 验证                  verify_enabled=True,  evaluate_enabled=False
  C: 生成 + 验证 + 价值评价       verify_enabled=True,  evaluate_enabled=True,  meta=False
  D: 生成 + 验证 + 价值评价 + 验证方法评价  (meta=True)
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

from .proposition import Proposition
from .knowledge_store import KnowledgeStore, Knowledge, STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from .operations import OperationRegistry, Context, Candidate
from .verification import Verification, VerificationResult, VALID, INVALID, UNKNOWN
from .evaluation import Evaluation
from .trace import Trace


MAX_DEPTH_DEFAULT = 3
MAX_CANDIDATES_PER_OBJECT = 6


class ComputeEngine:
    """承载一次实验配置的计算引擎。"""

    def __init__(self,
                 store: KnowledgeStore,
                 trace: Trace,
                 registry: OperationRegistry,
                 verification: Optional[Verification] = None,
                 evaluation: Optional[Evaluation] = None,
                 max_depth: int = MAX_DEPTH_DEFAULT,
                 max_candidates: int = MAX_CANDIDATES_PER_OBJECT):
        self.store = store
        self.trace = trace
        self.registry = registry
        self.verification = verification or Verification()
        self.evaluation = evaluation or Evaluation()
        self.max_depth = max_depth
        self.max_candidates = max_candidates

        # ---- 成本拆分 ----
        self.total_cost = 0.0
        self.raw_compute_cost = 0.0
        self.verification_cost = 0.0
        self.reuse_cost = 0.0
        self.cache_saved_cost = 0.0
        self.composite_saved_cost = 0.0

        # ---- 复合操作调用追踪 ----
        self.composite_usage: dict = {}

        # ---- 搜索空间统计（item 6）----
        self.generated_candidates = 0
        self.evaluated_candidates = 0
        self.passed_evaluation_gate = 0
        self.verified_candidates = 0
        self.valid_candidates = 0
        self.invalid_candidates = 0

        # ---- 预算统计（item 9）----
        self.budget_exhausted = False
        self.actual_steps = 0       # trace 中实际记录的步骤数
        self.useful_steps = 0       # 产生新命题的步骤（应用变换）
        self.verified_steps = 0     # 执行了验证的步骤

    def recursive_compute(self,
                          obj: Proposition,
                          goal: Any,
                          ctx: Context,
                          depth: int = 0,
                          parent_step: Optional[str] = None,
                          budget: Optional[int] = None) -> tuple:
        """返回 (compute_id, subtree_valid_props)。
        subtree_valid_props 是本次调用（含子递归）中新验证为 valid 的命题集合，
        用于精确的 evaluation feedback attribution（不使用全局 valid 计数）。
        """
        if ctx.budget_exhausted():
            self.budget_exhausted = True
            return "", set()
        if depth > self.max_depth:
            return "", set()
        if budget is not None and budget <= 0:
            return "", set()

        compute_id = self.trace.new_compute_id()
        subtree_valid: Set[Proposition] = set()

        # ---- 步骤1：识别/解析 ----
        self.trace.record(compute_id, depth, obj, "identify", obj,
                          parent_step=parent_step, cost=0.2,
                          decision="parse", meta={"kind": obj.kind})
        self.raw_compute_cost += 0.2
        self.actual_steps += 1

        if ctx.budget_exhausted():
            self.budget_exhausted = True
            return compute_id, subtree_valid

        # ---- 步骤2：产生候选变换 ----
        candidates = self.registry.generate(obj, ctx, budget=self.max_candidates)
        self.trace.record(compute_id, depth, obj, "generate_candidates",
                          f"{len(candidates)} candidates",
                          parent_step=parent_step, cost=0.3,
                          decision="expand",
                          meta={"n": len(candidates),
                                "ops": sorted({c.op_name for c in candidates})})
        self.raw_compute_cost += 0.3
        self.actual_steps += 1
        self.generated_candidates += len(candidates)

        candidates = candidates[: self.max_candidates]

        # C/D 组：评价 + 排序
        if ctx.evaluate_enabled and not ctx.budget_exhausted():
            scored = []
            for c in candidates:
                if ctx.budget_exhausted():
                    break
                ev = self.evaluation.evaluate(c.new_object, goal, ctx)
                self.trace.record(compute_id, depth, c.new_object, "evaluate_rank",
                                  c.new_object, parent_step=parent_step,
                                  usefulness=ev.value_score, cost=0.2,
                                  decision="rank", meta=ev.to_dict())
                self.raw_compute_cost += 0.2
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

            # 复合操作标记
            is_composite = c.meta.get("op_kind") == "learned" or c.op_name.startswith("composite_")
            if is_composite:
                self.composite_usage[c.op_name] = self.composite_usage.get(c.op_name, 0) + 1
                self.composite_saved_cost += 0.5

            # ---- 步骤3：应用变换 ----
            apply_step = self.trace.record(
                compute_id, depth, obj, c.op_name, o_prime,
                parent_step=parent_step, cost=c.cost,
                meta={"op_kind": c.meta.get("op_kind"), "is_composite": is_composite})
            self.raw_compute_cost += c.cost
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
                self.raw_compute_cost += 0.1
                self.actual_steps += 1
                if worth:
                    self.passed_evaluation_gate += 1
                if not worth:
                    k_low = Knowledge(
                        proposition=o_prime, status=STATUS_UNKNOWN, confidence=0.0,
                        usefulness=ev.value_score, source=compute_id,
                        parent_objects=[obj.to_str()], operation=c.op_name,
                        cost=c.cost + 0.1, kind="proposition", evaluated=True)
                    self.store.upsert(k_low)
                    continue

            # ---- 步骤5：验证 ----
            if ctx.verify_enabled:
                existing = self.store.get(o_prime)
                if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                    existing.usage_count += 1
                    saved = self._estimate_verification_cost(o_prime)
                    self.cache_saved_cost += saved
                    self.reuse_cost += 0.1
                    self.trace.record(
                        compute_id, depth, o_prime, "reuse_known", o_prime,
                        parent_step=apply_step.step_id,
                        verification=existing.verification_method,
                        verification_result=existing.verification_result,
                        confidence=existing.confidence, cost=0.1,
                        decision="retain" if existing.status != STATUS_INVALID else "stop",
                        meta={"reused": True, "cache_saved": round(saved, 2)})
                    self.total_cost += 0.1
                    self.actual_steps += 1
                    if existing.status == STATUS_INVALID:
                        continue
                    if existing.status == STATUS_VALID:
                        subtree_valid.add(o_prime)
                else:
                    result, all_results = self.verification.verify(o_prime, ctx)
                    # consensus-based self-estimation（非 ground-truth reliability）
                    self._record_consensus_verifier_use(all_results, ctx)
                    status = {VALID: STATUS_VALID, INVALID: STATUS_INVALID,
                              UNKNOWN: STATUS_UNKNOWN}[result.result]
                    use_score = ev.value_score if (ev is not None) else None
                    # 逻辑推导溯源：如果验证方法是 logical 且有推导信息，
                    # 用 derived_from 作为 parents，derivation_operation 作为 operation
                    if result.method == "logical" and result.derived_from:
                        parents = result.derived_from
                        op = result.derivation_operation or "logical"
                    else:
                        parents = [obj.to_str()]
                        op = c.op_name
                    k = Knowledge(
                        proposition=o_prime, status=status, confidence=result.confidence,
                        usefulness=use_score, source=compute_id,
                        parent_objects=parents, operation=op,
                        verification_method=result.method, verification_result=result.result,
                        cost=result.cost + c.cost, kind="proposition",
                        evaluated=(ev is not None),
                        evidence_history=[{
                            "method": result.method,
                            "support": result.support,
                            "contradiction": result.contradiction,
                            "confidence": result.confidence,
                            "detail": result.evidence,
                        }],
                        verification_history=[{
                            "method": result.method,
                            "result": result.result,
                            "confidence": result.confidence,
                        }])
                    self.store.upsert(k)
                    self.trace.record(
                        compute_id, depth, o_prime, "verify", o_prime,
                        parent_step=apply_step.step_id,
                        verification=result.method, verification_result=result.result,
                        confidence=result.confidence, cost=result.cost,
                        decision="retain" if status != STATUS_INVALID else "stop",
                        meta={"all_methods": [r.to_dict() for r in all_results],
                              "is_composite": is_composite})
                    self.verification_cost += result.cost
                    self.total_cost += result.cost + c.cost
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
                # A 组：不验证
                k = Knowledge(
                    proposition=o_prime, status=STATUS_UNKNOWN, confidence=0.0,
                    usefulness=None, source=compute_id,
                    parent_objects=[obj.to_str()], operation=c.op_name,
                    cost=c.cost, kind="proposition", evaluated=False)
                self.store.upsert(k)
                self.trace.record(compute_id, depth, o_prime, "no_verify_save",
                                  o_prime, parent_step=apply_step.step_id,
                                  cost=0.0, decision="retain_unknown")
                self.total_cost += c.cost
                self.actual_steps += 1

            # ---- 步骤7：递归展开（meta-evaluation 不在此触发，由 run_abcd.py 统一调度）----
            _, child_valid = self.recursive_compute(
                o_prime, goal, ctx, depth + 1, parent_step=apply_step.step_id)
            subtree_valid.update(child_valid)

            # ---- 步骤8：运行时评价反馈（精确 attribution）----
            # 仅对当前 candidate 的子树产出的 valid 命题计数，不使用全局 valid 数量
            if ctx.evaluate_enabled and ev is not None:
                produced_valid = len(child_valid) > 0
                actual = 1.0 if produced_valid else 0.0
                self.evaluation.add_feedback(ctx, ev.method_tag, ev.value_score, actual)

        return compute_id, subtree_valid

    def _estimate_verification_cost(self, prop: Proposition) -> float:
        if prop.kind == "forall":
            return 4.0
        if prop.kind == "implies":
            return 2.5
        return 1.0

    def _record_consensus_verifier_use(self, results: List[VerificationResult], ctx: Context) -> None:
        """consensus-based self-estimation：用多数派一致性作为验证方法成功的代理。

        注意：这不是 ground-truth reliability。多数派本身可能错误。
        此机制仅提供"方法间一致性"的自估计，不能证明系统学会了真正的验证可靠性。
        """
        if not results:
            return
        votes = {VALID: 0, INVALID: 0, UNKNOWN: 0}
        for r in results:
            votes[r.result] = votes.get(r.result, 0) + 1
        majority = max(votes, key=votes.get)
        for r in results:
            success = (r.result == majority) and (r.result != UNKNOWN)
            self.store.record_verification_outcome(r.method, success)

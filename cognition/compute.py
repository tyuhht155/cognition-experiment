"""统一的递归计算机制 recursive_compute。

理论第 4 条过程：
  当前对象 O -> 识别/解析 -> 产生候选变换 -> 应用 -> O' -> 验证 ->
  错误则停分支；成立则保存 -> 计算价值 -> 有价值则继续；当前无价值则保留但降优先级 -> 重复

“产生候选变换”本身也是可记录的计算步骤；“验证”本身也是可被评价的对象。
搜索=寻找能作用于当前对象的计算结构；生成=应用之。二者属同一过程（第 5 条）。

A/B/C/D 实验组通过 ctx 的开关区分：
  A: 只有生成（不验证）           verify_enabled=False, evaluate_enabled=False
  B: 生成 + 验证                  verify_enabled=True,  evaluate_enabled=False
  C: 生成 + 验证 + 价值评价       verify_enabled=True,  evaluate_enabled=True,  meta=False
  D: 生成 + 验证 + 价值评价 + 验证方法评价  (meta=True)
"""

from __future__ import annotations

from typing import Any, List, Optional

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
        self.total_cost = 0.0

    def recursive_compute(self,
                          obj: Proposition,
                          goal: Any,
                          ctx: Context,
                          depth: int = 0,
                          parent_step: Optional[str] = None,
                          budget: Optional[int] = None) -> str:
        """返回本次调用的 compute_id。所有副作用写入 store / trace。"""
        if ctx.budget_exhausted():
            return ""
        if depth > self.max_depth:
            return ""
        if budget is not None and budget <= 0:
            return ""

        compute_id = self.trace.new_compute_id()

        # ---- 步骤1：识别/解析当前对象（也是计算）----
        self.trace.record(compute_id, depth, obj, "identify", obj,
                          parent_step=parent_step, cost=0.2,
                          decision="parse", meta={"kind": obj.kind})

        if ctx.budget_exhausted():
            return compute_id

        # ---- 步骤2：产生候选变换（也是计算，记录）----
        candidates = self.registry.generate(obj, ctx, budget=self.max_candidates)
        self.trace.record(compute_id, depth, obj, "generate_candidates",
                          f"{len(candidates)} candidates",
                          parent_step=parent_step, cost=0.3,
                          decision="expand",
                          meta={"n": len(candidates),
                                "ops": sorted({c.op_name for c in candidates})})

        # 限制候选数
        candidates = candidates[: self.max_candidates]

        # C/D 组：先对每个候选做廉价评价，按价值降序处理（评价消耗步数预算）
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
                scored.append((ev.value_score, c, ev))
            scored.sort(key=lambda x: -x[0])
            candidates_ranked = [(c, ev) for _, c, ev in scored]
        else:
            candidates_ranked = [(c, None) for c in candidates]

        for c, ev in candidates_ranked:
            if ctx.budget_exhausted():
                break
            o_prime = c.new_object

            # 跳过自指重复（O->O）
            if o_prime == obj:
                continue

            # ---- 步骤3：应用变换 = 生成 O'（已记录）----
            apply_step = self.trace.record(
                compute_id, depth, obj, c.op_name, o_prime,
                parent_step=parent_step, cost=c.cost,
                meta={"op_kind": c.meta.get("op_kind")})

            # ---- 步骤4：评价（预测）—— C/D 组用排序阶段算得的 ev 决定是否值得验证 ----
            # 理论上“评价”包含对“验证成本”的预测，因此可据此跳过低价值候选的验证，
            # 节省计算。B 组无评价，直接验证所有候选。
            if ctx.evaluate_enabled and ev is not None:
                self.trace.record(
                    compute_id, depth, o_prime, "evaluate_gate", o_prime,
                    parent_step=apply_step.step_id,
                    usefulness=ev.value_score,
                    cost=0.1,
                    decision="continue" if ev.worth_continuing else "retain_low",
                    meta=ev.to_dict())
                if not ev.worth_continuing:
                    # 正确但当前无价值：保留为 unknown（不删除，只降优先级），不验证不展开
                    k_low = Knowledge(
                        proposition=o_prime,
                        status=STATUS_UNKNOWN,
                        confidence=0.0,
                        usefulness=ev.value_score,
                        source=compute_id,
                        parent_objects=[obj.to_str()],
                        operation=c.op_name,
                        cost=c.cost + 0.1,
                        kind="proposition",
                    )
                    self.store.upsert(k_low)
                    self.total_cost += c.cost + 0.1
                    continue

            # ---- 步骤5：验证 ----
            # 捷径：若该命题已作为知识成立/不成立，直接复用，不再验证（知识增长后降本）
            if ctx.verify_enabled:
                existing = self.store.get(o_prime)
                if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                    # 复用已有结论
                    existing.usage_count += 1
                    self.trace.record(
                        compute_id, depth, o_prime, "reuse_known", o_prime,
                        parent_step=apply_step.step_id,
                        verification=existing.verification_method,
                        verification_result=existing.verification_result,
                        confidence=existing.confidence,
                        cost=0.1,
                        decision="retain" if existing.status != STATUS_INVALID else "stop",
                        meta={"reused": True})
                    self.total_cost += 0.1
                    if existing.status == STATUS_INVALID:
                        continue
                else:
                    result, all_results = self.verification.verify(o_prime, ctx)
                    # 记录验证方法使用（D 组元评价会用可靠性）
                    self._record_verifier_use(all_results, ctx)
                    status = {VALID: STATUS_VALID, INVALID: STATUS_INVALID,
                              UNKNOWN: STATUS_UNKNOWN}[result.result]
                    k = Knowledge(
                        proposition=o_prime,
                        status=status,
                        confidence=result.confidence,
                        usefulness=(ev.value_score if ev else 0.0),
                        source=compute_id,
                        parent_objects=[obj.to_str()],
                        operation=c.op_name,
                        verification_method=result.method,
                        verification_result=result.result,
                        cost=result.cost + c.cost,
                        kind="proposition",
                    )
                    self.store.upsert(k)
                    self.trace.record(
                        compute_id, depth, o_prime, "verify", o_prime,
                        parent_step=apply_step.step_id,
                        verification=result.method,
                        verification_result=result.result,
                        confidence=result.confidence,
                        cost=result.cost,
                        decision="retain" if status != STATUS_INVALID else "stop",
                        meta={"all_methods": [r.to_dict() for r in all_results]})
                    self.total_cost += result.cost + c.cost
                    # 错误 -> 停止当前分支
                    if status == STATUS_INVALID:
                        continue
            else:
                # A 组：不验证，全部作为 unknown 保留
                k = Knowledge(
                    proposition=o_prime,
                    status=STATUS_UNKNOWN,
                    confidence=0.0,
                    source=compute_id,
                    parent_objects=[obj.to_str()],
                    operation=c.op_name,
                    cost=c.cost,
                    kind="proposition",
                )
                self.store.upsert(k)
                self.trace.record(compute_id, depth, o_prime, "no_verify_save",
                                  o_prime, parent_step=apply_step.step_id,
                                  cost=0.0, decision="retain_unknown")
                self.total_cost += c.cost

            # ---- 步骤6：D 组元评价（周期性）----
            if ctx.meta_evaluate_enabled and ctx.evaluate_enabled and (len(self.trace) % 50 == 0):
                self.evaluation.evaluate_evaluation(ctx)

            # ---- 步骤7：有价值 -> 继续递归展开 ----
            # B 组（无评价）：ev 为 None，默认展开；C/D 组：仅值得继续的才到这一步
            self.recursive_compute(o_prime, goal, ctx, depth + 1,
                                   parent_step=apply_step.step_id)

        return compute_id

    def _record_verifier_use(self, results: List[VerificationResult], ctx: Context) -> None:
        """记录验证方法使用。是否“成功”以多数派结论代理（后续由环境确认回填）。"""
        if not results:
            return
        # 多数派
        votes = {VALID: 0, INVALID: 0, UNKNOWN: 0}
        for r in results:
            votes[r.result] = votes.get(r.result, 0) + 1
        majority = max(votes, key=votes.get)
        for r in results:
            # 把“与方法多数派一致”视为该方法的初步成功代理
            success = (r.result == majority) and (r.result != UNKNOWN)
            self.store.record_verification_outcome(r.method, success)

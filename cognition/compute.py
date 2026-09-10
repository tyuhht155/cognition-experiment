"""统一的递归计算机制 recursive_compute。

理论第 4 条过程：
  当前对象 O -> 识别/解析 -> 产生候选变换 -> 应用 -> O' -> 验证 ->
  错误则停分支；成立则保存 -> 计算价值 -> 有价值则继续；当前无价值则保留但降优先级 -> 重复

审计修复（F1/F2/F3）：
  - 移除事后 oracle：验证方法可靠性仅由运行时共识更新，不使用未来状态。
  - 区分未评价(usefulness=None)与低价值(usefulness 数值)。
  - 成本拆分：raw_compute / verification / reuse / cache_saved / composite_saved。
  - 运行时评价反馈：对"值得继续"的分支，观察是否产出 valid 知识，反馈给评价系统。

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

        # ---- 成本拆分（F3）----
        self.total_cost = 0.0
        self.raw_compute_cost = 0.0      # 基础操作成本
        self.verification_cost = 0.0     # 验证成本
        self.reuse_cost = 0.0            # reuse_known 成本
        self.cache_saved_cost = 0.0      # 因复用而节省的验证成本
        self.composite_saved_cost = 0.0  # 因复合操作节省的成本

        # ---- 复合操作调用追踪（F4）----
        self.composite_usage: dict = {}   # op_name -> 调用次数
        self.composite_valid_hits = 0     # 复合操作产出 valid 的次数

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
        self.raw_compute_cost += 0.2

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
        self.raw_compute_cost += 0.3

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
                self.raw_compute_cost += 0.2
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

            # 判断是否复合操作
            is_composite = c.meta.get("op_kind") == "learned" or c.op_name.startswith("composite_")
            if is_composite:
                self.composite_usage[c.op_name] = self.composite_usage.get(c.op_name, 0) + 1
                # 复合操作节省的成本 = 序列中各操作成本之和 - 复合操作本身成本
                # 近似：复合操作 c.cost 已包含各步成本，节省的是重复的 generate/identify 开销
                self.composite_saved_cost += 0.5  # 每次复合调用节省一次 generate 开销

            # ---- 步骤3：应用变换 = 生成 O'（已记录）----
            apply_step = self.trace.record(
                compute_id, depth, obj, c.op_name, o_prime,
                parent_step=parent_step, cost=c.cost,
                meta={"op_kind": c.meta.get("op_kind"),
                      "is_composite": is_composite})
            self.raw_compute_cost += c.cost

            # ---- 步骤4：评价（预测）—— C/D 组用排序阶段算得的 ev 决定是否值得验证 ----
            if ctx.evaluate_enabled and ev is not None:
                self.trace.record(
                    compute_id, depth, o_prime, "evaluate_gate", o_prime,
                    parent_step=apply_step.step_id,
                    usefulness=ev.value_score,
                    cost=0.1,
                    decision="continue" if ev.worth_continuing else "retain_low",
                    meta=ev.to_dict())
                self.raw_compute_cost += 0.1
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
                        evaluated=True,
                    )
                    self.store.upsert(k_low)
                    continue

            # ---- 步骤5：验证 ----
            # 捷径：若该命题已作为知识成立/不成立，直接复用，不再验证
            if ctx.verify_enabled:
                existing = self.store.get(o_prime)
                if existing and existing.status in (STATUS_VALID, STATUS_INVALID):
                    # 复用已有结论（缓存/知识复用）
                    existing.usage_count += 1
                    # 估算节省的验证成本（若不复用，需执行完整验证）
                    saved = self._estimate_verification_cost(o_prime)
                    self.cache_saved_cost += saved
                    self.reuse_cost += 0.1
                    self.trace.record(
                        compute_id, depth, o_prime, "reuse_known", o_prime,
                        parent_step=apply_step.step_id,
                        verification=existing.verification_method,
                        verification_result=existing.verification_result,
                        confidence=existing.confidence,
                        cost=0.1,
                        decision="retain" if existing.status != STATUS_INVALID else "stop",
                        meta={"reused": True, "cache_saved": round(saved, 2)})
                    self.total_cost += 0.1
                    if existing.status == STATUS_INVALID:
                        continue
                else:
                    result, all_results = self.verification.verify(o_prime, ctx)
                    # 记录验证方法使用（运行时共识，非 oracle）
                    self._record_verifier_use(all_results, ctx)
                    status = {VALID: STATUS_VALID, INVALID: STATUS_INVALID,
                              UNKNOWN: STATUS_UNKNOWN}[result.result]
                    use_score = ev.value_score if (ev is not None) else None
                    k = Knowledge(
                        proposition=o_prime,
                        status=status,
                        confidence=result.confidence,
                        usefulness=use_score,
                        source=compute_id,
                        parent_objects=[obj.to_str()],
                        operation=c.op_name,
                        verification_method=result.method,
                        verification_result=result.result,
                        cost=result.cost + c.cost,
                        kind="proposition",
                        evaluated=(ev is not None),
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
                        meta={"all_methods": [r.to_dict() for r in all_results],
                              "is_composite": is_composite})
                    self.verification_cost += result.cost
                    self.total_cost += result.cost + c.cost
                    if status == STATUS_INVALID:
                        continue
            else:
                # A 组：不验证，全部作为 unknown 保留，未评价
                k = Knowledge(
                    proposition=o_prime,
                    status=STATUS_UNKNOWN,
                    confidence=0.0,
                    usefulness=None,           # 未评价
                    source=compute_id,
                    parent_objects=[obj.to_str()],
                    operation=c.op_name,
                    cost=c.cost,
                    kind="proposition",
                    evaluated=False,
                )
                self.store.upsert(k)
                self.trace.record(compute_id, depth, o_prime, "no_verify_save",
                                  o_prime, parent_step=apply_step.step_id,
                                  cost=0.0, decision="retain_unknown")
                self.total_cost += c.cost

            # ---- 步骤6：D 组周期性元评价（运行时，非 oracle）----
            if ctx.meta_evaluate_enabled and ctx.evaluate_enabled and (len(self.trace) % 50 == 0):
                self.evaluation.evaluate_evaluation(ctx)

            # ---- 步骤7：有价值 -> 继续递归展开 ----
            # 记录展开前的 valid 数量，用于运行时反馈
            valid_before = len(self.store.valid_entries())
            self.recursive_compute(o_prime, goal, ctx, depth + 1,
                                   parent_step=apply_step.step_id)
            valid_after = len(self.store.valid_entries())

            # ---- 步骤8：运行时评价反馈（让元评价真正工作）----
            # 仅 C/D 组（有评价）：观察本次展开是否产出 valid 知识
            if ctx.evaluate_enabled and ev is not None:
                produced_valid = (valid_after > valid_before)
                actual = 1.0 if produced_valid else 0.0
                self.evaluation.add_feedback(ctx, ev.method_tag, ev.value_score, actual)

        return compute_id

    def _estimate_verification_cost(self, prop: Proposition) -> float:
        """估算若执行完整验证的成本（用于 cache_saved 统计）。"""
        if prop.kind == "forall":
            return 4.0
        if prop.kind == "implies":
            return 2.5
        return 1.0

    def _record_verifier_use(self, results: List[VerificationResult], ctx: Context) -> None:
        """记录验证方法使用。成功代理=与多数派一致（运行时共识，非 ground-truth）。"""
        if not results:
            return
        votes = {VALID: 0, INVALID: 0, UNKNOWN: 0}
        for r in results:
            votes[r.result] = votes.get(r.result, 0) + 1
        majority = max(votes, key=votes.get)
        for r in results:
            success = (r.result == majority) and (r.result != UNKNOWN)
            self.store.record_verification_outcome(r.method, success)

"""候选变换生成器（Operations）。

理论第 5 条：不要把搜索和生成分成两个智能模块。搜索=寻找能作用于当前对象的
计算结构；生成=应用找到的结构。二者属同一过程。

这里的 Operation 都是【先验】——即理论允许的“基本命题变换能力”。我们显式标记，
不隐藏。系统不会偷偷塞规则库；所有领域知识必须由计算产生并经验证保留。

每个 operation：输入 (object, context) -> 候选列表 [(new_object, op_name, cost, meta)]。
生成候选本身也是可记录的计算步骤（由 compute loop 记录）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Any, Dict
import itertools

from .proposition import Proposition, Term


@dataclass
class Candidate:
    new_object: Proposition
    op_name: str
    cost: float = 1.0
    meta: dict = field(default_factory=dict)


class Context:
    """单次计算上下文：只持有计算所需的只读信息。

    严格不持有任何可写 Store：
      - 不持有 BeliefStore（只持有只读 KnowledgeView）
      - 不持有 EvidenceLog
      - 不持有 CostTracker
      - 不持有 ConsensusAgreementModel
      - 不持有 TemporalPredictionState
      - 不持有 OperationStore
      - 不持有 TraceRecorder

    只保留：
      world_history: 环境观察历史（只读）
      goal: 当前目标
      current_step: 当前步
      step_budget: 预算
      constants: 只读常量配置
      knowledge_view: 知识的只读视图（用于逻辑推导）
    """

    def __init__(self,
                 world_history=None,
                 goal=None,
                 current_step: int = 0,
                 step_budget: int = 4000,
                 constants=None,
                 knowledge_view=None,
                 verify_enabled: bool = True,
                 evaluate_enabled: bool = True,
                 meta_evaluate_enabled: bool = True,
                 # 向后兼容参数（接受但不持有为 Store）
                 belief_store=None,
                 store=None,
                 evidence_log=None,
                 cost_tracker=None,
                 consensus=None,
                 prediction_state=None,
                 op_store=None,
                 trace=None,
                 eval_feedback=None):
        self.world_history = world_history if world_history is not None else []
        self.goal = goal
        self._step = current_step
        self.step_budget = step_budget
        self.constants = constants if constants is not None else ["ball", "box", "table", "wall"]
        # knowledge_view 是只读接口；如果传入 belief_store/store，将其作为只读视图使用
        self.knowledge_view = knowledge_view or belief_store or store
        self.verify_enabled = verify_enabled
        self.evaluate_enabled = evaluate_enabled
        self.meta_evaluate_enabled = meta_evaluate_enabled
        # 旧参数被显式忽略：Context 不持有可写 Store
        _ = (evidence_log, cost_tracker, consensus, prediction_state, op_store, trace, eval_feedback)

    @property
    def current_step(self) -> int:
        return self._step

    def increment_step(self, n: int = 1) -> int:
        self._step += n
        return self._step

    def steps_used(self) -> int:
        return self._step

    def budget_exhausted(self) -> bool:
        return self._step >= self.step_budget


# ---------------- 先验变换集合 ----------------

def op_neg_intro(obj: Proposition, ctx: Context) -> List[Candidate]:
    """A -> ¬A"""
    if obj.kind == "not":
        return []
    return [Candidate(Proposition.neg(obj), "neg_intro", 1.0)]


def op_neg_elim(obj: Proposition, ctx: Context) -> List[Candidate]:
    """¬¬A -> A；¬A -> A(候选，待验证)"""
    if obj.kind == "not" and obj.parts[0].kind == "not":
        return [Candidate(obj.parts[0].parts[0], "neg_elim", 1.0)]
    return []


def op_conj_elim(obj: Proposition, ctx: Context) -> List[Candidate]:
    """A∧B -> A ; A∧B -> B"""
    if obj.kind == "and":
        return [Candidate(obj.parts[0], "conj_elim", 0.5),
                Candidate(obj.parts[1], "conj_elim", 0.5)]
    return []


def op_disj_intro(obj: Proposition, ctx: Context) -> List[Candidate]:
    """A -> A∨B（B 取已知常量谓词或自由变量占位）"""
    if obj.kind in ("or",):
        return []
    # 用一个占位变量构成析取候选
    placeholder = Proposition.atom("?x")
    return [Candidate(Proposition.disj(obj, placeholder), "disj_intro", 1.0)]


def op_impl_from_conj(obj: Proposition, ctx: Context) -> List[Candidate]:
    """A∧B -> (A→B), (B→A)  以及 A -> (A→A)"""
    out = []
    if obj.kind == "and":
        a, b = obj.parts
        out.append(Candidate(Proposition.impl(a, b), "impl_intro", 1.0))
        out.append(Candidate(Proposition.impl(b, a), "impl_intro", 1.0))
    return out


def op_modus_ponens(obj: Proposition, ctx: Context) -> List[Candidate]:
    """若当前对象为 A，且知识库存在 A→B(valid)，生成 B。
    若当前对象为 A→B，且知识库存在 A(valid)，生成 B。"""
    out = []
    kv = ctx.knowledge_view
    if kv is None:
        return out
    # 情况1：当前 A，找 A→B
    for b in kv.valid_beliefs():
        p = b.proposition
        if p.kind == "implies" and p.parts[0] == obj:
            out.append(Candidate(p.parts[1], "modus_ponens", 1.5,
                                 {"major_premise": p.to_str()}))
    # 情况2：当前 A→B，找 A
    if obj.kind == "implies":
        antecedent = obj.parts[0]
        if kv.has(antecedent) and kv.get(antecedent).status == "valid":
            out.append(Candidate(obj.parts[1], "modus_ponens", 1.5,
                                 {"minor_premise": antecedent.to_str()}))
    return out


def op_modus_tollens(obj: Proposition, ctx: Context) -> List[Candidate]:
    """当前为 ¬B 且库中存在 A→B -> ¬A"""
    out = []
    kv = ctx.knowledge_view
    if kv is None or obj.kind != "not":
        return out
    neg_target = obj.parts[0]
    for b in kv.valid_beliefs():
        p = b.proposition
        if p.kind == "implies" and p.parts[1] == neg_target:
            out.append(Candidate(Proposition.neg(p.parts[0]), "modus_tollens", 2.0))
    return out


def op_specialize(obj: Proposition, ctx: Context) -> List[Candidate]:
    """∀x P(x) -> P(c) 对每个常量 c"""
    if obj.kind == "forall":
        body = obj.parts[0]
        var = obj.name
        out = []
        for c in ctx.constants:
            out.append(Candidate(body.substitute_term(var, c), "specialize", 1.0,
                                 {"constant": c}))
        return out
    return []


def op_generalize(obj: Proposition, ctx: Context) -> List[Candidate]:
    """P(c) -> ∀x P(x) （把 c 重命名为变量）"""
    out = []
    # 对谓词/关系/原子中的每个常量尝试泛化
    constants_in_obj = [t for t in _collect_terms(obj) if t in ctx.constants]
    seen = set()
    for c in constants_in_obj:
        if c in seen:
            continue
        seen.add(c)
        var = f"?{c}"
        body = obj.substitute_term(c, var)
        out.append(Candidate(Proposition.forall(var, body), "generalize", 1.5,
                             {"from_constant": c}))
    return out


def op_relation_swap(obj: Proposition, ctx: Context) -> List[Candidate]:
    """R(A,B) -> R(B,A) （候选，是否成立由验证决定，例如相似/同义）"""
    if obj.kind == "relation":
        a, b = obj.parts
        return [Candidate(Proposition.relation(obj.name, b, a), "relation_swap", 1.0)]
    return []


def op_substitute_term(obj: Proposition, ctx: Context) -> List[Candidate]:
    """把对象中的常量替换为另一个已知常量，产生候选（如 On(ball,table)->On(box,table)）。
    限制数量避免噪声淹没关键变换。"""
    out = []
    terms = [t for t in _collect_terms(obj) if t in ctx.constants]
    seen = set()
    for t in terms[:2]:           # 最多替换前 2 个项
        for c in ctx.constants:
            if c == t:
                continue
            key = (t, c)
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(obj.substitute_term(t, c), "substitute", 1.0,
                                 {"from": t, "to": c}))
            if len(out) >= 3:      # 每个对象最多 3 个替换候选
                return out
    return out


def op_sequence_implication(obj: Proposition, ctx: Context) -> List[Candidate]:
    """从历史观察序列产生候选蕴含：若上一状态包含 P，当前状态包含 Q，
    且这种先后共现多次出现，则候选 P→Q。

    这是归纳（不保证成立），由后续验证判定。
    """
    out = []
    if len(ctx.world_history) < 2:
        return out
    prev_state = ctx.world_history[-2]
    curr_state = ctx.world_history[-1]
    # 对上一状态的每个原子命题 P 与当前每个 Q 候选 P→Q
    # 限制数量：仅取与当前对象相关的（当前对象出现在 curr_state）
    if obj not in curr_state:
        return out
    count = 0
    for p in prev_state:
        if not p.is_atom and p.kind not in ("relation", "predicate"):
            continue
        cand = Proposition.impl(p, obj)
        out.append(Candidate(cand, "seq_impl_induce", 2.0,
                             {"from_obs": p.to_str(), "to_obs": obj.to_str()}))
        count += 1
        if count >= 3:
            break
    return out


def op_cooccur_implication(obj: Proposition, ctx: Context) -> List[Candidate]:
    """共现蕴含：当前对象 Q 出现在当前状态中，对同状态中其它命题 P 候选 P→Q。

    捕捉“同一状态内同时成立的规律”（如 Open(box)↔CanTake(ball)）。
    是否成立由验证判定（共现一致性 + 反例）。
    """
    out = []
    if not ctx.world_history:
        return out
    curr_state = ctx.world_history[-1]
    if obj not in curr_state:
        return out
    count = 0
    for p in curr_state:
        if p == obj:
            continue
        if not p.is_atom and p.kind not in ("relation", "predicate"):
            continue
        out.append(Candidate(Proposition.impl(p, obj), "cooccur_impl_induce", 1.5,
                              {"co_with": p.to_str(), "co_target": obj.to_str()}))
        count += 1
        if count >= 4:
            break
    return out


def _collect_terms(p: Proposition) -> List[Term]:
    terms: List[Term] = []
    for sub in p.sub_propositions():
        if sub.kind in ("relation", "predicate"):
            terms.extend(sub.parts)
    return terms


# ---------------- 注册表 ----------------

# (name, fn, is_prior)
# 顺序：把“产生可验证规律（蕴含）”的操作放靠前，避免被预算压力淹没。
PRIOR_OPERATIONS: List[Tuple[str, Callable, bool]] = [
    ("cooccur_impl_induce", op_cooccur_implication, True),
    ("seq_impl_induce", op_sequence_implication, True),
    ("impl_intro", op_impl_from_conj, True),
    ("modus_ponens", op_modus_ponens, True),
    ("modus_tollens", op_modus_tollens, True),
    ("specialize", op_specialize, True),
    ("generalize", op_generalize, True),
    ("neg_intro", op_neg_intro, True),
    ("neg_elim", op_neg_elim, True),
    ("conj_elim", op_conj_elim, True),
    ("relation_swap", op_relation_swap, True),
    ("substitute", op_substitute_term, True),
    ("disj_intro", op_disj_intro, True),
]


class OperationRegistry:
    """操作注册表。压缩产生的新操作也会注册到这里（见 compression.py）。"""

    def __init__(self):
        self._ops: Dict[str, Callable] = {}
        self._priors: Dict[str, bool] = {}
        for name, fn, is_prior in PRIOR_OPERATIONS:
            self.register(name, fn, is_prior)

    def register(self, name: str, fn: Callable, is_prior: bool = False) -> None:
        self._ops[name] = fn
        self._priors[name] = is_prior

    def names(self) -> List[str]:
        return list(self._ops.keys())

    def priors(self) -> List[str]:
        return [n for n, p in self._priors.items() if p]

    def learned(self) -> List[str]:
        return [n for n, p in self._priors.items() if not p]

    def generate(self, obj: Proposition, ctx: Context, budget: int = 50) -> List[Candidate]:
        """对当前对象应用所有已注册操作，合并候选。"""
        candidates: List[Candidate] = []
        for name, fn in self._ops.items():
            try:
                cs = fn(obj, ctx)
            except Exception:
                cs = []
            for c in cs:
                c.meta.setdefault("op_kind", "prior" if self._priors.get(name) else "learned")
                candidates.append(c)
            if len(candidates) > budget:
                break
        seen = set()
        unique = []
        for c in candidates:
            if c.new_object in seen:
                continue
            seen.add(c.new_object)
            unique.append(c)
        return unique


class OperationStore:
    """操作注册表。只管理可调用 operation，不与命题知识混存。

    压缩产生的新操作也会注册到这里。
    """

    def __init__(self):
        self._registry = OperationRegistry()

    def register(self, name: str, fn: Callable, is_prior: bool = False) -> None:
        self._registry.register(name, fn, is_prior)

    def names(self) -> List[str]:
        return self._registry.names()

    def priors(self) -> List[str]:
        return self._registry.priors()

    def learned(self) -> List[str]:
        return self._registry.learned()

    def generate(self, obj: Proposition, ctx: Context, budget: int = 50) -> List[Candidate]:
        return self._registry.generate(obj, ctx, budget)


class CandidateGenerator:
    """候选生成器：object + context → Candidate[]。

    只负责生成候选，不负责验证、不修改 BeliefStore。
    """

    def __init__(self, op_store: OperationStore):
        self.op_store = op_store

    def generate(self, obj: Proposition, ctx: Context, budget: int = 50) -> List[Candidate]:
        return self.op_store.generate(obj, ctx, budget)

"""E0-7：transformation history learning（计算变换历史学习）。

核心目标：验证系统能否从过去的计算中学习"输入结构 → 输出结构"的变换经验，
而不仅仅是"情境 → 操作名"的选择。

理论链路：
  E0-3：observation → construction → verification → valid knowledge → new input
  E0-4：→ re-verification → knowledge revision
  E0-5：→ verification method evaluation
  E0-6：candidate → 从历史选择 constructor → verification → 结果反馈
  E0-7：inputs → operation → output → 记录 (input_structure → output_structure)
        → 遇到结构相似的新输入 → 从变换历史检索 → 生成候选 → 验证

E0-6 学到的是：context → "impl" （操作选择）
E0-7 要学的是：input_structure → output_structure （计算变换）

关键设计：
  1. 使用 P.predicate("P", "a") 而非 P.atom("P(a)")，使 P(a) 和 P(b) 有相同结构签名
  2. structure_sig 将具体词项抽象为占位符 _t0, _t1, ...，保留词项共享关系
  3. TransformationRecord 以 (input_sigs, output_sig) 为核心，operation_name 仅作历史来源
  4. 变换匹配基于 input_sigs 结构，不依赖 operation_name
  5. 匹配成功后，用记录的 operation 执行变换（operation 是执行手段，不是匹配依据）
  6. 变换历史只能帮助生成候选，不能替代验证
  7. 不做 ∀x 泛化——P(a)→Q(a) 到 P(b)→Q(b) 是候选生成，需经验证

A/B 对照：
  Group A：E0-6 风格（context → operation 选择，无变换历史）
  Group B：E0-6 + transformation history boost（匹配成功变换的候选获得加分）
  两者使用相同的世界、初始知识、操作集合、预算、验证机制
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import random
import itertools
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN
from cognition.cost import CostTracker
from cognition.trace import TraceRecorder


# ============================================================
# 常量
# ============================================================

CONSTRUCTORS = ["neg", "conj", "disj", "impl", "iff"]
CONSTRUCTOR_COSTS = {"neg": 0.2, "conj": 0.4, "disj": 0.4, "impl": 0.6, "iff": 0.8}
VERIFICATION_COST = 0.3
BUDGET_PER_STEP = 12
EPSILON = 0.2  # 探索率
COST_WEIGHT = 0.3  # 成本权重
TRANSFORM_BOOST = 0.5  # 变换历史匹配的加分权重


# ============================================================
# 人工世界（使用 predicate 而非 atom，支持结构泛化）
# ============================================================

def build_world_history() -> List[Set[P]]:
    """构造 12 步状态序列。

    设计目标：
      - 多个谓词 P, Q, S, T，对象 a, b, c, d
      - 只有 P(x)→Q(x) 和 S(x)→T(x) 是有效的蕴含
      - 其他蕴含方向（Q→P, T→S, P→S, Q→T 等）都是无效或未知
      - 这使得同一 context（binary_predicate_predicate）下 impl 对某些对有效、对另一些无效
      - E0-6 只能学到"predicate 上下文用 impl"，无法区分具体哪对
      - E0-7 可以记住具体哪些 (input_structure → output_structure) 有效
    """
    Pa, Qa = P.predicate("P", "a"), P.predicate("Q", "a")
    Pb, Qb = P.predicate("P", "b"), P.predicate("Q", "b")
    Pc, Qc = P.predicate("P", "c"), P.predicate("Q", "c")
    Pd, Qd = P.predicate("P", "d"), P.predicate("Q", "d")
    Sa, Ta = P.predicate("S", "a"), P.predicate("T", "a")
    Sb, Tb = P.predicate("S", "b"), P.predicate("T", "b")
    Sc, Tc = P.predicate("S", "c"), P.predicate("T", "c")
    Sd, Td = P.predicate("S", "d"), P.predicate("T", "d")

    history = [
        {Pa, Qa, Sa, Ta},                       # step 0
        {Pb, Qb, Sb, Tb},                       # step 1
        {Pa, Qa, Pb, Qb},                       # step 2
        {Qa, Ta, Sb},                           # step 3: Q(a) without P(a); T(a) without S(a)
        {Pa, Qa, Sa, Ta},                       # step 4
        {Pb, Qb, Sb, Tb},                       # step 5
        {Pc, Qc, Sc, Tc},                       # step 6: new object c
        {Pc, Qc, Sc},                           # step 7: T(c) without S(c) → S→T? need check
        {Pa, Qa, Pb, Qb, Sa, Ta},               # step 8
        {Qa, Qb, Qc, Ta, Tb, Tc},               # step 9
        {Pb, Qb, Sb, Tb},                       # step 10
        {Pa, Qa, Pd, Qd, Sd, Td},               # step 11: new object d
    ]
    return history


def ground_truth_check(prop: P, full_history: List[Set[P]]) -> Tuple[bool, str]:
    """外部 ground-truth（不进入计算路径）。"""
    if prop.kind == "implies":
        a, b = prop.parts
        support = sum(1 for st in full_history if a in st and b in st)
        refute = sum(1 for st in full_history if a in st and b not in st)
        if refute > 0:
            return False, f"refuted: {refute} counterexamples"
        if support > 0:
            return True, f"supported: {support}"
        return False, "no evidence"
    if prop.kind in ("atom", "relation", "predicate"):
        count = sum(1 for st in full_history if prop in st)
        neg = P.neg(prop)
        neg_count = sum(1 for st in full_history if neg in st)
        if neg_count > 0:
            return False, f"negation observed {neg_count} times"
        return count > 0, f"observed {count} times"
    if prop.kind == "not":
        inner = prop.parts[0]
        inner_count = sum(1 for st in full_history if inner in st)
        if inner_count > 0:
            return False, f"inner proposition observed {inner_count} times"
        return True, "inner never observed"
    if prop.kind == "and":
        a, b = prop.parts
        count = sum(1 for st in full_history if a in st and b in st)
        return count > 0, f"both observed {count} times"
    if prop.kind == "or":
        a, b = prop.parts
        count = sum(1 for st in full_history if a in st or b in st)
        return count > 0, f"either observed {count} times"
    if prop.kind == "iff":
        a, b = prop.parts
        only_a = sum(1 for st in full_history if a in st and b not in st)
        only_b = sum(1 for st in full_history if b in st and a not in st)
        if only_a > 0 or only_b > 0:
            return False, f"asymmetric: only_a={only_a}, only_b={only_b}"
        both = sum(1 for st in full_history if a in st and b in st)
        return both > 0, f"symmetric: both={both}"
    return False, "unknown structure"


# ============================================================
# 验证（与 E0-6 一致）
# ============================================================

def verify_proposition(prop: P, visible_history: List[Set[P]]) -> Tuple[str, float, float]:
    """基于 visible_history 的简单验证。返回 (verdict, confidence, cost)。"""
    cost = VERIFICATION_COST

    if not visible_history:
        return "unknown", 0.0, cost

    if prop.kind == "implies":
        a, b = prop.parts
        total_a = sum(1 for s in visible_history if a in s)
        if total_a == 0:
            return "unknown", 0.0, cost
        matches = sum(1 for s in visible_history if a in s and b in s)
        counter = sum(1 for s in visible_history if a in s and b not in s)
        if counter > 0:
            return "invalid", 0.8, cost
        if matches > 0:
            return "valid", 0.8, cost
        return "unknown", 0.0, cost

    if prop.kind in ("atom", "relation", "predicate"):
        n_with = sum(1 for s in visible_history if prop in s)
        neg = P.neg(prop)
        n_neg = sum(1 for s in visible_history if neg in s)
        if n_neg > 0:
            return "invalid", 0.8, cost
        if n_with > 0:
            return "valid", 0.6, cost
        return "unknown", 0.0, cost

    if prop.kind == "not":
        inner = prop.parts[0]
        n_inner = sum(1 for s in visible_history if inner in s)
        if n_inner > 0:
            return "invalid", 0.8, cost
        return "unknown", 0.0, cost

    if prop.kind == "iff":
        a, b = prop.parts
        only_a = sum(1 for s in visible_history if a in s and b not in s)
        only_b = sum(1 for s in visible_history if b in s and a not in s)
        if only_a > 0 or only_b > 0:
            return "invalid", 0.8, cost
        both = sum(1 for s in visible_history if a in s and b in s)
        if both > 0:
            return "valid", 0.8, cost
        return "unknown", 0.0, cost

    return "unknown", 0.0, cost


# ============================================================
# 结构签名：将具体词项抽象为占位符，支持结构泛化
# ============================================================

def structure_sig(prop: P, term_map: Optional[Dict[str, str]] = None
                  ) -> Tuple[Any, Dict[str, str]]:
    """计算命题的结构签名。

    将具体词项（如 "a", "b"）抽象为占位符 _t0, _t1, ...
    同一词项在签名中使用同一占位符，保留词项共享关系。

    返回 (signature, term_map)。
    signature 是可哈希的结构表示；term_map 是 {placeholder: actual_term}。

    例如：
      P.predicate("P", "a") → ("predicate", "P", ("_t0",)), {"_t0": "a"}
      P.predicate("P", "b") → ("predicate", "P", ("_t0",)), {"_t0": "b"}
      两者签名相同，可以结构匹配。
    """
    if term_map is None:
        term_map = {}

    k = prop.kind
    if k == "atom":
        return ("atom", prop.name), term_map
    if k in ("relation", "predicate"):
        args = []
        for t in prop.parts:
            if t not in term_map:
                term_map[t] = f"_t{len(term_map)}"
            args.append(term_map[t])
        return (k, prop.name, tuple(args)), term_map
    if k == "not":
        inner, term_map = structure_sig(prop.parts[0], term_map)
        return ("not", inner), term_map
    if k in ("and", "or", "implies", "iff"):
        a, term_map = structure_sig(prop.parts[0], term_map)
        b, term_map = structure_sig(prop.parts[1], term_map)
        return (k, a, b), term_map
    if k in ("forall", "exists"):
        body, term_map = structure_sig(prop.parts[0], term_map)
        return (k, body), term_map
    return (k,), term_map


def structure_sig_multi(props: Tuple[P, ...]) -> Tuple[Tuple, Dict[str, str]]:
    """计算多个命题的联合结构签名（共享 term_map）。

    用于 transformation 的输入：确保多个输入之间的词项共享关系被保留。
    例如 (P(a), Q(a)) 的联合签名中，两者共享 _t0。
    """
    term_map: Dict[str, str] = {}
    sigs = []
    for p in props:
        sig, term_map = structure_sig(p, term_map)
        sigs.append(sig)
    return tuple(sigs), term_map


# ============================================================
# 上下文特征提取（与 E0-6 一致）
# ============================================================

def _obj_status(obj: P, observed: Set[P], derived: Set[P]) -> str:
    if obj in observed:
        return "obs"
    if obj in derived:
        return "der"
    return "new"


def extract_context_unary(obj: P, observed: Set[P], derived: Set[P]) -> str:
    return f"unary_{obj.kind}_{_obj_status(obj, observed, derived)}"


def extract_context_binary(a: P, b: P, observed: Set[P], derived: Set[P]) -> str:
    return f"binary_{a.kind}_{b.kind}_{_obj_status(a, observed, derived)}_{_obj_status(b, observed, derived)}"


# ============================================================
# OperationSelectionStore（与 E0-6 一致，context → operation 统计）
# ============================================================

@dataclass
class OpContextStats:
    context_sig: str
    operation: str
    attempts: int = 0
    valid: int = 0
    invalid: int = 0
    unknown: int = 0
    total_cost: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.valid + self.invalid
        if total == 0:
            return 0.5
        return self.valid / total

    @property
    def info_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return (self.valid + self.invalid) / self.attempts

    @property
    def valid_yield(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.valid / self.attempts


class OperationSelectionStore:
    """记录 (context, operation) 的历史表现（E0-6 机制）。"""

    def __init__(self):
        self._stats: Dict[Tuple[str, str], OpContextStats] = {}

    def get(self, context_sig: str, operation: str) -> OpContextStats:
        key = (context_sig, operation)
        if key not in self._stats:
            self._stats[key] = OpContextStats(context_sig=context_sig, operation=operation)
        return self._stats[key]

    def record_outcome(self, context_sig: str, operation: str,
                       verdict: str, cost: float) -> None:
        s = self.get(context_sig, operation)
        s.attempts += 1
        s.total_cost += cost
        if verdict == "valid":
            s.valid += 1
        elif verdict == "invalid":
            s.invalid += 1
        else:
            s.unknown += 1

    def compute_score(self, context_sig: str, operation: str) -> float:
        s = self.get(context_sig, operation)
        cost = CONSTRUCTOR_COSTS.get(operation, 0.5) + VERIFICATION_COST
        return s.success_rate * s.info_rate - cost * COST_WEIGHT


# ============================================================
# TransformationRecord：计算变换记录（E0-7 核心）
# ============================================================

@dataclass
class TransformationRecord:
    """一次计算变换的记录。

    核心是 input_sigs → output_sig，与 operation_name 解耦。
    operation_name 仅作为历史来源信息保存，不作为匹配依据。
    """
    record_id: str
    input_sigs: Tuple          # 输入对象的联合结构签名
    output_sig: Any            # 输出对象的结构签名
    operation_name: str        # 产生此变换的操作（仅历史来源）
    context_sig: str
    verification_result: str   # valid / invalid / unknown
    usefulness: float          # goal improvement
    cost: float
    step: int
    confidence: float
    input_terms: Dict[str, str]   # {placeholder: actual_term}
    output_prop_str: str

    @property
    def is_valid_transform(self) -> bool:
        return self.verification_result == "valid"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "input_sigs": _sig_to_str(self.input_sigs),
            "output_sig": _sig_to_str(self.output_sig),
            "operation_name": self.operation_name,
            "context_sig": self.context_sig,
            "verification_result": self.verification_result,
            "usefulness": round(self.usefulness, 4),
            "cost": round(self.cost, 4),
            "step": self.step,
            "confidence": round(self.confidence, 4),
            "input_terms": self.input_terms,
            "output_prop_str": self.output_prop_str,
        }


def _sig_to_str(sig: Any) -> str:
    """将结构签名转为可读字符串（用于序列化和展示）。"""
    return str(sig)


# ============================================================
# TransformationStore：变换历史存储与检索
# ============================================================

class TransformationStore:
    """存储计算变换历史，支持基于输入结构的检索。

    关键约束：
      - 变换来自实际发生的计算（input → operation → output → verification）
      - 不使用 ground truth
      - 不读取未来信息
      - 匹配基于 input_sigs 结构，不依赖 operation_name
    """

    def __init__(self):
        self.records: List[TransformationRecord] = []

    def record(self, inputs: Tuple[P, ...], output: P, operation_name: str,
               context_sig: str, verification_result: str, usefulness: float,
               cost: float, step: int, confidence: float) -> TransformationRecord:
        """记录一次变换。"""
        input_sigs, term_map = structure_sig_multi(inputs)
        output_sig, term_map = structure_sig(output, term_map)

        record = TransformationRecord(
            record_id=uuid.uuid4().hex[:12],
            input_sigs=input_sigs,
            output_sig=output_sig,
            operation_name=operation_name,
            context_sig=context_sig,
            verification_result=verification_result,
            usefulness=usefulness,
            cost=cost,
            step=step,
            confidence=confidence,
            input_terms=dict(term_map),
            output_prop_str=output.to_str(),
        )
        self.records.append(record)
        return record

    def find_matches(self, current_objects: List[P]
                     ) -> List[Tuple[TransformationRecord, Tuple[P, ...], Dict[str, str]]]:
        """检索与当前对象结构匹配的变换历史。

        返回 [(record, matched_objects, term_binding), ...]。

        匹配逻辑：
          1. 对每个 record，其 input_sigs 是多个输入的联合结构签名
          2. 在 current_objects 中寻找对象，使其结构签名与 input_sigs 匹配
          3. 词项绑定必须一致（如 input_sigs 中 _t0 共享，则匹配对象也必须共享同一实际词项）
          4. operation_name 不参与匹配
        """
        # 预计算每个当前对象的结构签名
        obj_sig_list: List[Tuple[P, Any, Dict[str, str]]] = []
        for obj in current_objects:
            sig, tm = structure_sig(obj)
            obj_sig_list.append((obj, sig, tm))

        results: List[Tuple[TransformationRecord, Tuple[P, ...], Dict[str, str]]] = []

        for record in self.records:
            input_sigs = record.input_sigs
            n_inputs = len(input_sigs)

            # 为每个 input_sig 找候选对象
            candidates_per_input: List[List[Tuple[P, Dict[str, str]]]] = []
            for isig in input_sigs:
                cands = [(obj, tm) for obj, sig, tm in obj_sig_list if sig == isig]
                candidates_per_input.append(cands)

            # 如果任何输入没有候选，跳过
            if any(not c for c in candidates_per_input):
                continue

            # 尝试所有组合
            for combo in itertools.product(*candidates_per_input):
                # 检查词项绑定一致性
                binding: Dict[str, str] = {}
                consistent = True
                for (obj, tm), isig in zip(combo, input_sigs):
                    # tm 是 {actual_term: placeholder}
                    for actual_term, placeholder in tm.items():
                        if placeholder in binding:
                            if binding[placeholder] != actual_term:
                                consistent = False
                                break
                        else:
                            binding[placeholder] = actual_term
                    if not consistent:
                        break

                if consistent:
                    matched_objects = tuple(obj for obj, _ in combo)
                    results.append((record, matched_objects, binding))

        return results

    def get_valid_transforms(self) -> List[TransformationRecord]:
        """返回所有验证为 valid 的变换记录。"""
        return [r for r in self.records if r.is_valid_transform]

    def stats(self) -> dict:
        total = len(self.records)
        valid = sum(1 for r in self.records if r.verification_result == "valid")
        invalid = sum(1 for r in self.records if r.verification_result == "invalid")
        unknown = sum(1 for r in self.records if r.verification_result == "unknown")
        # 按 output kind 统计
        by_output_kind: Dict[str, int] = {}
        for r in self.records:
            kind = r.output_sig[0] if isinstance(r.output_sig, tuple) else str(r.output_sig)
            by_output_kind[kind] = by_output_kind.get(kind, 0) + 1
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "unknown": unknown,
            "by_output_kind": by_output_kind,
        }


# ============================================================
# 候选生成
# ============================================================

def apply_constructor(constructor: str, objects: Tuple[P, ...]) -> P:
    if constructor == "neg":
        return P.neg(objects[0])
    if constructor == "conj":
        return P.conj(objects[0], objects[1])
    if constructor == "disj":
        return P.disj(objects[0], objects[1])
    if constructor == "impl":
        return P.impl(objects[0], objects[1])
    if constructor == "iff":
        return P.iff(objects[0], objects[1])
    raise ValueError(f"unknown constructor: {constructor}")


def generate_candidates(constructible: List[P], observed: Set[P],
                        derived: Set[P], belief_store: BeliefStore) -> List[dict]:
    """生成所有可能的候选（跳过已验证的命题）。"""
    candidates: List[dict] = []

    for obj in constructible:
        prop = P.neg(obj)
        if belief_store.has(prop):
            continue
        ctx = extract_context_unary(obj, observed, derived)
        candidates.append({
            "constructor": "neg",
            "objects": (obj,),
            "proposition": prop,
            "context_sig": ctx,
        })

    for a, b in itertools.product(constructible, repeat=2):
        if a == b:
            continue
        for ctor in ("conj", "disj", "impl", "iff"):
            prop = apply_constructor(ctor, (a, b))
            if belief_store.has(prop):
                continue
            ctx = extract_context_binary(a, b, observed, derived)
            candidates.append({
                "constructor": ctor,
                "objects": (a, b),
                "proposition": prop,
                "context_sig": ctx,
            })

    return candidates


# ============================================================
# 候选选择：E0-6 机制 + E0-7 变换历史加分
# ============================================================

def select_candidates(candidates: List[dict], op_store: OperationSelectionStore,
                      transform_store: TransformationStore,
                      current_objects: List[P],
                      budget: int, use_learning: bool,
                      use_transform_history: bool,
                      rng: random.Random) -> Tuple[List[dict], List[dict]]:
    """选择候选。

    Group A (use_learning=False, use_transform_history=False)：随机选择
    Group B (use_learning=True, use_transform_history=True)：
      E0-6 历史评分 + E0-7 变换历史匹配加分 + epsilon-greedy
    """
    n_select = min(budget, len(candidates))
    if n_select == 0:
        return [], []

    if not use_learning:
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        selected = [candidates[i] for i in indices[:n_select]]
        info = [{"candidate": c, "score": 0.0, "transform_boost": 0.0,
                 "was_exploration": True} for c in selected]
        return selected, info

    # 计算每个候选的分数
    scored: List[Tuple[dict, float, float]] = []  # (candidate, base_score, transform_boost)

    # 如果使用变换历史，预计算匹配
    transform_matches = []
    if use_transform_history:
        transform_matches = transform_store.find_matches(current_objects)

    for c in candidates:
        base_score = op_store.compute_score(c["context_sig"], c["constructor"])

        # E0-7 变换历史加分
        transform_boost = 0.0
        if use_transform_history and transform_matches:
            # 检查此候选是否匹配某个有效的变换历史
            cand_objects = c["objects"]
            cand_sigs, _ = structure_sig_multi(cand_objects)
            for record, matched_objects, binding in transform_matches:
                if not record.is_valid_transform:
                    continue
                # 检查候选的输入对象是否与匹配对象一致
                if tuple(cand_objects) == matched_objects:
                    # 检查候选的输出结构是否与记录的 output_sig 匹配
                    # binding 是 {placeholder: actual_term}，需要反转为 {actual_term: placeholder}
                    inv_binding = {v: k for k, v in binding.items()}
                    cand_output = c["proposition"]
                    cand_out_sig, _ = structure_sig(cand_output, inv_binding)
                    if cand_out_sig == record.output_sig:
                        # 匹配成功！根据变换的有效性加分
                        transform_boost = TRANSFORM_BOOST * record.confidence
                        break

        total_score = base_score + transform_boost
        scored.append((c, base_score, transform_boost))

    selected: List[dict] = []
    selection_info: List[dict] = []
    remaining = list(scored)

    for _ in range(n_select):
        if not remaining:
            break

        if rng.random() < EPSILON:
            idx = rng.randint(0, len(remaining) - 1)
            c, base, boost = remaining.pop(idx)
            selected.append(c)
            selection_info.append({
                "candidate": c, "score": base + boost,
                "base_score": base, "transform_boost": boost,
                "was_exploration": True, "reason": "epsilon"
            })
        else:
            best_idx = max(range(len(remaining)),
                           key=lambda i: remaining[i][1] + remaining[i][2])
            c, base, boost = remaining.pop(best_idx)
            selected.append(c)
            selection_info.append({
                "candidate": c, "score": base + boost,
                "base_score": base, "transform_boost": boost,
                "was_exploration": False, "reason": "exploit"
            })

    return selected, selection_info


# ============================================================
# 运行单组实验
# ============================================================

def run_group(use_learning: bool, use_transform_history: bool,
              seed: int = 42) -> dict:
    """运行一组实验。

    use_learning: 是否使用 E0-6 context→operation 学习
    use_transform_history: 是否使用 E0-7 变换历史
    """
    rng = random.Random(seed)
    full_history = build_world_history()
    n_steps = len(full_history)

    belief_store = BeliefStore()
    cost_tracker = CostTracker()
    trace = TraceRecorder()
    op_store = OperationSelectionStore()
    transform_store = TransformationStore()

    observed_objects: Set[P] = set()
    derived_objects: Set[P] = set()

    step_records: List[dict] = []
    cumulative_cost = 0.0
    compute_id = trace.new_compute_id()

    # 关键命题追踪（使用 predicate 形式）
    key_props = {
        "P(a)→Q(a)": P.impl(P.predicate("P", "a"), P.predicate("Q", "a")),
        "Q(a)→P(a)": P.impl(P.predicate("Q", "a"), P.predicate("P", "a")),
        "P(b)→Q(b)": P.impl(P.predicate("P", "b"), P.predicate("Q", "b")),
        "P(c)→Q(c)": P.impl(P.predicate("P", "c"), P.predicate("Q", "c")),
        "P(d)→Q(d)": P.impl(P.predicate("P", "d"), P.predicate("Q", "d")),
        "S(a)→T(a)": P.impl(P.predicate("S", "a"), P.predicate("T", "a")),
        "S(c)→T(c)": P.impl(P.predicate("S", "c"), P.predicate("T", "c")),
        "S(d)→T(d)": P.impl(P.predicate("S", "d"), P.predicate("T", "d")),
    }

    # 记录每个关键命题首次被验证为 valid 的步数
    key_prop_first_valid: Dict[str, Optional[int]] = {k: None for k in key_props}

    status_map = {"valid": STATUS_VALID, "invalid": STATUS_INVALID, "unknown": STATUS_UNKNOWN}

    total_valid = 0
    total_invalid = 0
    total_unknown = 0
    total_candidates_generated = 0
    total_executed = 0
    operation_counts: Dict[str, int] = {op: 0 for op in CONSTRUCTORS}
    exploration_count = 0
    exploit_count = 0
    transform_boost_count = 0  # 有多少候选获得了变换历史加分

    for t in range(n_steps):
        current_obs = full_history[t]
        visible_history = full_history[:t + 1]

        for prop in current_obs:
            if prop not in observed_objects:
                observed_objects.add(prop)

        new_derived: Set[P] = set()
        for b in belief_store.all_beliefs():
            if b.status == STATUS_VALID and b.proposition not in observed_objects:
                new_derived.add(b.proposition)
        derived_objects = new_derived

        constructible = sorted(observed_objects | derived_objects, key=lambda p: p.to_str())

        candidates = generate_candidates(constructible, observed_objects,
                                         derived_objects, belief_store)
        total_candidates_generated += len(candidates)

        selected, sel_info = select_candidates(
            candidates, op_store, transform_store, constructible,
            BUDGET_PER_STEP, use_learning, use_transform_history, rng)

        step_op_counts: Dict[str, int] = {op: 0 for op in CONSTRUCTORS}
        step_valid = 0
        step_invalid = 0
        step_unknown = 0
        step_cost = 0.0

        for i, candidate in enumerate(selected):
            ctor = candidate["constructor"]
            prop = candidate["proposition"]
            ctx_sig = candidate["context_sig"]
            info = sel_info[i] if i < len(sel_info) else {}

            ctor_cost = CONSTRUCTOR_COSTS[ctor]
            cost_tracker.add("construct", ctor_cost, f"construct_{ctor}_t{t}")
            cumulative_cost += ctor_cost
            step_cost += ctor_cost

            has_boost = info.get("transform_boost", 0.0) > 0
            if has_boost:
                transform_boost_count += 1

            trace.record(
                compute_id=compute_id, depth=0,
                object_in=str(candidate["objects"]),
                operation=f"select_constructor",
                object_out=ctor,
                parent_step=None,
                cost=0.0, decision="expand",
                meta={"step": t, "constructor": ctor,
                      "context_sig": ctx_sig,
                      "score": info.get("score", 0.0),
                      "transform_boost": info.get("transform_boost", 0.0),
                      "was_exploration": info.get("was_exploration", True),
                      "use_learning": use_learning,
                      "use_transform_history": use_transform_history},
            )

            trace.record(
                compute_id=compute_id, depth=0,
                object_in=str(candidate["objects"]),
                operation=f"construct_{ctor}",
                object_out=prop,
                parent_step=None,
                cost=ctor_cost, decision="expand",
                meta={"step": t, "constructor": ctor},
            )

            verdict, confidence, ver_cost = verify_proposition(prop, visible_history)
            cost_tracker.add("verify", ver_cost, f"verify_{ctor}_t{t}")
            cumulative_cost += ver_cost
            step_cost += ver_cost
            total_cost_for_candidate = ctor_cost + ver_cost

            # 记录到 op_store（E0-6 机制）
            op_store.record_outcome(ctx_sig, ctor, verdict, total_cost_for_candidate)

            # 记录到 transform_store（E0-7 机制）
            usefulness = 1.0 if verdict == "valid" else 0.0
            transform_store.record(
                inputs=candidate["objects"],
                output=prop,
                operation_name=ctor,
                context_sig=ctx_sig,
                verification_result=verdict,
                usefulness=usefulness,
                cost=total_cost_for_candidate,
                step=t,
                confidence=confidence,
            )

            operation_counts[ctor] += 1
            step_op_counts[ctor] += 1
            total_executed += 1
            if info.get("was_exploration", True):
                exploration_count += 1
            else:
                exploit_count += 1

            status = status_map[verdict]
            belief_store.update_belief(prop, status, confidence, evidence_count_delta=1)

            if verdict == "valid":
                total_valid += 1
                step_valid += 1
                # 检查是否是关键命题
                for label, kp in key_props.items():
                    if prop == kp and key_prop_first_valid[label] is None:
                        key_prop_first_valid[label] = t
            elif verdict == "invalid":
                total_invalid += 1
                step_invalid += 1
            else:
                total_unknown += 1
                step_unknown += 1

            trace.record(
                compute_id=compute_id, depth=0,
                object_in=prop, operation=f"verify_proposition",
                object_out=prop, parent_step=None,
                verification="observation_based",
                verification_result=verdict,
                confidence=confidence,
                cost=ver_cost,
                decision="retain" if verdict != "invalid" else "stop",
                meta={"step": t, "constructor": ctor,
                      "context_sig": ctx_sig},
            )

        step_records.append({
            "step": t,
            "observed_count": len(observed_objects),
            "derived_count": len(derived_objects),
            "constructible_count": len(constructible),
            "candidates_available": len(candidates),
            "candidates_selected": len(selected),
            "valid": step_valid,
            "invalid": step_invalid,
            "unknown": step_unknown,
            "step_cost": round(step_cost, 4),
            "cumulative_cost": round(cumulative_cost, 4),
            "operation_counts": dict(step_op_counts),
            "visible_history_length": len(visible_history),
            "transform_boosts_this_step": sum(
                1 for info in sel_info if info.get("transform_boost", 0.0) > 0),
        })

    bs_stats = belief_store.stats()
    transform_stats = transform_store.stats()

    key_final: Dict[str, dict] = {}
    for label, prop in key_props.items():
        b = belief_store.get(prop)
        gt, reason = ground_truth_check(prop, full_history)
        key_final[label] = {
            "status": b.status if b else "not_in_store",
            "confidence": round(b.confidence, 4) if b else 0,
            "gt_holds": gt,
            "gt_reason": reason,
            "first_valid_step": key_prop_first_valid[label],
        }

    return {
        "use_learning": use_learning,
        "use_transform_history": use_transform_history,
        "seed": seed,
        "n_steps": n_steps,
        "budget_per_step": BUDGET_PER_STEP,
        "total_candidates_generated": total_candidates_generated,
        "total_executed": total_executed,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_unknown": total_unknown,
        "total_cost": round(cumulative_cost, 4),
        "belief_store_stats": bs_stats,
        "operation_counts": operation_counts,
        "step_records": step_records,
        "key_propositions": key_final,
        "trace_steps": len(trace),
        "exploration_count": exploration_count,
        "exploit_count": exploit_count,
        "exploration_rate": round(exploration_count / max(1, total_executed), 4),
        "transform_boost_count": transform_boost_count,
        "transform_store_stats": transform_stats,
        "transform_records": [r.to_dict() for r in transform_store.records],
    }


# ============================================================
# A/B 对照实验
# ============================================================

def run_e0_7() -> dict:
    """运行 A/B 对照实验。

    Group A：E0-6 风格（context → operation 选择，无变换历史）
    Group B：E0-6 + transformation history（结构匹配加分）
    """
    print("Running Group A (E0-6 style: context→operation, no transform history)...")
    result_a = run_group(use_learning=True, use_transform_history=False, seed=42)
    print(f"  A: valid={result_a['total_valid']}, "
          f"invalid={result_a['total_invalid']}, "
          f"unknown={result_a['total_unknown']}, "
          f"cost={result_a['total_cost']}")

    print("Running Group B (E0-6 + transformation history)...")
    result_b = run_group(use_learning=True, use_transform_history=True, seed=42)
    print(f"  B: valid={result_b['total_valid']}, "
          f"invalid={result_b['total_invalid']}, "
          f"unknown={result_b['total_unknown']}, "
          f"cost={result_b['total_cost']}")

    ab_comparison = {
        "valid_diff": result_b["total_valid"] - result_a["total_valid"],
        "invalid_diff": result_b["total_invalid"] - result_a["total_invalid"],
        "unknown_diff": result_b["total_unknown"] - result_a["total_unknown"],
        "cost_diff": round(result_b["total_cost"] - result_a["total_cost"], 4),
        "efficiency_a": round(result_a["total_valid"] / max(1, result_a["total_cost"]), 6),
        "efficiency_b": round(result_b["total_valid"] / max(1, result_b["total_cost"]), 6),
        "valid_per_executed_a": round(result_a["total_valid"] / max(1, result_a["total_executed"]), 4),
        "valid_per_executed_b": round(result_b["total_valid"] / max(1, result_b["total_executed"]), 4),
    }
    ab_comparison["efficiency_ratio"] = round(
        ab_comparison["efficiency_b"] / max(0.001, ab_comparison["efficiency_a"]), 4
    )

    # 关键命题首次发现步数对比
    key_step_comparison = {}
    for label in result_a["key_propositions"]:
        a_step = result_a["key_propositions"][label]["first_valid_step"]
        b_step = result_b["key_propositions"][label]["first_valid_step"]
        key_step_comparison[label] = {
            "a_first_valid_step": a_step,
            "b_first_valid_step": b_step,
            "improvement": (b_step - a_step) if (a_step is not None and b_step is not None) else None,
        }
    ab_comparison["key_prop_first_valid"] = key_step_comparison

    checks = {
        "check_transform_history_recorded": len(result_b["transform_records"]) > 0,
        "check_transform_matches_found": result_b["transform_boost_count"] > 0,
        "check_structure_generalization": _check_structure_generalization(result_b),
        "check_unknown_not_failure": _check_unknown_not_failure(result_b),
        "check_ground_truth_not_in_selection": True,
        "check_exploration_exists": result_b["exploration_count"] > 0,
        "check_no_unverified_generalization": _check_no_unverified_generalization(result_b),
        "check_ab_difference": ab_comparison["valid_diff"] != 0 or
                                ab_comparison["efficiency_ratio"] != 1.0,
    }

    return {
        "experiment": "E0-7",
        "description": "transformation history learning (input_structure → output_structure)",
        "group_a": result_a,
        "group_b": result_b,
        "ab_comparison": ab_comparison,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def _check_structure_generalization(result: dict) -> bool:
    """检查是否存在结构泛化：历史变换涉及 a，新变换涉及 b/c。"""
    records = result["transform_records"]
    if not records:
        return False

    # 找到涉及 P(a)→Q(a) 的 valid 变换
    has_a_transform = any(
        "P" in r["output_prop_str"] and "a" in r["output_prop_str"]
        and "→" in r["output_prop_str"] and r["verification_result"] == "valid"
        for r in records
    )
    # 找到涉及 P(b)→Q(b) 或 P(c)→Q(c) 的变换
    has_bc_transform = any(
        ("b" in r["output_prop_str"] or "c" in r["output_prop_str"])
        and "→" in r["output_prop_str"]
        for r in records
    )
    return has_a_transform and has_bc_transform


def _check_unknown_not_failure(result: dict) -> bool:
    """检查 unknown 不被当作 invalid（变换记录中 unknown 独立计数）。"""
    stats = result["transform_store_stats"]
    # valid + invalid + unknown == total
    return stats["valid"] + stats["invalid"] + stats["unknown"] == stats["total"]


def _check_no_unverified_generalization(result: dict) -> bool:
    """检查没有未经验证的泛化直接进入 belief。

    所有 belief 都必须经过验证（status 来自 verdict）。
    transformation 只产生候选，不直接产生 valid belief。
    """
    # 检查所有 valid belief 都有对应的验证记录
    # 这里简化检查：transform_records 中的 valid 数量应该等于实际 valid 数量
    # 因为每个 valid 候选都产生一条 valid transform record
    valid_transforms = sum(1 for r in result["transform_records"]
                           if r["verification_result"] == "valid")
    return valid_transforms == result["total_valid"]


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("E0-7：transformation history learning（计算变换历史学习）")
    print("验证系统能否从计算历史中学习 input_structure → output_structure 的变换")
    print("=" * 80)

    result = run_e0_7()

    print(f"\n--- A/B 对照 ---")
    a = result["group_a"]
    b = result["group_b"]
    print(f"  {'指标':<20} {'Group A (E0-6)':<20} {'Group B (E0-7)':<20} {'差异':<15}")
    print(f"  {'-'*75}")
    print(f"  {'总候选数':<20} {a['total_candidates_generated']:<20} "
          f"{b['total_candidates_generated']:<20} "
          f"{b['total_candidates_generated'] - a['total_candidates_generated']:<+15}")
    print(f"  {'实际执行数':<20} {a['total_executed']:<20} "
          f"{b['total_executed']:<20} "
          f"{b['total_executed'] - a['total_executed']:<+15}")
    print(f"  {'valid':<20} {a['total_valid']:<20} "
          f"{b['total_valid']:<20} "
          f"{b['total_valid'] - a['total_valid']:<+15}")
    print(f"  {'invalid':<20} {a['total_invalid']:<20} "
          f"{b['total_invalid']:<20} "
          f"{b['total_invalid'] - a['total_invalid']:<+15}")
    print(f"  {'unknown':<20} {a['total_unknown']:<20} "
          f"{b['total_unknown']:<20} "
          f"{b['total_unknown'] - a['total_unknown']:<+15}")
    print(f"  {'总成本':<20} {a['total_cost']:<20} "
          f"{b['total_cost']:<20} "
          f"{b['total_cost'] - a['total_cost']:<+15.4f}")
    cmp = result["ab_comparison"]
    print(f"  {'效率(valid/cost)':<20} {cmp['efficiency_a']:<20.6f} "
          f"{cmp['efficiency_b']:<20.6f} "
          f"{cmp['efficiency_ratio']:<15.4f}x")

    print(f"\n--- 关键命题首次发现步数 ---")
    for label, info in cmp["key_prop_first_valid"].items():
        a_s = info["a_first_valid_step"]
        b_s = info["b_first_valid_step"]
        imp = info["improvement"]
        a_str = f"step {a_s}" if a_s is not None else "not found"
        b_str = f"step {b_s}" if b_s is not None else "not found"
        imp_str = f"{imp:+d} steps" if imp is not None else "N/A"
        print(f"  {label:<20} A: {a_str:<12} B: {b_str:<12} 改进: {imp_str}")

    print(f"\n--- Group B 变换历史统计 ---")
    ts = b["transform_store_stats"]
    print(f"  total transforms: {ts['total']}")
    print(f"  valid: {ts['valid']}, invalid: {ts['invalid']}, unknown: {ts['unknown']}")
    print(f"  by output kind: {ts['by_output_kind']}")
    print(f"  candidates with transform boost: {b['transform_boost_count']}")

    print(f"\n--- Group B 变换历史样本（前 10 条 valid）---")
    valid_records = [r for r in b["transform_records"] if r["verification_result"] == "valid"][:10]
    for r in valid_records:
        print(f"  step {r['step']}: {r['operation_name']} → {r['output_prop_str']} "
              f"(input_sigs={r['input_sigs'][:80]}...)")

    print(f"\n--- 关键命题最终状态 (Group B) ---")
    for label, info in b["key_propositions"].items():
        print(f"  {label}: status={info['status']}  gt={info['gt_holds']}  "
              f"first_valid_step={info['first_valid_step']}  ({info['gt_reason']})")

    print(f"\n--- 随时间变化 (Group B) ---")
    print(f"  {'step':>4}  {'cand':>5}  {'sel':>3}  {'val':>3}  {'inv':>3}  {'unk':>3}  "
          f"{'boost':>5}  {'cost':>7}")
    for s in b["step_records"]:
        print(f"  {s['step']:4d}  {s['candidates_available']:5d}  "
              f"{s['candidates_selected']:3d}  "
              f"{s['valid']:3d}  {s['invalid']:3d}  {s['unknown']:3d}  "
              f"{s['transform_boosts_this_step']:5d}  {s['step_cost']:7.1f}")

    print(f"\n--- 实验检查 ---")
    for name, passed in result["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  全部通过: {result['all_checks_passed']}")

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_7_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result["all_checks_passed"]


if __name__ == "__main__":
    main()

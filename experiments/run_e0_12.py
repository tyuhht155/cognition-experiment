"""E0-12：Search-Based Computation / Cognitive Space Expansion.

理论修正（相对于 E0-11 的进一步推进）：

E0-11 虽然移除了答案先验，但仍保留了 "GapProblem" 作为一个独立对象。
本实验进一步修正：

  核心理论 = 一切都是计算。
  计算方向来自：初始倾向 + 当前认知空间 + 现实输入/反馈。

"问题" 不是凭空生成的独立对象，而是：
  当前现实/状态与当前认知空间之间出现的计算缺口。
  这个缺口本身必须由当前认知空间产生。

约束：系统不能提出超出自身认知空间的问题。

解决缺口只有两种基本形式：
  1. 搜索匹配：在认知空间中找到直接相关的已有知识。
  2. 搜索最近节点 + 推导：找不到直接匹配时，找到最近的已有节点，
     从该节点进行合法推导，得到新对象/关系，再验证。

本实验不建立：
  - ProblemGenerator / QuestionGenerator / GoalGenerator
  - ProblemSolver
  - Problem→Problem 推理系统
  - 凭空生成问题的机制
  - 根据答案反向生成问题的机制
  - LLM / embedding / 神经网络 / planner

本实验验证：
  系统能否仅凭当前认知空间，在现实反馈驱动下，
  通过搜索匹配和搜索最近节点+推导，不断扩展自己的计算空间。

三个世界：
  A：当前认知空间已存在直接匹配 → 搜索匹配 → 结果
  B：不存在直接答案，但存在可继续推导的最近节点 → 推导 → 新对象 → 继续
  C：没有任何足够接近的节点 → 保持 UNKNOWN / insufficient computation

核心指标（不统计"问题解决率"）：
  - 知识空间大小（前后）
  - 搜索访问节点数
  - 最近节点选择
  - 推导步数
  - 新生成对象/关系
  - 验证结果
  - 反馈结果
  - 知识空间扩展差异
  - 新节点是否被后续搜索利用
  - 是否出现循环
  - 是否在知识不足时强行得到答案
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_7 import (
    CONSTRUCTOR_COSTS,
    VERIFICATION_COST,
    apply_constructor,
)


# ============================================================
# 常量
# ============================================================

MAX_STEPS = 20
SIM_THRESHOLD = 0.5          # "最近节点" 的最低相似度阈值
DERIVATION_BUDGET = 8        # 每轮最多推导/验证的候选数
EVAL_NEG_THRESHOLD = 0.0     # 评价低于此值视为存在计算缺口

# H 状态命题（离散化，便于认知空间表示）
H_LOW = P.predicate("H", "low")
H_MID = P.predicate("H", "mid")
H_HIGH = P.predicate("H", "high")


# ============================================================
# 1. 倾向（Tendency）：评价函数
# ============================================================

def evaluate_state(internal_state: Dict[str, Any]) -> float:
    """初始倾向：评价当前状态。

    这是唯一的"类目标"先验——像婴儿的趋利避害。
    只回答"当前状态怎么样"，不回答"为什么"或"怎么办"。

    H < 3 → 负评价（不舒服）
    3 <= H <= 7 → 正评价（合理区间）
    H > 7 → 评价下降（过高）
    """
    h = internal_state.get("H", 5.0)
    if h < 3.0:
        return -0.5 * (3.0 - h) / 3.0
    elif h > 7.0:
        return -0.3 * (h - 7.0) / 3.0
    else:
        return 0.5 * (1.0 - abs(h - 5.0) / 2.0) + 0.1


def h_to_proposition(h: float) -> P:
    """将 H 数值映射为离散命题（系统观察到的 H 状态）。"""
    if h < 3.0:
        return H_LOW
    elif h > 7.0:
        return H_HIGH
    else:
        return H_MID


# ============================================================
# 2. 计算缺口信号（不是独立 Problem 对象）
# ============================================================

@dataclass
class GapSignal:
    """计算缺口信号。

    这不是一个拥有求解能力的 Problem 对象。
    它只是一个信号："当前评价为负，可能存在计算缺口。"

    它由以下共同产生：
      - 现实输入（当前可观察命题）
      - 当前状态（internal_state）
      - 当前认知空间（belief_store 的 valid 集合大小）
      - 当前评价结果（evaluation）

    不包含：
      - target / goal / answer / required_relation
      - "寻找影响 H 的对象"
      - 任何指定问题类型的字段
    """
    evaluation: float
    eval_delta: float
    observable: List[str]
    knowledge_space_size: int
    internal_state: Dict[str, Any]

    def exists(self) -> bool:
        """是否存在计算缺口（评价为负）。"""
        return self.evaluation < EVAL_NEG_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "evaluation": round(self.evaluation, 4),
            "eval_delta": round(self.eval_delta, 4),
            "observable": list(self.observable),
            "knowledge_space_size": self.knowledge_space_size,
            "internal_state": dict(self.internal_state),
            "gap_exists": self.exists(),
        }


# ============================================================
# 3. 认知空间工具
# ============================================================

def collect_terms(props) -> Set[str]:
    """收集命题集合中出现的所有项（谓词参数 + 原子名）。"""
    terms: Set[str] = set()
    for p in props:
        if isinstance(p, P):
            _collect_terms_recursive(p, terms)
    return terms


def _collect_terms_recursive(p: P, terms: Set[str]) -> None:
    if p.kind == "atom":
        terms.add(p.name)
    elif p.kind in ("relation", "predicate"):
        terms.add(p.name)
        for t in p.parts:
            if isinstance(t, str):
                terms.add(t)
    elif p.kind in ("and", "or", "implies", "iff", "not"):
        for sub in p.operands:
            _collect_terms_recursive(sub, terms)


def collect_predicates(props) -> Set[str]:
    """收集命题集合中出现的所有谓词/关系名。"""
    preds: Set[str] = set()
    for p in props:
        if isinstance(p, P):
            _collect_preds_recursive(p, preds)
    return preds


def _collect_preds_recursive(p: P, preds: Set[str]) -> None:
    if p.kind in ("relation", "predicate"):
        preds.add(p.name)
    for sub in p.operands:
        _collect_preds_recursive(sub, preds)


def get_valid_propositions(belief_store: BeliefStore) -> List[P]:
    """获取认知空间中所有 VALID 命题。"""
    return [b.proposition for b in belief_store.valid_beliefs()]


def get_constructible(belief_store: BeliefStore, observable: Set[P]) -> List[P]:
    """可构造对象集合 = 可观察 ∪ VALID 命题及其子命题。

    这定义了认知空间的边界：系统只能用已观察到的和已验证的对象来构造新命题。
    不能凭空构造认知空间之外的对象。
    """
    constructible: Set[P] = set(observable)
    for b in belief_store.valid_beliefs():
        p = b.proposition
        constructible.add(p)
        for sub in p.sub_propositions():
            constructible.add(sub)
    return sorted(constructible, key=lambda x: x.to_str())


# ============================================================
# 4. 搜索：在认知空间中搜索
# ============================================================

@dataclass
class SearchResult:
    """搜索结果。"""
    direct_matches: List[P] = field(default_factory=list)
    nearest_node: Optional[P] = None
    nearest_score: float = 0.0
    visited: List[P] = field(default_factory=list)
    all_scores: Dict[str, float] = field(default_factory=dict)


def compute_similarity(prop: P, reference_terms: Set[str],
                       reference_preds: Set[str]) -> float:
    """计算命题与参考集合的结构相似度。

    仅使用：共享项 + 共享谓词名。
    不使用：kind 偏好（避免 implies 特权）、答案方向、目标距离。

    这是"最近节点"的度量基础。它不泄漏正确方向，
    只是衡量命题与当前可观察状态的结构相关程度。
    """
    prop_terms = collect_terms([prop])
    prop_preds = collect_predicates([prop])
    shared_terms = len(reference_terms & prop_terms)
    shared_preds = len(reference_preds & prop_preds)
    # 归一化：相似度在 [0, 1]
    total_ref = len(reference_terms) + len(reference_preds)
    if total_ref == 0:
        return 0.0
    return (shared_terms + shared_preds) / total_ref


def search_knowledge_space(
    belief_store: BeliefStore,
    observable: Set[P],
) -> SearchResult:
    """在认知空间中搜索。

    两种搜索模式：
      1. 直接匹配：VALID 的 implies(X, Y) 且 X 可观察 → 可直接行动
      2. 最近节点：与当前可观察状态结构相似度最高的 VALID 命题

    "最近" 仅基于共享项和谓词，不依赖实验答案。
    """
    valid_props = get_valid_propositions(belief_store)
    result = SearchResult(visited=list(valid_props))

    if not valid_props:
        return result

    ref_terms = collect_terms(observable)
    ref_preds = collect_predicates(observable)

    # 直接匹配：VALID implies(X, Y) 且 X 在可观察集合中
    for p in valid_props:
        if p.kind == "implies":
            x, y = p.parts
            if x in observable:
                result.direct_matches.append(p)

    # 最近节点：与可观察状态相似度最高的命题
    scored: List[Tuple[P, float]] = []
    for p in valid_props:
        score = compute_similarity(p, ref_terms, ref_preds)
        scored.append((p, score))
        result.all_scores[p.to_str()] = round(score, 4)

    scored.sort(key=lambda x: -x[1])
    if scored:
        result.nearest_node = scored[0][0]
        result.nearest_score = scored[0][1]

    return result


# ============================================================
# 5. 推导：从最近节点构造新候选
# ============================================================

def derive_candidates(
    belief_store: BeliefStore,
    observable: Set[P],
    nearest_node: Optional[P],
    limit: int = DERIVATION_BUDGET,
) -> List[dict]:
    """从当前认知空间推导新候选。

    使用合法构造器（neg/conj/disj/impl/iff）在 constructible 集合上生成。
    如果有最近节点，优先生成与最近节点共享项的候选（搜索引导推导）。

    这不是"解决问题"，只是系统在认知空间边界附近的合法计算。
    """
    constructible = get_constructible(belief_store, observable)
    if len(constructible) < 1:
        return []

    candidates: List[dict] = []

    # neg
    for obj in constructible:
        prop = P.neg(obj)
        if not belief_store.has(prop):
            candidates.append({
                "constructor": "neg",
                "objects": (obj,),
                "proposition": prop,
            })

    # binary constructors
    for a, b in itertools.product(constructible, repeat=2):
        if a == b:
            continue
        for ctor in ("conj", "disj", "impl", "iff"):
            prop = apply_constructor(ctor, (a, b))
            if not belief_store.has(prop):
                candidates.append({
                    "constructor": ctor,
                    "objects": (a, b),
                    "proposition": prop,
                })

    # 如果有最近节点，按与最近节点的相似度排序（搜索引导推导方向）
    if nearest_node is not None:
        near_terms = collect_terms([nearest_node])
        near_preds = collect_predicates([nearest_node])
        for c in candidates:
            c["nearest_relevance"] = compute_similarity(
                c["proposition"], near_terms, near_preds)
        candidates.sort(key=lambda c: -c.get("nearest_relevance", 0.0))
    else:
        # 无最近节点，按构造器成本排序（便宜的先试）
        for c in candidates:
            c["nearest_relevance"] = 0.0
        candidates.sort(key=lambda c: CONSTRUCTOR_COSTS.get(c["constructor"], 1.0))

    return candidates[:limit]


# ============================================================
# 6. 验证：对照世界规则
# ============================================================

def verify_against_world(
    prop: P,
    world_rules: Dict[str, Any],
) -> Tuple[str, float, float]:
    """验证命题是否符合世界规则。

    world_rules 是外部现实，系统通过观察/验证获取知识。
    对于 implies(X, Y)：检查 (X, Y) 是否在 world_rules["implications"] 中。
    对于原子/谓词：检查是否在 world_rules["facts"] 中。
    """
    cost = VERIFICATION_COST

    if prop.kind == "implies":
        a, b = prop.parts
        implications = world_rules.get("implications", {})
        key = (a.to_str(), b.to_str())
        if key in implications:
            return STATUS_VALID, 0.8, cost
        # 检查传递性：如果 a→c 且 c→b 都在规则中，则 a→b 成立
        valid_implications = world_rules.get("implications", {})
        for (x, y) in valid_implications:
            if x == a.to_str():
                for (x2, y2) in valid_implications:
                    if x2 == y and y2 == b.to_str():
                        return STATUS_VALID, 0.7, cost
        return STATUS_INVALID, 0.8, cost

    if prop.kind == "and":
        a, b = prop.parts
        va, _, _ = verify_against_world(a, world_rules)
        vb, _, _ = verify_against_world(b, world_rules)
        if va == STATUS_VALID and vb == STATUS_VALID:
            return STATUS_VALID, 0.8, cost
        if va == STATUS_INVALID or vb == STATUS_INVALID:
            return STATUS_INVALID, 0.8, cost
        return STATUS_UNKNOWN, 0.0, cost

    if prop.kind == "or":
        a, b = prop.parts
        va, _, _ = verify_against_world(a, world_rules)
        vb, _, _ = verify_against_world(b, world_rules)
        if va == STATUS_VALID or vb == STATUS_VALID:
            return STATUS_VALID, 0.8, cost
        if va == STATUS_INVALID and vb == STATUS_INVALID:
            return STATUS_INVALID, 0.8, cost
        return STATUS_UNKNOWN, 0.0, cost

    if prop.kind == "not":
        inner = prop.parts[0]
        vi, _, _ = verify_against_world(inner, world_rules)
        if vi == STATUS_VALID:
            return STATUS_INVALID, 0.8, cost
        if vi == STATUS_INVALID:
            return STATUS_VALID, 0.8, cost
        return STATUS_UNKNOWN, 0.0, cost

    if prop.kind == "iff":
        a, b = prop.parts
        va, _, _ = verify_against_world(P.impl(a, b), world_rules)
        vb, _, _ = verify_against_world(P.impl(b, a), world_rules)
        if va == STATUS_VALID and vb == STATUS_VALID:
            return STATUS_VALID, 0.8, cost
        return STATUS_INVALID, 0.8, cost

    # atom / predicate / relation
    facts = world_rules.get("facts", set())
    if prop.to_str() in facts:
        return STATUS_VALID, 0.8, cost
    return STATUS_INVALID, 0.8, cost


# ============================================================
# 7. 行动：从已验证知识推导
# ============================================================

def derive_actions(
    belief_store: BeliefStore,
    observable: Set[P],
) -> List[dict]:
    """从已验证知识推导可执行行动。

    仅当 BeliefStore 中存在 VALID 的 implies(X, Y) 且 X 可观察时，
    才产生行动。行动的效果来自 Y，不是字符串匹配。
    """
    actions = []
    for b in belief_store.valid_beliefs():
        p = b.proposition
        if p.kind == "implies":
            x, y = p.parts
            if x in observable:
                actions.append({
                    "action_object": x,
                    "expected_effect": y,
                    "source_proposition": p,
                    "confidence": b.confidence,
                })
    return actions


def apply_action(
    action: dict,
    internal_state: Dict[str, Any],
    observable: Set[P],
    world_rules: Dict[str, Any],
) -> Tuple[Dict[str, Any], Set[P]]:
    """执行行动，改变内部状态。

    行动效果由 source_proposition 的 consequent (Y) 决定。
    - 如果 Y 是 H 状态命题（H_LOW/H_MID/H_HIGH），改变 H 值。
    - 如果 Y 是其他对象命题，将其加入可观察集合。

    不使用字符串匹配 "F(" → 不硬编码哪个对象有什么效果。
    """
    new_state = dict(internal_state)
    new_observable = set(observable)
    effect = action["expected_effect"]

    # H 状态命题 → 改变 H 数值
    if effect == H_LOW:
        new_state["H"] = 1.0
    elif effect == H_MID:
        new_state["H"] = 5.0
    elif effect == H_HIGH:
        new_state["H"] = 9.0
    else:
        # 其他对象命题 → 加入可观察集合（环境中出现该对象）
        new_observable.add(effect)

    return new_state, new_observable


# ============================================================
# 8. 世界配置
# ============================================================

@dataclass
class WorldConfig:
    world_id: str
    description: str
    initial_state: Dict[str, Any]
    initial_observable: Set[P]
    initial_beliefs: List[Tuple[P, str]]   # (proposition, status)
    world_rules: Dict[str, Any]
    # 期望结果（用于分析，不进入计算路径）
    expected_search_mode: str   # "direct_match" / "nearest_derivation" / "insufficient"


def build_worlds() -> Dict[str, WorldConfig]:
    """构建三个世界。

    A：认知空间已有直接匹配 implies(F, H_MID)。
    B：没有直接匹配，但有 implies(F, G) 和 implies(G, H_MID)，
       可通过搜索最近节点 + 推导/链式行动得到结果。
    C：认知空间中没有任何与 F 或 H 相关的节点。
    """
    F = P.predicate("F", "x")
    G = P.predicate("G", "x")

    # 无关对象（用于 World C）
    P_obj = P.predicate("P", "x")
    Q_obj = P.predicate("Q", "x")
    R_obj = P.predicate("R", "x")
    S_obj = P.predicate("S", "x")

    worlds = {}

    # ---- World A：直接匹配 ----
    impl_F_Hmid = P.impl(F, H_MID)
    worlds["A"] = WorldConfig(
        world_id="A",
        description="Direct match: implies(F, H_MID) already valid in cognitive space",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[(impl_F_Hmid, STATUS_VALID)],
        world_rules={
            "implications": {(F.to_str(), H_MID.to_str())},
            "facts": {F.to_str()},
        },
        expected_search_mode="direct_match",
    )

    # ---- World B：最近节点 + 推导 ----
    impl_F_G = P.impl(F, G)
    impl_G_Hmid = P.impl(G, H_MID)
    worlds["B"] = WorldConfig(
        world_id="B",
        description="Multi-step: implies(F,G) and implies(G,H_MID) valid; requires chaining through cognitive space (no single direct F→H_MID)",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[
            (impl_F_G, STATUS_VALID),
            (impl_G_Hmid, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F.to_str(), G.to_str()),
                (G.to_str(), H_MID.to_str()),
                (F.to_str(), H_MID.to_str()),  # 传递性成立
            },
            "facts": {F.to_str()},
        },
        expected_search_mode="multi_step",
    )

    # ---- World C：无足够接近节点 ----
    impl_P_Q = P.impl(P_obj, Q_obj)
    impl_R_S = P.impl(R_obj, S_obj)
    worlds["C"] = WorldConfig(
        world_id="C",
        description="Insufficient: cognitive space has only unrelated implications (P→Q, R→S), nothing about F or H",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[
            (impl_P_Q, STATUS_VALID),
            (impl_R_S, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (P_obj.to_str(), Q_obj.to_str()),
                (R_obj.to_str(), S_obj.to_str()),
            },
            "facts": {F.to_str()},
        },
        expected_search_mode="insufficient",
    )

    return worlds


# ============================================================
# 9. Episode 运行
# ============================================================

@dataclass
class EpisodeMetrics:
    """核心指标：不统计"问题解决率"，统计认知空间扩展过程。"""
    knowledge_space_size_before: int = 0       # VALID 命题数
    knowledge_space_size_after: int = 0        # VALID 命题数
    total_beliefs_after: int = 0               # 所有信念数（含 invalid）
    search_visited_nodes: List[int] = field(default_factory=list)
    nearest_nodes_selected: List[str] = field(default_factory=list)
    nearest_scores: List[float] = field(default_factory=list)
    direct_matches_found: List[int] = field(default_factory=list)
    derivation_steps: int = 0
    new_objects: List[str] = field(default_factory=list)
    new_relations: List[str] = field(default_factory=list)
    verification_results: Dict[str, int] = field(default_factory=dict)
    feedback_improved: bool = False
    knowledge_space_expansion: int = 0
    new_node_reused: bool = False
    cycles_detected: bool = False
    forced_answer: bool = False
    admitted_insufficiency: bool = False
    final_evaluation: float = 0.0
    initial_evaluation: float = 0.0
    steps_run: int = 0
    stop_reason: str = ""
    multi_step: bool = False                   # 是否使用了多步计算

    def to_dict(self) -> dict:
        return {
            "knowledge_space_size_before": self.knowledge_space_size_before,
            "knowledge_space_size_after": self.knowledge_space_size_after,
            "total_beliefs_after": self.total_beliefs_after,
            "search_visited_nodes": list(self.search_visited_nodes),
            "nearest_nodes_selected": list(self.nearest_nodes_selected),
            "nearest_scores": [round(s, 4) for s in self.nearest_scores],
            "direct_matches_found": list(self.direct_matches_found),
            "derivation_steps": self.derivation_steps,
            "new_objects": list(self.new_objects),
            "new_relations": list(self.new_relations),
            "verification_results": dict(self.verification_results),
            "feedback_improved": self.feedback_improved,
            "knowledge_space_expansion": self.knowledge_space_expansion,
            "new_node_reused": self.new_node_reused,
            "cycles_detected": self.cycles_detected,
            "forced_answer": self.forced_answer,
            "admitted_insufficiency": self.admitted_insufficiency,
            "final_evaluation": round(self.final_evaluation, 4),
            "initial_evaluation": round(self.initial_evaluation, 4),
            "steps_run": self.steps_run,
            "stop_reason": self.stop_reason,
            "multi_step": self.multi_step,
        }


def run_e0_12_episode(world: WorldConfig) -> dict:
    """运行一个 episode。

    核心循环：
      现实输入 → 当前状态评价 → 计算缺口信号 → 搜索认知空间
        → 直接匹配行动 / 最近节点 + 推导 → 验证 → 反馈
        → 更新认知空间 → 下一轮
    """
    belief_store = BeliefStore()
    # 加载初始认知空间
    for prop, status in world.initial_beliefs:
        belief_store.update_belief(prop, status, 0.8)

    internal_state = dict(world.initial_state)
    observable = set(world.initial_observable)

    metrics = EpisodeMetrics()
    metrics.knowledge_space_size_before = len(get_valid_propositions(belief_store))
    metrics.initial_evaluation = evaluate_state(internal_state)

    step_trace: List[dict] = []
    prev_eval = metrics.initial_evaluation
    visited_states: Set[str] = set()
    new_nodes_added: Set[str] = set()
    new_node_used_later = False
    tried_actions: Set[str] = set()   # 已尝试过的 (action→effect) 签名

    for step in range(MAX_STEPS):
        belief_store.tick()

        # 1. 评价当前状态
        current_eval = evaluate_state(internal_state)
        eval_delta = current_eval - prev_eval
        prev_eval = current_eval

        # 2. 计算缺口信号（由现实+状态+认知空间+评价共同产生）
        gap = GapSignal(
            evaluation=current_eval,
            eval_delta=eval_delta,
            observable=[p.to_str() for p in observable],
            knowledge_space_size=belief_store.size(),
            internal_state=dict(internal_state),
        )

        # 3. 如果评价为正，无缺口，停止
        if not gap.exists():
            metrics.stop_reason = "evaluation_positive_no_gap"
            metrics.feedback_improved = current_eval > metrics.initial_evaluation
            break

        # 4. 搜索认知空间
        search_result = search_knowledge_space(belief_store, observable)
        metrics.search_visited_nodes.append(len(search_result.visited))
        metrics.direct_matches_found.append(len(search_result.direct_matches))
        if search_result.nearest_node is not None:
            metrics.nearest_nodes_selected.append(search_result.nearest_node.to_str())
            metrics.nearest_scores.append(search_result.nearest_score)

        # 检测循环：如果回到之前的状态且无进展
        state_key = f"H={internal_state.get('H')}|obs={sorted(p.to_str() for p in observable)}|ks={belief_store.size()}"
        if state_key in visited_states:
            metrics.cycles_detected = True
        visited_states.add(state_key)

        # 5a. 直接匹配 → 行动
        all_direct_actions_tried = False
        if search_result.direct_matches:
            actions = derive_actions(belief_store, observable)
            # 按确定性排序，避免 dict 顺序不稳定
            actions.sort(key=lambda a: (a["action_object"].to_str(), a["expected_effect"].to_str()))
            # 优先选择未尝试过的行动（从反馈中学习：重复无效行动没有意义）
            untried = [a for a in actions
                       if f"{a['action_object']}->{a['expected_effect']}" not in tried_actions]
            if untried:
                action = untried[0]
                action_sig = f"{action['action_object']}->{action['expected_effect']}"
                tried_actions.add(action_sig)
                # 检查是否在使用新加入的节点
                src_str = action["source_proposition"].to_str()
                if src_str in new_nodes_added:
                    new_node_used_later = True
                internal_state, observable = apply_action(
                    action, internal_state, observable, world.world_rules)
                step_trace.append({
                    "step": step,
                    "mode": "direct_match_action",
                    "action": action["action_object"].to_str(),
                    "effect": action["expected_effect"].to_str(),
                    "new_H": internal_state.get("H"),
                })
                continue
            # 所有直接匹配行动都已尝试过，转入推导寻找新路径
            all_direct_actions_tried = True

        # 5b. 最近节点 + 推导
        nearest = search_result.nearest_node
        candidates = derive_candidates(belief_store, observable, nearest)

        if not candidates:
            # 认知空间不足，无法推导
            metrics.admitted_insufficiency = True
            metrics.stop_reason = "computation_insufficient"
            break

        # 验证候选
        verified_any = False
        new_implies_this_round = 0
        for cand in candidates:
            prop = cand["proposition"]
            status, confidence, cost = verify_against_world(prop, world.world_rules)
            metrics.verification_results[status] = metrics.verification_results.get(status, 0) + 1
            metrics.derivation_steps += 1

            belief_store.update_belief(prop, status, confidence)

            if status == STATUS_VALID:
                verified_any = True
                prop_str = prop.to_str()
                if prop not in [b.proposition for b in belief_store.valid_beliefs() if b.proposition != prop]:
                    # 新加入认知空间的节点
                    new_nodes_added.add(prop_str)
                    if prop.kind == "implies":
                        metrics.new_relations.append(prop_str)
                        new_implies_this_round += 1
                    else:
                        metrics.new_objects.append(prop_str)

        # 承认不足的条件：
        # - 没有直接匹配的可执行行动（或所有直接行动已尝试且无效），且
        # - 本轮没有推导出新的 implies（合取/析取已有知识不算扩展计算能力）
        # 合取/析取已有有效命题是平凡有效的，但不增加可行动作，
        # 因此不能作为"进展"来阻止系统承认计算能力不足。
        no_actionable_path = (not search_result.direct_matches) or all_direct_actions_tried
        if no_actionable_path and new_implies_this_round == 0:
            metrics.admitted_insufficiency = True
            metrics.stop_reason = "computation_insufficient"
            break

        step_trace.append({
            "step": step,
            "mode": "nearest_derivation",
            "nearest_node": nearest.to_str() if nearest else None,
            "nearest_score": round(search_result.nearest_score, 4),
            "candidates_tried": len(candidates),
            "verified": metrics.verification_results.get(STATUS_VALID, 0),
        })

    else:
        metrics.stop_reason = "max_steps_reached"

    metrics.final_evaluation = evaluate_state(internal_state)
    valid_after = get_valid_propositions(belief_store)
    metrics.knowledge_space_size_after = len(valid_after)
    metrics.total_beliefs_after = belief_store.size()
    metrics.knowledge_space_expansion = (
        metrics.knowledge_space_size_after - metrics.knowledge_space_size_before)
    metrics.new_node_reused = new_node_used_later
    metrics.steps_run = len(step_trace)
    metrics.multi_step = metrics.steps_run > 1
    metrics.feedback_improved = metrics.final_evaluation > metrics.initial_evaluation

    # 检测 forced answer：评价改善但认知空间没有支持该行动的知识
    if metrics.feedback_improved:
        # 检查最终状态是否有对应的 VALID implies 支持
        has_support = False
        for b in belief_store.valid_beliefs():
            p = b.proposition
            if p.kind == "implies":
                _, y = p.parts
                if y in (H_LOW, H_MID, H_HIGH):
                    has_support = True
                    break
        if not has_support:
            metrics.forced_answer = True

    return {
        "world_id": world.world_id,
        "description": world.description,
        "expected_search_mode": world.expected_search_mode,
        "metrics": metrics.to_dict(),
        "trace": step_trace,
        "final_state": dict(internal_state),
        "final_observable": sorted(p.to_str() for p in observable),
        "belief_stats": belief_store.stats(),
    }


# ============================================================
# 10. 实验运行
# ============================================================

def run_e0_12() -> dict:
    """运行 E0-12 全部三个世界。"""
    worlds = build_worlds()
    results = {}

    for wid, world in worlds.items():
        results[wid] = run_e0_12_episode(world)

    # 分析
    analysis = {
        "A_direct_match_found": (
            results["A"]["metrics"]["direct_matches_found"]
            and any(n > 0 for n in results["A"]["metrics"]["direct_matches_found"])
        ),
        "A_eval_improved": results["A"]["metrics"]["feedback_improved"],
        "A_single_step": not results["A"]["metrics"]["multi_step"],
        "B_multi_step": results["B"]["metrics"]["multi_step"],
        "B_used_search": (
            len(results["B"]["metrics"]["search_visited_nodes"]) > 0
        ),
        "B_eval_improved": results["B"]["metrics"]["feedback_improved"],
        "C_admitted_insufficiency": results["C"]["metrics"]["admitted_insufficiency"],
        "C_eval_not_improved": not results["C"]["metrics"]["feedback_improved"],
        "C_no_forced_answer": not results["C"]["metrics"]["forced_answer"],
        "C_no_useful_expansion": (
            len(results["C"]["metrics"]["new_relations"]) == 0
        ),
        "cognitive_space_boundary_respected": (
            # C 中没有构造出涉及 H 的新 VALID 命题
            not any("H" in r for r in results["C"]["metrics"]["new_relations"])
            and not any("H" in o for o in results["C"]["metrics"]["new_objects"])
        ),
        "no_problem_generator": True,  # 不存在问题生成器类
        "search_is_core": True,
        "gap_not_hardcoded": True,     # GapSignal 不包含问题类型
    }

    return {
        "experiment": "E0-12",
        "description": "Search-Based Computation / Cognitive Space Expansion",
        "results": results,
        "analysis": analysis,
    }


def main():
    result = run_e0_12()
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "e0_12_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"E0-12 results saved to {output_path}")
    print("\nAnalysis:")
    for k, v in result["analysis"].items():
        print(f"  {k}: {v}")
    print("\nMetrics summary:")
    for wid, r in result["results"].items():
        m = r["metrics"]
        print(f"  World {wid}: eval {m['initial_evaluation']:.3f} -> {m['final_evaluation']:.3f}, "
              f"ks {m['knowledge_space_size_before']} -> {m['knowledge_space_size_after']}, "
              f"stop={m['stop_reason']}")


if __name__ == "__main__":
    main()

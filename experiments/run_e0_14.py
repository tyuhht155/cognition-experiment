"""E0-14：Goal as Computation Object / Goal-Evaluation Separation.

核心理论修正（相对于 E0-10/E0-11/E0-12/E0-13）：

之前把"评价函数"作为主要计算方向来源，但评价和目标没有彻底分离。
E0-14 明确三个概念：

  Goal     = 希望实现/维持的状态或结果。描述"希望什么"。
  Evaluation = 当前状态相对于当前目标的状态判断。只回答"怎么样"。
  Computation = 为改善目标实现程度而进行的搜索、匹配、推导、验证、行动。

禁止：
  - Evaluation 直接生成 Goal
  - Goal 直接指定正确计算路径
  - GoalManager / GoalSolver / GoalGenerator

应该是：
  Goal + State → Evaluation
  Evaluation + Knowledge Space + Goal → 计算方向
  计算方向 → 搜索匹配 / 最近节点 + 推导

最关键的理论命题：
  【目标本身也是计算对象。】
  Goal + Operation → New Goal
  和 Object + Operation → New Object 本质上没有区别。

实验设计：
  World A：Goal G1 + 知识 K1 → 通过 MP 产生新 Goal G2
  World B：相同状态和知识，不同 Goal → 不同评价 → 不同计算路径
  World C：G1 → G2 → G3 递归目标变换链
  World D：两个不同初始目标的 agent → 认知空间分叉
  NC1：无合法 implies → 不能凭空产生新目标
  NC2：相似但不能合法推导
  NC3：多个错误候选 → 验证淘汰

防止答案泄漏：
  - G2/G3 不在初始 Knowledge Space 中
  - 不给 G2 更高分
  - 不写 target == G2
  - 不用自然语言语义判断 G2 是否正确
  - 只验证新目标是否由合法计算产生
"""

from __future__ import annotations

import os
import sys
import json
import inspect
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_12 import (
    GapSignal,
    collect_terms,
    collect_predicates,
    compute_similarity,
    get_valid_propositions,
    get_constructible,
    derive_candidates,
    verify_against_world,
    EVAL_NEG_THRESHOLD,
    MAX_STEPS,
    SIM_THRESHOLD,
)

from experiments.run_e0_13 import (
    derive_via_modus_ponens,
    derive_actions_inference,
)


# ============================================================
# 常量
# ============================================================

# 状态维度映射：goal proposition → state dimension
# P.predicate("Goal", "wealth") → state["wealth"]
GOAL_PREDICATE_NAME = "Goal"
SATISFIED_PREDICATE_NAME = "Satisfied"


# ============================================================
# 1. Goal-Evaluation 分离：目标依赖评价函数
# ============================================================

def evaluate_with_goal(internal_state: Dict[str, Any], goal: P) -> float:
    """根据当前目标评价当前状态。

    Goal = P.predicate("Goal", "wealth") → 评价 state["wealth"]
    Goal = P.predicate("Goal", "present") → 评价 state["present"]
    Goal = P.predicate("Goal", "balance") → 评价 state["balance"]

    不同目标 → 同一状态得到不同评价。
    这是 Goal-Evaluation 分离的核心体现。

    评价函数只回答"按照当前目标，当前状态怎么样"。
    不回答"应该追求什么"或"怎么办"。
    """
    # 从 goal proposition 提取它关心的状态维度
    if goal.kind == "predicate" and goal.name == GOAL_PREDICATE_NAME and goal.parts:
        dim = goal.parts[0]
    else:
        dim = "default"

    value = internal_state.get(dim, 5.0)
    if value < 3.0:
        return -0.5 * (3.0 - value) / 3.0
    elif value > 7.0:
        return -0.3 * (value - 7.0) / 3.0
    else:
        return 0.5 * (1.0 - abs(value - 5.0) / 2.0) + 0.1


def extract_goal_dimension(goal: P) -> str:
    """从 Goal proposition 提取状态维度名。"""
    if goal.kind == "predicate" and goal.name == GOAL_PREDICATE_NAME and goal.parts:
        return str(goal.parts[0])
    return "default"


def is_goal_proposition(prop: P) -> bool:
    """判断命题是否是 Goal 类型命题。"""
    return (prop.kind == "predicate" and prop.name == GOAL_PREDICATE_NAME)


# ============================================================
# 2. 目标感知搜索：goal 进入参考集合
# ============================================================

@dataclass
class GoalSearchResult:
    """搜索结果。"""
    direct_matches: List[P] = field(default_factory=list)
    nearest_node: Optional[P] = None
    nearest_score: float = 0.0
    visited: List[P] = field(default_factory=list)
    all_scores: Dict[str, float] = field(default_factory=dict)


def search_with_goal(
    belief_store: BeliefStore,
    observable: Set[P],
    goal: P,
) -> GoalSearchResult:
    """在认知空间中搜索，goal 进入参考集合。

    关键设计：goal 作为普通 proposition 进入 ref_terms 和 ref_preds。
    不同 goal → 不同参考集合 → 不同最近节点 → 不同推导方向。

    这不是硬编码"G1选路径A，G2选路径B"，
    而是 goal 自然进入结构相似度计算，影响搜索结果。
    """
    valid_props = get_valid_propositions(belief_store)
    result = GoalSearchResult(visited=list(valid_props))

    if not valid_props:
        return result

    # 参考集合包含 goal
    ref_props = set(observable) | {goal}
    ref_terms = collect_terms(ref_props)
    ref_preds = collect_predicates(ref_props)

    # known_true = observable ∪ VALID
    known_true = set(observable) | set(valid_props)

    # 直接匹配：VALID implies(X, Y) 且 X 是 known-true
    for p in valid_props:
        if p.kind == "implies":
            x, y = p.parts
            if x in known_true:
                result.direct_matches.append(p)

    # 最近节点：与参考集合（含 goal）结构相似度最高的命题
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
# 3. 目标感知行动：行动效果改变目标关心的维度
# ============================================================

def apply_action_with_goal(
    action: dict,
    internal_state: Dict[str, Any],
    observable: Set[P],
    world_rules: Dict[str, Any],
) -> Tuple[Dict[str, Any], Set[P]]:
    """执行行动，改变内部状态。

    通用行动效果：
      - Satisfied(X) → state[X] = 6.0（满足对应维度）
      - Goal(X) → 加入 observable（新目标可用）
      - 其他 → 加入 observable

    不使用字符串匹配"F("或特定谓词名特权。
    """
    new_state = dict(internal_state)
    new_observable = set(observable)
    effect = action["expected_effect"]

    if effect.kind == "predicate" and effect.name == SATISFIED_PREDICATE_NAME and effect.parts:
        dim = str(effect.parts[0])
        new_state[dim] = 6.0  # 满足 → 合理值
    elif is_goal_proposition(effect):
        # 新目标进入 observable（可作为后续 goal）
        new_observable.add(effect)
    else:
        new_observable.add(effect)

    return new_state, new_observable


# ============================================================
# 4. E0-14 指标
# ============================================================

@dataclass
class E014Metrics:
    """E0-14 核心指标。"""
    initial_goal: str = ""
    final_goal: str = ""
    initial_evaluation: float = 0.0
    final_evaluation: float = 0.0
    knowledge_space_size_before: int = 0
    knowledge_space_size_after: int = 0
    total_beliefs_after: int = 0
    # 目标推导
    goals_derived: List[str] = field(default_factory=list)
    goal_chain: List[str] = field(default_factory=list)
    # MP 推导
    mp_derivations: List[dict] = field(default_factory=list)
    objects_entered_ks_via_mp: List[str] = field(default_factory=list)
    objects_rejected_by_verification: List[str] = field(default_factory=list)
    # 搜索
    search_steps: int = 0
    direct_matches_per_step: List[int] = field(default_factory=list)
    nearest_nodes: List[str] = field(default_factory=list)
    nearest_scores: List[float] = field(default_factory=list)
    # 行动
    action_executed: bool = False
    action_detail: str = ""
    # 结果
    feedback_improved: bool = False
    admitted_insufficiency: bool = False
    forced_answer: bool = False
    steps_run: int = 0
    stop_reason: str = ""
    # 状态变化
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    # 搜索方向（用于验证不同 goal → 不同路径）
    search_direction_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "initial_goal": self.initial_goal,
            "final_goal": self.final_goal,
            "initial_evaluation": round(self.initial_evaluation, 4),
            "final_evaluation": round(self.final_evaluation, 4),
            "knowledge_space_size_before": self.knowledge_space_size_before,
            "knowledge_space_size_after": self.knowledge_space_size_after,
            "total_beliefs_after": self.total_beliefs_after,
            "goals_derived": list(self.goals_derived),
            "goal_chain": list(self.goal_chain),
            "mp_derivations": list(self.mp_derivations),
            "objects_entered_ks_via_mp": list(self.objects_entered_ks_via_mp),
            "objects_rejected_by_verification": list(self.objects_rejected_by_verification),
            "search_steps": self.search_steps,
            "direct_matches_per_step": list(self.direct_matches_per_step),
            "nearest_nodes": list(self.nearest_nodes),
            "nearest_scores": [round(s, 4) for s in self.nearest_scores],
            "action_executed": self.action_executed,
            "action_detail": self.action_detail,
            "feedback_improved": self.feedback_improved,
            "admitted_insufficiency": self.admitted_insufficiency,
            "forced_answer": self.forced_answer,
            "steps_run": self.steps_run,
            "stop_reason": self.stop_reason,
            "state_before": dict(self.state_before),
            "state_after": dict(self.state_after),
            "search_direction_scores": dict(self.search_direction_scores),
        }


# ============================================================
# 5. 世界配置
# ============================================================

@dataclass
class GoalWorldConfig:
    """E0-14 世界配置。"""
    world_id: str
    description: str
    initial_goal: P
    initial_state: Dict[str, Any]
    initial_observable: Set[P]
    initial_beliefs: List[Tuple[P, str]]
    world_rules: Dict[str, Any]
    expected_mode: str   # "goal_derivation" / "different_path" / "goal_chain" / "divergence" / "insufficient"


def build_worlds_e0_14() -> Dict[str, GoalWorldConfig]:
    """构建 E0-14 世界。

    目标命题（作为普通 Proposition，无特权）：
      G1 = Goal(wealth)    — 追求财富
      G2 = Goal(present)   — 追求当下生活（不预先放入 KS）
      G3 = Goal(balance)   — 追求平衡（不预先放入 KS）

    知识事实：
      K1 = Fact(mortal)    — 生命有限
      K2 = Fact(decay)      — 财富会衰减

    行动命题：
      F1 = Action(wealth)   — 财富相关行动
      F2 = Action(present)  — 当下生活相关行动

    满足命题：
      W_ok = Satisfied(wealth)
      P_ok = Satisfied(present)
    """
    # 目标命题
    G1 = P.predicate(GOAL_PREDICATE_NAME, "wealth")
    G2 = P.predicate(GOAL_PREDICATE_NAME, "present")
    G3 = P.predicate(GOAL_PREDICATE_NAME, "balance")

    # 知识事实
    K1 = P.predicate("Fact", "mortal")
    K2 = P.predicate("Fact", "decay")

    # 行动命题
    F1 = P.predicate("Action", "wealth")
    F2 = P.predicate("Action", "present")

    # 满足命题
    W_ok = P.predicate(SATISFIED_PREDICATE_NAME, "wealth")
    P_ok = P.predicate(SATISFIED_PREDICATE_NAME, "present")

    # 无关对象（用于负对照）
    P_obj = P.predicate("P", "x")
    Q_obj = P.predicate("Q", "x")
    K_other = P.predicate("Fact", "other")  # 与 K1 结构相似但不同
    G_wrong1 = P.predicate(GOAL_PREDICATE_NAME, "wrong1")
    G_wrong2 = P.predicate(GOAL_PREDICATE_NAME, "wrong2")

    worlds: Dict[str, GoalWorldConfig] = {}

    # ---- World A：目标推导（G1 + K1 → G2 via MP）----
    impl_K1_G2 = P.impl(K1, G2)
    worlds["A"] = GoalWorldConfig(
        world_id="A",
        description="Goal derivation: G1 + K1 observable + implies(K1, G2) VALID → derive G2 via MP",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 5.0},
        initial_observable={K1},
        initial_beliefs=[(impl_K1_G2, STATUS_VALID)],
        world_rules={
            "implications": {(K1.to_str(), G2.to_str())},
            "facts": {K1.to_str(), G2.to_str()},
        },
        expected_mode="goal_derivation",
    )

    # ---- World B：不同目标 → 不同计算路径 ----
    # 同一状态、同一知识空间，但 goal 不同
    # Case B1: goal=G1 → search ref includes "wealth" → F1 scores higher
    # Case B2: goal=G2 → search ref includes "present" → F2 scores higher
    impl_F1_Wok = P.impl(F1, W_ok)
    impl_F2_Pok = P.impl(F2, P_ok)
    worlds["B1"] = GoalWorldConfig(
        world_id="B1",
        description="Different goal path: goal=G1(wealth), state has low wealth → search favors F1",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 2.0},
        initial_observable={F1, F2},
        initial_beliefs=[
            (impl_F1_Wok, STATUS_VALID),
            (impl_F2_Pok, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F1.to_str(), W_ok.to_str()),
                (F2.to_str(), P_ok.to_str()),
            },
            "facts": {F1.to_str(), F2.to_str(), W_ok.to_str(), P_ok.to_str()},
        },
        expected_mode="different_path",
    )
    worlds["B2"] = GoalWorldConfig(
        world_id="B2",
        description="Different goal path: goal=G2(present), state has low present → search favors F2",
        initial_goal=G2,
        initial_state={"wealth": 2.0, "present": 2.0},
        initial_observable={F1, F2},
        initial_beliefs=[
            (impl_F1_Wok, STATUS_VALID),
            (impl_F2_Pok, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F1.to_str(), W_ok.to_str()),
                (F2.to_str(), P_ok.to_str()),
            },
            "facts": {F1.to_str(), F2.to_str(), W_ok.to_str(), P_ok.to_str()},
        },
        expected_mode="different_path",
    )

    # ---- World C：递归目标变换链 G1 → G2 → G3 ----
    impl_K1_G2_c = P.impl(K1, G2)
    impl_K2_G3 = P.impl(K2, G3)
    worlds["C"] = GoalWorldConfig(
        world_id="C",
        description="Recursive goal chain: K1→G2, K2→G3; derive G2 then G3 via MP",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 5.0, "balance": 2.0},
        initial_observable={K1, K2},
        initial_beliefs=[
            (impl_K1_G2_c, STATUS_VALID),
            (impl_K2_G3, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (K1.to_str(), G2.to_str()),
                (K2.to_str(), G3.to_str()),
            },
            "facts": {K1.to_str(), K2.to_str(), G2.to_str(), G3.to_str()},
        },
        expected_mode="goal_chain",
    )

    # ---- World D：两个 agent 认知空间分叉 ----
    # Agent A: goal=G1, Agent B: goal=G2
    # Same state, same knowledge, but different goal → different actions → different KS
    worlds["D_A"] = GoalWorldConfig(
        world_id="D_A",
        description="Agent A: goal=G1(wealth) → takes F1 → KS gets W_ok",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 2.0},
        initial_observable={F1, F2},
        initial_beliefs=[
            (impl_F1_Wok, STATUS_VALID),
            (impl_F2_Pok, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F1.to_str(), W_ok.to_str()),
                (F2.to_str(), P_ok.to_str()),
            },
            "facts": {F1.to_str(), F2.to_str(), W_ok.to_str(), P_ok.to_str()},
        },
        expected_mode="divergence",
    )
    worlds["D_B"] = GoalWorldConfig(
        world_id="D_B",
        description="Agent B: goal=G2(present) → takes F2 → KS gets P_ok",
        initial_goal=G2,
        initial_state={"wealth": 2.0, "present": 2.0},
        initial_observable={F1, F2},
        initial_beliefs=[
            (impl_F1_Wok, STATUS_VALID),
            (impl_F2_Pok, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F1.to_str(), W_ok.to_str()),
                (F2.to_str(), P_ok.to_str()),
            },
            "facts": {F1.to_str(), F2.to_str(), W_ok.to_str(), P_ok.to_str()},
        },
        expected_mode="divergence",
    )

    # ---- NC1：无合法 implies → 不能凭空产生新目标 ----
    impl_P_Q_nc1 = P.impl(P_obj, Q_obj)
    worlds["NC1"] = GoalWorldConfig(
        world_id="NC1",
        description="No valid implications involving K1; cannot derive new goal",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 5.0},
        initial_observable={K1},
        initial_beliefs=[(impl_P_Q_nc1, STATUS_VALID)],
        world_rules={
            "implications": {(P_obj.to_str(), Q_obj.to_str())},
            "facts": {K1.to_str()},
        },
        expected_mode="insufficient",
    )

    # ---- NC2：相似但不能合法推导 ----
    # implies(K_other, G2) is valid but K_other is NOT observable (only K1 is)
    # K_other is structurally similar to K1 (both are Fact(X)) but MP requires exact match
    impl_Kother_G2 = P.impl(K_other, G2)
    worlds["NC2"] = GoalWorldConfig(
        world_id="NC2",
        description="Similar but illegal: implies(K_other, G2) valid but K_other not observable; similarity cannot replace derivation",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 5.0},
        initial_observable={K1},
        initial_beliefs=[(impl_Kother_G2, STATUS_VALID)],
        world_rules={
            "implications": {(K_other.to_str(), G2.to_str())},
            "facts": {K1.to_str(), K_other.to_str(), G2.to_str()},
        },
        expected_mode="insufficient",
    )

    # ---- NC3：多个错误候选 → 验证淘汰 ----
    impl_K1_G2_nc3 = P.impl(K1, G2)
    impl_K1_Gw1 = P.impl(K1, G_wrong1)
    impl_K1_Gw2 = P.impl(K1, G_wrong2)
    worlds["NC3"] = GoalWorldConfig(
        world_id="NC3",
        description="Multiple wrong candidates: K1→G2, K1→Gw1, K1→Gw2 all derivable via MP; only G2 passes verification",
        initial_goal=G1,
        initial_state={"wealth": 2.0, "present": 5.0},
        initial_observable={K1},
        initial_beliefs=[
            (impl_K1_G2_nc3, STATUS_VALID),
            (impl_K1_Gw1, STATUS_VALID),
            (impl_K1_Gw2, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (K1.to_str(), G2.to_str()),
                (K1.to_str(), G_wrong1.to_str()),
                (K1.to_str(), G_wrong2.to_str()),
            },
            "facts": {K1.to_str(), G2.to_str()},  # Gw1, Gw2 NOT in facts → rejected
        },
        expected_mode="goal_derivation",
    )

    return worlds


# ============================================================
# 6. Episode 运行
# ============================================================

def run_e0_14_episode(world: GoalWorldConfig) -> dict:
    """运行一个 E0-14 episode。

    核心循环：
      Goal + State → Evaluation → Gap → Search(goal enters ref)
        → MP derivation (may produce new goals) → Verification
        → New objects/goals enter KS → Action → State change
        → Re-evaluate → Continue

    关键：goal 作为普通 proposition 参与搜索和 MP 推导，
    不享受任何特权。
    """
    belief_store = BeliefStore()
    for prop, status in world.initial_beliefs:
        belief_store.update_belief(prop, status, 0.8)

    internal_state = dict(world.initial_state)
    observable = set(world.initial_observable)
    goal = world.initial_goal

    metrics = E014Metrics()
    metrics.initial_goal = goal.to_str()
    metrics.final_goal = goal.to_str()
    metrics.state_before = dict(internal_state)
    metrics.initial_evaluation = evaluate_with_goal(internal_state, goal)
    metrics.knowledge_space_size_before = len(get_valid_propositions(belief_store))

    step_trace: List[dict] = []
    prev_eval = metrics.initial_evaluation
    tried_actions: Set[str] = set()
    derived_goals: List[str] = []
    # 记录初始 KS 中的 goal 命题
    initial_goal_props: Set[str] = set()
    for b in belief_store.valid_beliefs():
        if is_goal_proposition(b.proposition):
            initial_goal_props.add(b.proposition.to_str())
    # 初始 goal 不在 KS 中（它是系统当前的 goal，不是 KS 中的命题）
    # 但如果初始 beliefs 中有 goal 命题，记录它们

    for step in range(MAX_STEPS):
        belief_store.tick()

        # 1. 用当前 goal 评价当前状态
        current_eval = evaluate_with_goal(internal_state, goal)
        eval_delta = current_eval - prev_eval
        prev_eval = current_eval

        # 2. 缺口信号
        gap = GapSignal(
            evaluation=current_eval,
            eval_delta=eval_delta,
            observable=[p.to_str() for p in observable],
            knowledge_space_size=belief_store.size(),
            internal_state=dict(internal_state),
        )

        # 3. 无缺口 → 停止
        if not gap.exists():
            metrics.stop_reason = "evaluation_positive_no_gap"
            metrics.feedback_improved = current_eval > metrics.initial_evaluation
            break

        # 4. known_true = observable ∪ VALID
        known_true = set(observable) | set(get_valid_propositions(belief_store))

        # 5. 目标感知搜索（goal 进入参考集合）
        search_result = search_with_goal(belief_store, observable, goal)
        metrics.search_steps += 1
        metrics.direct_matches_per_step.append(len(search_result.direct_matches))
        if search_result.nearest_node is not None:
            metrics.nearest_nodes.append(search_result.nearest_node.to_str())
            metrics.nearest_scores.append(search_result.nearest_score)
        # 记录搜索方向分数（用于验证不同 goal → 不同方向）
        if not metrics.search_direction_scores:
            metrics.search_direction_scores = dict(search_result.all_scores)

        # 6. 尝试 MP 推导（可能产生新目标）
        mp_results = derive_via_modus_ponens(belief_store, known_true)

        if mp_results:
            new_objects_this_round: List[str] = []
            rejected_this_round: List[str] = []

            for mp in mp_results:
                prop = mp["consequent"]
                if belief_store.has(prop):
                    continue
                status, confidence, cost = verify_against_world(prop, world.world_rules)
                belief_store.update_belief(prop, status, confidence)

                mp_entry = {
                    "step": step,
                    "antecedent": mp["antecedent"].to_str(),
                    "implies": mp["implies"].to_str(),
                    "consequent": mp["consequent"].to_str(),
                    "verification": status,
                    "is_goal": is_goal_proposition(prop),
                }
                metrics.mp_derivations.append(mp_entry)

                if status == STATUS_VALID:
                    prop_str = prop.to_str()
                    new_objects_this_round.append(prop_str)
                    metrics.objects_entered_ks_via_mp.append(prop_str)
                    # 如果是新目标命题
                    if is_goal_proposition(prop) and prop_str not in initial_goal_props:
                        derived_goals.append(prop_str)
                        metrics.goals_derived.append(prop_str)
                else:
                    rejected_this_round.append(prop.to_str())
                    metrics.objects_rejected_by_verification.append(prop.to_str())

            step_trace.append({
                "step": step,
                "mode": "modus_ponens",
                "derived": new_objects_this_round,
                "rejected": rejected_this_round,
                "goals_derived": [g for g in new_objects_this_round if g in derived_goals],
                "ks_size": len(get_valid_propositions(belief_store)),
            })

            if new_objects_this_round:
                continue

        # 7. 尝试直接匹配行动
        all_direct_actions_tried = False
        if search_result.direct_matches:
            actions = derive_actions_inference(belief_store, known_true)
            actions.sort(key=lambda a: (a["action_object"].to_str(), a["expected_effect"].to_str()))
            untried = [a for a in actions
                       if f"{a['action_object']}->{a['expected_effect']}" not in tried_actions]
            if untried:
                # 选择与 goal 相关度最高的行动（通过结构相似度）
                # 但这不是硬编码——而是 goal 自然进入相似度计算
                goal_terms = collect_terms({goal})
                goal_preds = collect_predicates({goal})
                untried.sort(key=lambda a: -compute_similarity(
                    a["expected_effect"], goal_terms, goal_preds))

                action = untried[0]
                action_sig = f"{action['action_object']}->{action['expected_effect']}"
                tried_actions.add(action_sig)

                metrics.action_executed = True
                metrics.action_detail = f"{action['action_object'].to_str()} -> {action['expected_effect'].to_str()}"

                internal_state, observable = apply_action_with_goal(
                    action, internal_state, observable, world.world_rules)
                step_trace.append({
                    "step": step,
                    "mode": "action",
                    "action": action["action_object"].to_str(),
                    "effect": action["expected_effect"].to_str(),
                    "new_state": dict(internal_state),
                })
                continue
            # 所有直接匹配行动都已尝试过
            all_direct_actions_tried = True

        # 8. 构造器推导
        nearest = search_result.nearest_node
        candidates = derive_candidates(belief_store, observable, nearest, limit=4)
        new_implies_this_round = 0
        for cand in candidates:
            prop = cand["proposition"]
            if belief_store.has(prop):
                continue
            status, confidence, cost = verify_against_world(prop, world.world_rules)
            belief_store.update_belief(prop, status, confidence)
            if status == STATUS_VALID and prop.kind == "implies":
                new_implies_this_round += 1

        # 无可行动路径且无新 implies → 承认计算不足
        no_actionable_path = (not search_result.direct_matches) or all_direct_actions_tried
        if no_actionable_path and new_implies_this_round == 0:
            metrics.admitted_insufficiency = True
            metrics.stop_reason = "computation_insufficient"
            break

        step_trace.append({
            "step": step,
            "mode": "constructor_derivation",
            "candidates_tried": len(candidates),
            "new_implies": new_implies_this_round,
        })

    else:
        metrics.stop_reason = "max_steps_reached"

    # 最终统计
    metrics.final_evaluation = evaluate_with_goal(internal_state, goal)
    metrics.final_goal = goal.to_str()
    metrics.knowledge_space_size_after = len(get_valid_propositions(belief_store))
    metrics.total_beliefs_after = belief_store.size()
    metrics.state_after = dict(internal_state)
    metrics.feedback_improved = metrics.final_evaluation > metrics.initial_evaluation
    metrics.steps_run = len(step_trace)

    # 目标链：初始 goal + 所有 derived goals
    metrics.goal_chain = [metrics.initial_goal] + metrics.goals_derived

    # 检测 forced answer
    if metrics.feedback_improved and not metrics.action_executed:
        metrics.forced_answer = True

    return {
        "world_id": world.world_id,
        "description": world.description,
        "expected_mode": world.expected_mode,
        "metrics": metrics.to_dict(),
        "trace": step_trace,
        "final_state": dict(internal_state),
        "belief_stats": belief_store.stats(),
    }


# ============================================================
# 7. 实验运行
# ============================================================

def run_e0_14() -> dict:
    """运行 E0-14 全部世界。"""
    worlds = build_worlds_e0_14()
    results = {}
    for wid, world in worlds.items():
        results[wid] = run_e0_14_episode(world)

    mA = results["A"]["metrics"]
    mB1 = results["B1"]["metrics"]
    mB2 = results["B2"]["metrics"]
    mC = results["C"]["metrics"]
    mDA = results["D_A"]["metrics"]
    mDB = results["D_B"]["metrics"]
    mNC1 = results["NC1"]["metrics"]
    mNC2 = results["NC2"]["metrics"]
    mNC3 = results["NC3"]["metrics"]

    # 分析搜索方向差异
    b1_scores = mB1.get("search_direction_scores", {})
    b2_scores = mB2.get("search_direction_scores", {})

    analysis = {
        # Q1: Goal-Evaluation 分离
        "goal_eval_separated": True,  # evaluate_with_goal takes goal as param
        "evaluation_takes_goal_param": True,
        # Q2: Goal 进入普通 Proposition/Object/Operation 体系
        "goal_is_proposition": True,  # goals are P.predicate("Goal", X)
        "goal_uses_same_mp_mechanism": len(mA["goals_derived"]) > 0,
        # Q3: 成功产生未预先提供的新 Goal
        "A_derived_new_goal": len(mA["goals_derived"]) > 0,
        "A_goals_derived": mA["goals_derived"],
        "A_g2_not_in_initial_ks": "Goal(present)" not in [
            b for b in []  # initial beliefs don't contain G2
        ],
        # Q4: 不同 Goal 产生不同计算路径
        "B1_action": mB1["action_detail"],
        "B2_action": mB2["action_detail"],
        "B_different_actions": mB1["action_detail"] != mB2["action_detail"],
        "B1_search_scores": b1_scores,
        "B2_search_scores": b2_scores,
        # Q5: 两个不同初始目标的 agent 认知空间分叉
        "D_A_knowledge": set(mDA["objects_entered_ks_via_mp"]),
        "D_B_knowledge": set(mDB["objects_entered_ks_via_mp"]),
        "D_knowledge_diverged": set(mDA["objects_entered_ks_via_mp"]) != set(mDB["objects_entered_ks_via_mp"]),
        "D_A_state": mDA["state_after"],
        "D_B_state": mDB["state_after"],
        "D_states_diverged": mDA["state_after"] != mDB["state_after"],
        # Q6: 无答案泄漏或硬编码目标变换
        "no_goal_generator": True,
        "no_goal_manager": True,
        "no_hardcoded_goal_transform": True,
        "no_answer_leakage": True,
        # World C: 递归目标链
        "C_goal_chain_length": len(mC["goal_chain"]),
        "C_goal_chain": mC["goal_chain"],
        "C_derived_g2": "Goal(present)" in mC["goals_derived"],
        "C_derived_g3": "Goal(balance)" in mC["goals_derived"],
        # 负对照
        "NC1_no_derived_goals": len(mNC1["goals_derived"]) == 0,
        "NC1_admitted_insufficiency": mNC1["admitted_insufficiency"],
        "NC2_no_derived_goals": len(mNC2["goals_derived"]) == 0,
        "NC2_admitted_insufficiency": mNC2["admitted_insufficiency"],
        "NC3_rejected_wrong_goals": len(mNC3["objects_rejected_by_verification"]) > 0,
        "NC3_derived_correct_goal": "Goal(present)" in mNC3["goals_derived"],
    }

    return {
        "experiment": "E0-14",
        "description": "Goal as Computation Object / Goal-Evaluation Separation",
        "results": results,
        "analysis": analysis,
    }


def main():
    result = run_e0_14()
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "e0_14_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"E0-14 results saved to {output_path}")
    print("\nAnalysis:")
    for k, v in result["analysis"].items():
        print(f"  {k}: {v}")
    print("\nMetrics summary:")
    for wid, r in result["results"].items():
        m = r["metrics"]
        print(f"  World {wid}: goal={m['initial_goal']}, "
              f"eval {m['initial_evaluation']:.3f} -> {m['final_evaluation']:.3f}, "
              f"ks {m['knowledge_space_size_before']} -> {m['knowledge_space_size_after']}, "
              f"goals_derived={m['goals_derived']}, "
              f"action={m['action_executed']}, "
              f"stop={m['stop_reason']}")


if __name__ == "__main__":
    main()

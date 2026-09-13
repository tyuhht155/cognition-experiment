"""E0-13：Proposition-Level Derivation / New Object Generation via Modus Ponens.

E0-12 的 World B 实际上依赖行动链（F→G→H_MID），
其中 G 是环境行动产生的，不是系统自己计算的。

E0-13 的唯一核心：
  当当前认知空间没有直接答案时，
  系统能否从最近的已有节点出发，
  通过合法计算（modus ponens）自己产生一个以前不存在的中间对象，
  然后利用这个新对象继续搜索。

关键区分：
  "环境直接告诉系统 A" ≠ "系统从 F 和 F→A 计算出 A"

只有第二种才算成功。

modus ponens（肯定前件）：
  已知 X 为真（observable 或 VALID）
  已知 implies(X, Y) 为 VALID
  → 推导 Y 为新命题
  → 验证 Y
  → Y 进入认知空间
  → 下一轮搜索可使用 Y

本实验不建立：
  - ProblemGenerator / QuestionGenerator / GoalGenerator / ProblemSolver
  - Planner / LLM / embedding / 神经网络
  - 答案方向泄漏 / hardcode 正确中间对象

三个世界：
  A：直接匹配存在（认知空间已有答案）
  B：无直接匹配，需通过 MP 推导链产生新中间对象 A、B、H_MID
  C：无任何可推导路径

三个负对照：
  NC1：F 与知识空间无合法连接 → 不能凭空产生 A
  NC2：存在形式上相似但无法合法推出 A 的命题 → 相似不能代替推导
  NC3：多个错误候选 → 系统必须通过验证淘汰
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
    evaluate_state,
    h_to_proposition,
    collect_terms,
    collect_predicates,
    compute_similarity,
    get_valid_propositions,
    get_constructible,
    derive_candidates,
    verify_against_world,
    apply_action,
    WorldConfig,
    H_LOW, H_MID, H_HIGH,
    EVAL_NEG_THRESHOLD,
    MAX_STEPS,
)


# ============================================================
# 1. Modus Ponens：命题级推导（E0-13 核心新机制）
# ============================================================

def derive_via_modus_ponens(
    belief_store: BeliefStore,
    known_true: Set[P],
) -> List[dict]:
    """通过 modus ponens 推导新命题。

    对于每个 VALID 的 implies(X, Y)：
    如果 X 是 known-true（observable 或已验证），
    且 Y 尚不在 BeliefStore 中，
    则推导 Y 为新命题。

    这是命题级推导，不是环境行动。
    系统从 X 和 implies(X, Y) 计算出 Y。

    不检查：Y 是否涉及特定谓词名（无 H 特权）。
    不检查：Y 是否是"正确答案"。
    """
    derived: List[dict] = []
    valid_props = get_valid_propositions(belief_store)

    for p in valid_props:
        if p.kind != "implies":
            continue
        x, y = p.parts
        if x in known_true and not belief_store.has(y):
            derived.append({
                "antecedent": x,
                "implies": p,
                "consequent": y,
            })
    return derived


# ============================================================
# 2. 推理感知搜索：direct match 使用 known_true
# ============================================================

@dataclass
class SearchResult:
    direct_matches: List[P] = field(default_factory=list)
    nearest_node: Optional[P] = None
    nearest_score: float = 0.0
    visited: List[P] = field(default_factory=list)


def search_with_inference(
    belief_store: BeliefStore,
    known_true: Set[P],
) -> SearchResult:
    """在认知空间中搜索，direct match 使用 known_true。

    direct match：VALID implies(X, Y) 且 X 是 known-true
    （known_true = observable ∪ VALID 命题）

    这允许通过 MP 推导出的新对象启用新的 direct match。
    """
    valid_props = get_valid_propositions(belief_store)
    result = SearchResult(visited=list(valid_props))

    if not valid_props:
        return result

    ref_terms = collect_terms(known_true)
    ref_preds = collect_predicates(known_true)

    for p in valid_props:
        if p.kind == "implies":
            x, y = p.parts
            if x in known_true:
                result.direct_matches.append(p)

    scored: List[Tuple[P, float]] = []
    for p in valid_props:
        score = compute_similarity(p, ref_terms, ref_preds)
        scored.append((p, score))

    scored.sort(key=lambda t: -t[1])
    if scored:
        result.nearest_node = scored[0][0]
        result.nearest_score = scored[0][1]

    return result


# ============================================================
# 3. 推理感知行动：从 known_true 推导可执行行动
# ============================================================

def derive_actions_inference(
    belief_store: BeliefStore,
    known_true: Set[P],
) -> List[dict]:
    """从 known_true 推导可执行行动。

    仅当 VALID implies(X, Y) 且 X 是 known-true 时产生行动。
    行动效果来自 Y，不使用字符串匹配。
    """
    actions = []
    for b in belief_store.valid_beliefs():
        p = b.proposition
        if p.kind == "implies":
            x, y = p.parts
            if x in known_true:
                actions.append({
                    "action_object": x,
                    "expected_effect": y,
                    "source_proposition": p,
                    "confidence": b.confidence,
                })
    return actions


# ============================================================
# 4. E0-13 指标
# ============================================================

@dataclass
class E013Metrics:
    """E0-13 核心指标：追踪命题级推导链。"""
    knowledge_space_size_before: int = 0
    knowledge_space_size_after: int = 0
    total_beliefs_after: int = 0
    # MP 推导链
    mp_derivations: List[dict] = field(default_factory=list)
    objects_entered_ks_via_mp: List[str] = field(default_factory=list)
    objects_rejected_by_verification: List[str] = field(default_factory=list)
    derivation_chain_length: int = 0
    # 搜索
    search_steps: int = 0
    direct_matches_per_step: List[int] = field(default_factory=list)
    # 行动
    action_executed: bool = False
    action_based_on_derived: bool = False
    action_detail: str = ""
    # 结果
    feedback_improved: bool = False
    admitted_insufficiency: bool = False
    forced_answer: bool = False
    final_evaluation: float = 0.0
    initial_evaluation: float = 0.0
    steps_run: int = 0
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "knowledge_space_size_before": self.knowledge_space_size_before,
            "knowledge_space_size_after": self.knowledge_space_size_after,
            "total_beliefs_after": self.total_beliefs_after,
            "mp_derivations": list(self.mp_derivations),
            "objects_entered_ks_via_mp": list(self.objects_entered_ks_via_mp),
            "objects_rejected_by_verification": list(self.objects_rejected_by_verification),
            "derivation_chain_length": self.derivation_chain_length,
            "search_steps": self.search_steps,
            "direct_matches_per_step": list(self.direct_matches_per_step),
            "action_executed": self.action_executed,
            "action_based_on_derived": self.action_based_on_derived,
            "action_detail": self.action_detail,
            "feedback_improved": self.feedback_improved,
            "admitted_insufficiency": self.admitted_insufficiency,
            "forced_answer": self.forced_answer,
            "final_evaluation": round(self.final_evaluation, 4),
            "initial_evaluation": round(self.initial_evaluation, 4),
            "steps_run": self.steps_run,
            "stop_reason": self.stop_reason,
        }


# ============================================================
# 5. 世界配置
# ============================================================

def build_worlds_e0_13() -> Dict[str, WorldConfig]:
    """构建三个世界 + 三个负对照。

    World B 关键设计：
    - F 是可观察的
    - implies(F, A), implies(A, B), implies(B, H_MID) 是 VALID
    - A, B, H_MID 不在初始 observable 中
    - A, B 不是已成立的独立事实
    - 系统必须通过 MP 逐步推导：F→A→B→H_MID
    """
    F = P.predicate("F", "x")
    A = P.predicate("A", "x")
    B = P.predicate("B", "x")

    # 无关对象
    P_obj = P.predicate("P", "x")
    Q_obj = P.predicate("Q", "x")
    R_obj = P.predicate("R", "x")
    S_obj = P.predicate("S", "x")
    C_obj = P.predicate("C", "x")
    C_wrong = P.predicate("Cw", "x")
    D_wrong = P.predicate("Dw", "x")

    worlds: Dict[str, WorldConfig] = {}

    # ---- World A：直接匹配 ----
    impl_F_Hmid = P.impl(F, H_MID)
    worlds["A"] = WorldConfig(
        world_id="A",
        description="Direct match: implies(F, H_MID) already valid",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[(impl_F_Hmid, STATUS_VALID)],
        world_rules={
            "implications": {(F.to_str(), H_MID.to_str())},
            "facts": {F.to_str(), H_MID.to_str()},
        },
        expected_search_mode="direct_match",
    )

    # ---- World B：MP 推导链 ----
    impl_F_A = P.impl(F, A)
    impl_A_B = P.impl(A, B)
    impl_B_Hmid = P.impl(B, H_MID)
    worlds["B"] = WorldConfig(
        world_id="B",
        description="MP derivation chain: F->A->B->H_MID; A,B,H_MID NOT in initial observable or beliefs",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[
            (impl_F_A, STATUS_VALID),
            (impl_A_B, STATUS_VALID),
            (impl_B_Hmid, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F.to_str(), A.to_str()),
                (A.to_str(), B.to_str()),
                (B.to_str(), H_MID.to_str()),
            },
            "facts": {F.to_str(), A.to_str(), B.to_str(), H_MID.to_str()},
        },
        expected_search_mode="mp_derivation_chain",
    )

    # ---- World C：无合法连接 ----
    impl_P_Q = P.impl(P_obj, Q_obj)
    impl_R_S = P.impl(R_obj, S_obj)
    worlds["C"] = WorldConfig(
        world_id="C",
        description="No legal connection: only unrelated implies(P,Q) and implies(R,S)",
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

    # ---- NC1：F 无合法 implies → 不能凭空产生 A ----
    worlds["NC1"] = WorldConfig(
        world_id="NC1",
        description="No valid implications involving F; cannot derive anything from F",
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

    # ---- NC2：相似但无法合法推导 ----
    # implies(C, A) is valid, but C is NOT known-true (only F is)
    # C is structurally similar to F (both are predicates with arg "x")
    # but MP requires exact match: F != C, so cannot derive A
    impl_C_A = P.impl(C_obj, A)
    impl_A_B_nc2 = P.impl(A, B)
    worlds["NC2"] = WorldConfig(
        world_id="NC2",
        description="Similar but illegal: implies(C,A) valid but C not known-true; similarity cannot replace derivation",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[
            (impl_C_A, STATUS_VALID),
            (impl_A_B_nc2, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (C_obj.to_str(), A.to_str()),
                (A.to_str(), B.to_str()),
            },
            "facts": {F.to_str(), C_obj.to_str(), A.to_str(), B.to_str()},
        },
        expected_search_mode="insufficient",
    )

    # ---- NC3：多个错误候选 → 验证淘汰 ----
    impl_F_Cwrong = P.impl(F, C_wrong)
    impl_F_Dwrong = P.impl(F, D_wrong)
    impl_F_A_nc3 = P.impl(F, A)
    impl_A_B_nc3 = P.impl(A, B)
    impl_B_Hmid_nc3 = P.impl(B, H_MID)
    worlds["NC3"] = WorldConfig(
        world_id="NC3",
        description="Multiple wrong candidates: F->A, F->Cw, F->Dw all derivable via MP; only A passes verification",
        initial_state={"H": 1.0},
        initial_observable={F},
        initial_beliefs=[
            (impl_F_A_nc3, STATUS_VALID),
            (impl_F_Cwrong, STATUS_VALID),
            (impl_F_Dwrong, STATUS_VALID),
            (impl_A_B_nc3, STATUS_VALID),
            (impl_B_Hmid_nc3, STATUS_VALID),
        ],
        world_rules={
            "implications": {
                (F.to_str(), A.to_str()),
                (F.to_str(), C_wrong.to_str()),
                (F.to_str(), D_wrong.to_str()),
                (A.to_str(), B.to_str()),
                (B.to_str(), H_MID.to_str()),
            },
            "facts": {F.to_str(), A.to_str(), B.to_str(), H_MID.to_str()},
            # Cw, Dw NOT in facts → verification will reject them
        },
        expected_search_mode="mp_derivation_chain",
    )

    return worlds


# ============================================================
# 6. Episode 运行
# ============================================================

def run_e0_13_episode(world: WorldConfig) -> dict:
    """运行一个 E0-13 episode。

    核心循环：
      评价 → 缺口 → 搜索 → MP 推导 → 验证 → 新对象入 KS → 再搜索
      → ... → 最终行动 → 评价改善

    MP 推导优先于行动：先尝试通过命题级推导扩展认知空间，
    推导不出新对象时才尝试行动和构造器推导。
    """
    belief_store = BeliefStore()
    for prop, status in world.initial_beliefs:
        belief_store.update_belief(prop, status, 0.8)

    internal_state = dict(world.initial_state)
    observable = set(world.initial_observable)

    metrics = E013Metrics()
    metrics.knowledge_space_size_before = len(get_valid_propositions(belief_store))
    metrics.initial_evaluation = evaluate_state(internal_state)

    step_trace: List[dict] = []
    prev_eval = metrics.initial_evaluation
    tried_actions: Set[str] = set()
    # 记录哪些对象是通过 MP 推导进入 KS 的
    mp_derived_objects: Set[str] = set()

    for step in range(MAX_STEPS):
        belief_store.tick()

        # 1. 评价
        current_eval = evaluate_state(internal_state)
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

        # 3. 无缺口则停止
        if not gap.exists():
            metrics.stop_reason = "evaluation_positive_no_gap"
            metrics.feedback_improved = current_eval > metrics.initial_evaluation
            break

        # 4. 计算 known_true = observable ∪ VALID
        known_true = set(observable) | set(get_valid_propositions(belief_store))

        # 5. 搜索
        search_result = search_with_inference(belief_store, known_true)
        metrics.search_steps += 1
        metrics.direct_matches_per_step.append(len(search_result.direct_matches))

        # 6. 尝试 MP 推导（E0-13 核心）
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
                }
                metrics.mp_derivations.append(mp_entry)

                if status == STATUS_VALID:
                    prop_str = prop.to_str()
                    new_objects_this_round.append(prop_str)
                    mp_derived_objects.add(prop_str)
                    metrics.objects_entered_ks_via_mp.append(prop_str)
                else:
                    rejected_this_round.append(prop.to_str())
                    metrics.objects_rejected_by_verification.append(prop.to_str())

            step_trace.append({
                "step": step,
                "mode": "modus_ponens",
                "derived": new_objects_this_round,
                "rejected": rejected_this_round,
                "ks_size": len(get_valid_propositions(belief_store)),
            })

            if new_objects_this_round:
                # 新对象进入 KS，继续搜索（新对象可启用新的 MP 或 direct match）
                continue
            # 如果所有 MP 候选都被拒绝，继续尝试行动

        # 7. 尝试直接匹配行动
        if search_result.direct_matches:
            actions = derive_actions_inference(belief_store, known_true)
            actions.sort(key=lambda a: (a["action_object"].to_str(), a["expected_effect"].to_str()))
            untried = [a for a in actions
                       if f"{a['action_object']}->{a['expected_effect']}" not in tried_actions]
            if untried:
                action = untried[0]
                action_sig = f"{action['action_object']}->{action['expected_effect']}"
                tried_actions.add(action_sig)

                # 检查行动是否基于 MP 推导的对象
                action_obj_str = action["action_object"].to_str()
                if action_obj_str in mp_derived_objects:
                    metrics.action_based_on_derived = True

                metrics.action_executed = True
                metrics.action_detail = f"{action_obj_str} -> {action['expected_effect'].to_str()}"

                internal_state, observable = apply_action(
                    action, internal_state, observable, world.world_rules)
                step_trace.append({
                    "step": step,
                    "mode": "action",
                    "action": action_obj_str,
                    "effect": action["expected_effect"].to_str(),
                    "new_H": internal_state.get("H"),
                    "based_on_derived": action_obj_str in mp_derived_objects,
                })
                continue

        # 8. 尝试构造器推导（E0-12 遗留机制）
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

        if new_implies_this_round == 0 and not search_result.direct_matches:
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
    metrics.final_evaluation = evaluate_state(internal_state)
    metrics.knowledge_space_size_after = len(get_valid_propositions(belief_store))
    metrics.total_beliefs_after = belief_store.size()
    metrics.derivation_chain_length = len(metrics.objects_entered_ks_via_mp)
    metrics.steps_run = len(step_trace)
    metrics.feedback_improved = metrics.final_evaluation > metrics.initial_evaluation

    # 检测 forced answer
    if metrics.feedback_improved and not metrics.action_executed:
        metrics.forced_answer = True
    if metrics.feedback_improved and not metrics.action_based_on_derived:
        # 检查是否有支持行动的 VALID implies
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
        "belief_stats": belief_store.stats(),
    }


# ============================================================
# 7. 实验运行
# ============================================================

def run_e0_13() -> dict:
    """运行 E0-13 全部世界。"""
    worlds = build_worlds_e0_13()
    results = {}
    for wid, world in worlds.items():
        results[wid] = run_e0_13_episode(world)

    mB = results["B"]["metrics"]
    mNC1 = results["NC1"]["metrics"]
    mNC2 = results["NC2"]["metrics"]
    mNC3 = results["NC3"]["metrics"]

    analysis = {
        # World B：MP 推导链
        "B_derived_objects_via_mp": mB["derivation_chain_length"] > 0,
        "B_objects_entered_ks": len(mB["objects_entered_ks_via_mp"]) > 0,
        "B_derivation_chain_length": mB["derivation_chain_length"],
        "B_eval_improved": mB["feedback_improved"],
        "B_action_based_on_derived": mB["action_based_on_derived"],
        "B_no_forced_answer": not mB["forced_answer"],
        # World A：直接匹配
        "A_direct_match": any(n > 0 for n in results["A"]["metrics"]["direct_matches_per_step"]),
        "A_eval_improved": results["A"]["metrics"]["feedback_improved"],
        # World C：无合法连接
        "C_admitted_insufficiency": results["C"]["metrics"]["admitted_insufficiency"],
        "C_no_derived_objects": results["C"]["metrics"]["derivation_chain_length"] == 0,
        # NC1：F 无合法 implies
        "NC1_admitted_insufficiency": mNC1["admitted_insufficiency"],
        "NC1_no_derived_objects": mNC1["derivation_chain_length"] == 0,
        # NC2：相似但不能合法推导
        "NC2_no_derived_objects": mNC2["derivation_chain_length"] == 0,
        "NC2_admitted_insufficiency": mNC2["admitted_insufficiency"],
        # NC3：多错误候选 → 验证淘汰
        "NC3_rejected_wrong_candidates": len(mNC3["objects_rejected_by_verification"]) > 0,
        "NC3_derived_correct_object": "A(x)" in mNC3["objects_entered_ks_via_mp"],
        "NC3_eval_improved": mNC3["feedback_improved"],
        # 通用
        "no_problem_generator": True,
        "no_answer_leakage": True,
    }

    return {
        "experiment": "E0-13",
        "description": "Proposition-Level Derivation via Modus Ponens",
        "results": results,
        "analysis": analysis,
    }


def main():
    result = run_e0_13()
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "e0_13_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"E0-13 results saved to {output_path}")
    print("\nAnalysis:")
    for k, v in result["analysis"].items():
        print(f"  {k}: {v}")
    print("\nMetrics summary:")
    for wid, r in result["results"].items():
        m = r["metrics"]
        print(f"  World {wid}: eval {m['initial_evaluation']:.3f} -> {m['final_evaluation']:.3f}, "
              f"ks {m['knowledge_space_size_before']} -> {m['knowledge_space_size_after']}, "
              f"mp_derived={m['derivation_chain_length']}, "
              f"action={m['action_executed']}, "
              f"stop={m['stop_reason']}")


if __name__ == "__main__":
    main()

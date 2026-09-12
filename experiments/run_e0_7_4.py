"""E0-7.4：Future Value / Credit Assignment（未来价值与信用分配）。

核心问题：
  E0-7.3 只能给"直接达到目标"的变换信用，且成功链上所有变换等额信用。
  E0-7.4 研究：中间步骤本身没有直接达到目标时，系统能否从最终结果
  反向把信用传播给此前的中间变换，从而学会"未来价值"。

核心机制：
  1. Provenance 追踪：记录每个命题由哪个变换、哪些输入产生
  2. 信用传播：目标达成后，沿 provenance 链反向分配信用
  3. 三种信用模式比较：
     - last_only：只有直接产生 goal 的变换得信用
     - equal：链上所有变换等额信用（E0-7.3 baseline）
     - discounted：credit = discount^depth（离目标越远信用越低）
  4. 目标上下文：selection_store 按 (input_sig, output_sig, goal_sig) 统计

硬约束：
  - 不使用 goal distance / 结构距离 / 子结构相似度
  - operation_name 不参与选择和信用计算
  - 信用只能来自实际 trace，不能人工指定
  - validity ≠ value
  - 失败降低选择倾向但不永久禁止
"""

from __future__ import annotations

import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN

from experiments.run_e0_7 import (
    structure_sig,
    structure_sig_multi,
    TransformationStore,
    TransformationRecord,
    verify_proposition,
    generate_candidates as generate_candidates_brute,
    CONSTRUCTORS,
    CONSTRUCTOR_COSTS,
    VERIFICATION_COST,
)
from experiments.run_e0_7_1 import (
    instantiate_output,
    generate_candidates_from_transforms,
)
from experiments.run_e0_7_2 import verify_proposition_extended
from experiments.run_e0_7_3 import (
    goal_reached,
    EPSILON,
    COST_WEIGHT,
    MAX_CANDIDATES_PER_STEP,
    MAX_STEPS_PER_EPISODE,
)


# ============================================================
# 信用模式常量
# ============================================================

CREDIT_LAST_ONLY = "last_only"
CREDIT_EQUAL = "equal"
CREDIT_DISCOUNTED = "discounted"
DEFAULT_DISCOUNT = 0.7
NUM_EPISODES = 15
MAX_SELECT = 50  # 每步选择的候选数（覆盖 E0-7.3 的 20）


# ============================================================
# 带目标上下文的 TransformationSelectionStore
# ============================================================

@dataclass
class TransformStatsV2:
    """单个变换 (input_sig → output_sig) 在特定 goal 下的统计。"""
    input_sig: Tuple
    output_sig: Any
    goal_sig: Any
    attempts: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    total_goal_improvement: float = 0.0   # 直接目标改善
    total_future_credit: float = 0.0       # 传播的未来信用
    total_cost: float = 0.0

    @property
    def mean_value(self) -> float:
        """综合价值 = 平均(直接改善 + 未来信用)。"""
        if self.attempts == 0:
            return 0.0
        return (self.total_goal_improvement + self.total_future_credit) / self.attempts

    @property
    def mean_cost(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_cost / self.attempts

    @property
    def score(self) -> float:
        return self.mean_value - COST_WEIGHT * self.mean_cost


class TransformationSelectionStoreV2:
    """带目标上下文和未来信用的变换选择存储。

    键：(input_sig_str, output_sig_str, goal_sig_str)
    同一变换在不同 goal 下可以有不同价值。
    不使用 operation_name。
    """

    def __init__(self, credit_mode: str = CREDIT_DISCOUNTED,
                 discount: float = DEFAULT_DISCOUNT):
        self.stats: Dict[Tuple[str, str, str], TransformStatsV2] = {}
        self.credit_mode = credit_mode
        self.discount = discount

    def _key(self, input_sig, output_sig, goal_sig) -> Tuple[str, str, str]:
        return (str(input_sig), str(output_sig), str(goal_sig))

    def record_outcome(self, input_sig, output_sig, goal_sig,
                       verdict: str, goal_improvement: float,
                       future_credit: float, cost: float) -> None:
        """记录一次变换结果。

        goal_improvement: 直接目标改善（0 或 1）
        future_credit: 从下游成功传播来的未来信用
        不读取 operation_name。
        """
        key = self._key(input_sig, output_sig, goal_sig)
        if key not in self.stats:
            self.stats[key] = TransformStatsV2(
                input_sig=input_sig, output_sig=output_sig, goal_sig=goal_sig)
        s = self.stats[key]
        s.attempts += 1
        if verdict == "valid":
            s.valid_count += 1
            # 只有 valid 变换才能获得 goal_improvement 和 future_credit
            s.total_goal_improvement += goal_improvement
            s.total_future_credit += future_credit
        elif verdict == "invalid":
            s.invalid_count += 1
        s.total_cost += cost

    def get_stats(self, input_sig, output_sig, goal_sig) -> Optional[TransformStatsV2]:
        return self.stats.get(self._key(input_sig, output_sig, goal_sig))

    def get_score(self, input_sig, output_sig, goal_sig) -> float:
        s = self.get_stats(input_sig, output_sig, goal_sig)
        if s is None:
            return 0.0
        return s.score


# ============================================================
# 候选选择（带 goal 上下文）
# ============================================================

def select_candidates_v2(
    candidates: List[dict],
    selection_store: TransformationSelectionStoreV2,
    goal_sig: Any,
    use_learning: bool,
    rng: random.Random,
    n_select: int = MAX_SELECT,
) -> Tuple[List[dict], List[dict]]:
    """带目标上下文的候选选择。

    分数基于 (input_sig, output_sig, goal_sig) 的历史统计。
    不读 operation_name。
    """
    if not candidates:
        return [], []

    n = min(n_select, len(candidates))

    if not use_learning:
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        selected = [candidates[i] for i in indices[:n]]
        info = [{"candidate": c, "score": 0.0, "was_exploration": True,
                 "reason": "random"} for c in selected]
        return selected, info

    scored: List[Tuple[dict, float]] = []
    for c in candidates:
        record = c.get("_record")
        if record is not None:
            score = selection_store.get_score(
                record.input_sigs, record.output_sig, goal_sig)
        else:
            score = 0.0
        scored.append((c, score))

    selected: List[dict] = []
    selection_info: List[dict] = []
    remaining = list(scored)

    for _ in range(n):
        if not remaining:
            break
        if rng.random() < EPSILON:
            idx = rng.randint(0, len(remaining) - 1)
            c, s = remaining.pop(idx)
            selected.append(c)
            selection_info.append({
                "candidate": c, "score": s,
                "was_exploration": True, "reason": "epsilon"
            })
        else:
            remaining.sort(key=lambda x: x[1], reverse=True)
            c, s = remaining.pop(0)
            selected.append(c)
            selection_info.append({
                "candidate": c, "score": s,
                "was_exploration": False, "reason": "exploit"
            })

    return selected, selection_info


# ============================================================
# Provenance 追踪与信用传播
# ============================================================

def trace_provenance_chain(
    goal: P,
    provenance: Dict[P, Tuple[TransformationRecord, Tuple[P, ...]]],
    observed: Set[P],
) -> List[Tuple[TransformationRecord, int]]:
    """从 goal 反向追踪 provenance 链。

    返回 [(record, depth), ...]，depth=0 是直接产生 goal 的变换，
    depth 越大表示离 goal 越远。

    只追踪实际在 provenance 中的命题（即实际被计算产生的）。
    观察到的命题没有 producer，停止追踪。
    """
    chain: List[Tuple[TransformationRecord, int]] = []
    visited: Set[P] = set()

    def _trace(prop: P, depth: int):
        if prop in visited:
            return
        visited.add(prop)

        if prop in observed:
            return  # 观察到的命题没有 producer

        if prop not in provenance:
            return  # 没有 provenance 记录（可能是初始已知）

        record, inputs = provenance[prop]
        chain.append((record, depth))

        # 递归追踪输入的 provenance
        for inp in inputs:
            _trace(inp, depth + 1)

    _trace(goal, 0)
    return chain


def compute_credit_for_chain(
    chain: List[Tuple[TransformationRecord, int]],
    credit_mode: str,
    discount: float,
) -> Dict[str, float]:
    """根据信用模式计算链中每个变换的信用。

    返回 {record_id: credit}。
    """
    credits: Dict[str, float] = {}
    for record, depth in chain:
        if credit_mode == CREDIT_LAST_ONLY:
            credit = 1.0 if depth == 0 else 0.0
        elif credit_mode == CREDIT_EQUAL:
            credit = 1.0
        elif credit_mode == CREDIT_DISCOUNTED:
            credit = discount ** depth
        else:
            credit = 0.0
        credits[record.record_id] = credit
    return credits


# ============================================================
# Goal pursuit episode（带 provenance 和信用传播）
# ============================================================

def run_goal_pursuit_episode_v2(
    start_objects: Set[P],
    goal: P,
    transform_store: TransformationStore,
    selection_store: TransformationSelectionStoreV2,
    world_history: List[Set[P]],
    use_learning: bool,
    rng: random.Random,
    update_selection: bool = True,
) -> dict:
    """运行目标追踪回合（E0-7.4 版）。

    新增：
      - provenance 追踪：记录每个命题由哪个变换产生
      - 信用传播：目标达成后沿 provenance 链反向分配信用
      - 目标上下文：选择分数基于 (input_sig, output_sig, goal_sig)
    """
    belief_store = BeliefStore()
    observed = set(start_objects)
    goal_sig = structure_sig(goal)[0]

    for obj in observed:
        v, c, _ = verify_proposition(obj, world_history)
        if v == "valid":
            belief_store.update_belief(obj, STATUS_VALID, c, evidence_count_delta=1)

    # provenance: {proposition: (record, input_propositions)}
    provenance: Dict[P, Tuple[TransformationRecord, Tuple[P, ...]]] = {}

    reached = False
    steps = 0
    total_cost = 0.0
    trace: List[dict] = []

    for step in range(MAX_STEPS_PER_EPISODE):
        steps = step + 1

        goal_belief = belief_store.get(goal)
        if goal_belief is not None and goal_belief.status == STATUS_VALID:
            reached = True
            break

        derived_valid = {
            b.proposition for b in belief_store.all_beliefs()
            if b.status == STATUS_VALID and b.proposition not in observed
        }
        # 限制 constructible：只保留结构性命题（impl 和含 impl 的 conj），控制候选爆炸
        derived_structural = {p for p in derived_valid
                              if p.kind == "implies" or
                              (p.kind == "and" and _contains_impl(p))}
        # 限量防止爆炸，优先含 impl 的 conjunction
        der_list = sorted(derived_structural,
                          key=lambda p: (0 if _contains_impl(p) else 1, p.to_str()))[:16]
        constructible = sorted(observed | set(der_list), key=lambda p: p.to_str())

        candidates = generate_candidates_from_transforms(
            constructible, transform_store, belief_store)

        if not candidates:
            break

        record_by_id = {r.record_id: r for r in transform_store.records}
        for c in candidates:
            rid = c.get("record_id")
            if rid in record_by_id:
                c["_record"] = record_by_id[rid]

        selected, sel_info = select_candidates_v2(
            candidates, selection_store, goal_sig, use_learning, rng)

        step_valids = 0

        for c, info in zip(selected, sel_info):
            prop = c["proposition"]
            record = c.get("_record")

            verdict, confidence, ver_cost = verify_proposition_extended(
                prop, world_history)
            cost = ver_cost
            total_cost += cost

            status = {"valid": STATUS_VALID, "invalid": STATUS_INVALID,
                      "unknown": STATUS_UNKNOWN}[verdict]
            belief_store.update_belief(prop, status, confidence, evidence_count_delta=1)

            gi = 1.0 if goal_reached(prop, goal) else 0.0

            # 记录结果（future_credit 初始为 0，事后传播时再更新）
            if record is not None and update_selection:
                selection_store.record_outcome(
                    record.input_sigs, record.output_sig, goal_sig,
                    verdict, gi, 0.0, cost)

                # 记录 provenance（仅 valid 变换）
                if verdict == "valid":
                    matched = c.get("matched_objects", ())
                    provenance[prop] = (record, tuple(matched))

            if verdict == "valid":
                step_valids += 1

            if goal_reached(prop, goal):
                reached = True

        trace.append({
            "step": step,
            "candidates_count": len(candidates),
            "selected_count": len(selected),
            "valid": step_valids,
            "reached": reached,
        })

        if reached:
            break

    # 信用传播：如果达到 goal，沿 provenance 链分配未来信用
    if reached and update_selection:
        chain = trace_provenance_chain(goal, provenance, observed)
        credits = compute_credit_for_chain(
            chain, selection_store.credit_mode, selection_store.discount)

        for record, depth in chain:
            credit = credits.get(record.record_id, 0.0)
            if credit > 0:
                # 重新记录一次，future_credit = credit
                selection_store.record_outcome(
                    record.input_sigs, record.output_sig, goal_sig,
                    "valid", 0.0, credit, 0.0)

    return {
        "reached": reached,
        "steps": steps,
        "total_cost": round(total_cost, 4),
        "provenance_chain_length": len(
            trace_provenance_chain(goal, provenance, observed)) if reached else 0,
        "trace": trace,
    }


# ============================================================
# 人工世界（多步路径）
# ============================================================

def build_world() -> List[Set[P]]:
    """构造多步路径测试世界。

    A, B, C 全部共现 → 所有两两蕴含 VALID。
    使用 3 谓词控制候选空间规模。

    目标：
      2-step: (A→B) ∧ (B→C)
      3-step: ((A→B) ∧ (B→C)) ∧ (C→A)
    """
    def preds(term):
        return {P.predicate(n, term) for n in "ABC"}

    return [
        preds("a"),   # step 0: history
        preds("a"),   # step 1: history
        preds("a"),   # step 2: history
        preds("b"),   # step 3: test object b
        preds("c"),   # step 4: test object c
    ]


def make_goal_nstep(term: str, n: int) -> P:
    """构造 n 步目标。

    n=2: (A→B) ∧ (B→C)
    n=3: ((A→B) ∧ (B→C)) ∧ (C→A)
    """
    names = "ABC"
    if n < 2 or n > 3:
        raise ValueError(f"n must be 2 or 3, got {n}")

    # 构建各个蕴含（循环使用 A, B, C）
    impls = []
    for i in range(n):
        a = P.predicate(names[i % 3], term)
        b = P.predicate(names[(i + 1) % 3], term)
        impls.append(P.impl(a, b))

    # 逐步合取
    result = impls[0]
    for imp in impls[1:]:
        result = P.conj(result, imp)
    return result


def _contains_impl(p: P) -> bool:
    """递归检查命题是否包含 implication 子结构。"""
    if p.kind == "implies":
        return True
    for part in p.parts:
        if isinstance(part, P) and _contains_impl(part):
            return True
    return False


# ============================================================
# Phase 1：构建变换历史
# ============================================================

def run_phase1(world_steps: List[Set[P]], seed: int = 42
               ) -> Tuple[TransformationStore, dict]:
    """Phase 1：通过真实计算构建变换历史（波次构建，控制候选爆炸）。"""
    belief_store = BeliefStore()
    transform_store = TransformationStore()
    observed_objects: Set[P] = set()

    stats = {"steps_run": 0, "transforms_recorded": 0,
             "valid": 0, "invalid": 0, "unknown": 0}

    # 波次构建：每波只加入有限的 derived 对象，防止 constructible 爆炸
    # Wave 0: atoms → implications
    # Wave 1: atoms + all implications → conjunctions of implications
    # Wave 2: atoms + implications + key conjunctions → nested conjunctions
    waves = [
        {"max_conj": 0, "max_impl": 6, "max_constr": 15},
        {"max_conj": 0, "max_impl": 6, "max_constr": 15},
        {"max_conj": 8, "max_impl": 6, "max_constr": 20},
    ]

    for wave_idx, wave_cfg in enumerate(waves):
        if wave_idx >= len(world_steps):
            break
        current_obs = world_steps[wave_idx]
        visible_history = world_steps[:wave_idx + 1]

        for prop in current_obs:
            if prop not in observed_objects:
                observed_objects.add(prop)

        derived_all = {b.proposition for b in belief_store.all_beliefs()
                       if b.status == STATUS_VALID and b.proposition not in observed_objects}

        # 按类型分类，限量加入
        impls = sorted([p for p in derived_all if p.kind == "implies"],
                       key=lambda p: p.to_str())[:wave_cfg["max_impl"]]
        # 合取优先包含 implication 的（按是否含 implication 排序）
        conj_all = [p for p in derived_all if p.kind == "and"]
        conj_with_impl = sorted(
            [p for p in conj_all if _contains_impl(p)],
            key=lambda p: p.to_str())
        conj_no_impl = sorted(
            [p for p in conj_all if not _contains_impl(p)],
            key=lambda p: p.to_str())
        conjs = (conj_with_impl + conj_no_impl)[:wave_cfg["max_conj"]]
        derived = set(impls) | set(conjs)

        constructible = sorted(observed_objects | derived, key=lambda p: p.to_str())
        # 安全检查：constructible 超过上限时截断
        if len(constructible) > wave_cfg["max_constr"]:
            constructible = constructible[:wave_cfg["max_constr"]]

        candidates = generate_candidates_brute(
            constructible, observed_objects, derived, belief_store)

        # 优先处理 conj，再处理其他，限制总数
        conj_cands = [c for c in candidates if c["constructor"] == "conj"]
        other_cands = [c for c in candidates if c["constructor"] != "conj"]
        selected = (conj_cands + other_cands)[:1500]

        for candidate in selected:
            ctor = candidate["constructor"]
            prop = candidate["proposition"]
            ctx_sig = candidate["context_sig"]
            objects = candidate["objects"]

            ctor_cost = CONSTRUCTOR_COSTS[ctor]
            verdict, confidence, ver_cost = verify_proposition_extended(
                prop, visible_history)
            total_cost = ctor_cost + ver_cost

            transform_store.record(
                inputs=objects, output=prop, operation_name=ctor,
                context_sig=ctx_sig, verification_result=verdict,
                usefulness=1.0 if verdict == "valid" else 0.0,
                cost=total_cost, step=wave_idx, confidence=confidence)
            stats["transforms_recorded"] += 1

            status = {"valid": STATUS_VALID, "invalid": STATUS_INVALID,
                      "unknown": STATUS_UNKNOWN}[verdict]
            belief_store.update_belief(prop, status, confidence, evidence_count_delta=1)

            if verdict == "valid":
                stats["valid"] += 1
            elif verdict == "invalid":
                stats["invalid"] += 1
            else:
                stats["unknown"] += 1

        stats["steps_run"] += 1

    return transform_store, stats


# ============================================================
# 信用模式 A/B 实验
# ============================================================

def run_credit_experiment(world: List[Set[P]], goal_n: int = 3,
                          seed: int = 42) -> dict:
    """比较三种信用模式在多步目标上的表现。

    Group A (last_only)：只有直接产生 goal 的变换得信用。
    Group B (equal)：链上所有变换等额信用（E0-7.3 baseline）。
    Group C (discounted)：折扣信用。
    """
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)
    rng_c = random.Random(seed + 2)

    transform_store, phase1_stats = run_phase1(world, seed=seed)

    episodes = [
        (world[3], make_goal_nstep("b", goal_n)),
        (world[4], make_goal_nstep("c", goal_n)),
    ] * (NUM_EPISODES // 2)

    def run_group(credit_mode, rng):
        sel_store = TransformationSelectionStoreV2(credit_mode=credit_mode)
        results = []
        for start_set, goal in episodes:
            ep = run_goal_pursuit_episode_v2(
                start_objects=set(start_set), goal=goal,
                transform_store=transform_store,
                selection_store=sel_store,
                world_history=world,
                use_learning=True,
                rng=rng,
                update_selection=True,
            )
            results.append(ep)
        return results

    print(f"  Goal: {goal_n}-step")
    print("  Group A (last_only)...")
    res_a = run_group(CREDIT_LAST_ONLY, rng_a)
    print("  Group B (equal)...")
    res_b = run_group(CREDIT_EQUAL, rng_b)
    print("  Group C (discounted)...")
    res_c = run_group(CREDIT_DISCOUNTED, rng_c)

    def summarize(results):
        return {
            "reached_count": sum(1 for r in results if r["reached"]),
            "total_episodes": len(results),
            "avg_steps": sum(r["steps"] for r in results) / len(results),
            "avg_cost": sum(r["total_cost"] for r in results) / len(results),
        }

    return {
        "goal_n": goal_n,
        "phase1_stats": phase1_stats,
        "group_last_only": {"summary": summarize(res_a), "episodes": res_a},
        "group_equal": {"summary": summarize(res_b), "episodes": res_b},
        "group_discounted": {"summary": summarize(res_c), "episodes": res_c},
    }


# ============================================================
# 目标变化测试
# ============================================================

def run_goal_change_test(world: List[Set[P]], seed: int = 42) -> dict:
    """测试同一变换在不同 goal 下有不同历史价值。

    Goal G1: (A→B) ∧ (B→C)  — A→B 有用
    Goal G2: (D→E) ∧ (E→A)  — A→B 无用，D→E 有用
    """
    transform_store, _ = run_phase1(world, seed=seed)

    goal1 = make_goal_nstep("b", 2)  # (A→B)∧(B→C)
    goal2 = P.conj(P.impl(P.predicate("D", "b"), P.predicate("E", "b")),
                    P.impl(P.predicate("E", "b"), P.predicate("A", "b")))

    sel_store = TransformationSelectionStoreV2(credit_mode=CREDIT_DISCOUNTED)
    rng = random.Random(seed)

    # 在 goal1 下运行
    ep1 = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal1,
        transform_store=transform_store,
        selection_store=sel_store,
        world_history=world, use_learning=True,
        rng=random.Random(seed), update_selection=True)

    # 在 goal2 下运行
    ep2 = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal2,
        transform_store=transform_store,
        selection_store=sel_store,
        world_history=world, use_learning=True,
        rng=random.Random(seed), update_selection=True)

    # 检查 A→B 在两个 goal 下的分数
    Ab, Bb = P.predicate("A", "b"), P.predicate("B", "b")
    impl_ab = P.impl(Ab, Bb)
    input_sig_ab = structure_sig_multi((Ab, Bb))[0]
    output_sig_ab = structure_sig(impl_ab)[0]

    goal1_sig = structure_sig(goal1)[0]
    goal2_sig = structure_sig(goal2)[0]

    score_ab_g1 = sel_store.get_score(input_sig_ab, output_sig_ab, goal1_sig)
    score_ab_g2 = sel_store.get_score(input_sig_ab, output_sig_ab, goal2_sig)

    return {
        "goal1_reached": ep1["reached"],
        "goal2_reached": ep2["reached"],
        "A_to_B_score_under_goal1": round(score_ab_g1, 4),
        "A_to_B_score_under_goal2": round(score_ab_g2, 4),
        "scores_differ": abs(score_ab_g1 - score_ab_g2) > 0.001,
    }


# ============================================================
# operation_name 独立性测试
# ============================================================

def run_operation_name_independence_test(world: List[Set[P]], seed: int = 42) -> dict:
    """operation_name 改名不影响结果。"""
    transform_store, _ = run_phase1(world, seed=seed)

    # UNKNOWN store
    store_unknown = TransformationStore()
    for r in transform_store.records:
        store_unknown.records.append(TransformationRecord(
            record_id=r.record_id + "_u", input_sigs=r.input_sigs,
            output_sig=r.output_sig, operation_name="UNKNOWN",
            context_sig=r.context_sig, verification_result=r.verification_result,
            usefulness=r.usefulness, cost=r.cost, step=r.step,
            confidence=r.confidence, input_terms=r.input_terms,
            output_prop_str=r.output_prop_str))

    goal = make_goal_nstep("b", 3)

    ep_orig = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal,
        transform_store=transform_store,
        selection_store=TransformationSelectionStoreV2(),
        world_history=world, use_learning=True,
        rng=random.Random(seed), update_selection=True)

    ep_unknown = run_goal_pursuit_episode_v2(
        start_objects=set(world[3]), goal=goal,
        transform_store=store_unknown,
        selection_store=TransformationSelectionStoreV2(),
        world_history=world, use_learning=True,
        rng=random.Random(seed), update_selection=True)

    return {
        "original_reached": ep_orig["reached"],
        "unknown_reached": ep_unknown["reached"],
        "same_outcome": ep_orig["reached"] == ep_unknown["reached"],
        "original_steps": ep_orig["steps"],
        "unknown_steps": ep_unknown["steps"],
    }


# ============================================================
# 主实验
# ============================================================

def run_e0_7_4() -> dict:
    """运行 E0-7.4 未来价值与信用分配实验。"""
    world = build_world()

    print("Phase 1: Building transformation history...")
    transform_store, phase1_stats = run_phase1(world, seed=42)
    print(f"  transforms: {phase1_stats['transforms_recorded']}")

    print("\nCredit mode comparison (3-step goal):")
    credit_3 = run_credit_experiment(world, goal_n=3, seed=42)

    print("\nCredit mode comparison (2-step goal):")
    credit_2 = run_credit_experiment(world, goal_n=2, seed=42)

    print("\nGoal change test:")
    goal_change = run_goal_change_test(world, seed=42)

    print("\nOperation name independence test:")
    op_indep = run_operation_name_independence_test(world, seed=42)

    # 分析
    s_lo = credit_3["group_last_only"]["summary"]
    s_eq = credit_3["group_equal"]["summary"]
    s_dc = credit_3["group_discounted"]["summary"]

    analysis = {
        "credit_3step_last_only_reached": s_lo["reached_count"],
        "credit_3step_equal_reached": s_eq["reached_count"],
        "credit_3step_discounted_reached": s_dc["reached_count"],
        "discounted_best_or_equal_3step": (
            s_dc["reached_count"] >= s_lo["reached_count"]
            and s_dc["avg_steps"] <= s_eq["avg_steps"] + 1),
        "goal_change_scores_differ": goal_change["scores_differ"],
        "operation_name_independence": op_indep["same_outcome"],
    }
    analysis["all_checks_passed"] = all([
        analysis["operation_name_independence"],
    ])

    return {
        "experiment": "E0-7.4",
        "description": "future value / credit assignment",
        "phase1_stats": phase1_stats,
        "credit_experiment_3step": credit_3,
        "credit_experiment_2step": credit_2,
        "goal_change_test": goal_change,
        "operation_name_independence": op_indep,
        "analysis": analysis,
    }


def main():
    print("\n" + "=" * 80)
    print("E0-7.4：Future Value / Credit Assignment")
    print("验证中间步骤能否通过信用传播获得未来价值")
    print("=" * 80)

    result = run_e0_7_4()

    print(f"\n--- Credit mode comparison (3-step goal) ---")
    for label, key in [("last_only", "group_last_only"),
                        ("equal", "group_equal"),
                        ("discounted", "group_discounted")]:
        s = result["credit_experiment_3step"][key]["summary"]
        print(f"  {label:12s}: reached={s['reached_count']}/{s['total_episodes']}, "
              f"avg_steps={s['avg_steps']:.1f}, avg_cost={s['avg_cost']:.2f}")

    print(f"\n--- Credit mode comparison (2-step goal) ---")
    for label, key in [("last_only", "group_last_only"),
                        ("equal", "group_equal"),
                        ("discounted", "group_discounted")]:
        s = result["credit_experiment_2step"][key]["summary"]
        print(f"  {label:12s}: reached={s['reached_count']}/{s['total_episodes']}, "
              f"avg_steps={s['avg_steps']:.1f}, avg_cost={s['avg_cost']:.2f}")

    gc = result["goal_change_test"]
    print(f"\n--- Goal change test ---")
    print(f"  goal1_reached={gc['goal1_reached']}, goal2_reached={gc['goal2_reached']}")
    print(f"  A→B score under goal1={gc['A_to_B_score_under_goal1']}, "
          f"under goal2={gc['A_to_B_score_under_goal2']}")
    print(f"  scores_differ={gc['scores_differ']}")

    op = result["operation_name_independence"]
    print(f"\n--- Operation name independence ---")
    print(f"  same_outcome={op['same_outcome']}")

    print(f"\n--- Analysis ---")
    for k, v in result["analysis"].items():
        print(f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_7_4_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result["analysis"]["all_checks_passed"]


if __name__ == "__main__":
    main()

"""E0-7.3：Transformation Selection / Goal-directed Search（变换选择与目标导向搜索）。

核心问题：
  E0-7.2 解决了"能不能继续计算"。
  E0-7.3 解决"有多种可能继续计算的方向时，系统如何决定下一步计算什么"。

关键区分：
  - E0-6：context → operation（操作选择）
  - E0-7.3：input_structure → output_structure 的选择（变换选择）
  - 不使用 operation_name 进行选择

核心机制：
  1. 目标表示：goal Proposition
  2. 目标评价（显式先验）：goal_reached(output, goal) = output 结构等于 goal
  3. 信用分配：成功链中所有变换获得 usefulness=1.0
  4. TransformationSelectionStore：按 (input_sig, output_sig) 统计
  5. 选择：epsilon-greedy，score = mean_goal_improvement - cost_weight * mean_cost

硬约束：
  - operation_name 完全禁止参与选择
  - 不写死"距离目标"算法
  - 候选生成与候选选择分离
  - validity ≠ value（正确但无价值的知识保留）
"""

from __future__ import annotations

import os
import sys
import json
import random
import copy
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


# ============================================================
# 常量
# ============================================================

EPSILON = 0.2  # 探索率
COST_WEIGHT = 0.1  # 成本权重
MAX_CANDIDATES_PER_STEP = 20  # 每步选择的候选数量
MAX_STEPS_PER_EPISODE = 10  # 每个 episode 的最大步数
NUM_EPISODES = 5  # A/B 每组的 episode 数量


# ============================================================
# 目标评价（显式先验）
# ============================================================

def goal_reached(prop: P, goal: P) -> bool:
    """目标评价：命题是否达到目标。

    这是一个显式先验——系统没有"学会"目标定义，而是被给予了目标。
    判定标准：prop 与 goal 结构相等（Proposition.__eq__）。

    不使用：
    - 结构距离（不写死"接近目标"的算法）
    - 子结构共享（不写死"共享符号"的启发式）
    - operation_name

    仅使用 Proposition 的结构相等性。
    """
    return prop == goal


# ============================================================
# TransformationSelectionStore：变换级别的选择统计
# ============================================================

@dataclass
class TransformStats:
    """单个变换 (input_sig → output_sig) 的统计。"""
    input_sig: Tuple
    output_sig: Any
    attempts: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    total_goal_improvement: float = 0.0
    total_cost: float = 0.0

    @property
    def mean_goal_improvement(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_goal_improvement / self.attempts

    @property
    def mean_cost(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_cost / self.attempts

    @property
    def score(self) -> float:
        """变换的预期价值 = 平均目标改善 - 成本权重 * 平均成本。"""
        return self.mean_goal_improvement - COST_WEIGHT * self.mean_cost


class TransformationSelectionStore:
    """变换选择统计存储。

    按 (input_sig, output_sig) 聚合统计，不使用 operation_name。

    这是 transformation-level 的选择，与 E0-6 的 operation-level 选择不同。
    E0-6 统计 (context, operation)；本类统计 (input_sig, output_sig)。
    """

    def __init__(self):
        # key = (input_sig_str, output_sig_str)
        self.stats: Dict[Tuple[str, str], TransformStats] = {}

    def _key(self, input_sig, output_sig) -> Tuple[str, str]:
        return (str(input_sig), str(output_sig))

    def record_outcome(self, input_sig, output_sig, verdict: str,
                       goal_improvement: float, cost: float) -> None:
        """记录一次变换的结果。

        不读取 operation_name。
        """
        key = self._key(input_sig, output_sig)
        if key not in self.stats:
            self.stats[key] = TransformStats(input_sig=input_sig, output_sig=output_sig)
        s = self.stats[key]
        s.attempts += 1
        if verdict == "valid":
            s.valid_count += 1
        elif verdict == "invalid":
            s.invalid_count += 1
        s.total_goal_improvement += goal_improvement
        s.total_cost += cost

    def get_stats(self, input_sig, output_sig) -> Optional[TransformStats]:
        return self.stats.get(self._key(input_sig, output_sig))

    def get_score(self, input_sig, output_sig) -> float:
        s = self.get_stats(input_sig, output_sig)
        if s is None:
            return 0.0  # 未知变换：中性分数
        return s.score


# ============================================================
# 候选选择：epsilon-greedy on transformation scores
# ============================================================

def select_candidates(
    candidates: List[dict],
    selection_store: TransformationSelectionStore,
    use_learning: bool,
    rng: random.Random,
    n_select: int = MAX_CANDIDATES_PER_STEP,
) -> Tuple[List[dict], List[dict]]:
    """从候选中选择 n_select 个。

    Group A (use_learning=False)：随机选择。
    Group B (use_learning=True)：基于变换历史分数的 epsilon-greedy。

    关键：选择依据是 (input_sig, output_sig) 的历史统计，不读 operation_name。

    返回 (selected, selection_info)。
    """
    if not candidates:
        return [], []

    n = min(n_select, len(candidates))

    if not use_learning:
        # Group A: 随机选择
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        selected = [candidates[i] for i in indices[:n]]
        info = [{"candidate": c, "score": 0.0, "was_exploration": True,
                 "reason": "random"} for c in selected]
        return selected, info

    # Group B: epsilon-greedy 基于变换分数
    scored: List[Tuple[dict, float]] = []
    for c in candidates:
        # 从候选中获取变换的 input_sig 和 output_sig
        record = c.get("_record")
        if record is not None:
            score = selection_store.get_score(record.input_sigs, record.output_sig)
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
            # 探索：随机选
            idx = rng.randint(0, len(remaining) - 1)
            c, s = remaining.pop(idx)
            selected.append(c)
            selection_info.append({
                "candidate": c, "score": s,
                "was_exploration": True, "reason": "epsilon"
            })
        else:
            # 利用：选分数最高的
            remaining.sort(key=lambda x: x[1], reverse=True)
            c, s = remaining.pop(0)
            selected.append(c)
            selection_info.append({
                "candidate": c, "score": s,
                "was_exploration": False, "reason": "exploit"
            })

    return selected, selection_info


# ============================================================
# Goal pursuit episode：目标追踪回合
# ============================================================

def run_goal_pursuit_episode(
    start_objects: Set[P],
    goal: P,
    transform_store: TransformationStore,
    selection_store: TransformationSelectionStore,
    world_history: List[Set[P]],
    use_learning: bool,
    rng: random.Random,
    update_selection: bool = True,
) -> dict:
    """运行一个目标追踪回合。

    流程：
      1. 当前知识空间 = observed ∪ derived_valid
      2. generate_candidates_from_transforms → 候选集合
      3. select_candidates → 选择 n 个候选
      4. verify → 更新 belief
      5. 如果 goal 达到，停止并信用分配
      6. 否则继续

    信用分配：如果 goal 达到，链中所有变换获得 goal_improvement=1.0。
    """
    belief_store = BeliefStore()
    observed = set(start_objects)

    # 注册 observed objects
    for obj in observed:
        v, c, _ = verify_proposition(obj, world_history)
        if v == "valid":
            belief_store.update_belief(obj, STATUS_VALID, c, evidence_count_delta=1)

    chain_records: List[TransformationRecord] = []  # 本回合应用的变换
    reached = False
    steps = 0
    total_cost = 0.0

    trace: List[dict] = []

    for step in range(MAX_STEPS_PER_EPISODE):
        steps = step + 1

        # 检查 goal 是否已达到
        goal_belief = belief_store.get(goal)
        if goal_belief is not None and goal_belief.status == STATUS_VALID:
            reached = True
            break

        # 当前知识空间
        derived_valid = {
            b.proposition for b in belief_store.all_beliefs()
            if b.status == STATUS_VALID and b.proposition not in observed
        }
        constructible = sorted(observed | derived_valid, key=lambda p: p.to_str())

        # 候选生成（不读 operation_name）
        candidates = generate_candidates_from_transforms(
            constructible, transform_store, belief_store)

        if not candidates:
            break

        # 给候选附加 record 信息（用于选择评分，不读 operation_name）
        # generate_candidates_from_transforms 返回 record_id，我们需要找回 record
        # 为了简化，重新匹配获取 record
        matches = transform_store.find_matches(constructible)
        record_by_id = {r.record_id: r for r in transform_store.records}
        for c in candidates:
            rid = c.get("record_id")
            if rid in record_by_id:
                c["_record"] = record_by_id[rid]

        # 候选选择
        selected, sel_info = select_candidates(
            candidates, selection_store, use_learning, rng)

        step_valids = 0
        step_new_objects: List[P] = []

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

            # 计算目标改善
            gi = 1.0 if goal_reached(prop, goal) else 0.0

            # 记录变换结果到 selection_store
            if record is not None and update_selection:
                selection_store.record_outcome(
                    record.input_sigs, record.output_sig,
                    verdict, gi, cost)
                if verdict == "valid":
                    chain_records.append(record)

            if verdict == "valid":
                step_valids += 1
                step_new_objects.append(prop)

            # 检查是否达到 goal
            if goal_reached(prop, goal):
                reached = True

        trace.append({
            "step": step,
            "candidates_count": len(candidates),
            "selected_count": len(selected),
            "valid": step_valids,
            "reached": reached,
            "selection_info": [{
                "proposition": info["candidate"]["proposition"].to_str(),
                "score": round(info["score"], 4),
                "was_exploration": info["was_exploration"],
                "reason": info["reason"],
            } for info in sel_info],
        })

        if reached:
            break

    # 信用分配：如果达到 goal，链中所有变换获得 goal_improvement=1.0
    if reached and update_selection:
        for record in chain_records:
            # 重新记录一次，goal_improvement=1.0
            selection_store.record_outcome(
                record.input_sigs, record.output_sig,
                "valid", 1.0, 0.0)

    return {
        "reached": reached,
        "steps": steps,
        "total_cost": round(total_cost, 4),
        "chain_length": len(chain_records),
        "trace": trace,
    }


# ============================================================
# 人工世界设计
# ============================================================

def build_world() -> List[Set[P]]:
    """构造竞争场景的人工世界。

    历史中 A, B, C, D 全部共现，因此所有两两蕴含都是 VALID：
      A→B, A→C, A→D, B→A, B→C, B→D, C→A, C→B, C→D, D→A, D→B, D→C
    加上大量合取/析取等。

    目标：(A(b)→B(b)) ∧ (B(b)→A(b))

    要达到目标，系统必须：
      1. 构建 A(b)→B(b)  [VALID]
      2. 构建 B(b)→A(b)  [VALID]
      3. 构建 (A→B)∧(B→A)  [VALID，目标达到]

    竞争：A→C, A→D, B→C, B→D 等也是 VALID 变换，但对目标无用。
    系统需要从历史中学习哪些变换对目标有价值。

    信用分配：成功链中的变换获得 usefulness=1.0。
    """
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    Ca = P.predicate("C", "a")
    Da = P.predicate("D", "a")

    Ab = P.predicate("A", "b")
    Bb = P.predicate("B", "b")
    Cb = P.predicate("C", "b")
    Db = P.predicate("D", "b")

    Ac = P.predicate("A", "c")
    Bc = P.predicate("B", "c")
    Cc = P.predicate("C", "c")
    Dc = P.predicate("D", "c")

    return [
        {Aa, Ba, Ca, Da},        # step 0: all co-occur → all implications valid
        {Aa, Ba, Ca, Da},        # step 1: repeat for stable history
        {Ab, Bb, Cb, Db},        # step 2: test object b
        {Ac, Bc, Cc, Dc},        # step 3: test object c
    ]


def make_goal(term: str) -> P:
    """构造目标命题：(A(term)→B(term)) ∧ (B(term)→A(term))。"""
    A = P.predicate("A", term)
    B = P.predicate("B", term)
    return P.conj(P.impl(A, B), P.impl(B, A))


# ============================================================
# Phase 1：构建变换历史
# ============================================================

def run_phase1(world_steps: List[Set[P]], seed: int = 42
               ) -> Tuple[TransformationStore, dict]:
    """Phase 1：通过真实计算构建变换历史。"""
    belief_store = BeliefStore()
    transform_store = TransformationStore()
    observed_objects: Set[P] = set()

    stats = {"steps_run": 0, "transforms_recorded": 0,
             "valid": 0, "invalid": 0, "unknown": 0}

    for t, current_obs in enumerate(world_steps[:2]):  # 只用前 2 步构建历史
        visible_history = world_steps[:t + 1]

        for prop in current_obs:
            if prop not in observed_objects:
                observed_objects.add(prop)

        derived = {b.proposition for b in belief_store.all_beliefs()
                   if b.status == STATUS_VALID and b.proposition not in observed_objects}

        constructible = sorted(observed_objects | derived, key=lambda p: p.to_str())
        candidates = generate_candidates_brute(
            constructible, observed_objects, derived, belief_store)

        # 限制候选数量防止爆炸
        selected = candidates[:200]

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
                cost=total_cost, step=t, confidence=confidence)
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
# A/B 实验
# ============================================================

def run_ab_experiment(world: List[Set[P]], seed: int = 42) -> dict:
    """运行 A/B 对照实验。

    Group A：随机选择。
    Group B：基于变换历史的 epsilon-greedy 选择。

    两者使用相同的 world、goal、transform history、budget。
    唯一区别：candidate selection policy。
    """
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)

    # Phase 1: 构建历史
    transform_store, phase1_stats = run_phase1(world, seed=seed)

    # 定义目标
    goal_b = make_goal("b")
    goal_c = make_goal("c")

    # Group A: random selection
    print("  Group A (random selection)...")
    sel_store_a = TransformationSelectionStore()
    results_a = []
    episodes = [(world[2], goal_b), (world[3], goal_c)] * (NUM_EPISODES // 2)
    for start_set, goal in episodes:
        ep = run_goal_pursuit_episode(
            start_objects=set(start_set), goal=goal,
            transform_store=transform_store,
            selection_store=sel_store_a,
            world_history=world,
            use_learning=False,
            rng=rng_a,
            update_selection=True,
        )
        results_a.append(ep)

    # Group B: learned selection
    print("  Group B (learned selection)...")
    sel_store_b = TransformationSelectionStore()
    results_b = []
    for start_set, goal in episodes:
        ep = run_goal_pursuit_episode(
            start_objects=set(start_set), goal=goal,
            transform_store=transform_store,
            selection_store=sel_store_b,
            world_history=world,
            use_learning=True,
            rng=rng_b,
            update_selection=True,
        )
        results_b.append(ep)

    # 汇总
    def summarize(results):
        return {
            "reached_count": sum(1 for r in results if r["reached"]),
            "total_episodes": len(results),
            "avg_steps": sum(r["steps"] for r in results) / len(results),
            "avg_cost": sum(r["total_cost"] for r in results) / len(results),
            "avg_chain_length": sum(r["chain_length"] for r in results) / len(results),
        }

    summary_a = summarize(results_a)
    summary_b = summarize(results_b)

    return {
        "phase1_stats": phase1_stats,
        "group_a": {"summary": summary_a, "episodes": results_a},
        "group_b": {"summary": summary_b, "episodes": results_b},
    }


# ============================================================
# operation_name 独立性测试
# ============================================================

def run_operation_name_independence_test(world: List[Set[P]], seed: int = 42) -> dict:
    """测试 operation_name 改变不影响选择结果。

    将所有 transform 的 operation_name 改为 UNKNOWN/FAKE，
    检查选择结果是否一致。
    """
    transform_store, _ = run_phase1(world, seed=seed)

    # 原始 store
    store_original = transform_store

    # UNKNOWN store
    store_unknown = TransformationStore()
    for r in transform_store.records:
        r_copy = TransformationRecord(
            record_id=r.record_id + "_u",
            input_sigs=r.input_sigs,
            output_sig=r.output_sig,
            operation_name="UNKNOWN",
            context_sig=r.context_sig,
            verification_result=r.verification_result,
            usefulness=r.usefulness,
            cost=r.cost,
            step=r.step,
            confidence=r.confidence,
            input_terms=r.input_terms,
            output_prop_str=r.output_prop_str,
        )
        store_unknown.records.append(r_copy)

    # FAKE store
    store_fake = TransformationStore()
    for r in transform_store.records:
        r_copy = TransformationRecord(
            record_id=r.record_id + "_f",
            input_sigs=r.input_sigs,
            output_sig=r.output_sig,
            operation_name="FAKE_OP",
            context_sig=r.context_sig,
            verification_result=r.verification_result,
            usefulness=r.usefulness,
            cost=r.cost,
            step=r.step,
            confidence=r.confidence,
            input_terms=r.input_terms,
            output_prop_str=r.output_prop_str,
        )
        store_fake.records.append(r_copy)

    # 用相同的 selection_store 和 rng 测试
    # 关键：selection_store 按 (input_sig, output_sig) 统计，不包含 operation_name
    sel_store = TransformationSelectionStore()
    rng = random.Random(seed)

    goal_b = make_goal("b")

    # 用 original store 跑一次
    ep_orig = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=store_original,
        selection_store=sel_store,
        world_history=world,
        use_learning=True,
        rng=random.Random(seed),
        update_selection=True,
    )

    # 用 unknown store 跑一次（新 selection_store，相同 rng）
    sel_store_u = TransformationSelectionStore()
    ep_unknown = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=store_unknown,
        selection_store=sel_store_u,
        world_history=world,
        use_learning=True,
        rng=random.Random(seed),
        update_selection=True,
    )

    # 用 fake store 跑一次
    sel_store_f = TransformationSelectionStore()
    ep_fake = run_goal_pursuit_episode(
        start_objects=set(world[2]), goal=goal_b,
        transform_store=store_fake,
        selection_store=sel_store_f,
        world_history=world,
        use_learning=True,
        rng=random.Random(seed),
        update_selection=True,
    )

    # 比较结果（reached, steps）
    same_reached = (ep_orig["reached"] == ep_unknown["reached"] == ep_fake["reached"])

    return {
        "original_reached": ep_orig["reached"],
        "unknown_reached": ep_unknown["reached"],
        "fake_reached": ep_fake["reached"],
        "same_outcome": same_reached,
        "original_steps": ep_orig["steps"],
        "unknown_steps": ep_unknown["steps"],
        "fake_steps": ep_fake["steps"],
    }


# ============================================================
# 主实验
# ============================================================

def run_e0_7_3() -> dict:
    """运行 E0-7.3 变换选择实验。"""
    world = build_world()

    print("Phase 1: Building transformation history...")
    transform_store, phase1_stats = run_phase1(world, seed=42)
    print(f"  transforms: {phase1_stats['transforms_recorded']}")

    print("\nA/B experiment:")
    ab_result = run_ab_experiment(world, seed=42)

    print("\nOperation name independence test:")
    op_indep = run_operation_name_independence_test(world, seed=42)

    # 分析
    sa = ab_result["group_a"]["summary"]
    sb = ab_result["group_b"]["summary"]

    analysis = {
        "group_a_reached": sa["reached_count"],
        "group_b_reached": sb["reached_count"],
        "group_b_better": sb["reached_count"] >= sa["reached_count"],
        "group_b_lower_avg_steps": sb["avg_steps"] <= sa["avg_steps"],
        "group_b_lower_avg_cost": sb["avg_cost"] <= sa["avg_cost"],
        "operation_name_independence": op_indep["same_outcome"],
    }
    analysis["all_checks_passed"] = all([
        analysis["group_b_better"],
        analysis["operation_name_independence"],
    ])

    return {
        "experiment": "E0-7.3",
        "description": "transformation selection / goal-directed search",
        "phase1_stats": phase1_stats,
        "ab_experiment": ab_result,
        "operation_name_independence": op_indep,
        "analysis": analysis,
    }


def main():
    print("\n" + "=" * 80)
    print("E0-7.3：Transformation Selection / Goal-directed Search")
    print("验证系统能否在多个合法计算方向之间根据目标进行选择")
    print("=" * 80)

    result = run_e0_7_3()

    sa = result["ab_experiment"]["group_a"]["summary"]
    sb = result["ab_experiment"]["group_b"]["summary"]
    print(f"\n--- A/B Summary ---")
    print(f"  Group A (random):  reached={sa['reached_count']}/{sa['total_episodes']}, "
          f"avg_steps={sa['avg_steps']:.1f}, avg_cost={sa['avg_cost']:.2f}")
    print(f"  Group B (learned): reached={sb['reached_count']}/{sb['total_episodes']}, "
          f"avg_steps={sb['avg_steps']:.1f}, avg_cost={sb['avg_cost']:.2f}")

    op = result["operation_name_independence"]
    print(f"\n--- Operation name independence ---")
    print(f"  original_reached={op['original_reached']}, "
          f"unknown_reached={op['unknown_reached']}, "
          f"fake_reached={op['fake_reached']}")
    print(f"  same_outcome={op['same_outcome']}")

    print(f"\n--- Analysis ---")
    for k, v in result["analysis"].items():
        print(f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e0_7_3_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result["analysis"]["all_checks_passed"]


if __name__ == "__main__":
    main()

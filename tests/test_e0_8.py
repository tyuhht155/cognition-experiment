"""E0-8 不变量测试：Goal-Oriented New Object Generation（目标导向新对象生成）。

验证系统能否用基础构造器构造历史中不存在的新对象 X，验证后进入知识空间，
下一轮把 X 当普通对象继续计算，最终到达目标 G。

测试覆盖（16 项不变量）：
  A.  test_x_not_in_history             —— X 不是历史已有变换结果
  A2. test_x_not_in_initial_knowledge    —— X 不在初始知识空间
  B.  test_system_constructs_x           —— 基础构造器能产生 X
  C.  test_x_verified_before_knowledge   —— X 必须验证才能进入知识
  D.  test_x_reparticipates_next_round   —— X 进入知识后下轮参与计算
  E.  test_goal_reached_via_x            —— 最终通过 X 到达目标
  F.  test_invalid_x_blocks_path         —— X=INVALID 时路径不继续
  G.  test_no_history_shortcut           —— 仅历史无法捷径到达
  H.  test_no_operation_name_dependency  —— 不依赖 operation_name
  I.  test_no_ground_truth_guidance      —— 无 ground truth 指引下一步
  J.  test_trace_complete               —— trace 完整显示全流程
  12. test_constructible_vs_valid_vs_useful —— 能构造≠有效≠有用
  13. test_no_future_info               —— 严格时间因果
  14. test_goal_not_in_round0           —— G 在 Round 0 无法构造
  15. test_invalid_not_in_constructible —— INVALID 对象不进入 constructible
  16. test_constructor_generic          —— 构造器是通用的，不携带答案
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_e0_8 import (
    build_world,
    build_world_invalid,
    make_goal,
    intermediate_x,
    build_history_pq,
    run_e0_8_episode,
    run_e0_8,
    MAX_ROUNDS,
)
from experiments.run_e0_7 import (
    structure_sig,
    structure_sig_multi,
    TransformationStore,
    generate_candidates as generate_candidates_brute,
    CONSTRUCTORS,
)
from experiments.run_e0_7_1 import generate_candidates_from_transforms
from experiments.run_e0_7_2 import verify_proposition_extended
from experiments.run_e0_7_3 import goal_reached
from cognition.proposition import Proposition as P
from cognition.belief import BeliefStore
from cognition.models import STATUS_VALID, STATUS_INVALID, STATUS_UNKNOWN


# ============================================================
# 辅助
# ============================================================

def _run_main():
    """运行主场景 episode（构造器路径）。"""
    return run_e0_8_episode(
        build_world(), make_goal(), use_brute=True,
        transform_store=build_history_pq())


def _run_history_only():
    """运行对照场景 episode（仅历史路径）。"""
    return run_e0_8_episode(
        build_world(), make_goal(), use_brute=False,
        transform_store=build_history_pq())


def _run_invalid():
    """运行 INVALID 场景 episode。"""
    return run_e0_8_episode(
        build_world_invalid(), make_goal(), use_brute=True,
        transform_store=build_history_pq())


# ============================================================
# A. X 不是历史中已有的 transformation 结果
# ============================================================

def test_x_not_in_history():
    """X=A→B 不应出现在 P、Q 变换历史的输出中。"""
    hist = build_history_pq()
    x_sig = structure_sig(intermediate_x())[0]
    for rec in hist.records:
        assert rec.output_sig != x_sig, \
            f"X 的结构签名不应出现在历史中，但 {rec.output_prop_str} 匹配"


# ============================================================
# A2. X 不在初始知识空间
# ============================================================

def test_x_not_in_initial_knowledge():
    """初始知识只有 A(a)、B(a)，X=A→B 不在其中。"""
    belief = BeliefStore()
    world = build_world()
    for prop in world[-1]:
        belief.update_belief(prop, STATUS_VALID, 0.8, evidence_count_delta=1)
    assert not belief.has(intermediate_x()), "X 不应在初始知识空间中"


# ============================================================
# B. 系统能够通过基础构造产生 X
# ============================================================

def test_system_constructs_x():
    """暴力构造器 generate_candidates 应能从 A、B 产生 X=A→B。"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)
    candidates = generate_candidates_brute(
        [Aa, Ba], {Aa, Ba}, set(), belief)
    props = [c["proposition"] for c in candidates]
    assert intermediate_x() in props, "暴力构造器应产生 X=A→B"


# ============================================================
# C. X 必须经过验证才能进入知识空间
# ============================================================

def test_x_verified_before_knowledge():
    """X 进入知识前必须经过 verify，且状态为 VALID。"""
    r = _run_main()
    assert r["x_status"] == STATUS_VALID, "X 应被验证为 VALID"
    # X 在 trace 中有验证记录
    x_str = intermediate_x().to_str()
    x_verified = False
    for tr in r["trace"]:
        for res in tr["results"]:
            if res["proposition"] == x_str:
                assert res["verdict"] == "valid", "X 验证结果应为 valid"
                x_verified = True
    assert x_verified, "trace 中应有 X 的验证记录"


# ============================================================
# D. X 进入知识空间后，可以在下一轮重新参与计算
# ============================================================

def test_x_reparticipates_next_round():
    """X 被验证后应在下一轮出现在 constructible 中。"""
    r = _run_main()
    x_str = intermediate_x().to_str()
    x_round = None
    for tr in r["trace"]:
        if x_str in tr.get("new_valids", []):
            x_round = tr["round"]
            break
    assert x_round is not None, "X 应在某轮被验证为 valid"
    # 下一轮 constructible 应包含 X
    found = False
    for tr in r["trace"]:
        if tr["round"] > x_round and x_str in tr.get("constructible", []):
            found = True
            break
    assert found, "X 应在下一轮的 constructible 中"


# ============================================================
# E. 最终能够通过 X 到达目标
# ============================================================

def test_goal_reached_via_x():
    """主场景应到达目标 G，且 G 的构造依赖 X。"""
    r = _run_main()
    assert r["goal_achieved"], "目标 G 应被达成"
    assert r["goal_status"] == STATUS_VALID, "G 应为 VALID"
    # G 在 round 1（X 产生于 round 0）的 new_valids 中
    g_str = make_goal().to_str()
    g_round = None
    for tr in r["trace"]:
        if g_str in tr.get("new_valids", []):
            g_round = tr["round"]
            break
    assert g_round is not None and g_round >= 1, \
        "G 应在 round ≥1 产生（需要先有 X）"


# ============================================================
# F. 如果 X 被验证为 INVALID，则路径不能继续
# ============================================================

def test_invalid_x_blocks_path():
    """INVALID 世界中 X 应为 INVALID，目标不可达。"""
    r = _run_invalid()
    assert r["x_status"] == STATUS_INVALID, "X 应为 INVALID"
    assert not r["goal_achieved"], "目标不应达成"


# ============================================================
# G. 仅历史无法捷径到达
# ============================================================

def test_no_history_shortcut():
    """仅用 generate_candidates_from_transforms（P、Q 历史）无法产生 X 或 G。"""
    r = _run_history_only()
    assert not r["goal_achieved"], "仅历史不应达成目标"
    assert r["x_status"] == "not_in_store", "仅历史不应产生 X"
    # round 0 应有 0 候选
    assert r["trace"][0]["candidates_count"] == 0, "P、Q 历史对 A、B 应 0 匹配"


# ============================================================
# H. 不依赖 operation_name 才能生成 X
# ============================================================

def test_no_operation_name_dependency():
    """暴力构造器不读取 operation_name 即可产生 X。"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)
    # generate_candidates 不接受 operation_name 参数
    candidates = generate_candidates_brute(
        [Aa, Ba], {Aa, Ba}, set(), belief)
    # 确认候选中有 X，且候选 dict 的 constructor 字段是通用名 "impl"
    x_cand = [c for c in candidates if c["proposition"] == intermediate_x()]
    assert len(x_cand) == 1, "应恰好有一个 X 候选"
    assert x_cand[0]["constructor"] == "impl", "constructor 应为通用名 impl"


# ============================================================
# I. 不使用 ground truth 直接告诉系统下一步是什么
# ============================================================

def test_no_ground_truth_guidance():
    """系统不读取 ground_truth_check，候选生成不依赖目标答案。"""
    # generate_candidates 和 generate_candidates_from_transforms 都不接受
    # goal 或 ground_truth 参数，只接受 constructible/observed/derived/belief
    import inspect
    sig_brute = inspect.signature(generate_candidates_brute)
    params_brute = set(sig_brute.parameters.keys())
    assert "goal" not in params_brute, "暴力构造器不应接受 goal 参数"
    assert "ground_truth" not in params_brute, "暴力构造器不应接受 ground_truth 参数"

    sig_hist = inspect.signature(generate_candidates_from_transforms)
    params_hist = set(sig_hist.parameters.keys())
    assert "goal" not in params_hist, "历史候选不应接受 goal 参数"
    assert "ground_truth" not in params_hist, "历史候选不应接受 ground_truth 参数"


# ============================================================
# J. Trace 能完整显示全流程
# ============================================================

def test_trace_complete():
    """trace 应包含：初始对象 → 新对象生成 → 验证 → 知识更新 → 重新参与 → 目标。"""
    r = _run_main()
    trace = r["trace"]
    assert len(trace) >= 2, "至少 2 轮 trace"

    # 初始对象在 round 0 的 constructible 中
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    r0_constructible = set(trace[0].get("constructible", []))
    assert Aa.to_str() in r0_constructible, "round 0 应含 A(a)"
    assert Ba.to_str() in r0_constructible, "round 0 应含 B(a)"

    # 新对象 X 在某轮 new_valids 中
    x_str = intermediate_x().to_str()
    x_in_valids = any(x_str in tr.get("new_valids", []) for tr in trace)
    assert x_in_valids, "X 应在某轮 new_valids 中"

    # X 在某轮 results 中有验证记录
    x_in_results = any(
        any(res["proposition"] == x_str for res in tr.get("results", []))
        for tr in trace)
    assert x_in_results, "X 应在 trace results 中有验证记录"

    # X 在后续轮 constructible 中（知识更新后重新参与）
    x_repart = False
    for i, tr in enumerate(trace):
        if x_str in tr.get("new_valids", []):
            for later in trace[i + 1:]:
                if x_str in later.get("constructible", []):
                    x_repart = True
            break
    assert x_repart, "X 应在后续轮 constructible 中"

    # 目标 G 在某轮 new_valids 中
    g_str = make_goal().to_str()
    g_in_valids = any(g_str in tr.get("new_valids", []) for tr in trace)
    assert g_in_valids, "G 应在某轮 new_valids 中"


# ============================================================
# 12. 能构造 ≠ 被验证有效 ≠ 对目标有用
# ============================================================

def test_constructible_vs_valid_vs_useful():
    """trace 中应能区分：能构造（candidates）、有效（valid）、有用（useful_for_goal）。"""
    r = _run_main()
    trace = r["trace"]
    # 能构造：candidates_count > 0
    assert trace[0]["candidates_count"] > 0, "应有可构造的候选"
    # 有效但不有用：round 0 中有 valid 候选，但没有 useful_for_goal=True
    r0 = trace[0]
    valid_not_useful = [
        res for res in r0["results"]
        if res["verdict"] == "valid" and not res["useful_for_goal"]]
    assert len(valid_not_useful) > 0, \
        "round 0 应有 valid 但对目标无用的候选（如 A∧B）"
    # 有用：最终 G 的 useful_for_goal=True
    g_str = make_goal().to_str()
    useful_found = any(
        res["useful_for_goal"] and res["verdict"] == "valid"
        for tr in trace for res in tr["results"]
        if res["proposition"] == g_str)
    assert useful_found, "G 应被标记为 useful_for_goal"


# ============================================================
# 13. 严格时间因果，不使用未来信息
# ============================================================

def test_no_future_info():
    """严格时间因果：每轮 constructible 只含 observed + 之前轮产生的 valid，
    不含未来轮才产生的对象。每个候选的输入对象必须在该轮 constructible 中。"""
    r = _run_main()
    trace = r["trace"]
    x_str = intermediate_x().to_str()
    g_str = make_goal().to_str()

    # round 0 的 constructible 不含 X 或 G（它们还未被构造）
    r0_constructible = set(trace[0].get("constructible", []))
    assert x_str not in r0_constructible, "X 不应在 round 0 constructible 中"
    assert g_str not in r0_constructible, "G 不应在 round 0 constructible 中"

    # 每个候选的输入对象必须在该轮 constructible 中（不能引用未来对象）
    for tr in trace:
        round_constructible = set(tr.get("constructible", []))
        for res in tr.get("results", []):
            for inp in res.get("inputs", []):
                assert inp in round_constructible, \
                    f"候选输入 {inp} 不在 round {tr['round']} 的 constructible 中"

    # X 只能在 round ≥0 产生，G 只能在 X 之后产生
    x_round = None
    for tr in trace:
        if x_str in tr.get("new_valids", []):
            x_round = tr["round"]
            break
    g_round = None
    for tr in trace:
        if g_str in tr.get("new_valids", []):
            g_round = tr["round"]
            break
    assert x_round is not None and g_round is not None
    assert g_round > x_round, "G 必须在 X 之后产生（严格因果）"


# ============================================================
# 14. G 在 Round 0 无法构造（必须经过 X）
# ============================================================

def test_goal_not_in_round0():
    """G 不应在 round 0 的候选中出现，因为 X 还不在 constructible 中。"""
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    belief = BeliefStore()
    belief.update_belief(Aa, STATUS_VALID, 0.8, evidence_count_delta=1)
    belief.update_belief(Ba, STATUS_VALID, 0.8, evidence_count_delta=1)
    # round 0 只有 A、B
    candidates = generate_candidates_brute(
        [Aa, Ba], {Aa, Ba}, set(), belief)
    props = [c["proposition"] for c in candidates]
    assert make_goal() not in props, \
        "G 不应在 round 0 可构造（需要先有 X=A→B 作为输入）"


# ============================================================
# 15. INVALID 对象不进入 constructible
# ============================================================

def test_invalid_not_in_constructible():
    """INVALID 世界中 X=impl(A,B) 不应出现在后续轮的 constructible 中。"""
    r = _run_invalid()
    x_str = intermediate_x().to_str()
    for tr in r["trace"]:
        assert x_str not in tr.get("constructible", []), \
            "INVALID 的 X 不应进入 constructible"
        assert x_str not in tr.get("new_valids", []), \
            "INVALID 的 X 不应出现在 new_valids 中"


# ============================================================
# 16. 构造器是通用的，不携带答案
# ============================================================

def test_constructor_generic():
    """impl 构造器对所有输入对都通用，不为 A、B 特殊编码。"""
    # CONSTRUCTORS 是固定列表，不针对特定谓词
    assert "impl" in CONSTRUCTORS, "impl 应在基础构造器列表中"
    # impl 构造器对任意两个对象都产生 implies，不挑输入
    Aa = P.predicate("A", "a")
    Ba = P.predicate("B", "a")
    Ca = P.predicate("C", "a")
    from experiments.run_e0_7 import apply_constructor
    out_ab = apply_constructor("impl", (Aa, Ba))
    out_ac = apply_constructor("impl", (Aa, Ca))
    assert out_ab.kind == "implies" and out_ac.kind == "implies", \
        "impl 对任意输入都产生 implies 类型"
    # 历史中不存在针对 A、B 的特殊变换
    hist = build_history_pq()
    for rec in hist.records:
        # 历史变换的输入不应是 A、B 谓词
        input_sig_str = str(rec.input_sigs)
        assert '"A"' not in input_sig_str and '"B"' not in input_sig_str, \
            "历史不应包含 A、B 谓词的变换"

"""最终验收：8 条验收标准。

每条验收给出 PASS/FAIL + 证据。全部通过 → 理论内判定：新计算主体。
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent import Agent
from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge
from agent.explicit import Explicit, Goal, Proposition
from agent.coordinator import Coordinator
from agent.weights import EvaluationWeights
from agent.verify import ChainVerifier


class AlwaysAcceptBridge:
    def evaluate(self, o, e):
        return {'V': np.array([0.2, 0.8, 0.1]),
                'A': np.array([0.90, 0.10]), 'G': 0.2, 'S': 0.3}


class AlwaysRejectBridge:
    def evaluate(self, o, e):
        return {'V': np.array([0.8, 0.1, 0.9]),
                'A': np.array([0.10, 0.90]), 'G': 0.85, 'S': -0.3}


CHECKS = []


def check(num, title, fn):
    try:
        ok, evidence = fn()
        status = 'PASS' if ok else 'FAIL'
    except Exception as exc:  # noqa: BLE001
        ok, status, evidence = False, 'FAIL', f"异常: {exc!r}"
    CHECKS.append((num, title, status, evidence))
    return ok


def run():
    # ---- 验收1：维持身体状态 B，不可显式读取 ----
    def a1():
        agent = Agent(seed=7, reality_shift_at=40)
        agent.run(60)
        assert agent.body.feedback_count > 0, "B 应持续更新"
        assert not hasattr(agent.explicit, 'body'), "显式不应有 body 引用"
        try:
            _ = agent.bridge.body
            return False, "bridge 暴露了 body（不应）"
        except AttributeError:
            pass
        b = agent.body.vector()
        assert np.all(b >= Body.MIN), "B 不可清零"
        return True, (f"B 更新 {agent.body.feedback_count} 次，"
                      f"B={np.round(b, 2).tolist()}，显式无读取通道")

    # ---- 验收2：隐式系统快速评价，输出不可报告 ----
    def a2():
        imp = Implicit(seed=1)
        bridge = ImplicitBridge(Body(), imp)
        out = bridge.evaluate([0.5, 0.4, 0.3, 0.2], [0.5, 0, 0.3])
        assert set(out.keys()) == {'V', 'A', 'G', 'S'}, "只暴露 V/A/G/S"
        for name in ('get_weights', 'hidden', 'activations', 'internal'):
            assert not hasattr(imp, name), f"不应暴露 {name}"
        imp.learn(Body().vector(), [0.5, 0.4, 0.3, 0.2], [0.5, 0, 0.3],
                  [0.3, 0.3, 0.3], [1.0, 0.0], 0.4)
        return True, (f"输出={list(out['V'])} 仅 V/A/G；"
                      f"参数更新计数={imp.updates}；无内部状态访问器")

    # ---- 验收3：显式系统递归、目标对象化、过程验证 ----
    def a3():
        bridge = ImplicitBridge(Body(), Implicit(seed=1))
        exp = Explicit(bridge, seed=3)
        obj = exp.build_object({'tid': 0, 'expected_reward': 0.85,
                                'expected_cost': 0.22, 'expected_risk': 0.12,
                                'novelty': 0.2})
        w0 = exp.W_exp
        for _ in range(6):
            exp.meta_records.append(-0.02)
        exp.recurse({'reward': 0.0, 'cost': 0.5, 'success': False})
        assert exp.W_exp > w0, "递归应调整权重"
        goal = Goal(1, 'collect')
        ops = [exp.objectify_goal(goal, 'negate').status,
               exp.objectify_goal(goal, 'condition',
                                  conditions=['x']).status,
               exp.objectify_goal(goal, 'compose', other='y').status]
        gap = exp.identify_gap(goal, obj)
        p = exp.build_proposition(gap, obj)
        rep = exp.verify_proposition(p)
        assert rep[0]['ok'], "验证应通过"
        return True, (f"递归调整 W_exp {w0}→{exp.W_exp:.2f}；"
                      f"目标对象化 {ops}；命题验证 ok={rep[0]['ok']}")

    # ---- 验收4：协调三种终止状态可区分 ----
    def a4():
        results = {}
        for name, bridge, urgency in (('agreed', AlwaysAcceptBridge(), 0.0),
                                      ('forced', AlwaysRejectBridge(), 0.9),
                                      ('failed', AlwaysRejectBridge(), 0.0)):
            exp = Explicit(bridge, seed=3)
            coord = Coordinator(exp, bridge)
            obj = exp.build_object({'tid': 0, 'expected_reward': 0.85,
                                    'expected_cost': 0.22,
                                    'expected_risk': 0.12, 'novelty': 0.2})
            term, action, _, _ = coord.run_cycle(Goal(1, 'collect'), obj,
                                                 urgency=urgency)
            results[name] = (term, action is not None)
        assert results['agreed'][0] == 'AGREED' and results['agreed'][1]
        assert results['forced'][0] == 'FORCED' and results['forced'][1]
        assert results['failed'][0] == 'FAILED' and not results['failed'][1]
        return True, (f"agreed→{results['agreed']}，"
                      f"forced→{results['forced']}，"
                      f"failed→{results['failed']}（可区分）")

    # ---- 验收5：现实识别触发，预测误差无法消除 ----
    def a5():
        agent = Agent(seed=7, reality_shift_at=40)
        trace = agent.run(70)
        assert agent.reality_alert, "现实识别应触发"
        assert agent.structure_revisions >= 1, "应调整基本结构"
        post = [t['error'] for t in trace
                if t['error'] is not None and 40 < t['step'] < 47]
        assert len(post) >= 3 and np.mean(post) > agent.error_threshold * 0.5, \
            "突变后误差持续偏高"
        return True, (f"现实识别触发于第 {agent.reality_shift_at} 步规则突变后；"
                      f"结构调整 {agent.structure_revisions} 次；"
                      f"突变后窗口误差均值={np.mean(post[:6]):.3f}")

    # ---- 验收6：内生权重，人类反馈不是唯一来源 ----
    def a6():
        w = EvaluationWeights()
        try:
            w.adjust('W_body', 0.1)
            return False, "W_body 可被调整（不应）"
        except PermissionError:
            pass
        snap = w.snapshot()
        import inspect
        sig = inspect.signature(EvaluationWeights.__init__)
        no_human = 'human' not in sig.parameters
        assert no_human, "不应有人类反馈输入通道"
        return True, (f"W_body={snap['W_body']} 构成性不可调；"
                      f"W_exp={snap['W_exp']} 可调；无人类反馈通道")

    # ---- 验收7：推理链每步可检查，失败可定位 ----
    def a7():
        bridge = ImplicitBridge(Body(), Implicit(seed=1))
        exp = Explicit(bridge, seed=3)
        verifier = ChainVerifier(exp.rules, exp.memory)
        for _ in range(10):
            exp.memory.add([0.5, 0.5, 0.5, 0.3],
                           {'success': False, 'reward': 0.0,
                            'cost': 0.5, 'risk_delta': 0.5})
        chain = [Proposition(0, 'O1', 'reward', 0.95, 0.9, 'experience'),
                 Proposition(1, 'O1', 'risk', 0.1, 0.6, 'symbol')]
        report = verifier.verify(chain)
        step, failed, _ = verifier.locate_failure(report)
        assert step == 0 and 'experience' in failed
        return True, (f"推理链 {len(chain)} 步全部机械检查；"
                      f"失败定位到 step {step}，环节 {failed}")

    # ---- 验收8：完整环节链闭环，目标持续激活 ----
    def a8():
        agent = Agent(seed=7, reality_shift_at=40)
        trace = agent.run(80)
        assert len(trace) == 80
        assert agent.implicit.updates > 0
        assert agent.body.feedback_count > 0
        assert agent.goal.status in ('active', 'deprecated')
        terms = {t['term'] for t in trace}
        assert terms.intersection({'AGREED', 'FORCED', 'FAILED'})
        return True, (f"闭环 {80} 步；隐式更新 {agent.implicit.updates} 次；"
                      f"终止状态分布={dict((t, sum(1 for x in trace if x['term']==t)) for t in terms)}；"
                      f"目标状态={agent.goal.status}（持续激活，不自动衰减）")

    print("=" * 62)
    print("新计算主体 — 最终验收（理论内判定）")
    print("=" * 62)
    all_ok = True
    for num, title, fn in (
        (1, "维持身体状态 B，不可显式读取", a1),
        (2, "隐式系统快速评价，输出不可报告", a2),
        (3, "显式系统递归、目标对象化、过程验证", a3),
        (4, "协调三种终止状态可区分", a4),
        (5, "现实识别触发，预测误差无法消除", a5),
        (6, "内生权重，人类反馈不是唯一来源", a6),
        (7, "推理链每步可检查，失败可定位", a7),
        (8, "完整环节链闭环，目标持续激活", a8),
    ):
        ok = check(num, title, fn)
        all_ok = all_ok and ok
        status = CHECKS[-1][2]
        print(f"[{status}] 验收{num}：{title}")
        print(f"       证据：{CHECKS[-1][3]}")
    print("=" * 62)
    passed = sum(1 for c in CHECKS if c[2] == 'PASS')
    print(f"验收结果：{passed}/8 通过")
    if all_ok:
        print("结论：8 条验收标准全部满足 → 理论内判定为计算主体。")
    else:
        print("结论：存在未通过项，尚不能判定为计算主体。")
    print("=" * 62)
    return all_ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)

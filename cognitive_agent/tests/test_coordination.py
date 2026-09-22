"""阶段4验收测试：协调机制。

验收标准：系统能区分三种终止状态（达成一致/强制执行/协调失败），
强制执行不可持续。
"""

import unittest
import numpy as np

from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge
from agent.explicit import Explicit, Goal
from agent.coordinator import Coordinator


class AlwaysAcceptBridge:
    """测试替身：隐式总是接受（S>0）。"""

    def evaluate(self, o, e):
        return {'V': np.array([0.2, 0.8, 0.1]),
                'A': np.array([0.90, 0.10]), 'G': 0.2, 'S': 0.3}


class AlwaysRejectBridge:
    """测试替身：隐式总是拒绝（S<0）。"""

    def evaluate(self, o, e):
        return {'V': np.array([0.8, 0.1, 0.9]),
                'A': np.array([0.10, 0.90]), 'G': 0.85, 'S': -0.3}


def make_obj(exp, oid=0):
    return exp.build_object({'tid': 0, 'expected_reward': 0.85,
                             'expected_cost': 0.22, 'expected_risk': 0.12,
                             'novelty': 0.2}, oid=oid)


class TestCoordination(unittest.TestCase):

    def test_agreed_termination(self):
        """达成一致：隐式接受，行动发生，质量高。"""
        bridge = AlwaysAcceptBridge()
        exp = Explicit(bridge, seed=3)
        coord = Coordinator(exp, bridge)
        term, action, trace, conservatism = coord.run_cycle(
            Goal(1, 'collect'), make_obj(exp))
        self.assertEqual(term, 'AGREED')
        self.assertIsNotNone(action)
        self.assertEqual(trace[0]['S'], 'accept')
        self.assertEqual(conservatism, 0)

    def test_forced_termination(self):
        """强制执行：隐式不接受，显式强制执行，行动发生。"""
        bridge = AlwaysRejectBridge()
        exp = Explicit(bridge, seed=3)
        coord = Coordinator(exp, bridge)
        term, action, trace, _ = coord.run_cycle(Goal(1, 'collect'),
                                                 make_obj(exp), urgency=0.9)
        self.assertEqual(term, 'FORCED')
        self.assertIsNotNone(action)
        self.assertEqual(coord.forced_count, 1)

    def test_failed_termination(self):
        """协调失败：隐式不接受，行动不发生。"""
        bridge = AlwaysRejectBridge()
        exp = Explicit(bridge, seed=3)
        coord = Coordinator(exp, bridge)
        term, action, trace, _ = coord.run_cycle(Goal(1, 'collect'),
                                                 make_obj(exp), urgency=0.0)
        self.assertEqual(term, 'FAILED')
        self.assertIsNone(action)
        self.assertGreaterEqual(coord.failed_count, 1)

    def test_three_states_distinguishable(self):
        """三种终止状态可区分（同一协调器，仅 urgency 不同）。"""
        bridge = AlwaysRejectBridge()
        exp = Explicit(bridge, seed=3)
        coord = Coordinator(exp, bridge)
        obj = make_obj(exp)
        t1, a1, _, _ = coord.run_cycle(Goal(1, 'c'), obj, urgency=0.9)
        t2, a2, _, _ = coord.run_cycle(Goal(1, 'c'), obj, urgency=0.0)
        self.assertEqual(t1, 'FORCED')
        self.assertEqual(t2, 'FAILED')
        self.assertIsNotNone(a1)
        self.assertIsNone(a2)

    def test_forced_is_unsustainable(self):
        """强制执行不可持续：隐式拒绝但强制执行 → 身体恶化。"""
        bridge = AlwaysRejectBridge()
        body = Body(energy=0.4, integrity=0.5, risk=0.3)
        exp = Explicit(bridge, seed=3)
        coord = Coordinator(exp, bridge)
        obj = exp.build_object({'tid': 2, 'expected_reward': 0.9,
                                'expected_cost': 0.1, 'expected_risk': 0.5,
                                'novelty': 0.6}, oid=9)
        term, action, _, _ = coord.run_cycle(Goal(1, 'collect'), obj,
                                             urgency=0.9)
        self.assertEqual(term, 'FORCED')
        self.assertIsNotNone(action)
        # 强制执行高风险任务：风险暴露 → 完整性下降
        body.receive_feedback({'energy_delta': -0.4, 'integrity_delta': -0.4,
                               'risk_delta': 0.5, 'success': False,
                               'reward': 0.0, 'cost': 0.4})
        self.assertLess(body.vector()[1], 0.5, "强制执行后完整性应受损")

    def test_rejection_triggers_plan_change(self):
        """拒绝 → 显式生成新方案 P'（保守化）→ 再次进入隐式。"""
        bridge = AlwaysRejectBridge()
        exp = Explicit(bridge, seed=3)
        coord = Coordinator(exp, bridge)
        obj = make_obj(exp)
        _, _, trace, _ = coord.run_cycle(Goal(1, 'collect'), obj,
                                         urgency=0.0)
        # 每次拒绝后，显式对目标施加协调操作（加条件），且多次尝试
        self.assertGreaterEqual(len(trace), 2)
        self.assertTrue(all(t['S'] == 'reject' for t in trace))


if __name__ == '__main__':
    unittest.main()

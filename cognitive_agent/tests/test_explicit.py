"""阶段3验收测试：显式系统 E。

验收标准：显式系统能递归、能目标对象化、能过程验证。
"""

import unittest
import numpy as np

from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge
from agent.explicit import Explicit, Goal, Object


def make_explicit():
    bridge = ImplicitBridge(Body(), Implicit(seed=1))
    return Explicit(bridge, seed=3)


OFFER = {'tid': 0, 'expected_reward': 0.85, 'expected_cost': 0.22,
         'expected_risk': 0.12, 'novelty': 0.2}


class TestExplicit(unittest.TestCase):

    def test_object_construction(self):
        exp = make_explicit()
        obj = exp.build_object(OFFER, oid=1)
        self.assertIsInstance(obj, Object)
        self.assertEqual(obj.attrs['expected_reward'], 0.85)
        self.assertEqual(obj.features().shape, (4,))

    def test_gap_identification(self):
        exp = make_explicit()
        obj = exp.build_object(OFFER)
        goal = Goal(1, 'collect resources')
        gap = exp.identify_gap(goal, obj)
        self.assertIn('predicate', gap)
        self.assertIn('gap_signal', gap)
        self.assertIsInstance(gap['gap_signal'], float)

    def test_proposition_and_verification(self):
        exp = make_explicit()
        obj = exp.build_object(OFFER)
        gap = exp.identify_gap(Goal(1, 'collect'), obj)
        p = exp.build_proposition(gap, obj)
        report = exp.verify_proposition(p)
        self.assertEqual(len(report), 1)
        self.assertTrue(report[0]['symbol'][0], "符号检查应通过")
        self.assertTrue(report[0]['ok'], "完整验证应通过")

    def test_evaluation_splits_three_params(self):
        exp = make_explicit()
        obj = exp.build_object(OFFER)
        eval_ = exp.evaluate(obj)
        for k in ('cost', 'reward', 'risk', 'utility'):
            self.assertIn(k, eval_)
        self.assertIsInstance(eval_['utility'], float)

    def test_recursion_adjusts_weights(self):
        """递归：对自己的评价过程进行计算，调整权重。"""
        exp = make_explicit()
        w_exp0, w_ext0 = exp.W_exp, exp.W_ext
        # 连续 5 步差反馈（平均净收益 < 0.05）→ W_exp 上调
        for _ in range(6):
            exp.meta_records.append(-0.02)
        exp.recurse({'reward': 0.0, 'cost': 0.5, 'success': False})
        self.assertGreater(exp.W_exp, w_exp0)
        self.assertLess(exp.W_ext, w_ext0)

    def test_goal_objectification(self):
        """目标对象化：否定、加条件、组合、废弃。"""
        exp = make_explicit()
        goal = Goal(1, 'collect resources', priority=0.6)
        neg = exp.objectify_goal(goal, 'negate')
        self.assertEqual(neg.status, 'negated')
        self.assertTrue(neg.content.startswith('NOT('))
        cond = exp.objectify_goal(goal, 'condition',
                                  conditions=['reduce_risk'])
        self.assertEqual(cond.status, 'conditioned')
        self.assertIn('reduce_risk', cond.conditions)
        comp = exp.objectify_goal(goal, 'compose', other='survive')
        self.assertEqual(comp.status, 'composed')
        dep = exp.objectify_goal(goal, 'deprecate')
        self.assertEqual(dep.status, 'deprecated')
        # 原目标对象被操作后仍可被施加不同操作（目标是对象）
        goal2 = Goal(2, 'explore')
        neg2 = exp.objectify_goal(goal2, 'negate')
        self.assertNotEqual(neg2.content, goal2.content)


if __name__ == '__main__':
    unittest.main()

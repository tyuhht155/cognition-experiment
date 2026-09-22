"""阶段1验收测试：身体状态 B。

验收标准：显式系统无法直接读取 B，但 B 的变化能影响行为。
"""

import unittest
import numpy as np

from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge
from agent.explicit import Explicit


def train_g_mapping(imp, steps=300):
    """快速训练：让隐式学会 G ≈ 身体张力（B 驱动缺口信号）。"""
    rng = np.random.RandomState(0)
    for _ in range(steps):
        body = Body(energy=rng.uniform(0.05, 0.95),
                    integrity=rng.uniform(0.05, 1.0),
                    risk=rng.uniform(0.0, 0.9))
        target_G = body.drive()
        o = [0.5, 0.4, 0.3, 0.2]
        e = [0.5, 0.0, 0.3]
        imp.learn(body.vector(), o, e, [0.3, 0.3, 0.3], [0.5, 0.5], target_G)


class TestBody(unittest.TestCase):

    def test_B_updates_continuously(self):
        b = Body()
        before = b.vector().copy()
        b.receive_feedback({'energy_delta': -0.2, 'resource_delta': 0.3})
        after = b.vector()
        self.assertFalse(np.allclose(before, after))
        self.assertEqual(b.feedback_count, 1)

    def test_B_never_cleared(self):
        b = Body(energy=0.1, integrity=0.1)
        b.receive_feedback({'energy_delta': -10.0, 'integrity_delta': -10.0,
                            'resource_delta': -10.0, 'risk_delta': 10.0})
        self.assertGreaterEqual(b.vector()[0], Body.MIN)
        self.assertGreaterEqual(b.vector()[1], Body.MIN)
        self.assertGreaterEqual(b.vector()[2], Body.MIN)
        self.assertLessEqual(b.vector()[3], 1.0)

    def test_explicit_has_no_body_access(self):
        """显式系统没有任何读取 B 的通道。"""
        body = Body()
        bridge = ImplicitBridge(body, Implicit(seed=1))
        exp = Explicit(bridge)
        self.assertFalse(hasattr(exp, 'body'))
        self.assertFalse(hasattr(exp, '_B'))
        self.assertFalse(hasattr(exp, 'B'))
        with self.assertRaises(AttributeError):
            _ = bridge.body

    def test_B_change_influences_implicit_output(self):
        """B 是隐式输入：B 变化必然改变输出（输入敏感性）。"""
        imp = Implicit(seed=1)
        o, e = [0.5, 0.4, 0.3, 0.2], [0.5, 0.0, 0.3]
        out1 = imp.evaluate(Body(energy=0.9, risk=0.05).vector(), o, e)
        out2 = imp.evaluate(Body(energy=0.05, risk=0.9).vector(), o, e)
        self.assertFalse(np.allclose(out1['V'], out2['V']))

    def test_B_drive_affects_behavior_after_learning(self):
        """训练后：濒危 B → 缺口信号 G 更高（B 通过隐式影响行为）。"""
        imp = Implicit(seed=1)
        train_g_mapping(imp)
        o, e = [0.5, 0.4, 0.3, 0.2], [0.5, 0.0, 0.3]
        healthy = imp.evaluate(Body(energy=0.9, integrity=0.95,
                                    risk=0.05).vector(), o, e)
        endangered = imp.evaluate(Body(energy=0.05, integrity=0.1,
                                       risk=0.9).vector(), o, e)
        self.assertGreater(endangered['G'], healthy['G'])
        self.assertLess(endangered['A'][1], healthy['A'][1] + 1e-9
                        )  # 濒危时避免倾向不显著下降亦可接受，主断言为 G


if __name__ == '__main__':
    unittest.main()

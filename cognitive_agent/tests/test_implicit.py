"""阶段2验收测试：隐式系统 I。

验收标准：隐式输出不可报告（显式无法读取内部状态），
显式只能拿到 V/A/G，隐式每次反馈后更新参数。
"""

import unittest
import numpy as np

from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge


class TestImplicit(unittest.TestCase):

    def test_output_is_vector_not_text(self):
        imp = Implicit(seed=1)
        out = imp.evaluate(Body().vector(), [0.5, 0.4, 0.3, 0.2],
                           [0.5, 0.0, 0.3])
        self.assertIsInstance(out['V'], np.ndarray)
        self.assertEqual(out['V'].shape, (3,))
        self.assertEqual(out['A'].shape, (2,))
        self.assertIsInstance(out['G'], float)
        # 输出不是文本：没有任何字符串形式的评价
        self.assertNotIsInstance(out['V'], str)
        self.assertNotIn('cost_text', out)

    def test_no_internal_state_accessor(self):
        """不可报告性：无权重/激活/隐藏状态访问器。"""
        imp = Implicit(seed=1)
        for name in ('get_weights', 'weights', 'hidden', 'activations',
                     'internal', 'state'):
            self.assertFalse(hasattr(imp, name),
                             f"隐式系统不应暴露 {name}")

    def test_parameters_update_after_feedback(self):
        """每次行动反馈后更新参数（持续更新）。"""
        imp = Implicit(seed=1)
        u0 = imp.updates
        body = Body()
        o, e = [0.5, 0.4, 0.3, 0.2], [0.5, 0.0, 0.3]
        imp.learn(body.vector(), o, e, [0.3, 0.3, 0.3], [1.0, 0.0], 0.4)
        self.assertEqual(imp.updates, u0 + 1)
        imp.learn(body.vector(), o, e, [0.3, 0.3, 0.3], [0.0, 1.0], 0.6)
        self.assertEqual(imp.updates, u0 + 2)

    def test_bridge_exposes_only_VAG(self):
        """显式只能调用 V/A/G/S，不能调用内部状态。"""
        bridge = ImplicitBridge(Body(), Implicit(seed=1))
        out = bridge.evaluate([0.5, 0.4, 0.3, 0.2], [0.5, 0.0, 0.3])
        self.assertEqual(set(out.keys()), {'V', 'A', 'G', 'S'})

    def test_learning_reduces_prediction_error(self):
        """训练目标：预测行动结果——学习后预测收益误差下降。"""
        imp = Implicit(seed=1)
        rng = np.random.RandomState(3)
        # 用收益 0.8 的任务训练 120 步
        for _ in range(120):
            body = Body()
            o = [0.8, 0.2, 0.1, 0.2]
            e = [0.9, 0.5, 0.05]
            imp.learn(body.vector(), o, e, [0.2, 0.8, 0.05],
                      [1.0, 0.0], 0.2)
        out = imp.evaluate(Body().vector(), [0.8, 0.2, 0.1, 0.2],
                           [0.9, 0.5, 0.05])
        err = abs(out['V'][1] - 0.8)
        self.assertLess(err, 0.30, f"收益预测误差应下降，实际 {err:.3f}")


if __name__ == '__main__':
    unittest.main()

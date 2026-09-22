"""阶段6验收测试：评价权重。

验收标准：人类反馈不是唯一权重来源，系统有内生权重
（W_body 构成性、不可由显式调整）。
"""

import unittest

from agent.weights import EvaluationWeights


class TestWeights(unittest.TestCase):

    def test_W_body_untouchable(self):
        """W_body 是构成性权重，显式不可调整。"""
        w = EvaluationWeights()
        with self.assertRaises(PermissionError):
            w.adjust('W_body', 0.1)
        with self.assertRaises(PermissionError):
            w.adjust('W_body', -0.1)
        # 尝试调整后 W_body 不变
        self.assertEqual(w.snapshot()['W_body'], 1.0)

    def test_W_exp_W_ext_adjustable(self):
        """W_exp / W_ext 可被（显式递归）调整。"""
        w = EvaluationWeights(w_exp=0.6, w_ext=0.4)
        w.adjust('W_exp', 0.1)
        w.adjust('W_ext', -0.05)
        snap = w.snapshot()
        self.assertAlmostEqual(snap['W_exp'], 0.7)
        self.assertAlmostEqual(snap['W_ext'], 0.35)
        self.assertGreaterEqual(len(w.adjust_log), 2)

    def test_internal_weights_not_from_human_feedback(self):
        """内生权重：系统权重体系由身体状态（W_body）+ 经验构成，
        不存在人类反馈输入通道。"""
        w = EvaluationWeights()
        snap = w.snapshot()
        self.assertIn('W_body', snap)
        # 构造器不接受人类反馈参数
        import inspect
        sig = inspect.signature(EvaluationWeights.__init__)
        params = list(sig.parameters.keys())
        for p in ('human', 'human_feedback', 'rlhf'):
            self.assertNotIn(p, params)

    def test_synthesis_rule(self):
        """合成规则：身体状态是基础层，外部贡献不能直接进入。"""
        w = EvaluationWeights()
        snap = w.snapshot()
        # 外部贡献 W_ext 存在但有限；身体 W_body 始终在场且不可消除
        self.assertEqual(snap['W_body'], 1.0)
        self.assertLessEqual(snap['W_ext'], 0.9)


if __name__ == '__main__':
    unittest.main()

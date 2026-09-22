"""阶段7验收测试：过程验证。

验收标准：推理链每步可机械检查，失败可定位到环节。
"""

import unittest

from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge
from agent.explicit import Explicit, Goal, Proposition, Experience
from agent.verify import ChainVerifier


def make_verifier():
    bridge = ImplicitBridge(Body(), Implicit(seed=1))
    exp = Explicit(bridge, seed=3)
    return exp, ChainVerifier(exp.rules, exp.memory)


class TestVerify(unittest.TestCase):

    def test_chain_is_data_structure(self):
        """推理链是数据结构：每步是离散命题，可枚举检查。"""
        exp, verifier = make_verifier()
        p1 = Proposition(0, 'O1', 'reward', 0.8, 0.7, 'experience')
        p2 = Proposition(1, 'O1', 'risk', 0.3, 0.7, 'experience')
        chain = [p1, p2]
        report = verifier.verify(chain)
        self.assertEqual(len(report), 2)
        for r in report:
            self.assertIn('step', r)
            self.assertIn('symbol', r)
            self.assertIn('experience', r)
            self.assertIn('reality', r)
            self.assertIn('ok', r)

    def test_each_step_mechanically_checkable(self):
        """每一步经符号/经验/现实三种机械检查。"""
        exp, verifier = make_verifier()
        p = Proposition(0, 'O1', 'reward', 0.8, 0.7, 'experience')
        exp.memory.add([0.9, 0.2, 0.1, 0.2], {'success': True,
                                              'reward': 0.85, 'cost': 0.15})
        report = verifier.verify([p])
        r = report[0]
        self.assertTrue(r['symbol'][0])
        self.assertTrue(r['experience'][0])
        self.assertTrue(r['reality'][0])
        self.assertTrue(r['ok'])

    def test_failure_located_to_chain_step(self):
        """失败可定位到具体环节（经验矛盾命题）。"""
        exp, verifier = make_verifier()
        # 经验：成功率很低
        for _ in range(10):
            exp.memory.add([0.5, 0.5, 0.5, 0.3],
                           {'success': False, 'reward': 0.0,
                            'cost': 0.5, 'risk_delta': 0.5})
        bad = Proposition(0, 'O1', 'reward', 0.95, 0.9, 'experience')
        good = Proposition(1, 'O1', 'risk', 0.1, 0.6, 'symbol')
        chain = [bad, good]
        report = verifier.verify(chain)
        step, failed, prop = verifier.locate_failure(report)
        self.assertEqual(step, 0)
        self.assertIn('experience', failed)
        self.assertIn('reward', prop)

    def test_failure_located_to_reality_loop(self):
        """现实检查失败定位：行动反馈与命题预期矛盾。"""
        exp, verifier = make_verifier()
        p = Proposition(0, 'O1', 'reward', 0.95, 0.9, 'experience')
        fb = {'success': False, 'reward': 0.0, 'cost': 0.5}
        report = verifier.verify([p], feedback=fb)
        self.assertFalse(report[0]['reality'][0])
        step, failed, _ = verifier.locate_failure(report)
        self.assertEqual(step, 0)
        self.assertIn('reality', failed)


if __name__ == '__main__':
    unittest.main()

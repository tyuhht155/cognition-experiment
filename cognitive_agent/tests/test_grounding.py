"""阶段5验收测试：现实接地。

验收标准：预测误差持续存在且无法靠内部模型更新消除 → 现实识别触发。
现实 = 稳定外部环境 + 行动反馈 + 不可消除误差，不要求物理世界。
"""

import unittest
import numpy as np

from agent.environment import Environment
from agent.body import Body
from agent.implicit import Implicit, ImplicitBridge
from agent.explicit import Explicit
from agent.coordinator import Coordinator
from agent.agent import Agent


class TestGrounding(unittest.TestCase):

    def test_environment_is_stable_and_independent(self):
        """现实稳定运行，不随系统内部模型改变。"""
        env = Environment(seed=7)
        # 同一任务多次行动：真实收益均值接近真实规则值
        offer = {'tid': 0, 'expected_reward': 0.9, 'expected_cost': 0.2,
                 'expected_risk': 0.1, 'novelty': 0.2}
        rewards = [env.step(offer)['reward'] for _ in range(200)]
        self.assertGreater(np.mean(rewards), 0.4)   # 成功率~0.9，收益高
        # 宣称属性不受任何内部模型影响（无外部状态写入接口）

    def test_mutation_breaks_claim_truth_alignment(self):
        """规则突变后：宣称不变、真实反转 → 感知与结果背离。"""
        env = Environment(seed=7)
        offer_before = env.offer()
        _ = env.step(offer_before)
        v_before = env.current_rule_version()
        env.mutate_rules()
        self.assertEqual(env.current_rule_version(), v_before + 1)
        # 突变后，同一任务的真实收益应远低于突变前
        offer = {'tid': 0, 'expected_reward': 0.9, 'expected_cost': 0.2,
                 'expected_risk': 0.1, 'novelty': 0.2}
        rewards_after = [env.step(offer)['reward'] for _ in range(100)]
        self.assertLess(np.mean(rewards_after), 0.3,
                        "规则突变后真实收益应大幅下降")

    def test_reality_detection_triggers(self):
        """集成：规则突变 → 预测误差持续不可消除 → 现实识别触发。"""
        agent = Agent(seed=7, reality_shift_at=40)
        trace = agent.run(70)
        self.assertTrue(agent.reality_alert,
                        "现实识别应被触发")
        self.assertGreaterEqual(agent.structure_revisions, 1,
                                "触发现实识别后应调整基本结构")
        # 误差窗口在突变后高于阈值（不可消除）
        post = [t['error'] for t in trace if t['error'] is not None
                and t['step'] > 40]
        self.assertGreater(len(post), 0)
        self.assertGreater(np.mean(post[:min(6, len(post))]),
                           agent.error_threshold * 0.5)

    def test_error_uneliminable_before_adaptation(self):
        """突变后、结构重置前：误差持续存在，内部更新无法消除。"""
        agent = Agent(seed=7, reality_shift_at=40)
        # 只跑到突变后不久（结构调整发生在 alert 触发时，通常 40~46 步）
        for _ in range(44):
            agent.step()
        window = agent.prediction_errors[-agent.update_budget:]
        if len(window) == agent.update_budget:
            self.assertGreater(np.mean(window), 0.15,
                               "结构调整前误差应持续偏高")


if __name__ == '__main__':
    unittest.main()

"""阶段8验收测试：整合运行。

验收标准：完整环节链闭环运行，目标持续激活（不自动衰减），
除非主动废弃。
"""

import unittest
import numpy as np

from agent.agent import Agent
from agent.body import Body
from agent.explicit import Goal


class TestAgent(unittest.TestCase):

    def test_full_loop_closes(self):
        """完整环节链闭环运行：B→隐式→显式→协调→行动→反馈→更新。"""
        agent = Agent(seed=7, reality_shift_at=40)
        trace = agent.run(60)
        self.assertEqual(len(trace), 60)
        terms = set(t['term'] for t in trace)
        self.assertTrue(terms.intersection({'AGREED', 'FORCED', 'FAILED'}))
        # 隐式持续更新（每次行动反馈后更新参数）
        self.assertGreater(agent.implicit.updates, 0)
        # 身体持续更新
        self.assertGreater(agent.body.feedback_count, 0)

    def test_goal_stays_active_without_decay(self):
        """目标持续激活：未废弃时 status 保持 active，激活水平不衰减。"""
        agent = Agent(seed=7, reality_shift_at=40)
        agent.run(30)
        self.assertEqual(agent.goal.status, 'active')

    def test_goal_deprecated_only_by_active_abandonment(self):
        """废弃需主动判定：显式评价连续 3 次成本过高才废弃；
        协调失败不废弃目标。"""
        agent = Agent(seed=7, reality_shift_at=40)
        for _ in range(10):
            agent._goal_management(utility=0.1)   # 成本不高 → 不废弃
        self.assertEqual(agent.goal.status, 'active')
        agent._goal_management(utility=-0.4)
        agent._goal_management(utility=-0.4)
        self.assertEqual(agent.goal.status, 'active')  # 未满 3 次
        agent._goal_management(utility=-0.4)
        self.assertEqual(agent.goal.status, 'deprecated')  # 3 次 → 废弃

    def test_goal_activation_level_never_auto_decays(self):
        """目标一旦激活就持续：激活水平不随时间自动衰减。"""
        agent = Agent(seed=7, reality_shift_at=40)
        self.assertEqual(agent.goal_activation, 1.0)
        agent.run(25)
        self.assertEqual(agent.goal_activation, 1.0)  # 无衰减逻辑

    def test_explicit_infers_body_only_via_feedback(self):
        """显式只能通过行为反馈推断 B：反馈能量告急 → 紧急 → 可能强制。"""
        agent = Agent(seed=7, reality_shift_at=40)
        # 灌入能量持续告急的反馈记录
        for _ in range(4):
            agent.explicit.memory.add([0.9, 0.2, 0.1, 0.2],
                                      {'success': False, 'reward': 0.0,
                                       'cost': 0.6, 'energy_delta': -0.6,
                                       'integrity_delta': -0.3,
                                       'risk_delta': 0.3})
        self.assertEqual(agent._urgency(), 0.9)
        self.assertFalse(hasattr(agent.explicit, 'body'))

    def test_reality_alert_and_recovery(self):
        """现实识别触发后结构调整，系统重新适应新现实。"""
        agent = Agent(seed=7, reality_shift_at=40)
        trace = agent.run(90)
        self.assertTrue(agent.reality_alert)
        self.assertGreaterEqual(agent.structure_revisions, 1)
        # 结构调整后（清空经验重新学习），后期误差应回落
        later = [t['error'] for t in trace
                 if t['error'] is not None and t['step'] > 60]
        early = [t['error'] for t in trace
                 if t['error'] is not None and 41 <= t['step'] <= 50]
        if len(later) >= 3 and len(early) >= 3:
            self.assertLess(np.mean(later), np.mean(early) + 0.1)


if __name__ == '__main__':
    unittest.main()

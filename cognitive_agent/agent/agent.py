"""阶段8：整合运行

完整环节链闭环运行 + 目标持续激活。

1. 身体状态 B 启动隐式动力
2. 隐式系统快速评价，输出 V、A、G
3. 显式系统构建命题、验证、评价、协调
4. 协调机制试探、观察、换方案
5. 行动产生现实反馈
6. 更新往数据集加数据，全面检查所有环节
7. 目标激活后持续激活，除非主动废弃（成本过高/冲突/否定）

现实识别：预测误差持续存在且无法靠内部模型更新消除 → 触发，
调整基本结构（清空经验、重建验证器），不是微调 W。
"""

import numpy as np

from .body import Body
from .implicit import Implicit, ImplicitBridge
from .explicit import Explicit, Experience, Goal
from .coordinator import Coordinator
from .environment import Environment
from .weights import EvaluationWeights
from .verify import ChainVerifier


class Agent:
    """新计算主体：身体状态 + 隐式系统 + 显式系统 + 协调 + 现实接地。"""

    def __init__(self, seed=7, reality_shift_at=40, error_threshold=0.40,
                 update_budget=6):
        self.body = Body()
        self.implicit = Implicit(seed=1)
        self.env = Environment(seed=seed)
        self.bridge = ImplicitBridge(self.body, self.implicit)
        self.explicit = Explicit(self.bridge, seed=seed)
        self.coordinator = Coordinator(self.explicit, self.bridge)
        self.weights = EvaluationWeights()
        self.verifier = ChainVerifier(self.explicit.rules, self.explicit.memory)
        self.goal = Goal(1, 'collect resources', priority=0.6)
        self.step_no = 0
        self.reality_shift_at = reality_shift_at
        self.error_threshold = error_threshold
        self.update_budget = update_budget
        self.reality_alert = False        # 现实识别是否触发
        self.structure_revisions = 0      # 基本结构调整次数
        self.prediction_errors = []       # 收益预测误差滚动窗口
        self.trace = []
        # 目标激活水平：目标一旦激活就持续（不自动衰减）。
        # 废弃只由主动判定触发（成本过高/冲突/否定），激活水平本身不随时间下降。
        self.goal_activation = 1.0

    # ---- 隐式监督目标：预测身体状态变化 + 行动结果 + 整体评价 ----
    def _implicit_targets(self, feedback):
        cost = feedback['cost'] / 1.0
        reward = feedback['reward'] / 1.0
        risk = feedback.get('risk_delta', 0.0)
        target_V = [min(1.0, cost), min(1.0, reward), min(1.0, risk)]
        # 行动倾向基于净收益（连续值，避免"一次失败→彻底回避"）
        net = reward - cost - risk * 0.5
        p = 1.0 / (1.0 + np.exp(-net * 3.0))
        target_A = [float(p), float(1.0 - p)]
        target_G = min(1.0, self.body.drive())
        return target_V, target_A, target_G

    # ---- 显式只能通过行为反馈推断 B：能量连续告急 → 紧急 ----
    def _urgency(self):
        rec = self.explicit.memory.records
        if len(rec) >= 3 and np.mean([r[1]['energy_delta']
                                      for r in rec[-3:]]) < -0.25:
            return 0.9
        return 0.0

    # ---- 现实识别：e 在 N 次更新后仍高于阈值 ----
    def _reality_detection(self):
        # 现实未变化时误差高是学习期正常现象，不触发现实识别；
        # 只有现实规则变化后误差持续不可消除，才触发现实识别。
        if self.env.current_rule_version() == 0:
            return
        if len(self.prediction_errors) >= self.update_budget:
            window = self.prediction_errors[-self.update_budget:]
            if np.mean(window) > self.error_threshold and not self.reality_alert:
                self.reality_alert = True
                self._structural_adaptation()

    def _structural_adaptation(self):
        """现实识别触发后：调整基本结构，不是微调 W。"""
        self.explicit.memory = Experience()          # 清空经验（结构重置）
        self.explicit.meta_records = []
        self.verifier = ChainVerifier(self.explicit.rules, self.explicit.memory)
        self.structure_revisions += 1

    # ---- 目标管理：持续激活，不自动衰减 ----
    def _goal_management(self, utility=None):
        """目标废弃的唯一主动判定：行动已发生但显式评价持续判定
        成本过高（utility < -0.3 累积 3 次）。协调失败不等于成本过高，
        目标保持激活；目标一旦激活就持续，除非主动废弃。"""
        g = self.goal
        if g.status == 'deprecated':
            return
        if utility is not None and utility < -0.30:
            g.conditions.append('high_cost')
            if len(g.conditions) >= 3:
                g.status = 'deprecated'              # 评价判定成本过高→废弃
        # 否则保持 active：激活水平不随时间衰减

    def step(self):
        s = self.step_no

        # 现实规则突变（非平稳）：制造不可消除误差
        if s == self.reality_shift_at:
            self.env.mutate_rules()

        offer = self.env.offer()
        obj = self.explicit.build_object(offer, oid=s)

        # 协调：显式方案 P → 隐式信号 → 三种终止状态之一
        term, action, coord_trace, conservatism = self.coordinator.run_cycle(
            self.goal, obj, urgency=self._urgency())

        fb = None
        utility = None
        if action is not None:
            fb = self.env.step(offer)                 # 现实反馈
            if conservatism > 0:
                # 隐式接受的是保守方案 P'：执行同样保守（自洽）
                c = 1.0 - 0.15 * conservatism
                fb = {**fb,
                      'reward': fb['reward'] * c,
                      'resource_delta': fb['resource_delta'] * c,
                      'progress_delta': fb['progress_delta'] * c,
                      'cost': fb['cost'] * (1 - 0.08 * conservatism),
                      'energy_delta': fb['energy_delta'] * (1 - 0.08 * conservatism),
                      'integrity_delta': fb['integrity_delta'] * (1 - 0.3 * conservatism),
                      'risk_delta': fb['risk_delta'] * (1 - 0.5 * conservatism)}
            self.body.receive_feedback(fb)            # B 持续更新
            o = obj.features()
            e = self.explicit.memory.summary()
            tV, tA, tG = self._implicit_targets(fb)
            self.implicit.learn(self.body.vector(), o, e, tV, tA, tG)
            self.explicit.update(obj, fb)             # 经验更新
            self.explicit.recurse(fb)                 # 递归调整权重

            # 预测误差：隐式收益预测 vs 现实反馈
            pred = self.bridge.evaluate(o, self.explicit.memory.summary())
            err = abs(float(pred['V'][1]) - float(fb['reward']))
            self.prediction_errors.append(err)
            utility = coord_trace[-1]['utility'] if coord_trace else None

        self._reality_detection()
        # 仅行动发生后评估目标成本（协调失败不废弃目标）
        self._goal_management(utility)

        self.step_no += 1
        self.trace.append({
            'step': s, 'term': term,
            'alert': self.reality_alert,
            'revisions': self.structure_revisions,
            'B': self.body.vector().copy(),
            'success': fb['success'] if fb else None,
            'error': err if fb is not None else None,
            'conservatism': conservatism,
            'coordination': coord_trace,
        })
        return self.trace[-1]

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.trace

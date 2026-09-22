"""阶段5：现实环境 R（模拟现实）

现实的定义（按理论修正）：
  现实 = 稳定的外部环境 + 行动反馈 + 无法通过内部模型更新消除的预测误差。
  不要求物理世界。

本环境满足三个条件：
- 任务池稳定运行：规则是外部给定，不随系统内部模型改变
- 系统的行动产生真实反馈（用任务的真实属性计算）
- 支持规则突变（非平稳）：制造"内部模型无法消除"的预测误差，
  从而触发系统的现实识别

关键设计——感知（宣称值）与真实结果分离：
- offer() 给出任务的"宣称属性"（稳定不变），系统据此建模预测
- step() 用任务的"真实属性"（现实规则）给出结果
- 初始宣称 = 真实（感知可靠，模型可学会，误差低）
- 突变只反转真实属性、宣称不变 → 感知与结果背离 →
  模型按宣称预测、现实按真实反馈 → 误差持续且无法靠内部更新消除

注意：系统只能拿到 offer() 的宣称属性，永远拿不到真实属性；
真实属性只在 step() 的反馈中体现。
"""

import random


class Environment:
    """模拟现实：稳定规则 + 行动反馈 + 可注入的不可消除误差。"""

    def __init__(self, seed=7):
        rng = random.Random(seed)
        self.rng = rng
        # 现实规则：任务的真实属性（系统不可见）
        self._truth = {
            0: dict(reward=0.90, cost=0.20, risk=0.10, novelty=0.20),
            1: dict(reward=0.60, cost=0.30, risk=0.30, novelty=0.40),
            2: dict(reward=0.80, cost=0.40, risk=0.45, novelty=0.60),
            3: dict(reward=0.30, cost=0.10, risk=0.05, novelty=0.10),
        }
        # 宣称属性：系统感知到的世界面貌（初始 = 真实，突变后保持）
        self._claim = {tid: dict(t) for tid, t in self._truth.items()}
        self.rule_version = 0          # 现实规则版本（突变时 +1）
        self._noise = 0.08             # 感知噪声

    def offer(self):
        """当前被考虑的对象（任务）。现实稳定供给对象。
        返回的是宣称属性 + 噪声（系统永远不知道真实属性）。"""
        tid = self.rng.choice(list(self._claim.keys()))
        t = self._claim[tid]
        offer = {
            'tid': tid,
            'expected_reward': self._clamp(t['reward'] + self._noise * self.rng.uniform(-1, 1)),
            'expected_cost': self._clamp(t['cost'] + self._noise * self.rng.uniform(-1, 1)),
            'expected_risk': self._clamp(t['risk'] + self._noise * self.rng.uniform(-1, 1)),
            'novelty': t['novelty'],
        }
        return offer

    def step(self, offer):
        """行动 → 真实反馈。现实按自身规则（真实属性）给出结果，
        不随系统内部模型改变。
        能量经济学：成功行动带来净能量收益，失败行动纯消耗；
        成功行动还带来维护与风险管理（完整性微恢复、风险下降），
        失败则完整性受损、风险上升。低风险任务期望为正（可持续），
        高风险任务期望为负（不可持续）→ 隐式学会"预测身体状态变化"、
        系统学会选择低风险任务维持自身健康。"""
        t = self._truth[offer['tid']]
        r = self.rng.random()
        success = r >= t['risk']
        if success:
            reward = t['reward'] * (0.70 + 0.60 * self.rng.random())
            cost = t['cost'] * 0.60
            energy_delta = -cost + reward * 0.30      # 成功 → 净能量收益
            integrity_delta = 0.03                      # 成功 → 维护恢复
            risk_delta = -0.03                          # 成功 → 风险管理
            resource_delta = reward - cost * 0.20
        else:
            reward = 0.0
            cost = t['cost'] * 1.40
            risk_delta = t['risk']
            energy_delta = -cost                        # 失败 → 纯消耗
            integrity_delta = -risk_delta               # 失败 → 完整性受损
            resource_delta = -cost * 0.20
        return {
            'success': bool(success),
            'reward': reward,
            'cost': cost,
            'energy_delta': energy_delta,
            'resource_delta': resource_delta,
            'integrity_delta': integrity_delta,
            'risk_delta': risk_delta,
            'progress_delta': reward if success else 0.0,
        }

    def mutate_rules(self):
        """现实规则突变：真实收益/成本反转、风险上升，宣称属性不变 →
        感知与结果背离，旧模型结构性失效。"""
        for tid, t in self._truth.items():
            t['reward'], t['cost'] = t['cost'], t['reward']
            t['risk'] = min(1.0, t['risk'] + 0.15)
        self.rule_version += 1
        return self.rule_version

    def current_rule_version(self):
        return self.rule_version

    @staticmethod
    def _clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

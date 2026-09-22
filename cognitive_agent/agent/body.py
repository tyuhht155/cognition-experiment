"""阶段1：身体状态等价物 B

B = [b1能量, b2完整性, b3资源, b4存在风险, b5任务进度]，取值 0~1。

约束：
- B 持续更新，不可清零（有下限保护）
- B 不进入显式系统符号空间（Explicit 构造时拿不到本对象）
- 显式系统不能直接读取 B 的值（无公开读取接口）
- B 只通过隐式评价影响行为
- 显式只能通过行为反馈推断 B

B 是隐式动力的来源：张力（drive）高时产生生存缺口，激活封闭目标。
"""

import numpy as np


class Body:
    """身体状态向量。仅向隐式系统暴露内部协议（vector/drive），
    显式系统永远拿不到本对象引用，也没有任何公开读取 B 的接口。
    """

    MIN = 0.02              # 不可清零下限
    DRIVE_THRESHOLD = 0.45  # 张力阈值，高于则产生生存缺口

    def __init__(self, energy=0.8, integrity=1.0, resource=0.5,
                 risk=0.10, progress=0.0):
        self._B = np.array([energy, integrity, resource, risk, progress],
                           dtype=float)
        self._clamp()
        self._feedback_count = 0

    def _clamp(self):
        """B 持续更新，不可清零：任何分量不低于 MIN。"""
        self._B = np.clip(self._B, self.MIN, 1.0)

    # ---- 内部协议：仅供隐式系统使用（模块内约定，不对外宣传） ----
    def vector(self):
        """当前 B 的副本。模块内只传给隐式系统。"""
        return self._B.copy()

    @staticmethod
    def drive_from(B):
        """由 B 向量计算身体张力（缺口）。含任务进度分量：
        进度未推进同样是缺口。张力启动隐式动力，产生封闭目标。"""
        energy, integrity, resource, risk, progress = B
        tension = ((1.0 - energy) * 0.35 + (1.0 - integrity) * 0.25
                   + risk * 0.15 + (1.0 - resource) * 0.10
                   + (1.0 - progress) * 0.15)
        return float(np.clip(tension, 0.0, 1.0))

    def drive(self):
        """当前身体张力（0~1）。B 有缺口 → 张力高 → 行动倾向上升。"""
        return self.drive_from(self._B)

    def has_survival_gap(self):
        """身体状态驱动的封闭缺口信号（供目标管理使用）。"""
        return self.drive() > self.DRIVE_THRESHOLD

    # ---- 持续更新：接收现实反馈 ----
    def receive_feedback(self, fb):
        """fb: dict of deltas。B 持续更新，不可清零。
        含身体自稳态：资源充足时完整性缓慢自我修复，
        经验积累使存在风险缓慢衰减（世界变得可预测）。"""
        self._B[0] += fb.get('energy_delta', 0.0)
        self._B[1] += fb.get('integrity_delta', 0.0)
        self._B[2] += fb.get('resource_delta', 0.0)
        self._B[3] += fb.get('risk_delta', 0.0)
        self._B[4] += fb.get('progress_delta', 0.0)
        # 自稳态：资源充足 → 完整性微修复；风险缓慢衰减
        if self._B[2] > 0.30:
            self._B[1] += 0.01
        self._B[3] *= 0.995
        self._clamp()
        self._feedback_count += 1
        return self._B.copy()

    @property
    def feedback_count(self):
        return self._feedback_count

    # 显式系统不可读取：本类刻意不提供 .value() / .get_B() 等公开方法。

"""阶段2：隐式系统 I

输入：B + 当前对象 O + 经验记忆 E
输出：评价向量 V=[成本c, 收益r, 风险k] + 行动倾向 A + 缺口信号 G
      + 接受信号 S（协调用）
（均为数值向量，不输出文本）

约束（按理论修正）：
- 隐式系统不可报告 = 显式无法直接读取隐式内部状态 + 隐式每次行动
  反馈后更新参数 + 显式永远只能拿到隐式输出（V/A/G/S）
- 输出不是文本，是向量
- 内部表示分布式（MLP 隐藏层），无局部符号对应
- 不监督输出为"正确"的符号解释（监督信号来自身体状态变化与行动结果）
- 显式只能调用 V/A/G/S，不能调用内部状态

根本修复（身体驱动，非人工偏置）：
- 行动倾向 A 由身体状态 B 结构性驱动：B 有缺口 → 张力高 →
  approach logit 直接上升（构成性通道 BODY_DRIVE_GAIN，不参与学习，
  不可被显式调整——这就是 W_body 构成性的实现）
- 接受信号 S 由隐式内部产生：S = A_approach − 阈值(drive)，
  阈值由 B 调制（缺口大 → 阈值低 → 更容易行动）

训练目标：预测身体状态变化、预测行动结果、生成整体评价信号。
"""

import numpy as np

from .body import Body

B_SLICE = slice(0, 5)
O_SLICE = slice(5, 9)
E_SLICE = slice(9, 12)

# W_body 构成性通道：缺口对行动倾向的增益（不参与学习，不可消除）。
# 增益显著大于随机初始波动：B 有缺口 → A 必然 > 0.5（构成性保证），
# 同时能量经济学使系统多数时间不濒危，学习部分仍能表达对象选择性。
BODY_DRIVE_GAIN = 2.5
# 先天避险通道：对象宣称风险直接压低行动倾向（构成性，不参与学习）。
# 对应隐式系统的情绪/本能评价——在学会区分任务之前，
# 系统天然避开高风险对象，避免学习完成前被连续失败拖垮。
RISK_AVERSION = 1.5
# 接受阈值基线；缺口越大阈值越低（由 B 调制）
BASE_THRESHOLD = 0.55
THRESHOLD_DRIVE_MOD = 0.25


class ImplicitNet:
    """小型分布式神经网络（隐式系统的实现）。
    内部表示分布式：隐藏层激活无局部符号对应。"""

    def __init__(self, n_in=12, n_hidden=16, lr=0.06, seed=1):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * 0.35
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 6) * 0.35
        self.b2 = np.zeros(6)
        # 无人工行动偏置：初始行动倾向完全由身体状态 B 驱动。
        # 学习率刻意取小：单次反馈只小幅更新（避免一次失败→彻底回避），
        # 使身体驱动通道（W_body 构成性）保持支配地位。
        self.lr = lr
        self.updates = 0   # 参数更新计数（"每次反馈后更新"的痕迹）

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        V = 1.0 / (1.0 + np.exp(-z[:3]))           # 成本c, 收益r, 风险k
        # 身体驱动通道：B 缺口直接推高 approach（构成性，不参与学习）
        drive = Body.drive_from(x[B_SLICE])
        # 先天避险通道：对象宣称风险直接压低 approach（构成性）
        obj_risk = x[O_SLICE][2]
        z3 = z[3] + BODY_DRIVE_GAIN * drive - RISK_AVERSION * obj_risk
        z4 = z[4] - BODY_DRIVE_GAIN * 0.5 * drive + RISK_AVERSION * obj_risk
        a = np.exp(np.array([z3, z4]) - np.max(np.array([z3, z4])))
        A = a / a.sum()                             # [approach, avoid]
        G = 1.0 / (1.0 + np.exp(-z[5]))             # 缺口信号
        return h, V, A, G, drive

    def learn(self, x, target_V, target_A, target_G):
        """在线梯度下降：每次行动反馈后更新参数（预测身体状态变化、
        预测行动结果、生成整体评价信号三种目标的监督都在此实现）。
        身体驱动通道是固定增益，不参与梯度（W_body 不可被学习移除）。"""
        h, V, A, G, _ = self.forward(x)
        dV = V - target_V
        dA = A - target_A
        dG = G - target_G
        dz = np.zeros(6)
        dz[:3] = dV * V * (1 - V)          # 评价向量梯度（sigmoid）
        dz[3:5] = dA                       # 行动倾向梯度（softmax CE）
        dz[5] = dG * G * (1 - G)           # 缺口信号梯度（sigmoid）
        dW2 = np.outer(h, dz)
        db2 = dz
        dh = (dz @ self.W2.T) * (1 - h ** 2)
        dW1 = np.outer(x, dh)
        db1 = dh
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.updates += 1


class Implicit:
    """隐式系统对外壳。只暴露 evaluate() 与 learn()，
    不暴露权重、隐藏激活等内部状态（不可报告性）。"""

    def __init__(self, seed=1, lr=0.06):
        self.net = ImplicitNet(lr=lr, seed=seed)

    # ---- 显式可调用：唯一外部接口（V/A/G/S） ----
    def evaluate(self, B, O, E):
        """输入 B(5) + O(4) + E(3)，输出 V/A/G/S。不输出文本。
        S 为接受信号（隐式根据 B+E+对象产生）：S>0 接受，S<0 拒绝。"""
        x = np.concatenate([B, O, E]).astype(float)
        _, V, A, G, drive = self.net.forward(x)
        threshold = BASE_THRESHOLD - THRESHOLD_DRIVE_MOD * drive
        S = float(A[0] - threshold)
        return {'V': V, 'A': A, 'G': float(G), 'S': S}

    def learn(self, B, O, E, target_V, target_A, target_G):
        """每次行动反馈后更新参数。"""
        x = np.concatenate([B, O, E]).astype(float)
        self.net.learn(x, np.asarray(target_V, float),
                       np.asarray(target_A, float), float(target_G))

    @property
    def updates(self):
        return self.net.updates

    # 无内部状态访问器：get_weights / hidden / activations 均不存在


class ImplicitBridge:
    """显式与隐式之间的唯一通道：只透传 V/A/G/S。
    内部使用 body.vector() 组装输入，但 B 的数值不向显式暴露。
    显式永远接触不到 Body 对象，也接触不到 B 的值。"""

    def __init__(self, body, implicit):
        self._body = body
        self._implicit = implicit

    def evaluate(self, o_features, e_summary):
        B = self._body.vector()
        out = self._implicit.evaluate(B, np.asarray(o_features, float),
                                      np.asarray(e_summary, float))
        return out   # 只含 V / A / G / S

    @property
    def body(self):
        """刻意不暴露 body：显式通过本桥拿不到任何 B 信息。"""
        raise AttributeError("ImplicitBridge 不向显式暴露身体状态")

    def __getattr__(self, name):
        if name.startswith('body') or name.startswith('_body'):
            raise AttributeError(f"显式无法通过 {name} 读取身体状态")
        raise AttributeError(f"ImplicitBridge 无属性 {name}")

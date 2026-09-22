"""阶段3：显式系统 E

环节：对象构建→缺口识别→构建命题→验证→评价→协调→行动→更新
能力：递归（对自己的评价过程计算，调整权重）、目标对象化（否定/
      加条件/组合）、命题构建、过程验证
实现：符号推理层 + 工作记忆

约束：显式系统不能直接读取 B。它只持有 ImplicitBridge，
     只能调用隐式输出的 V/A/G，只能通过行为反馈推断 B。
"""

import numpy as np
from dataclasses import dataclass, field

from .verify import ChainVerifier


# ---------------- 符号对象 ----------------

@dataclass
class Object:
    """被构建的对象（符号化的现实片段）。"""
    oid: int
    kind: str
    attrs: dict

    def features(self):
        """对象数值特征（供隐式系统输入）：[收益, 成本, 风险, 新颖度]"""
        return np.array([self.attrs['expected_reward'],
                         self.attrs['expected_cost'],
                         self.attrs['expected_risk'],
                         self.attrs['novelty']], dtype=float)


@dataclass
class Proposition:
    """离散命题：推理链的最小单元。"""
    pid: int
    subject: str
    predicate: str
    value: float
    certainty: float
    source: str   # 'symbol' | 'experience' | 'reality'

    def __str__(self):
        return (f"{self.subject}.{self.predicate}={self.value:.2f}"
                f"[{self.source}]")


@dataclass
class Goal:
    """目标对象：可被否定、加条件、组合、废弃。"""
    gid: int
    content: str
    status: str = 'active'    # active/negated/conditioned/composed/deprecated
    conditions: list = field(default_factory=list)
    priority: float = 0.5

    def __str__(self):
        c = f"+{self.conditions}" if self.conditions else ""
        return f"Goal[{self.gid}]{self.content}({self.status}){c}"


@dataclass
class Rule:
    """符号规则：命题验证的基本操作规则。"""
    name: str
    predicate: str

    def matches(self, p):
        return p.predicate == self.predicate


# ---------------- 经验记忆 ----------------

class Experience:
    """经验记忆：行动结果的符号化积累（显式系统的一部分，
    也作为隐式系统的输入源 E）。"""

    def __init__(self, cap=200):
        self.records = []      # [(features, feedback), ...]
        self.cap = cap

    def add(self, features, feedback):
        self.records.append((np.asarray(features, float), feedback))
        if len(self.records) > self.cap:
            self.records.pop(0)

    def empty(self):
        return len(self.records) == 0

    def summary(self):
        """经验摘要 E（隐式系统输入）：[成功率, 平均净收益, 平均风险]"""
        if not self.records:
            return np.array([0.5, 0.0, 0.3], dtype=float)
        succ = [1.0 if r[1]['success'] else 0.0 for r in self.records]
        gain = [r[1]['reward'] - r[1]['cost'] for r in self.records]
        risk = [r[1].get('risk_delta', 0.0) for r in self.records]
        return np.array([np.mean(succ), np.mean(gain), np.mean(risk)],
                        dtype=float)

    def consistent(self, p):
        """经验检查：命题价值方向与历史均值一致（允许 ±0.3 偏差）。"""
        s = self.summary()
        if p.predicate == 'reward' and s[0] < 0.35 and p.value > 0.70:
            return False, "经验显示成功率低，与高收益命题矛盾"
        if p.predicate == 'risk' and s[2] < 0.15 and p.value > 0.60:
            return False, "经验显示风险低，与高风险命题矛盾"
        return True, "与经验一致"

    def reality_consistent(self, p, feedback):
        """现实检查：行动反馈与命题预期不矛盾。"""
        if p.predicate == 'reward' and feedback is not None:
            actual = feedback.get('reward', 0.0)
            if actual < 0.05 and p.value > 0.70:
                return False, "行动反馈与命题预期矛盾"
        return True, "与行动反馈一致"


# ---------------- 显式系统 ----------------

class Explicit:
    """符号推理层 + 工作记忆。环节：对象构建→缺口识别→构建命题→
    验证→评价→协调→行动→更新。"""

    def __init__(self, bridge, seed=3):
        self.bridge = bridge        # ImplicitBridge：只调 V/A/G，读不到 B
        self.memory = Experience()
        self.rules = [Rule('r1', 'reward'), Rule('r2', 'cost'),
                      Rule('r3', 'risk'), Rule('r4', 'feasible')]
        self.work_memory = {}       # 工作记忆
        self.next_pid = 0
        # 显式评价权重（阶段6）：W_exp 经验贡献 / W_ext 外部贡献
        self.W_exp = 0.6
        self.W_ext = 0.4
        self.meta_records = []      # 递归评价记录（近期净收益）
        self._rng = np.random.RandomState(seed)

    # ---- 环节1：对象构建 ----
    def build_object(self, env_offer, oid=0):
        obj = Object(oid, 'task', {
            'expected_reward': float(env_offer['expected_reward']),
            'expected_cost': float(env_offer['expected_cost']),
            'expected_risk': float(env_offer['expected_risk']),
            'novelty': float(env_offer['novelty']),
        })
        self.work_memory['object'] = obj
        return obj

    # ---- 环节2：缺口识别 ----
    def identify_gap(self, goal, obj):
        """缺口 = 目标与对象可行性的符号比较 + 隐式缺口信号 G。
        G 是隐式输出（显式可调用）；B 本身显式不可读。"""
        G = 0.0
        if self.bridge is not None:
            out = self.bridge.evaluate(obj.features(), self.memory.summary())
            G = out['G']
        gap_value = 0.5 + 0.5 * G
        return {'predicate': 'feasible', 'value': float(gap_value),
                'certainty': 0.6, 'gap_signal': float(G)}

    # ---- 环节3：构建命题 ----
    def build_proposition(self, gap, obj, source='symbol'):
        p = Proposition(self.next_pid, f"O{obj.oid}", gap['predicate'],
                        float(gap['value']), float(gap.get('certainty', 0.6)),
                        source)
        self.next_pid += 1
        return p

    # ---- 环节4：验证 ----
    def verify_proposition(self, p, feedback=None):
        return ChainVerifier(self.rules, self.memory).verify([p], feedback)

    # ---- 环节5：评价（拆开三参数，分别建模） ----
    def evaluate(self, obj):
        """显式评价：把隐式给出的 V=[c,r,k] 拆开，分别建模，
        再用显式权重（W_exp/W_ext）合成效用。"""
        out = self.bridge.evaluate(obj.features(), self.memory.summary())
        V = out['V']
        cost, reward, risk = float(V[0]), float(V[1]), float(V[2])
        exp_term = self.W_exp * reward
        ext_term = self.W_ext * reward * (1.0 - risk)
        utility = exp_term + ext_term - 0.5 * cost - 0.8 * risk
        return {'V': V, 'cost': cost, 'reward': reward, 'risk': risk,
                'utility': float(utility), 'W_exp': self.W_exp,
                'W_ext': self.W_ext}

    # ---- 环节6：协调 ----
    def coordinate(self, goal, obj, S):
        """显式侧协调：观察隐式接受/拒绝信号。
        S='reject' 时对象化目标（加条件降风险）再生成新方案。"""
        if S == 'reject':
            self.objectify_goal(goal, 'condition', conditions=['reduce_risk'])
        return {'coordinated': True, 'signal': S}

    # ---- 环节7：行动 ----
    def act(self, obj):
        return {'action': 'execute', 'target': obj.oid}

    # ---- 环节8：更新 ----
    def update(self, obj, feedback):
        self.memory.add(obj.features(), feedback)

    # ---- 递归：对自己的评价过程进行计算，调整权重 ----
    def recurse(self, feedback):
        """元评价：观察近期决策质量，递归调整 W_exp/W_ext。"""
        s = self.memory.summary()
        self.meta_records.append(float(s[1]))
        if len(self.meta_records) >= 5:
            recent = float(np.mean(self.meta_records[-5:]))
            if recent < 0.05:                      # 决策质量下滑 → 重经验
                self.W_exp = min(0.9, self.W_exp + 0.05)
                self.W_ext = max(0.1, self.W_ext - 0.05)
            elif recent > 0.30:                   # 决策质量好 → 信任外部
                self.W_ext = min(0.9, self.W_ext + 0.02)
        return {'W_exp': self.W_exp, 'W_ext': self.W_ext}

    # ---- 目标对象化：把目标当对象，施加否定/加条件/组合 ----
    def objectify_goal(self, goal, op, **kw):
        """对目标施加操作，返回新目标对象（原目标保留）。"""
        if op == 'negate':
            return Goal(goal.gid + 100, f"NOT({goal.content})",
                        status='negated', priority=0.0)
        if op == 'condition':
            return Goal(goal.gid + 200, goal.content, status='conditioned',
                        conditions=kw.get('conditions', []),
                        priority=goal.priority)
        if op == 'compose':
            return Goal(goal.gid + 300, f"{goal.content}∧{kw.get('other', '?')}",
                        status='composed', priority=goal.priority)
        if op == 'deprecate':
            goal.status = 'deprecated'
            return goal
        raise ValueError(f"未知目标操作: {op}")

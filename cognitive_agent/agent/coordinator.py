"""阶段4：协调机制

循环：
1. 显式生成方案 P
2. P 进入隐式系统
3. 隐式根据 B+E+P 产生接受/拒绝信号 S（S 由隐式内部产生，
   阈值由身体状态 B 调制：缺口大阈值低）
4. S=接受 → 行动发生，质量高
5. S=拒绝 → 显式观察拒绝信号，生成新方案 P'（保守化：降低宣称
   成本与风险），P' 再次进入隐式
6. 重复直到接受，或评价判定成本过高，或协调失败

三种终止状态（可区分）：
- AGREED   达成一致：隐式接受，行动发生，质量高
- FORCED   强制执行：隐式不接受但显式强制执行，行动发生，
           质量低，不可持续（风险暴露使 B 受损）
- FAILED   协调失败：隐式不接受，行动不发生

约束：隐式最终控制身体输出，显式只能试探。
"""


class Coordinator:
    """协调循环：显式试探，隐式把门。"""

    MAX_ATTEMPTS = 4

    def __init__(self, explicit, bridge):
        self.explicit = explicit
        self.bridge = bridge
        self.forced_count = 0
        self.failed_count = 0
        self.agreed_count = 0

    def _implicit_signal(self, feats):
        """隐式信号：基于 B+E+方案特征给出接受/拒绝。
        S 是隐式内部产生的接受信号（含 B 调制阈值），S>0 接受。"""
        out = self.bridge.evaluate(feats, self.explicit.memory.summary())
        return ('accept' if out['S'] > 0.0 else 'reject'), out

    def _plan_features(self, obj, attempt):
        """方案 P 的特征向量；拒绝后显式生成新方案 P'：
        保守化（降低宣称成本与宣称风险），再次进入隐式。"""
        feats = obj.features()
        if attempt > 0:
            feats = feats.copy()
            feats[1] *= max(0.4, 1.0 - 0.15 * attempt)   # 宣称成本下降
            feats[2] *= max(0.3, 1.0 - 0.25 * attempt)   # 宣称风险下降
        return feats

    def run_cycle(self, goal, obj, urgency=0.0):
        """返回 (终止状态, 行动, 轨迹, 保守度)。"""
        trace = []
        for attempt in range(self.MAX_ATTEMPTS):
            # 1. 显式生成方案（缺口识别→命题→验证→评价）
            gap = self.explicit.identify_gap(goal, obj)
            p = self.explicit.build_proposition(gap, obj)
            rep = self.explicit.verify_proposition(p)
            eval_ = self.explicit.evaluate(obj)
            # 2. 方案进入隐式（拒绝后换保守方案 P'）
            feats = self._plan_features(obj, attempt)
            S, out = self._implicit_signal(feats)
            trace.append({'attempt': attempt, 'S': S,
                          'utility': eval_['utility'],
                          'A': out['A'].tolist(), 'G': out['G'],
                          'S_raw': out['S'],
                          'verify_ok': rep[0]['ok']})
            if S == 'accept':
                self.agreed_count += 1
                return 'AGREED', self.explicit.act(obj), trace, attempt
            # 3. 拒绝 → 显式观察拒绝信号，生成新方案（协调/目标对象化）
            self.explicit.coordinate(goal, obj, S)
            # 显式强制执行（仅在紧迫时优先；强制不可持续）
            if urgency >= 0.80:
                self.forced_count += 1
                return 'FORCED', self.explicit.act(obj), trace, attempt
            # 否则：继续换方案 P'（成本过高不在此终止——
            # 协调失败与成本过高的判定分离，成本过高由目标管理层处理）
        # 多次换方案仍被拒 → 协调失败
        self.failed_count += 1
        return 'FAILED', None, trace, self.MAX_ATTEMPTS - 1

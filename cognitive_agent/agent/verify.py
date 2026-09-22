"""阶段7：过程验证

推理链 R=[r1..rn] 是数据结构；每一步 ri 是离散命题。
机械检查：符号检查 / 经验检查 / 现实检查。
输出对推理过程的验证（不是"真/假"结论），失败定位到具体环节。

验收：每一步可机械检查，失败可定位到环节。
"""


class ChainVerifier:
    """对推理链逐命题执行三种机械检查。"""

    def __init__(self, rule_base, experience):
        self.rule_base = rule_base      # 符号规则库
        self.experience = experience     # 经验记忆

    def symbol_check(self, p):
        """ri 是否遵循基本操作规则。"""
        for rule in self.rule_base:
            if rule.matches(p):
                return True, f"符号规则[{rule.name}]支持"
        return False, f"无符号规则支持 {p.predicate}"

    def experience_check(self, p):
        """ri 是否与存储的经验一致。"""
        if self.experience is None or self.experience.empty():
            return True, "无经验可对照（跳过）"
        return self.experience.consistent(p)

    def reality_check(self, p, feedback):
        """ri 是否与行动反馈矛盾。"""
        if feedback is None:
            return True, "无反馈可对照（跳过）"
        return self.experience.reality_consistent(p, feedback)

    def verify(self, chain, feedback=None):
        """返回每步检查报告：[{step, proposition, symbol, experience, reality, ok}]"""
        report = []
        for i, p in enumerate(chain):
            s_ok, s_msg = self.symbol_check(p)
            e_ok, e_msg = self.experience_check(p)
            r_ok, r_msg = self.reality_check(p, feedback)
            ok = s_ok and e_ok and r_ok
            report.append({
                'step': i,
                'proposition': str(p),
                'symbol': (s_ok, s_msg),
                'experience': (e_ok, e_msg),
                'reality': (r_ok, r_msg),
                'ok': ok,
            })
        return report

    def locate_failure(self, report):
        """失败定位到具体环节（返回失败步索引、失败检查类别、命题）。"""
        for r in report:
            if not r['ok']:
                failed = [k for k in ('symbol', 'experience', 'reality')
                          if not r[k][0]]
                return r['step'], failed, r['proposition']
        return None, None, None

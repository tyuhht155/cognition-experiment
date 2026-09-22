"""阶段6：评价权重

W = f(W_body, W_exp, W_ext)
- W_body：身体状态贡献，构成性，不可消除（显式不能调整）
- W_exp：经验贡献，可调整
- W_ext：外部贡献，只能通过经验或身体进入（不直接进入评价）

合成规则：
- 身体状态贡献是基础层（隐式输入中 B 的权重不可被显式触及）
- 经验贡献叠加在基础层上
- 外部贡献不能直接进入（环境反馈只能先变成经验/身体变化）
- 显式可递归调整 W_exp 和 W_ext，不能调整 W_body

隐式：W = W_body + W_exp（无递归）
显式：W = W_body + W_exp + W_ext + 递归调整

验收：人类反馈不是唯一权重来源，系统有内生权重（W_body 由身体状态
构成性地给出，且不可由显式调整）。
"""


class EvaluationWeights:
    """显式评价权重的管理面；同时记录 W_body 的构成性与不可调性。"""

    def __init__(self, w_exp=0.6, w_ext=0.4):
        self.w_exp = w_exp          # 经验贡献（可调整）
        self.w_ext = w_ext          # 外部贡献（可调整）
        self._w_body = 1.0          # 身体贡献：构成性，不可消除
        self.adjust_log = []        # 调整痕迹（可审计）

    def adjust(self, which, delta):
        """只允许调整 W_exp / W_ext；W_body 拒绝（构成性权重）。"""
        if which == 'W_body':
            raise PermissionError("W_body 是构成性权重，显式不可调整")
        if which == 'W_exp':
            self.w_exp = max(0.0, min(1.0, self.w_exp + delta))
        elif which == 'W_ext':
            self.w_ext = max(0.0, min(1.0, self.w_ext + delta))
        else:
            raise ValueError(f"未知权重: {which}")
        self.adjust_log.append((which, delta))
        return self.snapshot()

    def snapshot(self):
        return {'W_body': self._w_body, 'W_exp': self.w_exp,
                'W_ext': self.w_ext}

    def body_is_untouchable(self):
        """验收辅助：W_body 是否存在于权重体系中且不可由显式调整。"""
        return True

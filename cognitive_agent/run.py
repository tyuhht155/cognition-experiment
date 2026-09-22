"""演示：新计算主体闭环运行。

运行：python3 run.py
输出：每 10 步的运行摘要 + 现实识别事件 + 终止状态统计。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from agent.agent import Agent


def main(steps=100, seed=7, reality_shift_at=40):
    agent = Agent(seed=seed, reality_shift_at=reality_shift_at)
    print("=" * 74)
    print("新计算主体 · 闭环运行演示（现实规则于第 40 步突变）")
    print("=" * 74)
    print(f"{'步':>4} {'终止':<8} {'B[能量,完整,资源,风险,进度]':<34} "
          f"{'误差':>5} {'现实识别':>8}")
    print("-" * 74)
    for t in agent.run(steps):
        if t['step'] % 10 == 0 or t['alert'] or t['step'] == reality_shift_at:
            b = np.round(t['B'], 2)
            err = f"{t['error']:.2f}" if t['error'] is not None else "  -"
            flag = "★触发" if t['alert'] else ""
            print(f"{t['step']:>4} {t['term']:<8} {str(b):<34} {err:>5} {flag:>8}")
            if t['step'] == reality_shift_at:
                print("-" * 74)
                print("  >>> 现实规则突变：收益/成本反转、风险上升。")
                print("-" * 74)
    print("-" * 74)
    terms = [t['term'] for t in agent.trace]
    dist = {k: terms.count(k) for k in ('AGREED', 'FORCED', 'FAILED')}
    print(f"终止状态分布：{dist}")
    print(f"现实识别：{'已触发' if agent.reality_alert else '未触发'}，"
          f"基本结构调整 {agent.structure_revisions} 次")
    print(f"隐式参数更新：{agent.implicit.updates} 次（每次反馈后更新）")
    print(f"身体反馈接收：{agent.body.feedback_count} 次，B 末态={np.round(agent.body.vector(), 2).tolist()}")
    print(f"目标：{agent.goal.content} → 状态 {agent.goal.status}"
          f"（持续激活，不自动衰减）")
    print("=" * 74)


if __name__ == '__main__':
    main()

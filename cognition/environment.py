"""人工环境：极小的“球/盒子/桌子/墙”世界。

理论第 10 条：系统不能直接获得完整规则，只获得连续状态和结果。
给它简单目标（预测下一状态 / 维持某目标状态），让系统从观察中产生命题，
例如逐渐产生 On(ball,table)、Move(ball)、Open(box)、Open(box)->CanTake(ball) 等。

这里的转移规则是世界内部的“物理”，对系统是隐藏的。系统只通过 world_history
（观察到的状态序列）获得信息。验证方法基于观察（不读取隐藏规则）。
"""

from __future__ import annotations

import random
from typing import List, Set, Tuple

from .proposition import Proposition


# 命题构造便利
def Atom(n: str) -> Proposition:
    return Proposition.atom(n)


def Rel(name: str, a: str, b: str) -> Proposition:
    return Proposition.relation(name, a, b)


def Pred(name: str, *args: str) -> Proposition:
    return Proposition.predicate(name, *args)


class World:
    """确定性 + 随机事件的微型世界。隐藏规则如下（系统不可见）：

    - 球在桌上；推球(Move ball) 且盒子开 且盒子靠近桌子 -> 球进盒子。
    - 推球 且盒子关 -> 球掉到地上。
    - 球在地上 且抬起(Lift ball) -> 球回到桌上。
    - 开盒子后 CanTake(ball) 成立（能力谓词）。
    - 关盒子动作把 Open(box) 撤销。
    """

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        # 初始状态
        self.state: Set[Proposition] = {
            Rel("On", "ball", "table"),
            Rel("Near", "table", "box"),
            Pred("Closed", "box"),
        }
        self.history: List[Set[Proposition]] = []
        self.events: List[str] = []
        self._snapshot()

    # ---- 内部：状态转移（隐藏规则）----
    def _snapshot(self):
        self.history.append(set(self.state))

    def _has(self, p: Proposition) -> bool:
        return any(p == q for q in self.state)

    def _add(self, p: Proposition):
        self.state.add(p)

    def _del(self, p: Proposition):
        self.state = {q for q in self.state if q != p}

    def _apply_event(self, ev: str):
        if ev == "open_box":
            if self._has(Pred("Closed", "box")):
                self._del(Pred("Closed", "box"))
                self._add(Pred("Open", "box"))
                self._add(Pred("CanTake", "ball"))
        elif ev == "close_box":
            if self._has(Pred("Open", "box")):
                self._del(Pred("Open", "box"))
                self._add(Pred("Closed", "box"))
                self._del(Pred("CanTake", "ball"))
        elif ev == "move_ball":
            self._add(Pred("Move", "ball"))
            # 规则1：盒子开 + 靠近 -> 进盒子
            if (self._has(Pred("Open", "box")) and self._has(Rel("Near", "table", "box"))
                    and self._has(Rel("On", "ball", "table"))):
                self._del(Rel("On", "ball", "table"))
                self._add(Rel("In", "ball", "box"))
            # 规则2：盒子关 -> 掉地上
            elif (self._has(Pred("Closed", "box")) and self._has(Rel("On", "ball", "table"))):
                self._del(Rel("On", "ball", "table"))
                self._add(Pred("OnFloor", "ball"))
        elif ev == "lift_ball":
            if self._has(Pred("OnFloor", "ball")):
                self._del(Pred("OnFloor", "ball"))
                self._add(Rel("On", "ball", "table"))
        elif ev == "take_out":
            if self._has(Rel("In", "ball", "box")) and self._has(Pred("Open", "box")):
                self._del(Rel("In", "ball", "box"))
                self._add(Rel("On", "ball", "table"))
        # else: no-op

    def step(self, ev: str = None):
        if ev is None:
            ev = self.rng.choice(["open_box", "close_box", "move_ball",
                                  "lift_ball", "take_out", "idle"])
        self.events.append(ev)
        # 清除上一步的瞬时动作谓词（Move / Did），保持状态干净
        self.state = {q for q in self.state
                      if not (q.kind == "predicate" and q.name in ("Move", "Did"))}
        # 把“本步发生的事件”作为可观察谓词注入状态
        self._add(Pred("Did", ev))
        self._apply_event(ev)
        self._snapshot()
        return ev

    def run(self, n_steps: int) -> List[Set[Proposition]]:
        for _ in range(n_steps):
            self.step()
        return self.history

    # ---- 给系统的“目标谓词” ----
    @staticmethod
    def predict_next_goal() -> Proposition:
        return Atom("predict_next")

    @staticmethod
    def maintain_goal() -> Proposition:
        return Atom("maintain")

    # ---- 评估系统提出的规则是否真的成立（用全历史，含未见过的部分）----
    @staticmethod
    def ground_truth_check(rule: Proposition, history: List[Set[Proposition]]) -> Tuple[bool, int, int]:
        """对蕴含 A->B：共现语义——存在状态同时含 A,B，且没有状态含 A 但不含 B。
        返回 (是否成立, 支持数, 反例数)。"""
        if rule.kind != "implies":
            appeared = any(rule in s for s in history)
            return appeared, sum(1 for s in history if rule in s), 0
        a, b = rule.parts
        sup = 0
        ref = 0
        for s in history:
            if a in s:
                if b in s:
                    sup += 1
                else:
                    ref += 1
        holds = (ref == 0 and sup > 0)
        return holds, sup, ref

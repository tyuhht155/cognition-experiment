"""TemporalPredictionState：严格前向的时间预测状态。

核心约束（从数据结构保证）：
  - prediction 不能在创建它的 observation step 内被验证。
  - created_at < resolved_at 必然成立。
  - confirmation 只能来自 created_at 之后的 observation。

流程：
  t:   观察 A → 注册 prediction(A→B, trigger=A, expected=B)
  t+1: 环境产生新状态 → 检查 B → 更新 prediction feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TemporalPrediction:
    prediction_id: int
    proposition: str           # 命题字符串（如 "A → B"）
    trigger_observation: str   # 触发观察（前件 A）
    expected_next_observation: str  # 预期下一观察（后件 B）
    created_at: int            # 创建时间步
    resolved_at: Optional[int] = None  # 解决时间步（必须 > created_at）
    pending: bool = True
    confirmed: int = 0
    refuted: int = 0
    resolution_history: List[dict] = field(default_factory=list)

    def resolve(self, step: int, observed: bool) -> None:
        """在 step 时刻根据观察结果解决此预测。

        约束：step 必须 > created_at（不能在创建当步验证）。
        """
        if step <= self.created_at:
            return  # 拒绝在创建当步验证
        if not self.pending:
            return
        self.resolved_at = step
        self.pending = False
        if observed:
            self.confirmed += 1
        else:
            self.refuted += 1
        self.resolution_history.append({
            "step": step,
            "observed": observed,
            "confirmed": self.confirmed,
            "refuted": self.refuted,
        })

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "proposition": self.proposition,
            "trigger": self.trigger_observation,
            "expected": self.expected_next_observation,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "pending": self.pending,
            "confirmed": self.confirmed,
            "refuted": self.refuted,
        }


class TemporalPredictionState:
    """管理所有时间预测的状态。

    关键不变量：
      1. 任何 prediction 的 resolved_at（若存在）> created_at
      2. prediction 不能用创建时的 observation 作为 confirmation
      3. resolution 只能在新 observation 到达后触发
    """

    def __init__(self):
        self._predictions: Dict[int, TemporalPrediction] = {}
        self._next_id: int = 0
        self._pending_queue: List[int] = []  # 待解决的 prediction_id

    def register(self, proposition: str, trigger: str, expected: str,
                 current_step: int) -> TemporalPrediction:
        """在 current_step 注册一个预测。"""
        pid = self._next_id
        self._next_id += 1
        pred = TemporalPrediction(
            prediction_id=pid,
            proposition=proposition,
            trigger_observation=trigger,
            expected_next_observation=expected,
            created_at=current_step,
        )
        self._predictions[pid] = pred
        self._pending_queue.append(pid)
        return pred

    def resolve_pending(self, current_step: int, current_observation: set) -> List[dict]:
        """在新 observation 到达后解决所有 pending 预测。

        current_observation 必须是 current_step 的观察（新状态），
        不能是注册时的观察。
        """
        resolved = []
        still_pending = []
        for pid in self._pending_queue:
            pred = self._predictions[pid]
            if current_step <= pred.created_at:
                still_pending.append(pid)  # 还不能解决
                continue
            observed = pred.expected_next_observation in current_observation
            pred.resolve(current_step, observed)
            resolved.append(pred.to_dict())
        self._pending_queue = still_pending
        return resolved

    def get(self, pid: int) -> Optional[TemporalPrediction]:
        return self._predictions.get(pid)

    def all_predictions(self) -> List[TemporalPrediction]:
        return list(self._predictions.values())

    def total_confirmed(self) -> int:
        return sum(p.confirmed for p in self._predictions.values())

    def total_refuted(self) -> int:
        return sum(p.refuted for p in self._predictions.values())

    def pending_count(self) -> int:
        return len(self._pending_queue)

    def check_invariant(self) -> bool:
        """验证核心不变量：所有已解决预测的 resolved_at > created_at。"""
        for p in self._predictions.values():
            if p.resolved_at is not None and p.resolved_at <= p.created_at:
                return False
        return True

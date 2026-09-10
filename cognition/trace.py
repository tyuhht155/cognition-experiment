"""计算轨迹记录。

理论要求：“每一步保存完整计算轨迹”：
  当前对象 → 候选变换 → 新对象 → 验证 → 评价 → 是否保留 → 是否继续 → 最终结果

所有核心计算过程必须可追踪、可复现。这里只做被动记录，不做决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional
import json
import uuid


def _short(obj: Any) -> str:
    if obj is None:
        return "None"
    return str(obj)


@dataclass
class Step:
    """单个计算步骤。"""
    step_id: str
    compute_id: str           # 属于哪次 recursive_compute 调用
    depth: int                # 递归深度
    parent_step: Optional[str]
    object_in: str            # 输入对象（命题字符串）
    operation: str            # 使用的变换名（可为 "identify"/"verify"/"evaluate"）
    object_out: str           # 输出对象
    verification: Optional[str] = None    # 验证方法
    verification_result: Optional[str] = None  # valid/invalid/unknown
    confidence: Optional[float] = None
    usefulness: Optional[float] = None
    decision: Optional[str] = None        # retain/stop/continue/prune
    cost: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TraceRecorder:
    """一次实验的完整轨迹集合。

    ComputeEngine 只调用 record(...)，不理解内部数据结构。
    """

    def __init__(self):
        self.steps: List[Step] = []

    def new_compute_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def record(self,
               compute_id: str,
               depth: int,
               object_in: Any,
               operation: str,
               object_out: Any,
               parent_step: Optional[str] = None,
               verification: Optional[str] = None,
               verification_result: Optional[str] = None,
               confidence: Optional[float] = None,
               usefulness: Optional[float] = None,
               decision: Optional[str] = None,
               cost: float = 0.0,
               meta: Optional[dict] = None) -> Step:
        step = Step(
            step_id=uuid.uuid4().hex[:8],
            compute_id=compute_id,
            depth=depth,
            parent_step=parent_step,
            object_in=_short(object_in),
            operation=operation,
            object_out=_short(object_out),
            verification=verification,
            verification_result=verification_result,
            confidence=confidence,
            usefulness=usefulness,
            decision=decision,
            cost=cost,
            meta=meta or {},
        )
        self.steps.append(step)
        return step

    def to_dicts(self) -> List[dict]:
        return [s.to_dict() for s in self.steps]

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(), f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.steps)

    def summary(self) -> dict:
        ops = {}
        decisions = {}
        for s in self.steps:
            ops[s.operation] = ops.get(s.operation, 0) + 1
            if s.decision:
                decisions[s.decision] = decisions.get(s.decision, 0) + 1
        return {"steps": len(self.steps), "operations": ops, "decisions": decisions}


# 兼容旧代码
Trace = TraceRecorder

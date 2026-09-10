"""知识库 KnowledgeStore。

每条知识至少记录：
  proposition, status, confidence, usefulness, usage_count, success_count,
  source, parent_objects, operation, verification_method, verification_result, cost

重要约束（理论第 6/9 条）：
  - 不要因为 usefulness 很低就删除 proposition。正确但暂时无用的知识只降低调用优先级。
  - 验证方法本身也作为知识进入 KnowledgeStore（第 7 条）。
  - 知识空间不是预先建立的数据库，而是历史计算不断产生并保留下来的对象/命题/操作/验证结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import json

from .proposition import Proposition


STATUS_VALID = "valid"       # 成立
STATUS_INVALID = "invalid"   # 不成立
STATUS_UNKNOWN = "unknown"   # 未知


@dataclass
class Knowledge:
    """一条知识记录。"""
    proposition: Proposition
    status: str = STATUS_UNKNOWN
    confidence: float = 0.0        # 置信度 [0,1]
    usefulness: float = 0.0        # 历史价值（不删，只调优先级）
    usage_count: int = 0
    success_count: int = 0
    source: str = ""              # 由哪次计算产生（compute_id）
    parent_objects: List[str] = field(default_factory=list)
    operation: str = ""
    verification_method: str = ""
    verification_result: str = ""
    cost: float = 0.0
    kind: str = "proposition"     # proposition / operation / verification_method / composite
    created_step: int = 0         # 创建时的全局步序号

    def priority(self) -> float:
        """调用优先级：置信度 × (价值+使用奖励)。低 usefulness 不删，只降低优先级。"""
        value = self.usefulness + 0.1 * self.success_count - 0.05 * (self.usage_count - self.success_count)
        return max(0.0, self.confidence * max(0.0, value))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["proposition"] = self.proposition.to_str()
        return d


class KnowledgeStore:
    """知识空间。以 Proposition 为键（可哈希）。"""

    def __init__(self):
        self._entries: Dict[Proposition, Knowledge] = {}
        self._ops: Dict[str, Knowledge] = {}        # 可调用操作（含压缩产生的新操作）
        self._step_counter: int = 0
        # 验证方法可靠性统计：name -> (uses, successes)
        self._verifier_stats: Dict[str, Tuple[int, int]] = {}

    # ---- 基本计数 ----
    def tick(self) -> int:
        self._step_counter += 1
        return self._step_counter

    @property
    def now(self) -> int:
        return self._step_counter

    # ---- 命题知识 ----
    def has(self, prop: Proposition) -> bool:
        return prop in self._entries

    def get(self, prop: Proposition) -> Optional[Knowledge]:
        return self._entries.get(prop)

    def upsert(self, k: Knowledge) -> None:
        """新增或更新。低 usefulness 不删除。"""
        k.created_step = self._step_counter if not k.created_step else k.created_step
        existing = self._entries.get(k.proposition)
        if existing:
            # 合并：保留历史计数，更新状态字段
            existing.status = k.status
            existing.confidence = k.confidence
            existing.usefulness = max(existing.usefulness, k.usefulness)
            existing.usage_count += k.usage_count
            existing.success_count += k.success_count
            if k.verification_method:
                existing.verification_method = k.verification_method
                existing.verification_result = k.verification_result
            existing.cost = max(existing.cost, k.cost)
            if not existing.source and k.source:
                existing.source = k.source
        else:
            self._entries[k.proposition] = k

    def all_entries(self) -> List[Knowledge]:
        return list(self._entries.values())

    def valid_entries(self) -> List[Knowledge]:
        return [k for k in self._entries.values() if k.status == STATUS_VALID]

    def invalid_entries(self) -> List[Knowledge]:
        return [k for k in self._entries.values() if k.status == STATUS_INVALID]

    def query_by_status(self, status: str) -> List[Knowledge]:
        return [k for k in self._entries.values() if k.status == status]

    def find_relations(self, name: str) -> List[Knowledge]:
        """按关系名查询（如 On, In）。"""
        return [k for k in self._entries.values()
                if k.proposition.kind == "relation" and k.proposition.name == name]

    # ---- 操作（含压缩产生的新操作）作为知识 ----
    def register_operation(self, name: str, k: Knowledge) -> None:
        self._ops[name] = k
        self._entries.setdefault(k.proposition, k)

    def has_operation(self, name: str) -> bool:
        return name in self._ops

    def operations(self) -> List[str]:
        return list(self._ops.keys())

    # ---- 验证方法作为知识 ----
    def record_verification_outcome(self, method: str, success: bool) -> None:
        uses, succ = self._verifier_stats.get(method, (0, 0))
        uses += 1
        succ += 1 if success else 0
        self._verifier_stats[method] = (uses, succ)

    def verifier_reliability(self, method: str) -> float:
        uses, succ = self._verifier_stats.get(method, (0, 0))
        if uses == 0:
            return 0.5  # 未知，中性
        return succ / uses

    def verifier_stats(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._verifier_stats)

    # ---- 统计 ----
    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        valid = sum(1 for k in self._entries.values() if k.status == STATUS_VALID)
        invalid = sum(1 for k in self._entries.values() if k.status == STATUS_INVALID)
        unknown = sum(1 for k in self._entries.values() if k.status == STATUS_UNKNOWN)
        # 正确但无用（usefulness 很低）
        valid_low_use = sum(1 for k in self._entries.values()
                            if k.status == STATUS_VALID and k.usefulness < 0.05)
        return {
            "total": len(self._entries),
            "valid": valid,
            "invalid": invalid,
            "unknown": unknown,
            "valid_low_usefulness": valid_low_use,
            "operations": len(self._ops),
            "verifier_methods": len(self._verifier_stats),
        }

    def dump(self, path: str) -> None:
        out = {
            "entries": [k.to_dict() for k in self._entries.values()],
            "operations": list(self._ops.keys()),
            "verifier_stats": {m: list(v) for m, v in self._verifier_stats.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

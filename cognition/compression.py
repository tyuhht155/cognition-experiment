"""知识压缩：把反复成功的计算路径压缩成新的可调用操作。

理论第 13 条：A→B→C、A→B→D、A→E→C …… 发现重复结构。
若 operation1 + operation2 + operation3 反复有效，允许系统把它形成新的计算对象/新操作，
然后验证新操作，成功则加入知识空间可直接调用。
不要把“抽象”写成独立模块——它只是：多个已有结构 -> 产生新对象 -> 验证 -> 保留。
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Optional

from .proposition import Proposition
from .knowledge_store import KnowledgeStore, Knowledge, STATUS_VALID
from .operations import OperationRegistry, Candidate, Context
from .trace import Trace


def find_repeated_sequences(trace: Trace, min_len: int = 2, min_count: int = 2) -> List[Tuple[Tuple[str, ...], int]]:
    """在 trace 中按 compute_id 的应用步骤序列挖掘频繁操作序列。

    返回 [(op_sequence, 出现次数)]，按长度优先、频次降序。
    """
    # 收集每个 compute_id 内的“应用”操作序列（按记录顺序）
    per_compute: defaultdict = defaultdict(list)
    # 只取真正的变换应用，排除所有元步骤
    META_OPS = {"identify", "generate_candidates", "verify", "evaluate",
                "evaluate_rank", "evaluate_gate", "no_verify_save", "reuse_known"}
    for s in trace.steps:
        if s.operation in META_OPS:
            continue
        per_compute[s.compute_id].append(s.operation)

    # 构造所有子序列，统计频次
    seq_count: defaultdict = defaultdict(int)
    for cid, ops in per_compute.items():
        seen = set()
        for i in range(len(ops)):
            for L in range(min_len, len(ops) - i + 1):
                seq = tuple(ops[i:i + L])
                if seq in seen:
                    continue
                seen.add(seq)
                seq_count[seq] += 1

    frequent = [(seq, c) for seq, c in seq_count.items() if c >= min_count]
    # 长序列优先；同长按频次降序
    frequent.sort(key=lambda x: (-len(x[0]), -x[1]))
    return frequent


def compress(trace: Trace, registry: OperationRegistry, store: KnowledgeStore,
             ctx: Context, top_k: int = 5) -> int:
    """发现频繁序列 -> 注册为复合操作 -> 验证 -> 保留。

    返回成功新增的操作数。
    """
    frequent = find_repeated_sequences(trace, min_len=2, min_count=2)
    added = 0
    for seq, count in frequent[:top_k]:
        op_name = "composite_" + "_".join(seq)
        if registry.has(op_name) if hasattr(registry, "has") else op_name in registry.names():
            continue
        # 构造复合操作：依次应用序列中的每个操作
        maker = _make_composite_op(seq, registry)
        registry.register(op_name, maker, is_prior=False)

        # 把新操作作为知识对象记录并“验证”：统计该序列过去有多少次导致 valid 结果
        # 用 trace 中该序列对应记录的 decision 来判定
        valid_rate = _sequence_valid_rate(trace, seq, store)
        status = STATUS_VALID if valid_rate >= 0.5 else "unknown"
        k = Knowledge(
            proposition=Proposition.atom(op_name),  # 操作作为命题对象
            status=status,
            confidence=valid_rate,
            usefulness=float(count) / 10.0,
            source="compression",
            parent_objects=list(seq),
            operation="compression",
            verification_method="trace_replay",
            verification_result="valid" if status == STATUS_VALID else "unknown",
            cost=0.0,
            kind="operation",
        )
        store.register_operation(op_name, k)
        added += 1
    return added


def _make_composite_op(seq: Tuple[str, ...], registry: OperationRegistry):
    """构造复合操作函数：对对象依次应用 seq 中的每个操作，收集所有候选。"""
    def fn(obj: Proposition, ctx: Context) -> List[Candidate]:
        results: List[Candidate] = []
        current = [obj]
        for op_name in seq:
            op_fn = registry._ops.get(op_name)
            if op_fn is None:
                return []
            next_objs = []
            for o in current:
                for c in op_fn(o, ctx):
                    results.append(Candidate(c.new_object, op_name, c.cost,
                                             {**c.meta, "composite": "_".join(seq)}))
                    next_objs.append(c.new_object)
            current = next_objs[:3]  # 限制组合宽度
            if not current:
                break
        return results[:5]
    return fn


def _sequence_valid_rate(trace: Trace, seq: Tuple[str, ...], store: KnowledgeStore) -> float:
    """回放统计：该操作序列产生的对象中，被验证为 valid 的比例。"""
    total = 0
    valid = 0
    # 找 trace 中连续出现 seq 的位置，看其 object_out 是否在 store 中为 valid
    steps = trace.steps
    n = len(seq)
    for i in range(len(steps) - n + 1):
        window = tuple(s.operation for s in steps[i:i + n])
        if window != seq:
            continue
        last_obj_str = steps[i + n - 1].object_out
        # 反查 store（按 to_str 匹配，效率低但够用于实验）
        for k in store.all_entries():
            if k.proposition.to_str() == last_obj_str:
                total += 1
                if k.status == STATUS_VALID:
                    valid += 1
                break
    return valid / total if total else 0.0

"""统一命题结构 Proposition。

设计原则：
- 一个递归的、可哈希的、可序列化的数据结构，覆盖题目要求的所有形式：
  原子 A、否定 ¬A、合取 ∧、析取 ∨、蕴含 →、等价 ↔、二元关系 R(A,B)、
  一元谓词 P(A)、全称 ∀x P(x)、存在 ∃x P(x)，并允许任意嵌套。
- 这些形式只是“当前对象可被改变的结构”，不是写死的知识。
- 仅极少先验：结构本身 + 构造能力。没有任何领域规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional


# 项（term）：这里用字符串表示常量或变量。变量约定以 '?' 开头。
# 这样可以在量化命题里区分绑定变量与常量，但不强制类型系统。
Term = str


@dataclass(frozen=True)
class Proposition:
    """统一命题对象。

    内部用 (kind, name, parts) 三元组表达一切：
      - kind: 命题类型标签
      - name: 谓词名 / 关系名 / 量化变量名 / 空字符串
      - parts: 子成分元组。对于逻辑联结词，元素是 Proposition；
               对于谓词/关系，元素是 Term(字符串)；
               对于量化，parts[0] 是命题体 Proposition。
    """

    kind: str
    parts: Tuple = field(default_factory=tuple)
    name: str = ""

    # ---- 构造器：题目要求的所有形式 ----
    @staticmethod
    def atom(name: str) -> "Proposition":
        """A：原子命题。"""
        return Proposition(kind="atom", parts=(), name=name)

    @staticmethod
    def neg(p: "Proposition") -> "Proposition":
        """¬A：否定。"""
        return Proposition(kind="not", parts=(p,))

    @staticmethod
    def conj(a: "Proposition", b: "Proposition") -> "Proposition":
        """A ∧ B：合取。"""
        return Proposition(kind="and", parts=(a, b))

    @staticmethod
    def disj(a: "Proposition", b: "Proposition") -> "Proposition":
        """A ∨ B：析取。"""
        return Proposition(kind="or", parts=(a, b))

    @staticmethod
    def impl(a: "Proposition", b: "Proposition") -> "Proposition":
        """A → B：蕴含。"""
        return Proposition(kind="implies", parts=(a, b))

    @staticmethod
    def iff(a: "Proposition", b: "Proposition") -> "Proposition":
        """A ↔ B：等价。"""
        return Proposition(kind="iff", parts=(a, b))

    @staticmethod
    def relation(name: str, left: Term, right: Term) -> "Proposition":
        """R(A,B)：二元关系。"""
        return Proposition(kind="relation", parts=(left, right), name=name)

    @staticmethod
    def predicate(name: str, *args: Term) -> "Proposition":
        """P(A) 或 P(A,B,...)：n 元谓词。"""
        return Proposition(kind="predicate", parts=tuple(args), name=name)

    @staticmethod
    def forall(var: Term, body: "Proposition") -> "Proposition":
        """∀x P(x)：全称量化。变量存于 name。"""
        return Proposition(kind="forall", parts=(body,), name=var)

    @staticmethod
    def exists(var: Term, body: "Proposition") -> "Proposition":
        """∃x P(x)：存在量化。变量存于 name。"""
        return Proposition(kind="exists", parts=(body,), name=var)

    # ---- 基本访问 ----
    @property
    def is_atom(self) -> bool:
        return self.kind == "atom"

    @property
    def operands(self) -> Tuple["Proposition", ...]:
        """逻辑联结词的子命题。"""
        return tuple(p for p in self.parts if isinstance(p, Proposition))

    def relation_args(self) -> Tuple[Term, ...]:
        return tuple(self.parts)

    # ---- 规范字符串 / 可读性 ----
    def to_str(self) -> str:
        k = self.kind
        if k == "atom":
            return self.name
        if k == "not":
            return f"¬{self.parts[0].to_str()}"
        if k in ("and", "or", "implies", "iff"):
            sym = {"and": "∧", "or": "∨", "implies": "→", "iff": "↔"}[k]
            a, b = self.parts
            return f"({a.to_str()} {sym} {b.to_str()})"
        if k == "relation":
            a, b = self.parts
            return f"{self.name}({a},{b})"
        if k == "predicate":
            args = ",".join(self.parts) if self.parts else ""
            return f"{self.name}({args})" if self.parts else self.name
        if k == "forall":
            return f"∀{self.name} {self.parts[0].to_str()}"
        if k == "exists":
            return f"∃{self.name} {self.parts[0].to_str()}"
        return f"<{k}>"

    def __str__(self) -> str:  # 便于打印
        return self.to_str()

    __repr__ = __str__

    # ---- 结构工具：收集自由变量、子命题等 ----
    def free_vars(self, bound: Optional[set] = None) -> set:
        bound = set(bound) if bound else set()
        k = self.kind
        if k == "atom":
            return set()
        if k in ("not",):
            return self.parts[0].free_vars(bound)
        if k in ("and", "or", "implies", "iff"):
            return self.parts[0].free_vars(bound) | self.parts[1].free_vars(bound)
        if k in ("relation", "predicate"):
            return {t for t in self.parts if isinstance(t, str) and t not in bound}
        if k in ("forall", "exists"):
            inner = self.parts[0].free_vars(bound | {self.name})
            return inner
        return set()

    def sub_propositions(self) -> list:
        """返回所有子命题（含自身），去重保序。"""
        seen = []
        seen_set = set()

        def add(p: Proposition):
            if p not in seen_set:
                seen_set.add(p)
                seen.append(p)
                for sub in p.operands:
                    add(sub)
                if p.kind in ("forall", "exists"):
                    add(p.parts[0])

        add(self)
        return seen

    def substitute_term(self, old: Term, new: Term) -> "Proposition":
        """把项 old 全部替换为 new（不触碰绑定变量名，避免捕获）。"""
        return _subst_term(self, old, new, set())


def _subst_term(p: Proposition, old: Term, new: Term, bound: set) -> Proposition:
    k = p.kind
    if k == "atom":
        return p
    if k == "not":
        return Proposition.neg(_subst_term(p.parts[0], old, new, bound))
    if k in ("and", "or", "implies", "iff"):
        a, b = p.parts
        a2 = _subst_term(a, old, new, bound)
        b2 = _subst_term(b, old, new, bound)
        return Proposition(kind=k, parts=(a2, b2))
    if k == "relation":
        a, b = p.parts
        na = new if a == old and a not in bound else a
        nb = new if b == old and b not in bound else b
        return Proposition.relation(p.name, na, nb)
    if k == "predicate":
        args = tuple(new if t == old and t not in bound else t for t in p.parts)
        return Proposition.predicate(p.name, *args)
    if k in ("forall", "exists"):
        # 避免变量捕获：若新名与绑定变量冲突，先重命名
        var = p.name
        new_bound = bound | {var}
        body = _subst_term(p.parts[0], old, new, new_bound)
        ctor = Proposition.forall if k == "forall" else Proposition.exists
        return ctor(var, body)
    return p


# 便利别名
P = Proposition

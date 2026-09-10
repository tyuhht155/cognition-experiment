"""cognition: 递归计算认知系统的最小可运行实验包。

模块组成（按依赖顺序）：
  proposition     统一命题结构（嵌套/量化/关系）
  trace           计算轨迹记录
  knowledge_store 知识库（含验证方法作为知识）
  operations      候选变换生成器（先验变换集合）
  verification    多方法验证系统
  evaluation      价值评价 + 元评价
  compute         统一递归计算循环 recursive_compute
  environment     人工世界（球/盒子/桌子/墙）
  compression     知识压缩（重复路径 -> 新操作）

先验（理论第 13 条允许的极少先验，全部显式标记，不隐藏）：
  - 基本命题形式       -> proposition.Proposition 的构造器
  - 基本命题变换能力   -> operations.PRIOR_OPERATIONS（is_prior=True）
  - 一个初始目标/评价  -> evaluation.Evaluation（透明启发式）
  - 基本对象           -> 由 environment 通过观察注入，不预写领域规则
"""

from .proposition import Proposition
from .trace import Trace
from .knowledge_store import KnowledgeStore, Knowledge
from .operations import OperationRegistry, Context, Candidate
from .verification import Verification, VerificationResult
from .evaluation import Evaluation, EvaluationResult
from .compute import ComputeEngine
from .environment import World
from . import compression

__all__ = [
    "Proposition", "Trace", "KnowledgeStore", "Knowledge",
    "OperationRegistry", "Context", "Candidate",
    "Verification", "VerificationResult",
    "Evaluation", "EvaluationResult",
    "ComputeEngine", "World", "compression",
]

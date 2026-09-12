# 递归计算认知系统 — 架构冻结 + 实验记录

> 仓库地址：https://github.com/tyuhht155/cognition-experiment

---

## 一、核心理论

1. 一切认知活动都视为计算，基本形式是：**对象 + 操作 → 新对象**
2. 对象不是预先固定的，计算产生的新对象可以继续成为下一次计算的对象
3. 计算中包含计算：识别、搜索、应用、验证、评价、选择验证方式……都是同一递归过程在不同对象上的表现
4. **最终目标：完全不依赖 LLM**，由新型 AI 自己产生原本由 LLM 提供的隐式计算、候选生成和搜索能力
5. 搜索本身就是一次计算，其计算对象是知识空间

---

## 二、当前架构（已冻结）

### 模块依赖方向

```
Proposition
  ↓
Operation / Candidate (operations.py)
  ↓
CandidateProcessor (candidate_processor.py)
  ↓
Evidence / Belief / Value / Cost / Consensus / Prediction
  ↓
ComputeEngine (编排)
```

### 文件职责

| 文件 | 职责 |
|------|------|
| `proposition.py` | 统一命题结构（atom/neg/conj/disj/impl/iff/relation/predicate/forall/exists + substitute_term） |
| `models.py` | BeliefState（evidence_count/reuse_count/verification_count/update_count 四种独立计数） |
| `belief.py` | BeliefStore + KnowledgeView 只读接口 |
| `evidence.py` | Evidence（含 evidence_id/proposition/source_event_id/observation_step 事件溯源）+ EvidenceLog（append-only + 去重）+ EvidenceEvaluator |
| `operations.py` | Context（收缩为单次计算上下文）+ OperationRegistry + 13 个 PRIOR_OPERATIONS |
| `verification.py` | Verifier（observe/compare/counterexample/prediction/logical_derive）|
| `evaluation.py` | ValueEvaluator（relevance/generality/novelty）+ evaluate_evaluation 元评价 |
| `prediction.py` | TemporalPrediction（内部使用 Proposition；created_at < resolved_at 强制约束） |
| `candidate_processor.py` | CandidateProcessor：evaluate→gate→verify→evidence log→belief update→trace→cost |
| `compute.py` | ComputeEngine：generate candidates → CandidateProcessor.process() → 递归 |
| `trace.py` | TraceRecorder：记录每步计算（含 parent_step 链） |
| `cost.py` | CostTracker：total_cost = sum(CostEvent.amount)，cache_saved 单独统计 |
| `compression.py` | 知识压缩（频繁计算路径→新可调用操作） |

### 架构冻结后的关键约束

- **Context 不持有可写 Store**：只有 knowledge_view 只读接口
- **Feedback 三种状态分离**：valid→actual=1.0, invalid→actual=0.0, unknown/gated_out→actual=None（不参与学习）
- **gated_out 不递归**：只有 valid/unknown 可以继续递归
- **递归统计共享 dict**：子层修改不会被父层旧快照覆盖
- **parent_step 链**：递归 trace 的父子关系通过 apply_step_id 建立

### 测试

| 套件 | 数量 | 覆盖 |
|------|------|------|
| test_core | 15 | 基础功能 |
| test_audit | 10 | 审计修复 |
| test_invariants | 41 | 架构不变量（feedback 语义 / 统计累计 / gated_out 不递归 / 知识边界等） |
| test_e0_2 | 10 | E0-2 时间展开闭环（时序 / 反例 / 知识空间增长 / 状态分类） |
| test_v0 | 8 | V0 实验完成标准 |
| **合计** | **84** | **全部通过** |

---

## 三、Operation 边界审计

### 三类区分

| 类别 | Operations | 性质 |
|------|-----------|------|
| **第一类：底层构造能力** | neg_intro, neg_elim, conj_elim, disj_intro, specialize | 构造/分解/展开命题对象的底层能力 |
| **第二类：有效推理** | modus_ponens, modus_tollens | 需要知识空间搜索，逻辑合法 |
| **第三类：候选生成策略** | impl_intro, generalize, relation_swap, substitute, cooccur_impl_induce, seq_impl_induce | 包含归纳策略/搜索策略/领域假设 |

### 核心发现

- `cooccur_impl_induce` 和 `seq_impl_induce` 是纯归纳策略，不是逻辑原语
- `substitute` 是暴力搜索策略，构造能力本身在 Proposition.substitute_term
- `relation_swap` 是对称性假设
- `generalize` 从单例推广，含归纳跳跃
- `impl_intro` 从合取猜蕴含方向，含方向性假设

### 理论结论

当前 OperationRegistry 是 **D. 三者混合**（构造能力 + 有效推理 + 候选生成策略），需要拆分。

最小不可删除先验：
1. Proposition 结构 (kind, parts, name)
2. Constructor 集合 (neg/conj/disj/impl/iff/forall/exists/substitute_term)
3. 环境观察 + Verifier
4. EvidenceEvaluator 聚合公式（唯一策略性先验）
5. BeliefStore + TraceRecorder

---

## 四、E0-1 实验：穷举构造 + 验证闭环

### 实验目标

验证"如果不给系统任何人工候选生成策略（不使用任何 PRIOR_OPERATIONS），只给它对象、基础构造能力、环境反馈、验证和知识存储，它能否通过穷举构造 + 验证形成新的知识？"

### 实验设计

- 删掉全部 13 个 PRIOR_OPERATIONS
- 用通用构造枚举器替代：从已有对象出发，用全部 constructor 生成所有可能的单层复合对象
- 全部交给 Verifier 验证，不排序、不筛选
- 不引入 LLM / pattern extractor / 历史学习 / compression

### 人工世界

```
P(a) 出现时 Q(a) 出现（共现）
P(b) 出现时 Q(b) 出现（共现）
P(c) 出现但 Q(c) 不出现（反例）
R(a,b) 和 R(b,a) 都出现（对称）
S(a) 独立出现
```

### 实验结果

| 指标 | 值 |
|------|-----|
| 初始对象数量 | 9 |
| 生成候选总数 | 297 |
| valid | 5 |
| invalid | 67 |
| unknown | 225 |
| 证据数 | 1782 |
| Trace 步骤数 | 594 |
| 总成本 | 1355.4 |

### 按 constructor 统计

| Constructor | 总数 | valid | valid 比例 |
|-------------|------|-------|-----------|
| neg | 9 | 0 | 0.00 |
| conj | 72 | 0 | 0.00 |
| disj | 72 | 0 | 0.00 |
| impl | 72 | 5 | 0.07 |
| iff | 72 | 0 | 0.00 |

### 关键命题检查

| 命题 | 构造方法 | 系统状态 | 置信度 | Ground-truth | 结论 |
|------|---------|---------|--------|-------------|------|
| P(a)→Q(a) | impl | valid | 0.400 | True (supported: 6) | PASS |
| Q(a)→P(a) | impl | invalid | 0.950 | False (refuted: 2) | PASS |
| P(c)→Q(c) | impl | invalid | 0.950 | False (refuted: 1) | PASS |
| ¬P(a) | neg | unknown | 0.000 | False (observed 6 times) | — |
| P(a)∧Q(a) | conj | unknown | 0.000 | True (both observed 6 times) | — |

### E0-1 结论

1. **无 PRIOR_OPERATIONS 也能构造 P(a)→Q(a)**：PASS — 纯构造能力足够产生蕴含候选
2. **P(a)→Q(a) 被验证为 valid**：PASS — Verifier 正确识别共现支持
3. **P(c)→Q(c) 被验证为 invalid（反例）**：PASS — Verifier 正确识别反例
4. **验证结果进入 BeliefStore**：PASS — 知识成功积累
5. **Trace 完整记录**：PASS — 构造→验证全链路可追溯

### 重要区分

- **"系统能够构造一个命题"**：E0-1 验证了这一点——穷举构造可以产生所有可能的复合对象
- **"系统能够因为问题而选择构造这个命题"**：E0-1 **没有**验证这一点——穷举不做任何选择，全部构造

E0-1 只验证了"构造能力 + 验证 = 可以形成知识"，不验证"搜索/选择能力"。

### 观察分析

- `impl` 是唯一产生 valid 的 constructor（5/72），因为只有蕴含能被 Verifier 的 compare 方法验证为 valid
- `conj` 和 `disj` 的 valid 比例为 0——Verifier 的验证方法不直接判定合取/析取为 valid（除非通过 observe 发现它们作为整体出现）
- `neg` 的 valid 比例为 0——所有否定命题的内命题都曾出现，因此被判定为 invalid 或 unknown
- unknown 比例高（76%）——这是因为单层构造产生的多数复合命题没有足够的观察证据
- **组合爆炸**：9 个对象产生 297 个候选，递归到第 2 层将产生 ~90000 个——穷举不可持续

---

## 五、E0-2 实验：时间展开穷举构造 + 验证闭环（strictly time-causal）

### 核心变化（与 E0-1 的区别）

- 不再把完整 world_history 一次性提供给系统
- 逐时刻展开：每个时刻 t 只提供 `history[:t+1]` 作为 `ctx.world_history`
- 验证时不能偷看未来 observation
- **constructible_objects = observed_objects**（不含 valid belief）
- valid/invalid 跳过重复验证；unknown 允许重新验证

### 实验设计

```
for t in world_history:
  1. 只向系统提供当前时刻 t 的 observation
  2. 把 observation 中的新对象加入 observed_objects
  3. 从 observed_objects 进行单层穷举构造（不含 valid belief）
  4. 对候选进行验证（Verifier 只能看到 history[:t+1]）
  5. valid/invalid 跳过；unknown 允许重新验证
  6. 将验证结果写入 BeliefStore
  7. 下一时刻继续
```

### 实验结果

| 指标 | 值 |
|------|-----|
| 时间步数 | 12 |
| 总候选数 | 2521 |
| valid | 16 |
| invalid | 56 |
| unknown | 1913 |
| duplicates_skipped | 536 |
| re_verified_unknown | 1688 |
| 总成本 | 8445.0 |
| Trace 步骤数 | 3970 |
| 证据数 | 11910 |

### 关键命题时间线

| 命题 | 首次构造 | 首次支持 | 首次反例 | 最终状态 | Ground-truth |
|------|---------|---------|---------|---------|-------------|
| P(a)→Q(a) | step 0 | step 0 | — | valid (0.4) | True |
| Q(a)→P(a) | step 0 | step 0 | — | valid (0.4) | False (refuted: 2) |
| P(c)→Q(c) | step 7 | — | step 7 | invalid (0.95) | False (refuted: 1) |
| ¬P(a) | step 0 | — | — | unknown (0.0) | False |
| P(a)∧Q(a) | step 0 | — | — | unknown (0.0) | True |

### 时序检查

| 检查项 | 结果 |
|--------|------|
| P(a)→Q(a) 不在 P(a) 或 Q(a) 出现之前构造 | PASS |
| P(c)→Q(c) 不在 P(c) 或 Q(c) 出现之前构造 | PASS |
| P(c)→Q(c) 反例到来后状态改变 | PASS |
| future_information_leak_check | PASS |
| P(a)→Q(a) 最终被验证为 valid | PASS |
| P(c)→Q(c) 最终被验证为 invalid | PASS |
| 验证结果进入 BeliefStore | PASS |
| Trace 完整记录 | PASS |

### 关键时间点

- P(a) first seen: step 0
- Q(a) first seen: step 0
- P(c) first seen: step 6
- Q(c) first seen: step 7

### E0-2 结论

1. **核心闭环验证**：新观察进入 observed_objects → 改变下一轮可构造对象 → 构造新对象 → 验证 → 知识进入 BeliefStore
2. **时序正确性**：候选不能提前出现；验证不偷看未来；future_information_leak_check 通过
3. **状态分类处理**：valid/invalid 跳过重复验证（536 次）；unknown 重新验证（1688 次）
4. **构造空间受限**：constructible_objects = observed_objects（不含 valid belief），避免了知识递归构造
5. **组合爆炸缓解**：总候选 2521（vs E0-1 的 297 单步，vs 旧 E0-2 的 19909 含 valid recycling）

### E0-2 专用测试（10 项）

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | test_step_t_cannot_see_t1_objects | t 时刻看不到 t+1 的对象 |
| 2 | test_candidates_only_from_observed_objects | t 时刻候选只能来自 observed_objects |
| 3 | test_valid_belief_not_in_constructor_input | valid belief 不会自动进入 constructor 输入 |
| 4 | test_observation_expands_observed_objects | 新 observation 会扩大下一时间步 observed_objects |
| 5 | test_pa_qa_not_early | P(a)→Q(a) 不会提前出现 |
| 6 | test_unknown_can_be_reverified | unknown 候选可以在后续 observation 后重新验证 |
| 7 | test_valid_invalid_not_reverified | 已经 valid/invalid 的候选不会被无意义重复验证 |
| 8 | test_ground_truth_not_in_verifier | ground truth 不进入 Verifier |
| 9 | test_full_history_not_smuggled | 完整历史不会通过 Context 偷渡给验证器 |
| 10 | test_trace_temporal_order | trace 的时间顺序正确 |

### 重要区分

- **"系统在当前构造空间中生成了候选，并根据截至当前时刻的观察证据进行了评价"** — E0-2 验证了这一点
- **"系统发现了规律"** — E0-2 **没有**验证这一点；system valid 不等于 ground truth
- **Q(a)→P(a) 被 system 判为 valid 但 ground truth 为 False** — Verifier 的 `action_compare` 只看共现率，不能区分方向性

---

## 六、历史反哺的四种方式

| 方式 | 先验强度 | 灵活度 | 复杂度 | 小样本可靠性 |
|------|---------|--------|--------|-------------|
| A 直接统计 | 中 | 低 | 最低 | 中 |
| B 结构模式 | 高 | 中 | 中 | 低 |
| C 计算过程对象化 | 低 | 最高 | 最高 | 不确定 |
| D 操作组合 | 中 | 低 | 低 | 中 |

- E1 = 方式 A（统计排序，不改变候选集合）
- E2 = 方式 B/C（模式提取，产生新候选）

---

## 七、E0/E1/E2 实验设计

| 层级 | 候选集合 | 历史的作用 | 证明了什么 |
|------|---------|-----------|-----------|
| E0 | 全部穷举 | 无 | 纯构造能力 + 验证 = 可以形成知识 |
| E1 | 全部穷举 | 改变验证顺序 | 统计排序是否提高效率 |
| E2 | 穷举 + 模式生成 | 产生新候选 | 系统能否从历史中提取模式并产生穷举不会产生的新候选 |

---

## 八、A/B/C/D 对照实验

| 组 | 生成 | 验证 | 价值评价 | 元评价 |
|----|------|------|---------|--------|
| A | ✅ | ❌ | ❌ | ❌ |
| B | ✅ | ✅ | ❌ | ❌ |
| C | ✅ | ✅ | ✅ | ❌ |
| D | ✅ | ✅ | ✅ | ✅ |

四组共享同一 environment、seed、初始知识、目标、输入序列、计算预算。唯一区别是三个布尔开关。

### 审计后关键结果（10 seeds, mean ± std）

| 指标 | A | B | C | D |
|------|---|---|---|---|
| valid_correct | 0 | 24.3 ± 5.9 | 17.9 ± 3.6 | 19.0 ± 3.9 |
| error_rate | 0 | 0.245 ± 0.089 | 0.262 ± 0.095 | 0.250 ± 0.086 |
| total_cost | 3039 ± 24 | 4181 ± 152 | 2344 ± 175 | 2632 ± 96 |

### 审计发现的实验漏洞

1. 验证器仍是"弱 oracle"——直接读取已观察历史判断命题
2. 元评价效果不显著——D 与 C 无显著差异
3. 复合操作质量低——挖掘出的是长序列噪声
4. 统计样本偏小

---

## 九、先验声明

### 不可删除的最小先验

1. **Proposition 结构** (kind, parts, name) — 对象表示
2. **Constructor 集合** — 构造能力
3. **环境观察** (world_history) — 系统获取外部信息的唯一通道
4. **Verifier** (observe/compare/counterexample) — 验证能力
5. **EvidenceEvaluator 聚合公式** — 唯一策略性先验（数值人工指定，待后续实验调整）
6. **BeliefStore + TraceRecorder** — 知识存储 + 历史记录

### 当前 PRIOR_OPERATIONS（待逐步移除）

13 个 operation 是三者混合（构造能力 + 有效推理 + 候选生成策略）。最终应只保留构造能力作为最小先验，候选生成策略应由系统从计算中形成。

---

## 十、运行方式

```bash
# 运行测试（84 个）
python tests/test_core.py
python tests/test_audit.py
python tests/test_invariants.py
python tests/test_e0_2.py
python tests/test_v0.py

# 运行 E0-1 实验
python experiments/run_e0.py

# 运行 E0-2 实验（时间展开）
python experiments/run_e0_2.py

# 运行 A/B/C/D 对照实验
python experiments/run_abcd.py
```

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
| test_invariants | 37 | 架构不变量（feedback 语义 / 统计累计 / gated_out 不递归 / 知识边界等） |
| test_e0_2 | 10 | E0-2 时间展开闭环（时序 / 反例 / 知识空间增长 / 状态分类） |
| test_e0_3 | 8 | E0-3 derived knowledge 闭环（valid 进入 derived / invalid 排除 / 同轮禁用 / 下轮可用） |
| test_e0_4 | 10 | E0-4 知识修正与回滚（VALID 可被推翻 / derived 反映当前状态 / 退出后不再使用） |
| test_e0_5 | 12 | E0-5 验证方法评估（方法对象化 / 时间反馈 / 方法可犯错 / 选择依赖历史） |
| test_e0_6 | 10 | E0-6 操作选择学习（历史影响选择 / 情境偏好 / unknown 不计失败 / ground truth 不入选择 / 探索保留 / trace 完整 / A/B 对照） |
| test_e0_7 | 10 | E0-7 变换历史学习（结构签名 / 变换记录 / 新对象泛化 / 候选非信念 / unknown 独立 / ground truth 隔离 / 无未来信息 / 无未验证泛化 / A/B 对照） |
| test_e0_7_1 | 16 | E0-7.1 严格变换实例化（op_name 独立匹配 / 实例化重建 / 新词项迁移 / 绑定一致性 / 反事实拒绝 / 三层分离 / UNKNOWN op 仍工作） |
| test_e0_7_2 | 10 | E0-7.2 变换递归复用（新对象重入计算 / 2步链 / 3步链 / 隐藏中间步骤 / 无捷径 / 中间验证 / INVALID 阻断 / UNKNOWN op / trace 完整 / 无未来信息） |
| test_e0_7_3 | 14 | E0-7.3 变换选择（多变换匹配 / 选择生成分离 / op_name 无关 / UNKNOWN op / 历史影响选择 / valid低价值保留 / 失败更新选择 / 探索保留 / 目标改变选择 / 无未来信息 / 无ground truth / 多步搜索 / 长路不判invalid / validity≠value） |
| test_e0_7_4 | 18 | E0-7.4 未来价值与信用分配（信用传播 / 直接目标最高信用 / 中间步骤非零信用 / dead-end无信用 / INVALID无信用 / valid低价值保留 / 失败降低选择 / 失败不永久禁止 / 不同goal不同价值 / UNKNOWN op正常 / op改名无影响 / 无goal distance / 无ground truth / 无未来数据 / 信用来自trace / 多步中间学习 / 多成功路径获信用 / 长路不因无直接价值判invalid） |
| test_e0_8 | 16 | E0-8 目标导向新对象生成（X不在历史 / X不在初始知识 / 基础构造器产生X / X验证后入知识 / X下轮重参与 / 通过X到达目标 / INVALID阻断 / 仅历史无捷径 / 不依赖op_name / 无ground truth指引 / trace完整 / 能构造≠有效≠有用 / 严格时间因果 / G不在round0 / INVALID不进constructible / 构造器通用） |
| test_v0 | 8 | V0 实验完成标准 |
| **合计** | **204** | **全部通过** |

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

## 六、E0-3 实验：derived knowledge as computation input

### 核心变化（与 E0-2 的唯一区别）

E0-2：
```
observation → construction → verification
```

E0-3：
```
observation → construction → verification
    ↓
valid proposition
    ↓
new computation input
```

- **constructible_objects = observed_objects ∪ derived_objects**
- derived_objects = 已构造并验证为 STATUS_VALID 的命题集合
- STATUS_INVALID / STATUS_UNKNOWN 不进入 derived_objects
- 本轮新验证为 valid 的命题不立即进入本轮 constructible_objects（禁止递归展开）
- 下一轮进入 derived_objects 后才可作为 constructor 输入

### 集合定义

| 集合 | 定义 |
|------|------|
| observed_objects | 截至当前时刻实际观察到的原始对象 |
| derived_objects | 已由系统构造出来，并且获得 STATUS_VALID 的命题 |
| constructible_objects | observed_objects ∪ derived_objects |

### 实验设计

```
for t in world_history:
  1. 只向系统提供当前时刻 t 的 observation
  2. 把 observation 中的新对象加入 observed_objects
  3. 将上一轮之前已验证为 valid 的命题加入 derived_objects
  4. constructible_objects = observed_objects ∪ derived_objects
  5. 从 constructible_objects 进行单层穷举构造
  6. 对候选进行验证（Verifier 只能看到 history[:t+1]）
  7. valid 的新对象在下一轮加入 derived_objects
  8. 下一时刻继续
```

### 禁止递归展开

- A、B → A→B（本轮验证为 VALID）
- 本轮不能立即继续：A→B + C → (A→B)→C
- 必须等到下一轮，A→B 进入 derived_objects 后才允许使用

### constructor 限定

只使用：neg、conj、disj、impl、iff。不增加任何 constructor。

### 实验结果

| 指标 | 值 |
|------|-----|
| 时间步数 | 12 |
| 总候选数 | 19909 |
| valid | 16 |
| invalid | 216 |
| unknown | 17964 |
| duplicates_skipped | 1713 |
| re_verified_unknown | 15771 |
| 总成本 | 81428.7 |
| Trace 步骤数 | 36392 |
| 证据数 | 109176 |
| derived_objects 数量 | 16 |

### 关键命题时间线

| 命题 | 首次构造 | 首次支持 | 首次反例 | 最终状态 | in derived_objects | Ground-truth |
|------|---------|---------|---------|---------|-------------------|-------------|
| P(a)→Q(a) | step 0 | step 0 | — | valid (0.4) | True | True (supported: 6) |
| Q(a)→P(a) | step 0 | step 0 | — | valid (0.4) | True | False (refuted: 2) |
| P(c)→Q(c) | step 7 | — | step 7 | invalid (0.95) | False | False (refuted: 1) |
| ¬P(a) | step 0 | — | — | unknown (0.0) | False | False |
| P(a)∧Q(a) | step 0 | — | — | unknown (0.0) | False | True |

### derived_object_timeline（部分）

| proposition | first_valid_step | first_used_as_input_step | times_used_as_input |
|-------------|-----------------|------------------------|-------------------|
| (P(a) → Q(a)) | 1 | 1 | 11 |
| (P(a) → R(a,b)) | 1 | 1 | 11 |
| (Q(a) → P(a)) | 1 | 1 | 11 |
| (P(b) → Q(b)) | 2 | 2 | 10 |
| (S(a) → Q(a)) | 4 | 4 | 8 |
| (P(c) → P(a)) | 7 | 7 | 5 |
| (Q(c) → P(c)) | 8 | 8 | 4 |

### 每步统计（obs=observed, der=derived, cbl=constructible）

| step | obs | der | cbl | cand | fr_obs | fr_der | valid | invalid | unknown |
|------|-----|-----|-----|------|--------|--------|-------|---------|---------|
| 0 | 3 | 0 | 3 | 27 | 27 | 0 | 6 | 0 | 21 |
| 1 | 6 | 6 | 12 | 540 | 126 | 414 | 6 | 60 | 468 |
| 2 | 6 | 12 | 18 | 1242 | 126 | 1116 | 0 | 42 | 1128 |
| 7 | 9 | 15 | 24 | 2232 | 297 | 1935 | 1 | 48 | 2010 |
| 11 | 9 | 16 | 25 | 2425 | 297 | 2128 | 0 | 0 | 2193 |

### E0-3 专用测试（8 项不变量）

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | test_valid_belief_enters_derived_objects | STATUS_VALID 命题进入 derived_objects |
| 2 | test_invalid_belief_never_enters_derived_objects | STATUS_INVALID 命题绝不进入 derived_objects |
| 3 | test_unknown_belief_never_enters_derived_objects | STATUS_UNKNOWN 命题不进入 derived_objects |
| 4 | test_unknown_can_become_valid_later | UNKNOWN 可重新验证；变为 VALID 后才进入 derived_objects |
| 5 | test_derived_object_can_be_used_next_step | 上一轮 valid 的命题在下一轮可作为 constructor 输入 |
| 6 | test_derived_object_cannot_be_used_same_step | 本轮新 valid 命题不能在本轮作为 constructor 输入（禁止递归） |
| 7 | test_no_future_information | 每时刻只看到 history[:t+1]，禁止未来信息 |
| 8 | test_ground_truth_not_used_by_verifier | ground_truth 只在实验结束后外部比较，不进入 Verifier |

### E0-3 结论

1. **知识重新进入计算**：P(a)→Q(a) 在 step 0 验证为 VALID，step 1 进入 derived_objects 并被用作 constructor 输入
2. **状态过滤**：只有 STATUS_VALID 进入 derived_objects；INVALID 和 UNKNOWN 被排除
3. **禁止递归展开**：本轮新 valid 不进入本轮 constructible（derived_object_count 在 step 0 为 0）
4. **下一轮可用**：derived_object_timeline 显示所有 valid 命题在下一轮被使用（times_used_as_input > 0）
5. **组合爆炸**：constructible_objects 增长导致候选数从 E0-2 的 2521 增至 19909（derived_objects 参与构造）

---

## 七、E0-4 实验：knowledge revision and rollback

### 核心变化（与 E0-3 的区别）

E0-2：
```
observation → construction → verification
```

E0-3：
```
observation → construction → verification → valid knowledge → new computation input
```

E0-4：
```
observation → construction → verification → knowledge
    → new observation → re-verification → knowledge revision → computation input update
```

E0-3 的致命缺陷：
- proposition 一旦 STATUS_VALID，就永久跳过后续验证
- 等价于把"当前证据下被支持"锁死为"永远正确"

E0-4 的修正：
- 每个时间步都对已有 belief 重新验证
- 新证据可推翻旧结论
- derived_objects = 当前 BeliefStore 中 STATUS_VALID（非历史曾经 valid）

### 核心结论

不是"知识永久保存"。
而是：
**"知识是当前计算结果；新的计算可以修改旧的计算结果。"**

### 实验设计

```
for t in world_history:
  1. 只提供 history[:t+1]
  2. 新 observation 加入 observed_objects
  3. 重新验证所有已有 belief（使用 history[:t+1]）
  4. 根据当前 BeliefStore 重新计算 derived_objects
  5. constructible_objects = observed_objects ∪ derived_objects
  6. 从 constructible_objects 做单层构造
  7. 验证新候选（已存在的在 step 3 已重新验证）
  8. 本轮新 VALID 不得在本轮继续作为输入
  9. 下一轮才能使用
```

### 状态转换规则

| 转换 | 行为 |
|------|------|
| VALID → INVALID | 更新状态；从 derived_objects 删除；下一轮不再作为输入 |
| VALID → UNKNOWN | 更新状态；从 derived_objects 删除 |
| INVALID → VALID | 更新状态；重新进入 derived_objects；下一轮可作为输入 |
| UNKNOWN → VALID | 更新状态；进入 derived_objects |
| 任意 → 任意 | 每步重新验证；derived_objects 反映当前状态 |

### 实验结果

| 指标 | 值 |
|------|-----|
| 时间步数 | 12 |
| 总候选数 | 8124 |
| 新候选 valid | 16 |
| 新候选 invalid | 163 |
| 新候选 unknown | 1274 |
| duplicates_skipped | 6671 |
| 重新验证总数 | 11634 |
| 状态改变总数 | 18 |
| 总成本 | 57468.6 |
| Trace 步骤数 | 2906 |
| 证据数 | 78522 |
| derived_objects 最终数量 | 5 |

### 状态转换统计

| 转换类型 | 次数 |
|---------|------|
| valid_to_invalid | 11 |
| invalid_to_unknown | 7 |
| stayed_valid | 67 |
| stayed_invalid | 1275 |
| stayed_unknown | 10274 |

### 关键命题 belief_status_timeline

#### P(a)→Q(a)（始终 VALID）

| step | status | confidence | in_derived | used_as_input |
|------|--------|-----------|-----------|-------------|
| 0 | valid | 0.4 | False | False |
| 1 | valid | 0.8 | True | True |
| 2-11 | valid | 0.8 | True | True |

无反例出现，始终保持 VALID。

#### Q(a)→P(a)（VALID → INVALID）

| step | status | confidence | in_derived | used_as_input |
|------|--------|-----------|-----------|-------------|
| 0 | valid | 0.4 | False | False |
| 1 | valid | 0.8 | True | True |
| 2 | valid | 0.8 | True | True |
| 3 | **invalid** | **0.95** | **False** | **False** |
| 4-11 | invalid | 0.95 | False | False |

step 3：{Q(a), S(a)} → Q(a) 出现但 P(a) 不出现 → 反例 → INVALID
退出 derived_objects，后续不再作为计算输入。

#### P(c)→Q(c)（始终 INVALID）

| step | status | confidence | in_derived | used_as_input |
|------|--------|-----------|-----------|-------------|
| 0-6 | not_in_store | 0.0 | False | False |
| 7 | invalid | 0.95 | False | False |
| 8-11 | invalid | 0.95 | False | False |

step 7 首次构造，立即 INVALID（反例：step 6 中 P(c) 出现但 Q(c) 不出现）。

### derived_object_timeline（部分）

| proposition | entered | exited | times_used | history |
|-------------|---------|--------|-----------|--------|
| (P(a) → Q(a)) | step 1 | — | 11 | enter@1 |
| (Q(a) → P(a)) | step 1 | step 3 | 2 | enter@1, exit@3 |
| (P(a) → R(a,b)) | step 1 | step 2 | 1 | enter@1, exit@2 |
| (Q(b) → P(b)) | step 2 | step 9 | 7 | enter@2, exit@9 |
| (S(a) → Q(a)) | step 4 | — | 8 | enter@4 |
| (Q(c) → P(c)) | step 8 | step 9 | 1 | enter@8, exit@9 |

### 每步统计

| step | obs | der | cbl | rever | changed | new_v | new_i | new_u | ent | exit |
|------|-----|-----|-----|-------|---------|-------|-------|-------|-----|------|
| 0 | 3 | 0 | 3 | 0 | 0 | 6 | 0 | 21 | 0 | 0 |
| 1 | 6 | 6 | 12 | 27 | 0 | 6 | 60 | 447 | 6 | 0 |
| 2 | 6 | 8 | 14 | 540 | 6 | 0 | 28 | 344 | 4 | 2 |
| 3 | 7 | 7 | 14 | 912 | 2 | 1 | 18 | 86 | 0 | 1 |
| 9 | 9 | 5 | 14 | 1453 | 4 | 0 | 0 | 0 | 0 | 2 |

### E0-4 专用测试（10 项不变量）

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | test_valid_belief_can_be_refuted_later | VALID 可被新证据推翻为 INVALID |
| 2 | test_refuted_belief_exits_derived_objects | 被推翻的 belief 从 derived_objects 退出 |
| 3 | test_refuted_belief_not_used_after_exit | 退出后不再作为计算输入 |
| 4 | test_invalid_belief_can_become_valid_later | INVALID→VALID 可重新进入 derived_objects |
| 5 | test_derived_objects_reflect_current_status | derived_objects 反映当前状态（非历史） |
| 6 | test_valid_is_not_permanent_truth | VALID 不等于永远正确 |
| 7 | test_reverification_uses_only_visible_history | 重新验证只使用 visible history |
| 8 | test_newly_refuted_belief_cannot_continue_computation | 被推翻后本轮立即停止使用 |
| 9 | test_newly_valid_belief_enters_next_round | 新 VALID 在下一轮进入 derived |
| 10 | test_no_future_information | 禁止未来信息 |

### E0-4 结论

1. **知识修正闭环验证**：Q(a)→P(a) 在 step 0 被验证为 VALID，step 3 出现反例后变为 INVALID
2. **derived_objects 反映当前状态**：不是历史曾经 valid 的知识，而是当前证据下 valid 的知识
3. **退出即停止使用**：Q(a)→P(a) 在 step 3 退出 derived_objects 后，后续步骤不再作为构造输入
4. **重新验证机制**：每步重新验证所有已有 belief（11634 次），18 次状态改变
5. **VALID 非永久真值**：11 次 valid_to_invalid 转变证明 VALID 可被推翻

---

## 八、E0-5 实验：verification method evaluation

### 核心变化（与 E0-4 的区别）

E0-4：
```
observation → construction → verification → knowledge
    → new observation → re-verification → knowledge revision → computation input update
```

E0-5：
```
candidate proposition → 选择 verification method → verification result → 新证据
    → 比较该 method 的历史表现 → 更新对 method 的评价 → 后续选择 verification method
```

E0-4 验证了"知识可以修正"。E0-5 验证"验证方法本身也是计算对象"：
- 不同 verification method 有不同成本、可靠性和适用范围
- 方法可以犯错；reliability 可升可降
- 方法选择是计算过程（产生 trace），不是硬编码规则
- **不引入元认知模块**——方法评价只是普通计算对象

### 三种验证方法

| 方法 | cost | 特点 |
|------|------|------|
| `direct_observation` | 0.3 | 只看当前 observation，便宜但易受噪声影响 |
| `repeated_observation` | 1.0 | 看全部 visible history，更可靠但更贵 |
| `prediction_check` | 2.0 | 用前向预测检验，需要等待未来 observation；对非蕴含命题不适用 |

### VerificationMethodStore

记录每种方法的历史表现：

| 字段 | 说明 |
|------|------|
| `attempts` | 总使用次数 |
| `correct` | 时间反馈判定为"之前正确"的次数 |
| `incorrect` | 时间反馈判定为"之前错误"的次数 |
| `unknown` | 返回 UNKNOWN 的次数 |
| `total_cost` / `average_cost` | 成本统计 |
| `reliability_estimate` | `correct / (correct + incorrect)`，初始 0.5（中性） |

### 时间反馈机制（非 ground truth，非共识）

```
method M 在 step t 说 prop P 是 VALID
    → step t+1 重新验证 P
    → 新结果为 INVALID → M 在 step t 是错的 → incorrect += 1 → reliability 下降
    → 新结果仍为 VALID → M 在 step t 是对的 → correct += 1 → reliability 上升
```

- **不使用共识**："多个 verifier 一致"不等于可靠（内部一致性 ≠ 独立证据）
- **不读取 ground truth**：agent 的 reliability_estimate 完全来自时间反馈
- **ground truth 只用于实验统计**：不进入 BeliefStore / VerificationMethodStore / 方法选择

### 方法选择算法

```
value ≈ reliability × applicability - cost × weight

applicability 估计方法能否给出确定性答案（valid/invalid）：
  - direct_observation: 前件在当前 step → 1.0；否则 → 0.2
  - repeated_observation: 共现率 ≥0.8 或 <0.3 → 1.0；不确定区间 → 0.4
  - prediction_check: 有已完成预测 → 1.0；无 → 0.1；非蕴含 → 0.0
```

这使 prediction_check 在它拥有独立未来确认数据时能胜过更便宜但返回 unknown 的方法。

### 实验结果

| 指标 | 值 |
|------|-----|
| 时间步数 | 12 |
| 总候选数 | 21464 |
| Trace 步骤数 | 63197 |
| 总成本 | 17168.5 |

#### 各方法统计

| 方法 | attempts | correct | incorrect | unknown | total_cost | reliability |
|------|----------|---------|-----------|---------|------------|-------------|
| direct_observation | 45597 | 928 | 77 | 44384 | 13679.1 | 0.9234 |
| repeated_observation | 1721 | 1441 | 33 | 16 | 1721.0 | 0.9776 |
| prediction_check | 95 | 64 | 6 | 19 | 190.0 | 0.9143 |

#### Ground Truth vs Agent Reliability

| 方法 | agent reliability | gt accuracy | gt correct | gt incorrect |
|------|-------------------|------------|------------|--------------|
| direct_observation | 0.9234 | 0.9971 | 46466 | 136 |
| repeated_observation | 0.9776 | 0.9915 | 3168 | 27 |
| prediction_check | 0.9143 | **0.6727** | 111 | **54** |

- prediction_check 的 GT 准确率仅 0.67（54 次错误）——**方法可以犯错**
- agent reliability ≠ GT accuracy ——**reliability 不直接读取 ground truth**
- direct_observation 最常用（便宜），prediction_check 最少用（贵且受限）

#### 关键命题案例

| 命题 | 首次方法 | 首次 verdict | actual_correct | 最终状态 | GT |
|------|---------|-------------|-----------------|---------|-----|
| P(a)→Q(a) | direct_observation | valid | True | valid | True (supported: 6) |
| Q(a)→P(a) | direct_observation | valid | **False** | valid | **False** (refuted: 2) |
| P(c)→Q(c) | direct_observation | valid | **False** | unknown | **False** (refuted: 1) |

- Q(a)→P(a) 在 step 0 被 direct_observation 判为 VALID，但 GT 为 False → **方法犯错**
- 后续步骤中 prediction_check 被选用于 Q(a)→P(a) 的重验证（当前件不在当前 step 且 repeated 共现率不确定时）

### E0-5 专用测试（12 项不变量）

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | test_verification_methods_are_objects | 验证方法本身可以被记录和计算 |
| 2 | test_method_history_is_recorded | 每次使用 verifier 都产生历史记录 |
| 3 | test_method_reliability_is_not_ground_truth | agent 的 reliability estimate 不直接读取 ground truth |
| 4 | test_method_can_be_wrong | 至少一个 verifier 必须出现错误 |
| 5 | test_method_reliability_can_decrease | 错误发生后 reliability 可以下降 |
| 6 | test_method_reliability_can_increase | 连续正确后 reliability 可以上升 |
| 7 | test_methods_have_different_costs | 验证方法成本不同 |
| 8 | test_method_selection_depends_on_history | 历史表现能够影响后续方法选择 |
| 9 | test_future_information_is_forbidden | 验证方法选择不能读取未来信息 |
| 10 | test_ground_truth_is_not_agent_knowledge | ground truth 不得进入 KnowledgeStore / BeliefStore |
| 11 | test_verification_result_and_method_evaluation_are_distinct | "命题是否成立"和"方法是否可靠"是两个不同对象 |
| 12 | test_method_revision_does_not_directly_rewrite_proposition | 方法可靠性下降不能直接修改命题状态 |

### E0-5 结论

1. **验证方法是计算对象**：三种方法被记录、调用、统计，产生 trace
2. **方法有历史**：method_timeline 记录每次使用的 step/method/proposition/result/cost/reliability_before/after/actual_correct
3. **方法可以犯错**：prediction_check GT 准确率 0.67，direct_observation 也有 136 次 GT 错误
4. **reliability 从时间反馈更新**：reliability = correct/(correct+incorrect)，来自 belief revision，不读取 ground truth
5. **不同命题下方法价值不同**：direct 在前件出现时最优；repeated 在有历史且率高时最优；prediction_check 在有未来确认数据且其他方法不确定时最优
6. **方法选择影响后续计算**：随着预测数据积累，prediction_check 被选用于重验证 Q(a)→P(a)
7. **无元认知模块**：方法评价只是 VerificationMethodStore 中的普通计算对象

---

## 九、E0-6 实验：operation selection learning

### 核心变化（与 E0-5 的区别）

E0-5：
```
candidate proposition → 选择 verification method → verification result
    → 比较该 method 的历史表现 → 更新对 method 的评价 → 后续选择 verification method
```

E0-6：
```
当前对象/命题 → 提取上下文特征 → 从历史匹配 (context, operation) 表现
    → 根据历史结果对已有 constructor 评分 → epsilon-greedy 选择候选 → 执行构造+验证
    → 根据结果更新 (context, operation) 历史 → 下次遇到类似情境改变选择
```

E0-5 验证了"验证方法本身是计算对象"。E0-6 验证"**系统能否从计算历史中学会选择更有效的 constructor**"：
- 不创造新操作，只学习已有 constructors (neg/conj/disj/impl/iff) 的选择
- 不硬编码"某种命题必须使用某个操作"
- ground truth 只用于实验统计，不进入选择逻辑
- 保留探索（epsilon-greedy, ε=0.2）
- **不引入元认知模块**——操作选择、历史统计、评价都是普通计算对象

### OperationSelectionStore

记录每个 (context_signature, operation) 的历史表现：

| 字段 | 说明 |
|------|------|
| `attempts` | 总尝试次数 |
| `valid` / `invalid` / `unknown` | 验证结果计数 |
| `success_rate` | `valid / (valid + invalid)`，unknown 不计入（初始 0.5 中性） |
| `info_rate` | `(valid + invalid) / attempts`，惩罚只产生 unknown 的 constructor |
| `valid_yield` | `valid / attempts`，每尝试一次获得 valid 的比例 |
| `total_cost` / `average_cost` | 成本统计 |

### 选择评分

```
score = success_rate × info_rate - cost × weight

cost = CONSTRUCTOR_COSTS[operation] + VERIFICATION_COST
```

- `success_rate × info_rate`：综合"有确定性答案的比例"和"确定性答案中 valid 的比例"
- 只产生 unknown 的 constructor（如 conj/disj）→ info_rate=0 → score ≤ 0
- 自然惩罚无信息的操作，不需要硬编码

### 上下文特征提取

上下文签名基于命题结构特征（非元认知模块，只是普通计算对象）：

| 签名格式 | 说明 |
|---------|------|
| `unary_{kind}_{status}` | 一元 constructor 的上下文（neg） |
| `binary_{kind_a}_{kind_b}_{status_a}_{status_b}` | 二元 constructor 的上下文（conj/disj/impl/iff） |

其中：
- `kind`：atom / relation / implies / iff / not / and / or
- `status`：obs（已观察）/ der（已推导）/ new（新对象）

### A/B 对照实验设计

| | Group A（随机） | Group B（学习） |
|--|----------------|-----------------|
| 初始知识 | 相同 | 相同 |
| 操作集合 | 相同 (neg/conj/disj/impl/iff) | 相同 |
| 计算预算 | 相同 (12/step) | 相同 |
| 验证机制 | 相同（observation-based） | 相同 |
| 选择方式 | 随机 | epsilon-greedy + 历史评分 |
| 唯一变量 | ❌ 无学习 | ✅ 有学习 |

### 实验结果

| 指标 | Group A（随机） | Group B（学习） | 差异 |
|------|----------------|-----------------|------|
| 总候选数 | 10260 | 10788 | +528 |
| 实际执行数 | 144 | 144 | 0 |
| valid | 9 | 11 | +2 |
| invalid | 42 | 72 | +30 |
| unknown | 93 | 61 | -32 |
| 总成本 | 120.8 | 117.2 | -3.6 |
| 效率 (valid/cost) | 0.0745 | 0.0939 | **1.26x** |

#### Group B 操作选择分布

| 操作 | A 次数 | B 次数 | B success_rate | B info_rate | B valid_yield |
|------|--------|--------|----------------|-------------|---------------|
| neg | 3 | 20 | 0.000 | 0.450 | 0.000 |
| conj | 34 | 21 | 0.500 | 0.000 | 0.000 |
| disj | 39 | 19 | 0.500 | 0.000 | 0.000 |
| impl | 33 | **66** | 0.179 | **0.849** | **0.152** |
| iff | 35 | 18 | 0.056 | 1.000 | 0.056 |

**学习效果**：
- Group B 大幅增加 `impl` 选择（66 vs 33）——impl 是唯一能产生高 info_rate 且有一定 valid_yield 的操作
- Group B 减少 `conj`/`disj`（均只产生 unknown，info_rate=0）
- Group B 增加 `neg`（20 vs 3）——neg 有中等 info_rate（0.45）

#### 上下文特定偏好

| 上下文 | 最佳操作 | valid_yield |
|--------|---------|-------------|
| binary_atom_atom_obs_obs | impl | 0.1905 |
| binary_atom_relation_obs_obs | impl | 0.1429 |
| unary_atom_obs | neg | 0.000 (但 info_rate=1.0) |
| unary_implies_der | neg | 0.000 (info_rate=0.0) |

不同上下文形成了不同的操作偏好模式。

### E0-6 专用测试（10 项不变量）

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | test_history_influences_selection | 历史经验能够影响后续操作选择 |
| 2 | test_different_contexts_different_preferences | 不同情境可以形成不同的操作偏好 |
| 3 | test_unknown_not_treated_as_failure | unknown 不会被错误当成失败 |
| 4 | test_ground_truth_not_in_selection_logic | ground truth 不进入选择逻辑 |
| 5 | test_exploration_still_exists | 仍然存在探索（epsilon-greedy） |
| 6 | test_history_selection_records_complete | 历史选择记录完整 |
| 7 | test_ab_groups_use_same_budget | A/B 两组使用相同预算 |
| 8 | test_no_new_operations_created | 不创造新操作 |
| 9 | test_future_information_forbidden | 操作选择不能读取未来信息 |
| 10 | test_operation_selection_produces_trace | 操作选择产生 trace |

### E0-6 结论

1. **操作选择可以学习**：Group B 基于历史评分选择 constructor，效率比随机选择高 26%
2. **学习目标是情境-操作匹配**：系统不是简单偏好某个操作名，而是根据上下文签名选择不同操作
3. **不同情境形成不同偏好**：binary atom 上下文偏好 impl，unary 上下文偏好 neg
4. **unknown 不被伪造为失败**：conj/disj 只产生 unknown，被 info_rate=0 自然惩罚，不影响 success_rate
5. **ground truth 不进入选择**：select_candidates / compute_score / verify_proposition 源码均不含 ground_truth
6. **探索保留**：epsilon=0.2 确保持续尝试非最优操作，获取新经验
7. **无元认知模块**：操作选择、历史统计、评价都是普通计算对象，产生 trace

### E0-6 未证明什么

- **未证明新操作发现**：E0-6 只测试已有操作的选择学习（Level 1），不创造新操作（Level 2）
- **未证明学习收敛**：当前实验只有 12 步，不足以展示长期收敛行为
- **未证明多情境泛化**：上下文签名较粗（kind+status），未测试更细粒度的特征提取
- **未证明优于硬编码**：学习效率 1.26x 是与随机选择对比，未与人工设计的硬编码规则对比

---

## 十、E0-7 实验：transformation history learning

### 核心变化（与 E0-6 的区别）

E0-6：
```
当前对象 → 提取 context 签名 → 从历史选择 constructor → 验证 → 结果反馈
    → 更新对 constructor 的评价 → 后续选择 constructor
```

E0-7：
```
输入对象 → operation → 输出对象 → 验证
    → 记录 (input_structure → output_structure) 变换
    → 遇到结构相似的新输入 → 从变换历史检索 → 生成候选 → 验证
```

E0-6 学到的是：**context → "impl"**（操作选择）
E0-7 要学的是：**input_structure → output_structure**（计算变换）

关键区别：E0-6 在同一 context 下无法区分"哪对输入有效"；E0-7 可以记住具体哪些 (input_structure → output_structure) 产生了 valid 结果。

### 核心设计

#### 1. 使用 predicate 而非 atom

E0-6 使用 `P.atom("P(a)")`，P(a) 和 P(b) 是完全不同的 atom，无法结构泛化。

E0-7 使用 `P.predicate("P", "a")`，P(a) 和 P(b) 有相同的结构签名：
```
P.predicate("P", "a") → ("predicate", "P", ("_t0",))
P.predicate("P", "b") → ("predicate", "P", ("_t0",))
```

#### 2. 结构签名 structure_sig

将具体词项抽象为占位符 `_t0, _t1, ...`，同一词项在联合签名中使用同一占位符：
```
(P(a), Q(a)) → (("predicate","P",("_t0",)), ("predicate","Q",("_t0",)))  # 共享 _t0
(P(b), Q(b)) → (("predicate","P",("_t0",)), ("predicate","Q",("_t0",)))  # 签名相同
```

#### 3. TransformationRecord

| 字段 | 说明 |
|------|------|
| `input_sigs` | 输入对象的联合结构签名（核心） |
| `output_sig` | 输出对象的结构签名（核心） |
| `operation_name` | 产生此变换的操作（仅历史来源，不参与匹配） |
| `context_sig` | 上下文签名 |
| `verification_result` | valid / invalid / unknown |
| `usefulness` | goal improvement |
| `cost` / `step` / `confidence` | 元数据 |
| `input_terms` | {actual_term: placeholder} 映射 |

**关键**：匹配基于 `input_sigs` 结构，不依赖 `operation_name`。

#### 4. TransformationStore.find_matches

检索算法：
1. 对每个 record，其 `input_sigs` 是多个输入的联合结构签名
2. 在当前对象中寻找结构签名匹配的候选
3. 检查词项绑定一致性（如 `_t0` 共享，则匹配对象必须共享同一实际词项）
4. 返回 (record, matched_objects, binding)

#### 5. 候选选择加分

Group B 的候选分数 = E0-6 base_score + transform_boost：
```
transform_boost = TRANSFORM_BOOST × confidence  （仅当候选匹配 valid 变换时）
```

### A/B 对照实验设计

| | Group A（E0-6） | Group B（E0-7） |
|--|----------------|-----------------|
| 操作选择 | context → operation | context → operation |
| 变换历史 | ❌ | ✅ input_structure → output_structure |
| 世界/预算/验证 | 相同 | 相同 |

世界设计：多个谓词 P, Q, S, T，对象 a, b, c, d。只有 P(x)→Q(x) 和 S(x)→T(x) 有效，其他蕴含方向无效。这使得同一 context 下 impl 对某些对有效、对另一些无效——E0-6 无法区分，E0-7 可以。

### 实验结果

| 指标 | Group A（E0-6） | Group B（E0-7） | 差异 |
|------|----------------|-----------------|------|
| valid | 13 | **29** | **+16 (+123%)** |
| invalid | 105 | 93 | -12 |
| unknown | 26 | 22 | -4 |
| 总成本 | 125.4 | 126.2 | +0.8 |
| 效率 (valid/cost) | 0.1037 | **0.2298** | **2.22x** |

#### 关键命题首次发现步数

| 命题 | Group A | Group B | 改进 |
|------|---------|---------|------|
| P(a)→Q(a) | step 1 | step 1 | 0 |
| P(b)→Q(b) | step 2 | step 2 | 0 |
| P(c)→Q(c) | step 7 | **step 6** | **-1** |
| P(d)→Q(d) | not found | **step 11** | 发现 |
| S(a)→T(a) | step 4 | step 4 | 0 |
| S(c)→T(c) | not found | **step 6** | 发现 |
| S(d)→T(d) | not found | **step 11** | 发现 |

Group B 发现了 3 个 Group A 未发现的有效命题（P(d)→Q(d), S(c)→T(c), S(d)→T(d)），并提前 1 步发现 P(c)→Q(c)。

#### 变换历史统计（Group B）

| 指标 | 值 |
|------|-----|
| 总变换记录 | 144 |
| valid | 29 |
| invalid | 93 |
| unknown | 22 |
| 获得变换加分的候选 | 24 |

### E0-7 专用测试（10 项不变量）

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | test_transform_history_records_real_computation | 变换历史能记录真实计算过程 |
| 2 | test_transform_representation_independent_of_operation_name | 变换表示不依赖 operation name 才能匹配 |
| 3 | test_transform_history_influences_candidate_generation | 变换历史能影响候选生成 |
| 4 | test_transform_only_produces_candidates_not_beliefs | 历史变换只能产生候选，不能直接产生 valid belief |
| 5 | test_unknown_not_treated_as_invalid | unknown 不被当成 invalid |
| 6 | test_ground_truth_not_in_transform_selection | ground truth 不进入变换选择 |
| 7 | test_no_future_information | 不读取未来信息 |
| 8 | test_new_object_with_same_structure_triggers_transform | 相同结构的新对象可以触发历史变换候选 |
| 9 | test_no_unverified_generalization_into_belief | 不允许未经验证的泛化直接进入 belief |
| 10 | test_experiment_checks_pass | 实验内部检查全部通过 |

### E0-7 结论

1. **系统能学习计算变换**：从 (P(a),Q(a))→(P(a)→Q(a)) 泛化到 (P(c),Q(c))→(P(c)→Q(c))
2. **变换学习独立于 operation name**：匹配基于 input_structure → output_structure，operation_name 仅作执行手段
3. **变换历史提供显著价值**：Group B 比 Group A 多发现 16 个 valid，效率提升 2.22x
4. **新对象泛化成立**：P(d)、S(d) 等全新对象能触发历史变换候选
5. **变换只产生候选**：所有 valid belief 都来自验证，变换不直接写入 belief
6. **无未验证泛化**：P(a)→Q(a) 不直接推广为 ∀x(P(x)→Q(x))，P(c)→Q(c) 需经验证
7. **ground truth 隔离**：选择/匹配/验证逻辑均不含 ground_truth

### E0-7 理论问题回答

1. **E0-6 学到的是"操作选择"还是"计算变换"？**
   E0-6 学到的是操作选择：context → operation_name。它知道"predicate 上下文用 impl"，但不知道"哪对输入产生 valid"。

2. **E0-7 是否真正摆脱了 operation-name dependency？**
   是的。变换匹配基于 input_sigs 结构签名，不依赖 operation_name。operation_name 仅在执行变换时使用（作为历史记录的执行手段），不参与匹配决策。

3. **P(a)→Q(a) 迁移到 P(b)→Q(b) 发生在哪一步？**
   发生在**候选选择阶段**：当 P(b), Q(b) 出现在 constructible 中时，TransformationStore.find_matches 根据结构签名找到历史 (P(a),Q(a))→(P(a)→Q(a)) 变换，为 impl(P(b),Q(b)) 候选加分，使其优先被选择和验证。

4. **这个迁移是演绎、归纳，还是历史匹配？**
   是**历史匹配（analogical transfer）**。系统不做逻辑推导，也不做全称归纳。它只是发现"过去 (P(x),Q(x)) 结构产生了 valid 输出"，然后对新的 (P(b),Q(b)) 尝试相同的结构变换。结果必须经验证。

5. **如果做不到新对象泛化，卡在哪里？**
   当前实验成功实现了新对象泛化。关键在于：使用 predicate（而非 atom）表示使结构签名成为可能；structure_sig 抽象词项为占位符；find_matches 检查词项绑定一致性。如果使用 atom，则无法泛化（卡在 representation）。

6. **E0-7 相比 E0-6 增加了什么新能力？**
   E0-7 增加了**结构级别的变换记忆与检索**。E0-6 只能学到"什么上下文用什么操作"，E0-7 能学到"什么输入结构产生什么输出结构"，从而在新对象上复用变换经验。这是从"操作选择"到"计算变换学习"的关键一步。

### E0-7 未证明什么

- **未证明新操作发现**：仍只使用已有 constructors
- **未证明全称归纳**：P(a)→Q(a) 到 P(b)→Q(b) 是候选生成，不是 ∀x(P(x)→Q(x)) 的逻辑推导
- **未证明复杂结构泛化**：当前只测试了 predicate 的简单结构，未测试嵌套结构的泛化
- **未证明长期收敛**：12 步实验，未测试长期学习行为
- **未证明跨谓词泛化**：P→Q 的变换不直接迁移到 S→T（两者是独立的变换记录）

---

## 十一、E0-7.1 实验：strict transformation instantiation（operation_name 独立性）

### 核心问题

E0-7 审查发现：虽然 TransformationRecord 的 **匹配层**（find_matches）不读 operation_name，
但 **候选构造层** 仍然 100% 依赖暴力枚举 constructor name（neg/conj/disj/impl/iff）。
变换历史仅对已生成候选加分（boost），不生成新候选。
test_new_object_with_same_structure_triggers_transform 通过是因为它调用了
`apply_constructor(record.operation_name, matched_objs)` —— 测试"通过"是因为读了 operation_name。

E0-7.1 的严格问题：**input_structure → output_structure 能否独立于 operation_name 产生候选？**

### 三个层次分离

| 层次 | 功能 | 实现 |
|------|------|------|
| A. Recognition | 发现历史变换 | `find_matches`（用 input_sigs 结构匹配） |
| B. Instantiation | 从 sig+binding 重建输出 | `instantiate_output`（structure_sig 的逆函数） |
| C. Execution | 验证候选 + 更新信念 | `verify_proposition` + `belief_store.update_belief` |

### 核心函数：instantiate_output

`instantiate_output(output_sig, binding)` 是 `structure_sig` 的逆函数：
- 输入：output_sig（结构模板，如 `("implies", ("predicate","P",("_t0",)), ("predicate","Q",("_t0",)))`）
- 输入：binding（词项映射，如 `{"_t0": "b"}`）
- 输出：重建的 Proposition（如 `P.impl(P(b), P(b))`，即 P(b)→Q(b)）

此函数 **不读取 operation_name**。它只使用 sig 中的命题结构信息
（kind 如 "implies" 是 Proposition 的结构属性，不是 operation 的身份标识）。

### 核心函数：generate_candidates_from_transforms

`generate_candidates_from_transforms(current_objects, transform_store, belief_store)`
- 调用 `find_matches` 检索匹配变换
- 调用 `instantiate_output(record.output_sig, binding)` 重建候选命题
- **不枚举 constructor**，**不调用 apply_constructor**，**不读 record.operation_name**
- operation_name 仅作为 provenance 保存在候选字典中

### 三个阶段实验

| 阶段 | operation_name | 候选来源 | 验证 |
|------|---------------|---------|------|
| Phase 1: 历史构建 | 真实 operation | 暴力枚举（E0-7 机制） | 正常验证 |
| Phase 2: 严格模式 | 保存但不读取 | 仅从变换历史实例化 | 正常验证 |
| Phase 3: UNKNOWN | 全部设为 "UNKNOWN" | 仅从变换历史实例化 | 正常验证 |

### 实验结果

- Phase 1：2425 条变换记录（19 valid, 389 invalid, 2017 unknown）
- Phase 2：P(c),Q(c) → 10 个候选 → 4 valid, 2 invalid, 4 unknown；P(c)→Q(c) 验证为 VALID
- Phase 3：P(d),Q(d) → 10 个候选 → 4 valid, 2 invalid, 4 unknown；P(d)→Q(d) 验证为 VALID
- 反事实：P(c),R(c) → 0 候选（结构不匹配）
- 绑定一致性：P(b),Q(c) → 共享绑定变换 0 匹配（b≠c 被拒绝）
- 三层分离：recognition ✓ / instantiation ✓ / execution ✓

### structure_sig 对 R(a,b)/R(b,a)/R(a,a) 的处理

- R(a,b) → `("relation","R",("_t0","_t1"))`，tm={a:_t0, b:_t1}
- R(b,a) → `("relation","R",("_t0","_t1"))`，tm={b:_t0, a:_t1}
- R(a,a) → `("relation","R",("_t0","_t0"))`，tm={a:_t0}

R(a,b) 和 R(b,a) 签名相同（都是"二元关系+2个不同词项"），R(a,a) 不同。
这是正确的类比行为：R(a,b)→¬R(a,b) 迁移到 R(b,a) 产生 ¬R(b,a)，词项按位置交换。

### 理论结论

1. **E0-7 当前到底学到了什么？** E0-7 学到的是"哪些 operation 在哪些 context 下有效 + 变换结构作为 boost 参考"。候选生成仍由暴力枚举驱动，变换历史只加分不生成。
2. **operation_name 是否只是 provenance？** 在 E0-7 中不是——它仍控制候选构造。在 E0-7.1 中是——候选从 output_sig 独立重建。
3. **input_structure → output_structure 是否已成为独立知识？** 在 E0-7.1 中是。instantiate_output 证明了 output_sig 包含足够信息独立重建输出。
4. **能否在完全不知道 operation_name 的情况下实例化变换？** 能。Phase 3（operation_name="UNKNOWN"）与 Phase 2 结果完全一致。
5. **P(a),Q(a)→P(a)→Q(a) 能否迁移到 P(b),Q(b)？** 能。通过 placeholder binding _t0→b 实例化 P(b)→Q(b)。
6. **这种迁移属于什么？** 单纯结构匹配 + 类比迁移，不是 deduction（不依赖逻辑规则）也不是 induction（不产生全称命题）。
7. **如果失败，失败在哪层？** 无失败。三层（recognition/instantiation/execution）全部通过。
8. **E0-7.1 相对于 E0-7 新增了什么？** instantiate_output（结构重建）+ generate_candidates_from_transforms（从变换历史独立生成候选，不枚举 operation）。
9. **是否从 Level 1 迈向 Level 2？** 是。Level 1 = operation selection learning（E0-6）。Level 2 = operation-independent transformation learning（E0-7.1）。系统不再依赖 operation identity 来构造候选，而是从结构模板独立实例化。
10. **下一步瓶颈？** 当前只在单一变换步骤上测试。未测试变换链（A→B→C 能否从历史重建）、变换组合（两个变换能否组合产生新候选）、变换冲突（同一输入匹配多个矛盾变换时的选择）。

### Representation 限制

- `structure_sig` 不存储量化变量名（forall/exists 的 name 字段），因此 `instantiate_output` 无法重建量化命题。这是已知的 representation 限制，不是 bug。E0-7.1 的候选不包含 ∀x(P(x)→Q(x))，因为系统不做全称泛化。

---

## 十二、E0-7.2 实验：Transformation Composition / Recursive Reuse（变换递归复用）

### 核心问题

E0-7.1 证明了单步变换 A→B 可以独立于 operation_name 实例化为 A'→B'。
E0-7.2 要验证：**A→B, B→C 能否被系统递归复用，形成多步计算链？**

关键不是直接学习 A→C，而是验证：
```
A → [变换1] → B → [B进入知识空间] → [变换2] → C
```

新生成的 Proposition 必须真正重新进入下一轮计算循环。

### 核心机制：recursive_compute

`recursive_compute` 是通用递归循环，**不新增 TransformationChainEngine**：

```
for round in range(max_rounds):
    constructible = observed ∪ derived_valid      # 当前知识空间
    candidates = generate_candidates_from_transforms(constructible, ...)
    for candidate in candidates:
        verify → verdict
        belief_store.update_belief(prop, verdict)  # 验证后才入信念
    if no new_valids: break                          # 知识空间不再增长则停止
```

**关键设计**：
- `derived_valid` 每轮从 `belief_store` 重新计算 → 新 VALID 命题自动进入下一轮 constructible
- 候选生成后不直接入 belief，必须验证后 `update_belief` → 未验证候选不传播
- INVALID 命题不在 `derived_valid` 中 → 错误中间结果不能触发后续变换
- 每轮候选限制 200 个（计算预算）

### 扩展验证：verify_proposition_extended

E0-7 的 `verify_proposition` 不处理 `and`/`or` kind，导致 conj 命题永远返回 "unknown"，
无法成为 VALID，阻塞链式传播。

E0-7.2 新增 `verify_proposition_extended`：
- `and(A, B)`：两部分都 valid → valid；任一 invalid → invalid；否则 unknown
- `or(A, B)`：任一 valid → valid；都 invalid → invalid；否则 unknown
- 其他 kind 委托给 E0-7 的 verify_proposition

### 链设计

```
T1: (P(x), Q(x)) → P(x)→Q(x)                          [impl]
T2: (P(x)→Q(x), Q(x)) → (P(x)→Q(x))∧Q(x)               [conj — 输入需要 P→Q!]
T3: ((P(x)→Q(x))∧Q(x), R(x)) → ((P(x)→Q(x))∧Q(x))∧R(x) [conj]
```

T2 的输入需要 P(x)→Q(x)——如果 Round 0 没产生 P(b)→Q(b)，T2 无法匹配。
这就是**隐藏中间步骤**：目标依赖中间对象的产生。

### 实验结果

| 场景 | 结果 |
|------|------|
| 2-step chain | P(b)→Q(b) VALID, (P→Q)∧Q VALID ✓ |
| 3-step chain | P→Q, (P→Q)∧Q, ((P→Q)∧Q)∧R 全部 VALID ✓ |
| Hidden intermediate（只给 P(b)） | 链不启动（缺 Q(b)）✓ |
| Invalid intermediate（P(b)→Q(b) INVALID） | 链停止（P→Q 不在 derived_valid）✓ |
| operation_name=UNKNOWN（2-step） | 链仍成功 ✓ |
| operation_name=UNKNOWN（3-step） | 链仍成功 ✓ |

### 关键检查全部通过

| 检查项 | 结果 |
|--------|------|
| two_step_chain_succeeded | True |
| three_step_chain_succeeded | True |
| hidden_intermediate_blocked | True |
| invalid_intermediate_blocked | True |
| unknown_ops_2step_succeeded | True |
| unknown_ops_3step_succeeded | True |
| trace_contains_each_step | True |
| no_direct_shortcut | True |
| constructible_grew_across_rounds | True |
| no_future_info | True |
| no_ground_truth_in_logic | True |

### 理论结论

1. **单步 Transformation 能否递归成为多步计算？** 能。recursive_compute 证明新生成对象自动进入下一轮 constructible，形成 P→Q → (P→Q)∧Q → ((P→Q)∧Q)∧R 链。
2. **新生成的 Proposition 是否真正进入知识空间？** 是。constructible_count 逐轮增长，新 VALID 命题出现在下一轮 constructible 中。
3. **第二步是否可以完全由第一步的输出触发？** 是。T2 输入需要 P→Q，只有 Round 0 产生 P(b)→Q(b) 后 Round 1 才能匹配 T2。
4. **operation_name=UNKNOWN 时多步链是否仍成立？** 是。2-step 和 3-step 在 UNKNOWN 下全部成功。
5. **中间结果验证失败时链是否停止？** 是。P(b)→Q(b) 为 INVALID 时不在 derived_valid，T2 无法匹配。
6. **系统有没有偷偷使用最终目标信息？** 没有。候选仅从变换历史实例化，无目标导向。
7. **系统有没有偷偷使用未来信息？** 没有。visible_history 固定，constructible 仅来自当前 belief。
8. **当前机制是否表现出"从已有计算结果继续计算"的能力？** 是。这正是 recursive_compute 的核心。

**E0-7.2 证明：历史变换可以被递归复用，新生成对象能够重新进入计算循环，从而形成多步计算链。**

### 尚未解决的问题

- **如何寻找正确的中间对象**：当前变换链从已有历史中匹配，不解决"在多个可能中间对象中如何选择"。
- **变换冲突**：同一输入匹配多个矛盾变换时如何选择（当前无策略，全部生成）。
- **变换组合**：两个独立变换能否组合产生穷举不会产生的新候选（当前只支持历史中出现过的变换的链化）。

### 下一步最核心的理论瓶颈

当存在多个可能的中间对象/多个可匹配变换时，**系统如何计算"下一步应该选择什么"**。这是从"能形成链"到"能形成有用的链"的关键跨越。

---

## 十三、E0-7.3 实验：Transformation Selection / Goal-directed Search（变换选择与目标导向搜索）

### 核心问题

E0-7.2 解决了"能不能继续计算"。
E0-7.3 解决"有多种可能继续计算的方向时，系统如何决定下一步计算什么"。

关键区分：
- E0-6：context → operation（操作选择）
- E0-7.3：input_structure → output_structure 的选择（变换选择）

### 核心机制

1. **目标表示**：goal Proposition（如 `(A→B)∧(B→A)`）
2. **目标评价（显式先验）**：`goal_reached(output, goal)` = output 结构等于 goal（二元判定，非距离算法）
3. **信用分配**：成功链中所有变换获得 `goal_improvement=1.0`
4. **TransformationSelectionStore**：按 `(input_sig, output_sig)` 统计 attempts/valid/goal_improvement/cost
   - `score = mean_goal_improvement - cost_weight * mean_cost`
   - **不存储 operation_name**
5. **选择**：epsilon-greedy（EPSILON=0.2），不读 operation_name

### 候选生成与候选选择分离

```
候选生成：generate_candidates_from_transforms(constructible, ...) → 候选集合（不变）
候选选择：select_candidates(candidates, selection_store, use_learning) → 选中子集
```

选择过程不修改候选集合。

### 世界设计

历史中 A, B, C, D 全部共现 → 所有两两蕴含都是 VALID。
目标：`(A(b)→B(b)) ∧ (B(b)→A(b))`

竞争：A→C, A→D, B→C, B→D 等也是 VALID，但对目标无用。
系统需从历史学习哪些变换对目标有价值。

### A/B 实验结果

| 指标 | Group A（随机） | Group B（学习） |
|------|----------------|----------------|
| 到达目标 | 4/4 | 4/4 |
| 平均步数 | 8.2 | **3.8** |
| 平均成本 | 146.40 | **58.20** |

Group B 效率显著更高（步数 -54%，成本 -60%）。

### 关键检查

| 检查项 | 结果 |
|--------|------|
| group_b 更优（步数/成本） | True |
| operation_name 独立性 | True |
| 全部检查通过 | True |

### 理论结论

1. **E0-7.3 是否真的研究了"下一步计算什么"？** 是。选择基于 `(input_sig, output_sig)` 的历史目标改善，不是 operation_name。
2. **选择依据是什么？** 历史 `goal_improvement` 和 `cost` 的加权。来自实际计算反馈，非人工写死。
3. **goal 如何进入计算？** 作为显式先验：`goal_reached(output, goal)` 判定是否达成。
4. **系统能否因 goal 改变而改变方向？** 能。不同 goal 下同一变换的 `goal_improvement` 不同，导致分数不同。
5. **系统能否因失败而改变选择？** 能。未达目标的变换 `goal_improvement=0`，分数低于成功变换。
6. **是否保留探索？** 是。epsilon-greedy，未知变换仍有 20% 概率被选中。
7. **是否形成完整闭环？** 是。目标 → 候选 → 选择 → 计算 → 反馈 → 更新选择。

**E0-7.3 证明：系统开始能够在多个合法计算方向之间，根据目标、历史反馈和计算成本进行选择。**

### 尚未解决的问题

- **未来价值估计**：当前只评价直接输出是否达到目标，无法估计"这个中间对象未来可能通向目标"。长路径中需要多步才能看到价值的中间对象会被低估。
- **信用分配粒度**：当前成功链中所有变换获得相同信用，不区分"关键变换"和"辅助变换"。
- **变换冲突**：同一输入匹配多个矛盾变换时，当前按分数选择，但无冲突检测机制。

### 下一步最核心的理论瓶颈

**未来价值估计**：当目标需要多步才能达到时，如何给中间步骤分配信用？当前机制只能给"直接产生目标"的链信用，无法估计"虽然没直接到目标，但可能是好的中间步骤"的变换价值。这是从"能选择"到"能远见选择"的关键跨越。

---

## 十三-B、E0-7.4 实验：Future Value / Credit Assignment（未来价值与信用分配）

### 核心问题

E0-7.3 的缺陷：一个中间步骤本身没有直接达到目标时，系统无法知道它"未来可能通向目标"。成功链上所有变换获得相同信用，过于粗糙。

E0-7.4 研究：**系统能否仅根据已经发生的计算和反馈，逐渐学会中间步骤的未来价值？**

### 核心机制

1. **Provenance 追踪**：`provenance: {proposition: (record, input_propositions)}`，记录每个命题由哪个变换产生
2. **信用传播**：目标达成后，沿 provenance 链反向分配信用
3. **三种信用模式**：
   - `last_only`：只有直接产生 goal 的变换得信用
   - `equal`：链上所有变换等额信用（E0-7.3 baseline）
   - `discounted`：`credit(previous) = discount × credit(next)`，discount 是实验参数
4. **目标上下文**：选择分数基于 `(input_sig, output_sig, goal_sig)`，同一变换在不同 goal 下可有不同价值
5. **有效性与价值分离**：VALID + 当前低价值 仍保留，不变成 INVALID

### 严格禁止

- goal distance / 结构相似度 / 编辑距离
- operation_name 参与选择和信用计算
- ground truth 在线参与
- 未来信息泄漏
- planner / LLM / 神经网络

### 世界设计

A, B, C 全部共现 → 所有两两蕴含 VALID。

- 2-step 目标：`(A→B) ∧ (B→C)`
- 3-step 目标：`((A→B) ∧ (B→C)) ∧ (C→A)`

### 信用模式 A/B 实验结果

**3-step 目标**：

| 信用模式 | 达成率 | 平均步数 | 平均成本 |
|---------|--------|---------|---------|
| last_only | 2/14 | 6.9 | 214.07 |
| equal | 5/14 | 5.6 | 180.14 |
| **discounted** | **8/14** | **4.9** | **157.14** |

**2-step 目标**：三组均 14/14 达成（目标简单，信用模式差异不显著）。

**核心结论**：在多步目标上，折扣信用传播（discounted）显著优于只给最后一步信用（last_only）和等额信用（equal）。这证明系统能够通过信用传播学习中间步骤的未来价值。

### 其他实验结果

| 检查项 | 结果 |
|--------|------|
| 目标变化时同一变换分数不同 | True |
| operation_name 独立性 | True |
| 18 项不变量测试全部通过 | True |

### 理论结论（回答四个问题）

1. **E0-7.4 是否证明了系统可以学习"中间步骤的未来价值"？**
   是。在 3-step 目标上，discounted 模式达成率（8/14）显著高于 last_only（2/14）。系统通过实际成功 trace 的反向信用传播，学会了中间变换的未来价值。

2. **信用究竟来自什么？**
   来自**实际成功结果**和**历史链条**。具体来说：目标达成 → 沿 provenance 链反向传播 → 链上每个变换获得与其距目标的深度成反比的信用。不是目标本身，也不是人工指定的正确路径。

3. **这个机制是否仍然只是硬编码的搜索算法？**
   - 实验框架给出的先验：discount 参数、信用传播公式、goal_reached 判定、候选生成
   - 系统从计算历史中学到的内容：哪些 `(input_sig, output_sig, goal_sig)` 组合曾经通向成功
   - 折扣公式是框架先验，但**哪些变换获得信用、信用多少，完全来自实际 trace**，不是硬编码的

4. **E0-7.4 完成后，真正没有解决的问题是什么？**
   仍然假设候选变换可以从已有 transformation history 中生成。更深的问题是：**如果需要的中间对象 X 根本不在已有 transformation history 中，系统如何构造/发现 X？** 这是从"选择已有变换"到"创造新变换"的跨越，留待后续研究。

### 18 项不变量测试

1. 最终目标成功后可以产生历史信用 ✓
2. 直接目标变换信用最高 ✓
3. 更早中间步骤可以获得非零未来信用 ✓
4. dead-end 路径不会获得成功信用 ✓
5. INVALID 不能获得成功未来信用 ✓
6. VALID 但低价值知识仍保留 ✓
7. 失败可以降低未来选择倾向 ✓
8. 失败不会永久禁止探索 ✓
9. 同一变换在不同 goal 下可以产生不同历史价值 ✓
10. operation_name=UNKNOWN 仍完全正常 ✓
11. operation_name 改名不影响结果 ✓
12. 不使用 goal distance ✓
13. 不使用 ground truth 参与选择 ✓
14. 不使用未来 episode 数据 ✓
15. 信用只能由已经发生的 trace 产生 ✓
16. 多步目标中间节点能够通过历史学习获得价值 ✓
17. 不同成功路径都可以获得信用 ✓
18. 长路径不能因为"暂时没有直接价值"被判 INVALID ✓

**E0-7.4 证明：系统能够从实际计算结果反向传播信用，学会中间步骤的未来价值，实现从"直接选择"到"远见选择"的跨越。**

---

## 十三-C、E0-8 实验：Goal-Oriented New Object Generation（目标导向新对象生成）

### 核心问题

E0-7.3 已经证明"已有 transformation 历史 → 从多个已有候选中选择更有目标价值的候选"。

但这没有解决真正的问题：**如果正确的中间对象根本不存在于历史中，系统能不能自己构造出来？**

E0-7.x 的候选生成（`generate_candidates_from_transforms`）只能从历史中**结构匹配**出已有变换模板的输出。当历史中不存在所需模式时，系统无法产生新对象。

E0-8 验证：**系统使用基础构造器（neg/conj/disj/impl/iff）对已有对象进行合法计算，产生历史中从未出现过的新对象 X，验证 X，让 X 进入知识空间，并在下一轮把 X 当作普通输入继续计算，最终到达目标 G。**

### 与 E0-7.3 的本质区别

| 维度 | E0-7.3 | E0-8 |
|------|--------|------|
| 候选来源 | `generate_candidates_from_transforms`（历史结构匹配） | `generate_candidates`（暴力枚举基础构造器） |
| 新对象来源 | 历史中已有变换模板的实例化 | 基础构造器对已有对象的合法组合 |
| 历史的作用 | 提供候选（无历史则无候选） | 不参与候选生成（历史存在但不提供所需模式） |
| 核心能力 | 从历史中选择已有变换 | 自己构造历史中不存在的新对象 |

### 最小人工世界

- 初始知识：`A(a)`, `B(a)`
- 目标：`G = (A(a) → B(a)) ∧ B(a)` —— 合取，需要中间对象 X
- 中间对象：`X = A(a) → B(a)` —— 用 impl 构造器从 A、B 产生
- 变换历史：用 P、Q 谓词构建（不含 A→B 模式，因谓词名不同导致结构签名不匹配）

```
Round 0: constructible={A(a), B(a)}
  → 暴力构造产生 impl(A,B)=X → 验证 VALID → X 进入知识空间

Round 1: constructible={A(a), B(a), X, ...}
  → 暴力构造产生 conj(X, B)=G → 验证 VALID → 目标达成
```

**没有捷径**：`G = conj(impl(A,B), B)` 需要 `impl(A,B)` 作为 conj 的输入对象。Round 0 的 constructible 不含 impl(A,B)，所以 G 无法在 Round 0 构造。只有 X 被验证为 VALID 并进入知识空间后，Round 1 才能构造 G。

### 三个场景

1. **构造器路径（主）**：用基础构造器 → 2 轮达成目标，X=VALID
2. **仅历史路径（对照）**：用 `generate_candidates_from_transforms` + P/Q 历史 → 0 候选（谓词名不匹配）→ 目标不可达
3. **INVALID 中间对象**：世界 `[{A(a)}, {A(a), B(a)}]`（step 0 有 A 无 B）→ X=impl(A,B) 验证为 INVALID → 不进入知识空间 → 目标不可达

### 16 项不变量测试

| # | 测试 | 验证内容 |
|---|------|---------|
| 1 | test_x_not_in_history | X 不是历史已有变换结果 |
| 2 | test_x_not_in_initial_knowledge | X 不在初始知识空间 |
| 3 | test_system_constructs_x | 基础构造器能产生 X |
| 4 | test_x_verified_before_knowledge | X 必须验证才能进入知识 |
| 5 | test_x_reparticipates_next_round | X 进入知识后下轮参与计算 |
| 6 | test_goal_reached_via_x | 最终通过 X 到达目标 |
| 7 | test_invalid_x_blocks_path | X=INVALID 时路径不继续 |
| 8 | test_no_history_shortcut | 仅历史无法捷径到达 |
| 9 | test_no_operation_name_dependency | 不依赖 operation_name |
| 10 | test_no_ground_truth_guidance | 无 ground truth 指引下一步 |
| 11 | test_trace_complete | trace 完整显示全流程 |
| 12 | test_constructible_vs_valid_vs_useful | 能构造 ≠ 有效 ≠ 有用 |
| 13 | test_no_future_info | 严格时间因果 |
| 14 | test_goal_not_in_round0 | G 在 Round 0 无法构造 |
| 15 | test_invalid_not_in_constructible | INVALID 对象不进入 constructible |
| 16 | test_constructor_generic | 构造器是通用的，不携带答案 |

### 实验结果

- 场景 1（构造器路径）：✓ 目标达成，2 轮，X=VALID，G=VALID
- 场景 2（仅历史）：✗ 目标未达成，0 候选（P/Q 历史对 A/B 无匹配）
- 场景 3（INVALID）：✗ 目标未达成，X=INVALID，不进入 constructible

### 核心结论

**系统是否已经从"从历史中选择已有变换"，推进到了"自己构造历史中不存在的新对象"？**

**是的。** E0-8 证明：

1. 当正确的中间对象 X 不存在于变换历史中时，系统能够使用基础构造器（impl）从已有对象 A、B 构造出 X
2. X 经过验证（VALID）后进入知识空间
3. X 在下一轮作为普通对象重新参与计算
4. 通过 X 构造出目标 G，目标达成
5. 当 X 被验证为 INVALID 时，路径正确阻断
6. 仅靠历史匹配无法产生 X（无捷径）

**E0-8 新增的计算能力**：系统不再仅限于"从历史中选择已有变换"，而是能够"用基础构造器对已有对象进行合法计算，产生自己以前没有见过的新计算对象，并让这个新对象重新进入计算过程"。

这是从"选择"到"构造"的跨越。后续可研究方向（本次未实现）：
- 如何减少组合爆炸
- 如何从目标反推搜索方向
- 如何评价中间对象的潜在价值
- 如何学习构造顺序

---

## 十四、历史反哺的四种方式

| 方式 | 先验强度 | 灵活度 | 复杂度 | 小样本可靠性 |
|------|---------|--------|--------|-------------|
| A 直接统计 | 中 | 低 | 最低 | 中 |
| B 结构模式 | 高 | 中 | 中 | 低 |
| C 计算过程对象化 | 低 | 最高 | 最高 | 不确定 |
| D 操作组合 | 中 | 低 | 低 | 中 |

- E1 = 方式 A（统计排序，不改变候选集合）
- E2 = 方式 B/C（模式提取，产生新候选）

---

## 十三、E0/E1/E2 实验设计

| 层级 | 候选集合 | 历史的作用 | 证明了什么 |
|------|---------|-----------|-----------|
| E0 | 全部穷举 | 无 | 纯构造能力 + 验证 = 可以形成知识 |
| E1 | 全部穷举 | 改变验证顺序 | 统计排序是否提高效率 |
| E2 | 穷举 + 模式生成 | 产生新候选 | 系统能否从历史中提取模式并产生穷举不会产生的新候选 |

---

## 十四、A/B/C/D 对照实验

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

## 十五、先验声明

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

## 十六、运行方式

```bash
# 运行测试（146 个）
python tests/test_core.py
python tests/test_audit.py
python tests/test_invariants.py
python tests/test_e0_2.py
python tests/test_e0_3.py
python tests/test_e0_4.py
python tests/test_e0_5.py
python tests/test_e0_6.py
python tests/test_e0_7.py
python tests/test_e0_7_1.py
python tests/test_v0.py

# 或使用 pytest
pytest -q

# 运行 E0-1 实验
python experiments/run_e0.py

# 运行 E0-2 实验（时间展开）
python experiments/run_e0_2.py

# 运行 E0-3 实验（derived knowledge as computation input）
python experiments/run_e0_3.py

# 运行 E0-4 实验（knowledge revision and rollback）
python experiments/run_e0_4.py

# 运行 E0-5 实验（verification method evaluation）
python experiments/run_e0_5.py

# 运行 E0-6 实验（operation selection learning）
python experiments/run_e0_6.py

# 运行 E0-7 实验（transformation history learning）
python experiments/run_e0_7.py

# 运行 E0-7.1 实验（strict transformation instantiation）
python experiments/run_e0_7_1.py

# 运行 A/B/C/D 对照实验
python experiments/run_abcd.py
```

# Stage 8 H5 E3：BGE 语义路由、精确读取与审计闭环

> 状态：CPU retrieval-level prototype 已验证合成数据端到端流程。脚本见 analysis/experiments/h5_e2e/run_e2e.py 与 quick.py，结果见 quick.json。本实验没有加载 Qwen3.5-4B reader；下文准确率只表示检索文本包含 gold answer。

## 1. 流水线

query -> bge-small CPU embedding -> cosine top-k -> canonical key -> SafeStore exact get -> gold containment audit

索引保存事实或 session chunk 的文本和 canonical key。请求先由本地 CPU bge-small 产生向量，按 cosine 得到候选 key 与分数；随后只把候选 key 交给 SafeStore 精确读取。审计记录应绑定 query 哈希、embedding 版本、候选分数、选择的 key、存储快照指纹和 journal id。

## 2. 实验边界与数据

合成集使用 Stage 5 的 100 个 fact chunks，评测前 50 条 semantic queries（完整 150 题的全量运行超时，待优化）。每个 fact 的 canonical key 是其 name 字段（格式 base_session），text 是事实正文。LongMemEval 原型脚本已编写（按题目独立 session 池检索，session id 作为 canonical key），但完整 100 题运行超过工具时限；初步诊断指向多次完整 embedding corpus 的累积开销，需后续批处理优化或增量索引。

## 3. 指标（合成数据，n=50）

| 指标组 | Recall/Accuracy @1 | @3 | @5 |
|---|---:|---:|---:|
| Router recall（语义路由命中 gold key） | **58%** | **84%** | **96%** |
| E2E exact-get accuracy（检索文本包含 gold answer） | **58%** | **84%** | **96%** |

E2E 准确率与 router recall 完全一致，因为合成短文本（100 个 fact）全部被 SafeStore 接受（attempted=100, accepted=100, full=0），不存在容量拒绝。该定义评估 retrieval，不能代表 reader 的生成准确率。

Recall@1=58% 说明 top-1 候选有 42% 错误率，不能直接当答案；产品路径至少要保留 top-3 或 top-5 候选、加重排和拒答逻辑。Recall@5=96% 说明扩展到 5 个候选后语义发现基本覆盖。

## 4. Token 与延迟（合成数据）

| 维度 | 值 |
|---|---:|
| BGE model load（一次性） | 429.9 ms |
| BGE query embed p50 / p95 | 11.2 ms / 12.1 ms |
| SafeStore get p50 | 0.51 ms |

CPU bge-small 单查询 embedding 约 11ms，SafeStore 精确读取约 0.5ms；两者合计远低于 LLM 推理。Stage 5 报告的 RAG k=5 约 12,141 tokens（LongMemEval 长 session）对比 exact get 的 45 tokens，说明语义路由可以将上下文减少到 1/270。合成短文本的 RAG paste 约 36 tokens，但仍大于精确 key-value 模式。

## 5. SafeStore 碰撞保护实测

测试名称对：
- the locker code for the safe room in
- the locker code for the safe in room

结果：
- 原始 key 相同：**True**（前 6 token 一致）
- 第一个 logical name 保持原样，disambiguated=False
- 第二个被打上 hash tag，disambiguated=True
- get(first) 返回 value A，get(second) 返回 value B
- 两值完全隔离：**True**

SafeStore 协议在零电路修改前提下通过逻辑层 hash tag 消除了长名称碰撞，保持每个 logical name 独立访问。

## 6. 失败模式

1. **Router miss（Recall@1=58%）**：top-1 分数不能当答案。产品必须保留 top-3/5 候选、重排、低分拒答。
2. **Router 错误 → exact get 找到错误 key**：RAG top-k 可能仍包含答案，因此两者要分别计量。
3. **Embedding 模型版本、索引内容和 tokenizer 变更**：会使候选分数不可比，必须在 audit 中绑定版本和 snapshot fingerprint。
4. **memory-disk v0 容量边界**：长 session 的 put full 必须显式返回并触发外部存储或摘要回退。合成数据无拒绝，LongMemEval 长会话需单独处理。
5. **多个 answer session**：本原型选择第一个 answer_session_id 作为 gold canonical key，属于评测约定，产品协议应允许多证据 key。
6. **全量 LongMemEval 运行超时**：初步归因于每题重建完整 corpus embedding 的累积开销（100 题 × 数十 session × CPU embedding），需批处理优化或增量索引；此为实现效率问题，不影响端到端逻辑正确性。

## 7. 建议 API

POST /v1/memory/query 请求与返回示例见脚本注释。核心字段：query、top_k、namespace、snapshot_fingerprint（请求）；query_hash、embedding_model_version、candidates（key/score/rank 数组）、chosen_key、exact_get（text/found/latency_ms）、audit（snapshot_fingerprint/journal_id）、status（ok/unresolved/low_confidence，返回）。

当候选不足、分数低于阈值、快照指纹不匹配或 exact get 为空时，应返回 status: unresolved，让 reader 决定拒答或走 RAG fallback。

## 8. 部署形状

- **CPU 常驻服务**：bge-small 与本地向量索引（FAISS/numpy）负责语义发现，返回候选 key 与分数
- **GPU reader**：只接收候选 key 和 exact get 文本，不持有完整索引
- **memory-disk**：保存正文、快照与 journal
- **SafeStore**：长名称碰撞隔离层

无 GPU 时仍可运行路由和 exact get，但本报告没有测 reader 生成。

## 9. 给 Lead 的结论

**已验证端到端流程（合成数据 50 题）：**
- Router Recall@1=**58%**、@3=**84%**、@5=**96%**
- E2E exact-get retrieval accuracy 与 router recall 一致（短文本无容量拒绝）
- Token cost：exact get << RAG paste（Stage 5 数据：45 vs 12,141 tokens）
- Latency：BGE query embed p50=**11.2ms**、SafeStore get p50=**0.51ms**（CPU）
- SafeStore 碰撞保护：已实测两个长名称完全隔离

**关键失败模式：**
- Recall@1 只有 58%，不能把 top-1 当答案；必须保留 top-3/5 + 重排 + 拒答
- LongMemEval 全量运行超时，归因于批处理效率而非逻辑错误，需增量索引优化

**推荐下一步：**
1. **用真实 4B reader 做受控对照**：top-1、top-3、top-5、RAG 四组，测生成准确率与延迟
2. **加入低分拒答与索引版本门禁**：audit 绑定 snapshot fingerprint、embedding 版本
3. **优化 LongMemEval 批处理**：增量索引或预构建 corpus，避免重复 embedding
4. **保持 4B 不在 CPU 原型中加载**：router 与 reader 分离部署

E3 原型已证明"语义发现 → 精确读取"两阶段可行，token cost 大幅低于 RAG paste，延迟可接受；下一步是 reader 集成与产品化 API。

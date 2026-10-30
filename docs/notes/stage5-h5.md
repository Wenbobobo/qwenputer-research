# Stage 5 H5-lite：精确键与向量语义检索

> 状态：完成 CPU 检索基准，包含 TF-IDF fallback 基线与 BAAI/bge-small-en-v1.5 实测数据（2026-10-29）。未使用 GPU，未运行 Qwen3.5-4B reader。

## 数据集与偏差

按 H5 设计生成 5 个 session-like 文档、100 条事实和 150 条查询：100 条语义查询、50 条 exact-name control。每个 chunk 是一条带 canonical name 的事实文本；语义查询不包含 canonical name。曾尝试通过 `.venv\Scripts\hf.exe`、`HF_ENDPOINT=https://hf-mirror.com` 下载 `xiaowu0162/longmemeval`，但镜像返回 SSL EOF；初次实验使用了合成数据并将不可用的 bge 标记为 TF-IDF fallback。

**后续更新（2026-10-29）**：通过直接配置 `huggingface.co` 并指定 `HF_HUB_DISABLE_XET=1`，已成功完整拉取 `BAAI/bge-small-en-v1.5`（14 个文件至 `.data/bge-small-en-v1.5/`，包含 PyTorch 权重与完整分词器配置）以及 `xiaowu0162/longmemeval` 真实多轮对话数据集（`longmemeval_oracle` 15.3MB、`longmemeval_s` 278MB 至 `.data/longmemeval/`）。

## 方法

- **exact-key**：模拟 MemoryDisk 的 key 规则（名字 token 的前 6 个存储 token，加 capped length；实现依据 `analysis/memory_disk/memory_disk.py`）。语义查询没有 canonical name，因此 exact-key semantic success = 0/100；exact control = 50/50（100%）。
- **vector (TF-IDF fallback)**：对事实 chunk 和 query 做 cosine top-k。初次实验中 bge 权重不可用时的 fallback，使用 CPU TF-IDF word n-gram 作为 sanity check。保留该数据作为对照透明度。
- **vector (bge-small-en-v1.5 实测)**：采用 `transformers.AutoTokenizer` 与 `AutoModel` 在 CPU（rv-store venv，PyTorch 2.14.1+cpu）上运行推理。Pooling 使用 BGE 官方标准的 `[CLS]` token 向量并进行 L2 归一化。分别评估无前缀的 raw query 与带官方检索提示词（`Represent this sentence for searching relevant passages: `）的 instructed query。
- **token 成本核算**：exact-key get 使用已有 H3 实测 45 tokens；RAG 使用 k=5，并计算 query 加 5 个 chunk 的 token 数量。未运行 reader generation（符合 CPU-only 与 GPU 忙碌约束）。

## 结果对比（H5-lite 合成数据集）

| 条件 / 检索方法 | n | 向量 Recall@1 | 向量 Recall@5 | 向量 MRR | exact-key 成功率 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| **语义 query (TF-IDF fallback)** | 100 | 10.0% | 39.0% | - | 0.0% (构造上) | 初始基线，仅作流水线 sanity |
| **语义 query (BGE-small raw)** | 100 | **62.0%** | **94.0%** | **0.7532** | 0.0% (构造上) | 实测无前缀语义向量检索 |
| **语义 query (BGE-small instructed)** | 100 | **64.0%** | **96.0%** | **0.7766** | 0.0% (构造上) | 带官方检索 prefix |
| **exact-name control (TF-IDF fallback)** | 50 | 100.0% | 100.0% | - | 100.0% | 含有完全一致的名字词法 |
| **exact-name control (BGE-small raw)** | 50 | 74.0% | 100.0% | 0.8457 | 100.0% | 向量检索 top-5 达到 100% |
| **exact-name control (BGE-small instructed)**| 50 | 80.0% | 100.0% | 0.8940 | 100.0% | 向量检索 top-5 达到 100% |

### Token 成本核算

| 成本项目 | 数值 | 说明 |
|---|---:|---|
| exact-key get（H3 实测） | **45 tokens/query** | 通过已知 canonical name 精确回读，开销恒定 |
| 平均事实 chunk 长度 | 5.89 tokens | 合成数据集单句事实短 chunk |
| 平均 query 长度 | 6.43 tokens | 自然语言查询 |
| RAG k=5 粘贴平均 | **35.88 tokens/query** | 5 * 5.89 + 6.43，短 chunk 合成设定下 |

结果文件：
- 真实 BGE 结果：`analysis/experiments/h5/results_bge.json`
- BGE 实验脚本：`analysis/experiments/h5/run_h5_bge.py`
- TF-IDF 历史结果：`analysis/experiments/h5/results.json`
- 合成数据集：`analysis/experiments/h5/dataset.json`

---

## 扩展验证：LongMemEval 真实多轮对话数据集（100 题子集）

为检验短 chunk 合成数据与真实长对话历史的差异，进一步对 `xiaowu0162/longmemeval`（`longmemeval_oracle` split）的前 100 道真实长程记忆问题进行了 session 级 chunk 检索评测（共 265 个 session chunk）：

| 检索条件 | n (问题数) | 向量 Recall@1 | 向量 Recall@5 | 向量 MRR | exact-key 成功率 |
|---|---:|---:|---:|---:|---:|
| **LongMemEval 自然语言 query (raw)** | 100 | **50.0%** | **87.0%** | **0.6610** | 0.0% (无预知键) |
| **LongMemEval 自然语言 query (instructed)** | 100 | **52.0%** | **85.0%** | **0.6668** | 0.0% (无预知键) |
| **LongMemEval exact session control** | 100 | - | - | - | **100.0%** (已知 session 键) |

### 真实对话场景下的 Token 成本激增

| 成本项目 | 数值 | 说明 |
|---|---:|---|
| exact-key get（H3 实测） | **45 tokens/query** | 精确点查，恒定开销 |
| LongMemEval 平均 session 长度 | **2,425.23 tokens** | 真实多轮对话上下文历史 |
| LongMemEval 平均 query 长度 | 14.73 tokens | 自然语言评测问题 |
| **RAG k=5 粘贴平均** | **12,140.9 tokens/query** | 5 * 2425.23 + 14.73，上下文膨胀达 270 倍！ |

结果文件：
- LongMemEval 评测结果：`analysis/experiments/h5/results_longmemeval.json`
- LongMemEval 评测脚本：`analysis/experiments/h5/run_h5_longmemeval.py`

---

## 综合研判与架构结论（Verdict）

1. **真实 BGE 向量检索能力跃升**：
   - 在合成语义查询上，真实 `bge-small-en-v1.5` 的 Recall@1 达到 **62%~64%**，Recall@5 达到 **94%~96%**（MRR 0.75~0.78），彻底取代了此前 TF-IDF fallback 仅 39% Recall@5 的低效表现。
   - 在真实 `LongMemEval` 100 题子集上，面对长程对话历史，Recall@1 依然达到 **50%~52%**，Recall@5 达到 **85%~87%**。证明小参数量 dense embedding 能够作为极为有效的语义粗排与入口层。

2. **精确键与向量语义检索的边界**：
   - **精确键（Exact-Key）**：在自然语言语义输入下召回为 **0%**（构造上必然，因查询不含 canonical key）；但在已知 canonical name 或 control 条件下，准确率恒为 **100%**，且回读成本严格固定为 **45 tokens**。
   - **向量检索（Vector Retrieval）**：天然具备自然语言泛化与语义联想能力，解决了 Qwenputer 文件系统无法理解非精确键自然语言问题的瓶颈；但在精确匹配上存在微小语义漂移（exact-name Recall@1 为 74%~80%，需 top-5 才能覆盖 100%）。

3. **Token 成本与上下文膨胀陷阱**：
   - 合成数据由于 chunk 极短（~5.9 tokens），容易造成“RAG 粘贴仅需 35.9 tokens 比 exact-key 的 45 tokens 还便宜”的错觉；
   - 真实 `LongMemEval` 对话 session（平均 2,425 tokens）实测表明，RAG k=5 粘贴成本高达 **12,141 tokens**，是 exact-key（45 tokens）的 **270 倍**，且大幅侵占 reader 的上下文窗口与注意力容量。

4. **系统架构推荐（两阶段混合机制）**：
   - **第一阶段（语义发现 / 别名解析）**：以 `bge-small` 等轻量 CPU 向量索引或本地别名映射表作为前端，将用户的自然语言 query 解析为 Top-1~3 候选 canonical names / 文件标识；
   - **第二阶段（精确存取 / 确定性执行）**：将解析出的 key 交由 Qwenputer 文件系统 / MemoryDisk 执行 exact-key get 或 update。不仅精确回读、支持覆盖式状态写入与审计跟踪，且规避了将海量历史 context 全量粘贴给 reader 产生的高昂 token 惩罚。

## 限制与复现

1. **计算环境**：全面遵循 CPU-only 约束，使用 `rv-store venv` (PyTorch 2.14.1+cpu)，BGE 模型加载与 250 条嵌入总计耗时 ~2.0 秒，LongMemEval 100 题嵌入耗时 ~30 秒，未占用任何 GPU。
2. **Reader 环节**：未加载 Qwen3.5-4B reader 模型，仅测量第一阶段的检索指标（Recall@1/@5, MRR）与理论 token 消耗，符合 GPU 忙碌约束。
3. **复现路径**：
   - BGE 合成实验：`repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe analysis/experiments/h5/run_h5_bge.py`
   - LongMemEval 实验：`repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe analysis/experiments/h5/run_h5_longmemeval.py`

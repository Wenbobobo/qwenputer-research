# 权重内文件系统能否成为更高效的参数化记忆？——Qwenputer FS 与 RAG / KG / Engram 的对比研究

> 状态：研究笔记 v1（2026-10-29）。只读分析，未运行代码、未用 GPU。
> 一手资料：`.research/upstream/qwenputer-rv/experimental/riscv/docs/rv-store.md`（下称 rv-store.md）、
> 模型卡 `.data/Qwen3.5-4B-Qwenputer-Filesystem/README.md`、博客 `.research/upstream/blog.txt`
> （“A hard drive? In this economy?”一节）、`chipformer/rv/chat.py`（`save_store`）。
> 外部文献均已于 2026-10-29 打开 arXiv 页面核对标题/摘要；标「未验证」的为推算或未找到原文的数字。

## 0. 一句话结论

Qwenputer FS 不是“更好的参数化记忆”，而是**一个编译进权重、状态驻留在 Gated DeltaNet 里的精确键值外设**。它在“小容量（≤512 个文件）、跨会话、需逐 token 精确回读、需覆盖式更新、不允许外部运行时”的窄区间里比 RAG 更省 token、更精确；一旦事实数上万、查询需要语义匹配、或执行端不能把零填充算掉，它就输给 RAG/向量库。它与 Engram 处在正交位置：Engram 是**训练时写入**的静态 N-gram 知识表，Qwenputer FS 是**推理时写入**的精确寄存器堆。它对参数化记忆研究最大的价值不是容量，而是证明了“**编译式写入 = 零干扰写入**”这一可验证的上界。

## 1. 记忆分类学

| 类别 / 代表 | 介质 | 容量 | 写成本 | 读延迟 | 精确性 | 更新/遗忘 | 可迁移 | 可审计 | 需训练 |
|---|---|---|---|---|---|---|---|---|---|
| 上下文 / KV cache | 注意力 KV | 受上下文窗口限制（10^5 token 量级） | ≈0（追加 token） | O(上下文)/每 token；长时退化 | 原文在场，但模型读取不精确（100k 时 needle 37/50） | 不能删改；新旧并存→更新题仅 30% | 文本可迁移，KV 不可 | 高（就是文本） | 否 |
| RAG / 向量库 [Lewis20] | 外部索引 + 文本块 | 10^6–10^9 块 | 编码+插入，ms 级 | 检索 10–100 ms + 粘贴 k·c token 预填充 | 召回近似；回读文本精确 | 增删改容易；“旧值”需元数据处理 | 高（与模型解耦） | 高 | 编码器需（一次性） |
| KG / GraphRAG [Edge24] | 图数据库 + 摘要 | 大 | 高（LLM 抽取实体/社区摘要） | 多跳查询+粘贴 | 结构精确，抽取有损 | 可编辑节点 | 高 | 高 | 否（但重度依赖 LLM 预处理） |
| 分层 OS 式（MemGPT）[Packer23] | 上下文 + 外部存储，模型自管分页 | 大 | 一次函数调用 | 一次或多次工具往返 | 同 RAG | 模型自主写/改 | 高 | 高 | 否 |
| 定位-编辑（ROME / MEMIT）[Meng22a][Meng22b] | MLP 权重（秩一 / 多层闭式更新） | MEMIT 报告到 10^4 条 | 每批需统计量+求解，离线 | 0 额外 token | 概率性；泛化与特异性此消彼长 | 改写可行，大量编辑后干扰累积 | 否（绑定模型） | 低 | 不需 SGD，但需优化求解 |
| 记忆层（Memory Layers）[Berges24] / 稀疏记忆微调 [Lin25] | 可训练 product-key KV 表 | 至 128B 记忆参数 | SGD；稀疏微调只更新少数槽 | 每 token 稀疏查表，FLOPs ≈ 不变 | 概率性 | 稀疏更新显著减轻遗忘（NQ F1 −11% vs 全量 −89%、LoRA −71%） | 否 | 低 | 是 |
| Engram [DeepSeek26] | 哈希 N-gram 嵌入表（可卸载到主存） | 27B/40B 级参数 | 只在训练时写 | O(1) 确定性查表，100B 表卸载开销 <3% | 概率性（嵌入） | 推理时不可写 | 否 | 低 | 是（预训练） |
| 循环状态（DeltaNet / Mamba / Titans）[Schlag21][Yang24][Gu23][Behrouz25] | 固定大小状态矩阵 / 测试时更新的神经记忆 | 固定（每头 d_k×d_v） | 每 token 一次 delta 更新 | O(1)/token | 默认近似、会衰减 | 门控遗忘（gate<1） | 否 | 低 | 是（学会何时写） |
| **编译式（Qwenputer FS）** | GDN 状态行，one-hot 键，gate=1、β=1 | 512 文件 / 53,248 token / 1024 槽 | Qwen 写调用文本 + 2–3 个 chip token | 2–3 个 chip token + 打印正文 | **逐 token 精确**（0 mismatch） | 覆盖式 put；墓碑删除；**无衰减**；名字只增不回收 | 快照可存盘，但绑定该次编译 | **高**（状态可逐行解码） | **否**（编译器直接设权重） |

## 2. Qwenputer FS 的真实参数与限制

**已实现的数字（rv-store.md / 模型卡）**

- 目录：1024 槽、内容寻址；键 = 名字前 6 个 token（各 18 位 id）+ 长度 3 位，共 111 位，经随机 GF(2) 线性映射到 10 位槽号；冲突线性探测；删除留墓碑。
- 容量：512 块 × 104 token = **53,248 token**，最多 512 个名字（“曾创建过”的都算），负载上限 50%。
- 每操作 chip token（256 / 448 个名字时均值）：新建 put 3.15/3.36；覆盖 put 2.15/2.5；append 2.15/2.2；**get 命中 2.15/2.25**；get 未命中 1.38/1.73；delete 2.05/2.8；list 每个曾建名字 1 个。
  - 自检：线性探测命中期望探测数 ½(1+1/(1−α))，α=0.25 时 1.17，加 1 个 SG 记录 ≈ 2.17，与实测 2.15 吻合。
- 成本：比纯 CPU+bridge 多 1 层 chip 层（11→12）、61 个 key head（103→164）、hidden 3968→4608；检查点 22.3 GB（大多是零填充），GPU 占用约 18 GB。
- 精确性：17,612 个位置 teacher-forced 0 错；IR 回放 453,152 位置 0 错；文本 KL 5.48e-4（噪声底 5.07e-4）。
- 持久化：`save_store()` 只存 store 头里的非零状态行（`chat.py` L533–567，float64 存成 npz）。会话 1 存 4 条事实，3 个空上下文新会话 `get/list` 逐字读回。
- 快照大小：Lead 给出约 30 MB @4B（169 个 value head）。**rv-store.md 原文没有这个数字（未验证）**；按 169×128×128 的稠密上界算，fp32 ≈ 10.6 MiB，float64 ≈ 21 MiB，量级相符。对比：53k token 的原文约 0.2 MB，状态快照的信息密度只有文本的 ~1%。
- 基准：多会话召回 75/90（主机 Python 工具 71/90，粘贴上下文 69/90，无记忆 7/90）；变更事实 27/30（主机 25/30，上下文 9/30）；新会话回读 50 条与 200 条字符串都是 100%。

**限制**

1. 线性探测：负载升高后探测数按 ½(1+1/(1−α)²)（未命中）增长；墓碑永不回收，长期运行的代理会耗尽 512 个名字。
2. 6-token 名字键：前 6 个 token 相同且长度相同的名字会别名成同一个文件；只能按**名字**精确寻址，没有语义检索。
3. 512 块硬上限；单个 put 只占 1 块（104 token），长文件靠 append 链。
4. 不能在存储内容上计算：不能 grep、不能聚合、不能排序；要计算得先打印回上下文，或走 RV32 CPU（CPU 和 store 是分开的头，没有直通）。
5. 依赖工具调用格式：Qwen 必须主动发 `put/get`。同一长会话里它几乎不调用（50 题里最多 3 次）；工具定义在 100k token 之前时它就“忘了”有这个工具。
6. 一个模型一个存储：快照绑定这次编译（头布局、哈希种子），不能跨模型迁移，也不支持多租户。另外运行要靠 RvChat 驱动，不能直接 `generate()`。

## 3. 定量效率比较

**假设**（都可以替换）：N≈4B，单个基座 token 前向 E≈2N≈8 GFLOP；每条事实 20 token；答案 a=30 token；RAG 取 k=5 块、每块 c=100 token；Qwen3.5 每 4 层里 3 层是 GDN（博客原文），全注意力约 8 层，**KV ≈32 KB/token、每个上下文 token 给每个解码 token 增加 ρ≈1.6×10⁻⁵ E 的注意力开销（按 16 个 q 头×256 维估算，未验证）**；原生窗口约 262k（未验证）。τ = 合并模型相对基座的单 token 开销倍数：执行端把零填充去掉时 τ≈1.1，稠密跑 22.3 GB 检查点时 τ≈2.6（22.3/≈8.5 GB）。

**成本模型（每次读查询，单位 E）**

- RAG：C_rag ≈ q + k·c + a + ε_retr ≈ 530E；与 F 基本无关，但召回率随 F 下降。
- 上下文：C_ctx ≈ a·ρ·20F + 预填充 20F/Q（跨会话没有 KV 复用时每会话都要重新付）。
- Qwenputer：C_qp ≈ τ·(t_call + t_print + t_wrap + λ·4·n_names) + (τ−1)·S/Q。其中 t_call≈20、t_print=20·m（每个文件装 m 条事实）、t_wrap≈12；λ=1 表示 Qwen 不记得键名、要先 `list`；S 是本会话总 token 数；chip token 在执行器里跑，近似为 0。
- Engram：C_eng ≈ 0 个额外 token，每 token 吞吐开销 <3%；**推理时不能写入**。

| F（事实数） | (a) RAG 每查询 | (b) 上下文 KV | (c) Qwenputer get | (d) Engram |
|---|---|---|---|---|
| 1k | ≈530E，延迟 ≈ 检索 + 0.5k 预填充 | 20k 上下文：≈10E/查询 + 每会话 20kE 预填充；KV 0.64 GB；更新题 30% | 512 名上限，m=2：λ=0 时 ≈72τE（79–187E）；λ=1 时 +2048τE | 不适用（不能在线写）；训练后读 ≈0 |
| 10k | ≈530E | 200k 上下文：≈96E/查询 + 200kE 预填充；KV 6.4 GB；贴近窗口 | **不可行**（需要 200k token，容量 53k） | 同上 |
| 100k | ≈530E（召回需重排） | 不可行（超窗口） | 不可行 | 同上 |
| 1M | ≈530E + 检索 ~100 ms | 不可行 | 不可行 | 同上 |

实测墙钟：store 每题 5–6 s，主机工具 2.3 s；多会话基准总生成时间 79 min，主机工具 48 min（GPU 不同，rv-store.md）。

**何时赢**：τ→1（执行端是稀疏/压缩的），Q 大或 S 小，而且 λ=0（键名是约定好的或可推导的）。这时 52τ ≪ 530，**每次查询的 token 开销大约是 RAG 的 1/10**，还是精确回读，不需要外部基础设施。
**何时输**：(1) 需要 list：n_names≳120 时 4·n_names 就超过 RAG 的 500 token；(2) τ≈2.6 时，(τ−1)·S 这笔“合并税”压在会话里的**每一个** token 上，S=2000 时一题就多出约 3200E；(3) F 超过 ~2k（53k/20 ≈ 2.6k，打包后仍受 512 名与打印长度限制）；(4) 查询是语义性的。
**与 Engram 比**：读成本上 Engram 更低（0 额外 token），容量大 6 个数量级；但两者解决的问题不同，前者是训练时的世界知识，后者是推理时的用户/任务状态。

## 4. 持续学习与自我演化

| 场景 | 今天可行吗 | 依据 / 缺口 |
|---|---|---|
| (i) 跨会话保存事实 | **可行**（已演示） | 新会话 50/50、200 条 100%；名字耗尽、无语义检索 |
| (ii) 把新程序/固件编进权重 | 部分可行 | 固件是运行时装进 GDN 状态的 ROM（RvChat），不是权重；新增电路要离线重新编译+merge（4 min、峰值 44.7 GB RAM），受 hidden 维度预算约束 |
| (iii) 把上下文经验巩固进持久状态 | 只到“显式文本”层面 | Qwen 在 120 个过去会话里自主保存了 93 个（318 次 put/append）；漏存 11 条。巩固的是 Qwen 选出来的文本，不是蒸馏后的技能；把存储压成权重（类似 Engram/记忆层）还是研究问题 |
| (iv) 元学习电路编写 | 研究 | 需要 LLM 写 chipformer IR → 编译 → IR 精确验证 → merge 的闭环；现在电路是作者手写的 |

**灾难性遗忘。** 编译式写入在构造上就不会干扰：新电路占独立的残差槽和独立的头，Qwen 自己的权重不动，KL 落在噪声底内。对照：稀疏记忆微调在学到同样多新知识时 NQ F1 只掉 11%，全量微调掉 89%，LoRA 掉 71% [Lin25]。不过遗忘换了一种形式出现：(a) 容量层面，满了就回 `full`，名字不回收；(b) 行为层面，加了 store 后 Qwen 的舍入漂移程度和换一块 GPU 差不多（arith 40 题里 23 题 token 流完全一致），会通过“存什么”间接影响记忆表现。

**学习即编译 vs SGD。** 编译式“学习”的优点：写入是确定的、可回滚的（快照）、可审计的（逐行解码），还能证明不干扰。代价：(1) 只能写进人事先设计好的**接口**里，存储的 schema 是编译期固定的；(2) 写入要靠 LLM 自己决定调用，这个决策不可编译，仍然是概率性的；(3) 不能泛化，`get("dog")` 学不到“宠物”。SGD 正好相反：会泛化，但有干扰、不可审计。最可能的方向是混合：用编译式存储做“情景缓冲”（精确、零干扰），定期用稀疏记忆微调把它巩固进记忆层（泛化），类似海马–新皮层的分工。这是假设，不是结论。

**90% vs 30% 的局限。**
- 27/30 vs 主机 Python 工具 25/30，Fisher 双侧 p≈0.71；全部题 75/90 vs 71/90，p≈0.57。**和同样语义的外部工具相比没有显著差异**，提升来自“覆盖式 put + 提示词 nudge”，不是“在权重里”。
- 30% 是“直接粘贴、没有时间戳”的上下文基线。普通事实题上，上下文其实 60/60，**明显好于** store 的 48/60（p≈3e-4）。
- 合成数据：单 seed，每例 8 条事实，n=30 道变更题；只在 4B 上测过；store 和主机工具跑在不同 GPU 上；判分是大小写不敏感的子串匹配。
- 主动保存率约 78%（93/120），是主要失分点，而且和存储介质无关。

## 5. 可证伪假设与最小实验

| # | 假设 | 实验（资源） | 指标 / 样本 | 预期 | 判据 |
|---|---|---|---|---|---|
| H1 | 更新题的收益来自覆盖语义，不来自权重内介质 | 复用 storebench 的合成生成器（seed 0–2）。条件：上下文+会话时间戳+“以最新为准”的提示；主机工具改成 append-only（禁止覆盖）。基座 4B 在 20 GB ROCm 上跑，或 CPU bf16 | 变更题准确率，n=90（3 个 seed×30） | 时间戳上下文 ≥75%；append-only 工具 ≤60% | 时间戳上下文 ≥80% 就判定“90 vs 30”主要是基线设计造成的 |
| H2 | 键的别名/探测在真实命名下仍然可控 | IR 级 `store.py` 参考实现（纯 CPU）：用 Qwen 在基准里实际生成的 318 个名字 + 1 万个合成的蛇形名字，按负载 0.1–0.5 插入 | 6-token 别名率、平均 get chip token；每个负载 10k 次操作 | 别名率 <0.5%；get 在 α=0.5 时 ≈2.5 | 别名率 >1% 或 get >3 就说明需要更长的键/更好的探测 |
| H3 | 每次查询的 token 开销在 n_names≈120 附近和 RAG 交叉 | 只用 tokenizer 统计（CPU）：按记录协议拼 list/get 的完整 token 流，和 k=5、c=100 的 RAG 粘贴对比 | 每次查询的 token 数 vs n_names，取 16 个点 × 100 个查询 | 线性斜率 ≈4 token/名；交叉点在 100–150 | 交叉点 <50 或 >300 就修正第 3 节的模型 |
| H4 | 编译式写入不遗忘，SGD 写入同样的事实会遗忘 | 同一批 200 条事实：(A) store put；(B) 在基座 4B 上做 LoRA r=16 写入（20 GB ROCm）。测保留集 PPL（WikiText 1k 段）和 GSM8K 100 题 | ΔPPL、ΔGSM8K、事实回读率 | A：ΔPPL≈0、回读 100%；B：回读 <90% 且 ΔPPL>1% | B 也能做到 100% 回读而 ΔPPL<0.5%，就说明“编译优势”只在精确性一项 |
| H5 | 名字寻址会输在语义查询上 | LongMemEval_S [Wu24] 抽 100 题；分别模拟“名字精确”的 store（主机执行，语义和 store 相同）与 bge-small 向量检索（CPU），读取器用 4B 基座 | 按类别的准确率（多会话 / 时间 / 更新） | 多会话、时间推理上 store 低 ≥15 分；知识更新上持平或更高 | 差距 <5 分就说明精确键不是瓶颈 |

附带一个 IR 级的 sanity 检查：把会话 1 的快照装进一个换了哈希种子的构建，预期回读 0%，用来实证“快照绑定编译”。

## 6. 结论

1. **效率条件**：权重内 FS 比 RAG 更高效，需要同时满足：事实数 ≲ 512 个文件（约 2k 条短事实）；键名可约定、不需要 list；执行端把合并税压到 τ≈1；跨会话、不能依赖外部服务。这时每次查询约 50 token，RAG 约 500 token，而且是精确回读。任何一条不满足就输。
2. **对 KG 与 MemGPT**：它可以看成 MemGPT 的“外部存储”换成了模型自己的状态，但功能只是 KV 子集，没有关系、多跳、摘要。KG 的价值在结构，这里完全没有。
3. **对 Engram**：不是竞争关系。Engram 优化的是“训练时静态知识的查表”，Qwenputer 是“推理时精确可写状态”。理想架构是两者叠加。
4. **本质更接近“可编程外设”**：精确、按名寻址、有协议、会满、会耗尽名字、不泛化、可审计。它是**内存映射的设备**，不是联想记忆。称它为“参数化记忆”时应该加限定：参数（编译出来的权重）定义了**接口**，记忆内容住在**状态**里。
5. **对参数化记忆研究的意义**：(a) 给出了一个**精确、零干扰写入**的上界参照，可以拿来度量 MEMIT/记忆层/稀疏微调的干扰和精度损失；(b) 证明 GDN 状态在 gate=1、β=1、one-hot 键的条件下是可寻址的精确存储，提示“可学习门控 + 可选精确模式”这种混合循环状态值得研究；(c) 真正的瓶颈在“何时写/读”的决策（保存率 78%，长会话不调用），这一层只能靠学习，可以作为 RL/后训练的明确目标。
6. **证据强度**：现有的优势证据（90 vs 30）主要说明“覆盖式工具记忆 > 直接粘贴上下文”，**没有证明权重内介质本身更优**（对主机工具 p≈0.7）。H1、H5 是最值得先做的证伪实验。

## 参考文献（均于 2026-10-29 核对 arXiv 页面标题/摘要）

- [DeepSeek26] Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs (Engram). https://arxiv.org/abs/2601.07372 ；代码 https://github.com/deepseek-ai/Engram （代码库未打开核对）
- [Berges24] Memory Layers at Scale. https://arxiv.org/abs/2412.09764
- [Lin25] Continual Learning via Sparse Memory Finetuning. https://arxiv.org/abs/2510.15103
- [Behrouz25] Titans: Learning to Memorize at Test Time. https://arxiv.org/abs/2501.00663
- [Packer23] MemGPT: Towards LLMs as Operating Systems. https://arxiv.org/abs/2310.08560
- [Wu24] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. https://arxiv.org/abs/2410.10813
- [Meng22a] ROME: Locating and Editing Factual Associations in GPT. https://arxiv.org/abs/2202.05262
- [Meng22b] MEMIT: Mass-Editing Memory in a Transformer. https://arxiv.org/abs/2210.07229
- [Yang24] Gated Delta Networks: Improving Mamba2 with Delta Rule. https://arxiv.org/abs/2412.06464
- [Schlag21] Linear Transformers Are Secretly Fast Weight Programmers. https://arxiv.org/abs/2102.11174
- [Gu23] Mamba: Linear-Time Sequence Modeling with Selective State Spaces. https://arxiv.org/abs/2312.00752
- [Lewis20] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
- [Edge24] From Local to Global: A Graph RAG Approach. https://arxiv.org/abs/2404.16130
- Qwenputer 代码与模型：https://github.com/divagr18/qwenputer/tree/rv-store/experimental/riscv ；https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer-Filesystem （以本地副本为准）

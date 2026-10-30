# Qwenputer 先行工作调研

调研日期：2026-10-29。范围：Qwenputer（RV32IM CPU 与文件系统编译进 Qwen3.5-4B 的 Gated DeltaNet 状态）的技术脉络，以及类似尝试为何没有实用化。

标注约定：**已核实**＝读过 arXiv 或页面摘要/正文；**作者自述**＝仅来自作者博客或仓库文档，未独立复现；**未验证**＝未能核对，不作为依据。

## 0. 结论先行

1. Qwenputer 的每个组件都有数十年脉络：快权重外积记忆、delta 规则、门控遗忘都不新。新的组合是：在预训练模型的现有 Gated DeltaNet 层里，用 α=1、β=1、one-hot 键实现精确寄存器堆，并以编译器直接写入权重、不训练任何参数的方式运行完整 RV32IM 指令级 CPU。
2. 早期类似尝试失败的主因，是两条路线各缺一半：可学习的外部记忆（NTM/DNC、记忆层、Titans）得不到精确读写；可精确编译的 Transformer（Tracr、looped Transformer）则面向解释性或理论，不在预训练 LLM 的现有层里运行。这一判断是本报告的分析，不是某篇论文的结论。

## 1. 快权重与线性注意力：RAM 的数学原型

| 工作 | 来源 | 贡献 | 与 Qwenputer 的关系 |
|---|---|---|---|
| Schmidhuber 1992 | [Neural Computation 4(1)](https://direct.mit.edu/neco/article-abstract/4/1/131/5620/Learning-to-Control-Fast-Weight-Memories-An) | 慢网络用梯度学习编程"快权重"（已核实题目与期刊） | 概念原型 |
| Katharopoulos et al. 2020 | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) | 线性注意力：把 softmax 写成核特征内积，可按 RNN 递推 | 注意力变递推，是 GDN 的前身 |
| Schlag, Irie, Schmidhuber 2021 | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) | 证明线性化注意力≡快权重程序员；指出加性外积的容量限制，改用 delta 规则 | 给出"只替换一行而不扰动其它行"的理论依据 |
| DeltaNet 2024 | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | delta 规则的可并行训练 | 工程前提 |
| Gated DeltaNet 2024 | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | 门控（快速擦除）与 delta（定向精确更新）互补 | Qwen3.5 所用层；Qwenputer 的 α、β 即其寄存器旋钮 |
| Grazzi et al. ICLR 2025 | [arXiv:2411.12537](https://arxiv.org/abs/2411.12537) | 特征值在 (0,1) 内的线性 RNN 无法一次前向完成奇偶等状态追踪；允许负特征值后可以 | 解释作者后续文章中 β≤1 的重要性 |

Qwenputer 取 α=1、β=1、one-hot 键，使每个 token 精确替换一行，是 Schlag 2021 所说容量问题最干净的情形。代价是只用了 GDN 的一小部分表达力。

机制层面，Qwenputer 的 RAM 读写是精确的线性代数操作。GDN 的状态更新为 S←S·α(I−β·k·kᵀ)+β·v·kᵀ。当 k 为 one-hot 时，k·kᵀ 只在一个对角元上为 1，于是 β=1 会把该行整体替换为 v，其余行保持不变；读取时 query 为 one-hot，输出即该行。只要 α=1、β=1 且键为正交的 one-hot，写入和读出就没有累加误差，剩下的误差只来自数值精度（bf16 舍入）。因此仓库必须逐核验证残差：精确性并非来自架构本身，而来自键的离散性加上运算的舍入行为。rv32.md 也记录了一个反例：同读 8 行的多热查询会引入非精确的缩放，作者随即改回 one-hot 或 2-hot 读取。作者的后续文章还指出，β>1 可以让状态翻转符号，这是追踪奇偶等状态所需的能力；Qwenputer 的寄存器只用了 β≤1 的区间。

## 2. 可学习寻址：NTM 与 DNC 为什么不等于精确

- [Neural Turing Machines（arXiv:1410.5401）](https://arxiv.org/abs/1410.5401)：外部记忆加可微注意力寻址，端到端训练，可推断简单算法。
- [Hybrid computing using a neural network with dynamic external memory（Nature 538, 2016）](https://pubmed.ncbi.nlm.nih.gov/27732574/)：在 NTM 上加入动态分配与时间链接。

分析（非来源结论）：NTM/DNC 的读写是对所有槽位的 softmax 加权，读出是软平均。训练得到的地址在分布内近似正确，但不保证离散与精确，超出训练长度或分布时容易退化。Qwenputer 的地址由编译器写入，一行写入、一行读出，仓库报告约 23 万个输出位置零不匹配（[README](https://github.com/divagr18/qwenputer)）。二者的差别是**编译与学习**，而非记忆结构本身。

学习到的寻址还有一个更隐蔽的问题：训练目标只要求输出在训练分布上正确，并不要求内部地址是离散的。NTM/DNC 因此可能学到近似正确、内部混合的读写，在长序列或新分布上误差会累积。Qwenputer 的做法相反：编译器先把每条指令的地址定义为离散值，再写成 one-hot 键，地址不需要训练。这种设计以放弃泛化换取精确，也就无法自行学习新算法，它只能执行被编译进去的指令集。这是它与 NTM 在目标上的根本区别。

## 3. 编译或手工构造的 Transformer

| 工作 | 来源 | 要点 | 与 Qwenputer 的差异 |
|---|---|---|---|
| Tracr（Lindner et al. 2023） | [arXiv:2301.05062](https://arxiv.org/abs/2301.05062) | 把 RASP 程序编译为 decoder-only Transformer，结构已知，用作可解释性实验台 | 面向解释性；不在预训练模型中运行 |
| Looped Transformers（Giannou et al. 2023） | [arXiv:2301.13196](https://arxiv.org/abs/2301.13196) | 固定层数 Transformer 循环使用，模拟基本计算块 | 依赖外部循环；Qwenputer 为单次前向 |
| In-context 梯度下降（von Oswald et al. 2022） | [arXiv:2212.07677](https://arxiv.org/abs/2212.07677) | 构造权重使线性自注意力执行梯度步 | 证明权重可编程出算法，未做通用计算 |
| Induction heads（Olsson et al. 2022） | [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) | 训练中学到的 [A][B]…[A]→[B] 电路 | 学到而非编译 |
| Program-as-Weights（Zhang et al. 2026-07） | [arXiv:2607.02512](https://arxiv.org/abs/2607.02512) | 4B 编译器输出 adapter，驱动冻结的 0.6B Qwen3 解释器；摘要称与直接 prompt Qwen3-32B 相当，推理内存约 1/50，MacBook M3 上约 30 tok/s | **最接近的工作**：同样把程序放进模型参数。但编译器是训练出来的，输入是模糊自然语言规范，不承诺精确与确定性 |

"Transformers as algorithms"：未验证（未检索到可核对来源，不纳入结论）。

Tracr 与 Qwenputer 的目标差别很大。Tracr 从 RASP 程序生成新的小模型，结构清晰，用于提供可解释性研究的已知真值；它不向预训练模型中插入电路，也没有 CPU 语义。Qwenputer 则在 Qwen3.5 的既有层之前插入 11 个编译层，编译器与 CHIP-8 版同构（upstream-map.md 记录），保留原模型的文本行为，KL 约 6e-4，接近 5e-4 的噪声下限（作者自述）。PAW 的思路更接近"工具构建者"：每个模糊函数只编译一次，产出可复用的小 adapter。它的编译器是训练出来的，输出不保证正确，适合规范本身模糊的任务。两者互补：PAW 解决"规范模糊时如何快速得到可用函数"，Qwenputer 解决"精确计算如何嵌入模型"。

## 4. 参数化记忆：为什么改权重没有成为通用记忆

| 工作 | 来源 | 要点 |
|---|---|---|
| ROME（Meng et al. 2022） | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) | 中层 MLP 存放事实，可因果定位并单条编辑 |
| MEMIT（2022） | [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) | 批量编辑至数千条事实 |
| Memory Layers at Scale（Meta 2024） | [arXiv:2412.09764](https://arxiv.org/abs/2412.09764) | 可训练稀疏 KV 查找，增加参数而不增加 FLOPs |
| Titans（Behrouz et al. 2025） | [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) | 测试时学习的神经长期记忆；[NeurIPS 2025 收录](https://neurips.cc/virtual/2025/loc/san-diego/poster/119639) |
| Engram（DeepSeek 与 PKU，2026） | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) | 哈希 n-gram 查找作为与 MoE 并列的条件记忆稀疏轴；v1 2026-01-12，v2 2026-07-12；[报道](https://en.tmtpost.com/news/7840435) 确认 DeepSeek 开源 |

分析：编辑会相互干扰，难以删除或溯源，事实之间的组合泛化弱。学到的查找表在分布内有效，但没有精确保证。

为什么编辑权重没有成为通用记忆，主要不在容量，而在可操作性。ROME 与 MEMIT 的编辑是叠加在权重上的低秩修正，单条编辑可以成功，但随数量增长，编辑之间相互干扰；编辑结果不可枚举，难以撤销，也没有审计记录。Memory Layers 与 Engram 把记忆放进结构化查找表，减少了干扰，但映射仍是学习得到的，只在分布内可靠。Titans 把写入变成测试时的梯度步，写入的内容仍然不是可验证的符号。Qwenputer 的状态存放的是符号化的寄存器与文件内容，读写都能逐位校验，这是它与上述工作最清楚的区别。

## 5. 外部记忆：RAG、虚拟上下文与评测

| 工作 | 来源 | 要点 | 精确性与代价 |
|---|---|---|---|
| RAG（Lewis et al. 2020） | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | 检索加生成 | 文本精确、可溯源；检索可能漏掉 |
| MemGPT（2023） | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) | 类操作系统分页的虚拟上下文 | 每轮额外调用与 token 开销 |
| LongMemEval（ICLR 2025） | [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) | 500 题，覆盖信息抽取、多会话推理、时间推理、知识更新、弃答 | 评测标准，非系统 |

Qwenputer 文件系统版的数字（[README](https://github.com/divagr18/qwenputer)）：自建"多会话召回" 90 题，83.3，对照 base 7.8、宿主 Python 工具 78.9。这是**作者自测**，并非 LongMemEval。仓库文档（rv-store.md）也记录：同一长对话中，存储层"什么都不增加"，因为 Qwen 不调用它；10 万 token 的 prompt 无法装入 24 GB 显存。

就成本、延迟与精确性而言，外部记忆需要在三者间取舍。RAG 与向量库的查询代价低，精确性取决于检索是否召回正确文本，并且可直接引用来源；但长对话中的事实更新与冲突需要额外策略。MemGPT 类的虚拟上下文把记忆管理交给模型，代价是额外的调用轮次。Qwenputer 的文件系统把内容保存在 DeltaNet 状态中，读出逐字精确，状态可随模型一起保存和加载，不依赖外部服务；代价是每次读写都要额外解码 token，耗时明显高于向量检索（仓库未给出直接对比）。容量也很有限：512 个文件、约 5.3 万 token，因此它适合少量、需要逐字精确的内容，不适合数十万文档的场景。

## 6. 早期尝试为何没有实用化

| 因素 | 证据 | 性质 |
|---|---|---|
| 可学习寻址不精确 | NTM/DNC 的软寻址 | 根本 |
| 依赖链深度 | 11 层无法用调度压缩（rv32.md） | 架构内根本 |
| 数值依赖 | 一版同读 8 行时，fla 核留下 fp32 残差，改为 one-hot 或 2-hot 读（rv32.md） | 工程与根本并存 |
| 体积 | 磁盘 22.3 GB（rv-store.md），博客称 18.8 GB；零填充；chip 层不能量化到 e4m3（q35-size.md） | 工程 |
| 速度 | 约 3,200 CPU token/s（L4）；每条指令 1.0 到 1.55 个 token（rv32.md） | 工程 |
| 工具格式依赖 | 借用 Qwen3.5 的 tool_call 格式，依赖 Qwen 主动调用 | 设计选择 |

综合来看，早期尝试没有成为实用系统，原因有三。第一，可学习的记忆给不出精确保证，因此在算术或文件操作这类需要确定结果的场景不可靠。第二，可精确编译的 Transformer 大多是研究工具，结构或规模不适合真实模型；把它们搬进预训练模型，会遇到层耦合、数值舍入和推理速度的问题。第三，即使做到，工程代价也很高：Qwenputer 的体积、解码速度和驱动复杂度都远高于普通模型，而收益主要体现在算术与小文件的精确性上，对多数用户而言仍不如直接调用外部工具。因此它更像一项可检验的技术验证，而不是替代工具调用的产品路线。

## 7. 真正新的部分与限制

**新在何处**：(a) 在预训练模型现有 GDN 层中实现精确寄存器堆与完整指令级 CPU，且无参数训练（已核实 README）；(b) 通过 riscv-tests 的 rv32ui 与 rv32um 乘法子集（[rv32.md](https://github.com/divagr18/qwenputer/tree/rv-store/experimental/riscv)，作者自述）；(c) 文件系统状态可保存和加载。据我检索，未见同类公开工作直接做到这些；检索覆盖有限。

| 限制 | 判断 |
|---|---|
| bf16 与核依赖 | 多为工程，但精确性需逐核验证，属可接受的脆弱性 |
| 层耦合（11 层依赖链） | 在当前布局内根本 |
| 容量（头数、512 个文件） | 工程与架构并存 |
| tool-format 依赖 | 设计层面，长对话中可被忽略 |
| 22 GB 体积 | 工程（零填充可去除；chip 层需保持 bf16） |

后续研究的优先级应是：先量化数值依赖的脆弱点，即不同核、不同精度下的逐位一致性；再评估体积压缩，q35-size.md 作者估算 Qwen 原有 32 层可降到每权重约 1 字节、节省约 4.1 GB，但 chip 层需保持 bf16；最后检验"编译监控器"能否在不同模型与提示下复现。两篇后续的结论都依赖同一套编译设施，独立复现它们，是检验整条技术线最直接的方法。

## 8. 作者的两篇后续

| 文章 | 链接 | 要点 | 状态 |
|---|---|---|---|
| Hybrid models can hide reasoning in their RNN state | [divagr.com](https://divagr.com/blog/hybrid-models-can-hide-reasoning-in-their-rnn-state) | 仅用答案训练时，OLMo-Hybrid 学会在 RNN 状态中静默追踪，128 次翻转准确率 0.89（6 个种子）；Qwen3.5 的 β 被限制在 1 以内，未学会 | 作者自述，未独立复现 |
| Building oversight tools into a model's weights | [divagr.com](https://divagr.com/blog/building-oversight-tools-into-a-models-weights) | 在权重中编译线性探针与标志日志，报告捕获 91% 的谎言、28k token 零误报、探针 AUROC>0.99；困难提示下回复级 AUROC 降至 0.77 | 作者自述，未独立复现 |

## 9. 未验证与待复核

- 体积不一致：22.3 GB（仓库）与 18.8 GB（博客）。
- Qwen3.5 的 β 经 sigmoid 限制为 1 以内：来自作者博文，未核对 Qwen3.5 配置。
- Qwenputer 的所有基准数字均为作者自测，多为自建集。
- PAW 的代码与 arXiv 文档原定 2026-07-06 前公开：来自 HF 提交注记（[commit](https://huggingface.co/programasweights/paw-4b-gpt2/commit/9a0cb5e3d700f20dc7744e2a46f4fe2efbcf7ac3)），未核实。

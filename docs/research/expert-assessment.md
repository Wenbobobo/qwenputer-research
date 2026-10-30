# Qwenputer 关键难题评估报告

> **总判断**：这是一个已被工程证据部分证明的“可编译状态机”，不是通用的权重内计算机，也还不是成熟记忆系统。最强贡献是把 Gated DeltaNet 的递归状态约束成可寻址、可验证的精确 RAM，并用编译器生成电路；最强风险是数值、内核、基座耦合，以及“模型是否愿意调用”仍是概率行为。文中区分静态电路正确性、运行时正确性和任务收益。

## 1. 技术可行性

**判断：电路层面可行，通用部署层面脆弱。** one-hot key、α=β=1 把 delta 更新化为行覆盖；norm ballast 用 2^40 与可折叠的 2 的幂隔离范数，bf16 中把中间量限制在最多 8 个有效位，理论上可精确。依据：q35.md 的 “Memory is the delta rule”“Norm isolation”“bf16 exactness”；RV32 的 IR、参考机和数十万至百万位置 replay 通过。**反对意见**：这不是 bf16 普遍精确，而是精心设计的有限值域；未覆盖的溢出、长链、异常 token 仍可能破坏不变量。

关键脆弱点按严重度排序：
- **根本性约束**：Qwen3.5 的 GDN/状态布局、隐藏宽度和 tokenizer 必须匹配；11–12 个前置层及宽度扩展带来约 18–22GB checkpoint 和约 18GB 显存门槛。换基座不是换配置，而是重编译。tokenizer/vocab 扩展若 embedding 行无 ballast 会直接丢记录（rv-store.md “Bugs found along the way”）。
- **高风险工程项**：fla chunk prefill 与 recurrent decode 的 exp(cumsum)、conv 窗口/位置、状态原地更新和 kernel 顺序必须逐 kernel 对齐；文档曾记录 chunk/原地更新导致状态污染，且缺少 flash-linear-attention、causal-conv 时退回参考实现（rv32.md P2/P3 Not done）。
- **非根本性但昂贵**：层数、活跃维度、零填充、批处理可由 paired heads、list scheduling、overlay、稀疏执行器优化；q35-size.md 已实测 16→12 层约省 1GB、缩窄约省 0.56GB。

批处理已做到“与其他行无关”，但 batch-1 与批量 GEMM rounding 不保证 bit-identical；因此应把“跨 batch 语义不变”与“逐位复现”分开验收。基座版本、GPU、PyTorch/FAL kernel、chat template 都是发布 ABI。当前 exactness 主要是 teacher-forced/replay 证据；自由运行和未知 prompt 的覆盖仍**需验证**。

## 2. 前史与真正新意

Schlag 等（2021）fast weights 已把 delta rule 解释为 associative memory；NTM 早已研究可寻址读写；compiled transformers（Lindner 2023）和 Program-as-Weights（2026）也都把程序结构映射进参数/层；memory layers 与 Engram 则追求外部或参数化记忆的高效检索。**真正新意**不是“权重能表示程序”，而是三者组合：把现代 Gated DeltaNet 状态固定成精确 RAM；用 IR/发射器直接生成普通 Transformer 权重；再以 trace token 驱动一台可运行 RV32IM（含 Zba、大数单元）并桥接原生 tool-call 格式。依据：rv32.md P1–P3、q35.md compiler 文件表、blog.txt “The RNN inside…/Compiling a computer”。

此前没实用化的原因是：旧线性注意力缺少稳定的门控/内核生态；浮点误差会把软写入放大；缺少成熟工具调用格式来诱导模型；编译器必须处理 liveness、norm、overlay、mute、tokenizer 和 kernel 语义；硬件加速器也不擅长稀疏、短 burst、逐 token 状态机。Qwen3.5 的 GDN、工具格式预训练、bf16 可控值域、Triton/CUDA graph 和现成 HF loader 恰好同时满足时代条件。**反对意见**：成功也可能只是选择了最适合的基座和算术任务，不等于跨架构范式已成立；跨 3 个不同 GDN 基座的复现仍**需验证**。

## 3. 参数化记忆系统评估

它更像**可编程外设/协处理器**，记忆只是其 RAM；写入规则、哈希目录、更新语义和打印协议由编译电路决定，非向量相似度记忆。

|维度|相对 RAG/向量库/KG/MemGPT/Engram|
|---|---|
|容量/成本|固定约 512 文件、约 53k tokens；状态成本与上下文长度近似无关，胜过 KV/context；远逊可扩展磁盘、向量库、KG。|
|读写/延迟|哈希命中约 2–3 个 chip token，get 输出仍按 token 逐字生成；实测 store get 约 5–6秒、host 工具约2.3秒，故不能宣称更快。CPU trace 约3200 token/s；普通 RAG 检索可毫秒级但需外部系统。|
|精确/更新|命名 exact read-back、同名覆盖和 delete 强；语义召回、模糊查询、组合图查询弱；哈希碰撞、槽满、命名依赖需暴露。|
|审计/移植|trace、IR replay、快照可审计；但 checkpoint 与状态、driver、kernel 绑定，远不如 JSON/KG 可移植。跨会话持久化强于上下文，但快照约30MB且缺少权限、版本、事务和并发模型。|

在“单用户、小型结构化事实、跨会话、无需语义搜索、GPU 常驻、外部工具受限”时，它可能比 RAG/KG/Engram 更省检索 token、状态读写更稳定；在大规模知识、模糊召回、多人并发、低延迟或需增量可观测更新时不成立。现有 83.3/90 分优势来自合成 LongMemEval 风格、固定命名和工具 nudge；且 store 79 分钟而 host 48 分钟，收益主要是内聚性、精确更新，不是端到端效率（rv-store.md A/B）。

## 4. 持续学习与自我进化

(a) **事实写入状态：已实现**，但依赖模型主动调用，漏保存是主要误差（93/120 会话保存）。(b) **技能/程序编译进权重：已实现原型**，改变 checkpoint，不是运行时学习。(c) **编译器生产新能力：1–2 年可实现有限版**，前提是把 DSL/ISA/证明、资源预算和回归测试自动化；能力边界仍由人工设计的原语决定。(d) **元学习自动生成电路：近期近似幻想**：需要从自然语言规格综合可验证电路，并保证 bf16、kernel、安全不变量，远超当前证据。

关键证伪实验：跨 3 个基座、2 种 kernel、3 种 GPU 的长程 free-running（≥10^8 owned positions）；随机、对抗 tokenizer、碰撞、满槽、异常和中断恢复；不提供 nudge 的开放任务；自动编译器生成新算法并与 QEMU、形式规范逐位比较；持续数千次更新后的遗忘、回滚、并发和污染测试。任何一项失败都应下调“通用自进化”表述。

## 5. 优化与创意机会 Top 5

1. **零填充压缩**：按非零块、列做 safetensors 稀疏存储并在加载时恢复；预期下载、显存降数 GB；风险是 stock Transformers 与 kernel 失去兼容。q35-size.md 显示 3.57B 参数只是 padding。
2. **执行器、推理加速**：把 chip burst 做 persistent fused kernel，批量并行多行、减少每 token launch；目标把 get 从秒级压到亚秒；风险是状态别名、batch rounding 和 kernel 依赖。
3. **算法设备树**：编译排序、哈希、压缩、校验、日期、单位等可组合设备，统一 ISA/ABI；收益是可验证能力扩展；风险是层数、宽度和安全面线性膨胀。
4. **安全监控外设**：不可写 monitor、答案键、in-weight monitor；收益是监测模型输出、隐藏状态，且模型难以关闭；风险是监控器同样可能被 bridge、token 触发绕过，必须证明覆盖而非宣传“不可篡改”。
5. **文件系统 agent 接口**：加入版本、权限、事务、TTL、来源和 schema，支持结构化 get/query；收益是从 demo memory 变成可治理状态；风险是固定容量、隐私泄漏、恶意写入和快照回滚。

## 6. 给 owner 的六个月路线图

**月1：冻结 ABI。** 固定基座 commit、tokenizer、chat template、PyTorch/FAL/Triton、GPU 矩阵；发布 IR、reference、HF、executor 四层一致性规范和最小 checkpoint。

**月2：数值与内核审计。** 为 norm ballast、bf16 rounding、exp(cumsum)、conv、prefill/decode、batch invariance 建 property tests；每次构建自动产出 mismatch 定位和 digest。

**月3：压缩与速度。** 完成稀疏 checkpoint、加载原型、persistent executor、batched chip burst；以端到端 p50/p95 latency、显存和 token/FLOP 对比 host tool、RAG，而非只报 circuit tok/s。

**月4：真实记忆评测。** 用 LongMemEval、时间冲突、模糊别名、满槽、碰撞、多人并发、删除、回滚和隐私攻击；报告召回、更新正确率、漏写率、审计成本。

**月5：可组合能力。** 先编译一个非算术设备（排序或校验），验证 DSL→IR→bf16→kernel→QEMU 全链路；同时做 monitor 原型，明确覆盖边界。

**月6：决策门。** 若跨平台长程 exactness≥99.99%、端到端延迟不劣于 host 工具、真实记忆任务有显著优势，再做公开 beta；否则定位为研究型协处理器，停止“通用记忆、自我进化”叙事，优先开放 IR、快照格式和可审计 replay。

### 给 Lead 的 5 条关键结论

1. 核心创新是“GDN 精确 RAM + 编译器生成可运行 RV32”，不是泛化的参数化记忆。
2. 最大技术风险是 kernel、基座、tokenizer、批处理 ABI，而非 CPU ISA 本身；必须建立跨环境逐位回归。
3. 文件系统在小型结构化跨会话事实中有价值，但当前端到端比 host 工具更慢，不能称为普遍高效。
4. 事实状态写入和人工指定程序编译已成立；自动生成新电路可做，元学习自我进化尚属幻想。
5. 六个月应优先冻结 ABI、压缩零填充、加速 executor、做真实记忆和安全证伪，再决定是否 beta 化。

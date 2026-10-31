# Stage 8 计算增强组合评估：把“会算”变成可组合能力

> **给 Lead 的决策摘要（2026-10-30）**：未来 2–3 个阶段应把资源集中到三件事：**（1）fp32 chip 作为长序列正确性默认候选，并做可局部化的 fp32 消融；（2）设备树先做一个合并原型，优先 SORT4/ADD8/DADD 这类高频、固定形状、可证明的算子；（3）H5 作为 CPU 本地语义路由器接 exact get，与 memory-disk/SafeStore 组成可审计产品路径。in-weights 本身不应包装成“比 host 更快或更聪明的记忆”：现有 H1 只显示覆盖语义有效。所有速度数字均来自仓库已有实验；未运行模型、未新增 GPU 实验。文中“假设”明确标出。

## 1. 评价口径与基线

Qwenputer 的增强候选有三类价值：**计算能力**（精确算术、排序、位操作和确定性协议）、**状态能力**（跨会话写入、更新、审计）、**部署能力**（长序列稳定、CPU/GPU 资源效率）。评价不能只看 token/s：一个更快但破坏 lock 的电路是回归；一个更精确但需要每次打印 500 个 token 的记忆也可能没有产品价值。

基线事实如下。RV32 核心为 11–12 层、1,040,595 个非零权重，单条指令约 1.3–1.8 token，带 calc 调度和打印时一次调用约 254–684 token。设备树原型 ADD8、SORT4、DADD 均为 2 层、1 个 head，分别 899、3,876、1,454 个非零权重，IR 全量 exact、独立 HF argmax 均 0 错，尺寸约小 270–1,150 倍。发布模型经过 trim 后峰值约 13.15GB，可全驻留 RX 7900 XT；trim+FLA 的 put/get 为 16.84/13.62 tok/s，但 FLA 的 gpucheck 有约 1e-7 级微差。fp32 chip 试验增加约 0.65GB，paired lock divergence 从 bf16 的 3/12 降到 fp32 的 0/12，put/get 为 15.72/14.69 tok/s，Hi warm 为 18.87 tok/s。

记忆侧的事实边界更重要。H1 in-weights n=10 的 changed 严格正确率为 10/10，普通事实 17/20，全部问题 27/30；但 3 次历史 lock drift，且 host 覆盖式已经是 27/30=90%。因此 100% 是小样本的可行性信号，不是介质优势。H5 真实 LongMemEval 100 题中 bge-small 的 Recall@1/5 为 50%/87%，exact-key 对语义查询为 0%、已知名称为 100%；RAG 粘贴约 12,140 token，exact get 约 45 token。memory-disk v0 已有 40/40 测试通过，但受 512 文件、每文件 104 token、编译 hash/布局和名称 key 依赖约束。

## 2. 候选组合总表

评分定义：技术可行性 1–5（5=已有端到端证据）；工程成本为本项目熟悉代码栈的估计人日，**非仓库实测**；ROI 是 1–5 的“用户价值 × 证据强度 ÷ 成本风险”综合分，不是财务回报。部署场景写的是可落地边界。

| 候选 | 一句话价值 | 可行性 | 成本（人日） | ROI | 部署场景 | 主要风险/阻塞 | 判定 |
|---|---|---:|---:|---:|---|---|---|
| fp32 chip（默认或可选局部 fp32） | 用约 0.65GB 和 7–13% 级速度代价换长序列 lock 稳定性 | 5 | 2–4 | 5 | 长会话、审计、计算工具调用 | fp32 全覆盖显存增加；fp32 与 FLA 的组合尚未证明；长序列样本仍少 | **做** |
| ADD8/SORT4/DADD 设备树 | 把高频固定算子从 RV32 调度税中剥离，2 层即可复用 | 4 | 4–8 | 5 | 计算器、排序、校验、bignum、结构化 agent | 新 token 族和 owner/exit 合并；embedding 行成本；文本桥接未测 | **做，先做一个合并原型** |
| H5：bge-small 路由 + exact get | 以 CPU 语义召回解决 exact-key 的 0% 语义命中，再以 45 token 精确读回 | 4 | 3–5 | 5 | 本地隐私记忆盘、离线助手、企业审计 | bge 误召回、索引更新、模型/embedding 版本绑定；端到端未合并 | **做** |
| memory-disk v1 + SafeStore | 将状态快照变成有指纹、校验、journal、迁移和防覆盖的可运营存储 | 4 | 4–7 | 4 | 单用户离线设备、可回滚 agent 状态 | 512/104-token 硬限；指纹当前不含权重和 build hash；并发/权限缺失 | **做，作为产品壳** |
| in-weights 直接记忆 | 让记忆随模型状态保存，减少外部运行时依赖 | 4 | 2–4 | 3 | 无网络、无外部工具、单模型单用户 | 介质无已证速度优势；漏存、工具调用依赖、lock drift；不可跨模型 | **做限定性 demo，勿做主卖点** |
| 编译设备作 memory co-processor | 用 store/find/sort 形成可审计的状态处理链 | 3 | 8–15 | 4 | 结构化文件、键查找、批量排序 | 不能直接 grep/聚合；设备与 store 目前无直通；层/词表预算 | **做设计验证，后置合并** |
| 共享 chip layers 跨模型 | 同一设备图复用到多个基座，降低编译维护成本 | 2 | 15–30 | 3 | 多 Qwen3.5 变体、研究平台 | hidden/GDN/layout/tokenizer/kernel ABI 强耦合；尚无跨基座证据 | **暂缓** |
| 量化感知电路编译 | 让设备针对目标量化格式生成可证明电路 | 2 | 15–30 | 3 | 未来专用推理芯片、存储受限部署 | chip bf16 精确性约束；7900 XT FP8 无原生，NF4 已质量崩 | **暂缓** |
| CPU–GPU 混合执行 | CPU 承担稀疏控制，GPU 承担 Qwen 大矩阵 | 3 | 6–12 | 2 | GPU 不足或调试模式 | 已有 offload 73.6s put/39.8s get，PCIe/WSL 调度税巨大 | **杀近期方案** |
| filesystem as device bus | 以文件/状态协议串接设备、模型和外部程序 | 3 | 8–15 | 3 | 可审计插件协议、离线批处理 | 外部总线破坏“无运行时”叙事；版本、权限、事务和安全面 | **暂缓，先写协议** |
| 默认全 fp32 | 所有 chip/Qwen 子层统一使用 fp32 | 3 | 3–6 | 2 | 只追求稳定的实验机 | 显存和吞吐浪费；收益证据只针对 chip；可能遮蔽真实边界 | **杀默认全覆盖** |

## 3. 编译设备树：哪些任务值得专用化

### 3.1 成本模型

设备的优势不是“非零更少”本身，而是同时削掉三项：RV32 的程序计数器/寄存器簿记、约 200–250 token 的 calc/bridge 调度、以及不必要的层深。ADD8 的记录是 3 token，SORT4 是 8 token；DADD 为 2n+1 token，4 位 9、12 位 25、40 位 81。对照，RV32 8 位加法端到端估算约 5–10 token，4 元素排序约 45–70 token，DADD calc 4–12 位约 254–303 token。后一组不是完全同边界：DADD 假设操作数已经变成 LSB-first 的 P token，真实接入需捕获、对齐和 MSB 输出，估计增加 1–3 层、数千非零和约 2n–3n token。因此应把设备收益写成“在固定输入协议下成立”，不能宣称所有自然语言算术都自动加速。

| 任务 | 专用器件适配 | 每次成本（已知/假设） | 是否值得编译 | 理由 |
|---|---|---|---|---|
| 8 位加/减、比较、位掩码 | 高 | ADD8 3 token；每位扫描非零约 +66 | 高 | 输入域小、无循环、可穷举验证；适合协议字段和 checksum |
| 4 元素稳定排序 | 高 | SORT4 8 token；RV32 约45–70 token（估算） | 高 | 排序网络固定；可服务 top-k、优先级、时间戳小集合 |
| 十进制大数加法 | 中高 | DADD 2n+1 理想；真实端到端约2n–3n（假设） | 高 | bignum、金额、计数器频繁；必须显式 G 边界和位序 |
| 查表/有限状态 lookup | 高 | 取决于表宽；预计 2–4 层（假设） | 中高 | 协议码、单位、日期字段可审计；表过大时 embedding 和槽预算反噬 |
| CRC/校验、bitops | 高 | 固定多项式、串行状态；预计线性 token | 中高 | 纯函数、重复率高、正确性可证明 |
| regex/解析 | 低中 | 简单有限自动机可行；通用 regex 需循环/回溯 | 仅简单子集 | 适合邮箱/协议字段 DFA；不适合无界回溯和自然语言解析 |
| 通用排序、grep、聚合 | 低 | 随输入规模和内存增长 | 低 | 会退化成 CPU；直接交给 RV32 或外部索引 |

编译合并的技术事实是：最大器件只有 1 head、403 单元，而 RV32 每子层最多 16 head、9,216 单元，理论上可在现有 12 个 chip 层并行驻留，边际层数可能为 0。这里的“可能”是关键：owner/exit 的多路选择、code clear 顺序、RAM 读写冲突和共同 ballast 尚未在真实 merge.py 中验证。编译时间本身不是主要阻塞，真正风险是峰值 RAM、权重 merge、tokenizer 新行、以及一个设备出错会否影响核心回退。建议采用“设备树 + RV32 回退”：设备只拥有明确 token family，输入不匹配时保留通用路径。

研究价值高于短期商业价值。它能把“可编译神经外设”的尺度定量化：2 层、千级非零、0 mismatch 是可复用的论文级证据；商业上仅在计算器、离线审计、结构化 agent 中有价值，不能替代通用 CPU。

## 4. in-weights filesystem 与 SafeStore 产品化

### 4.1 决策树

使用顺序应是：

1. **是否必须离线、无外部服务、且状态要跟着单个模型快照走？** 是，进入 in-weights；否，优先普通数据库/向量库。
2. **查询是否有稳定 exact key？** 是，可直接 exact get；否，先用 CPU bge-small 路由到候选 key，再 exact get。
3. **是否要求逐 token 审计、可回滚、可证明覆盖？** 是，memory-disk + journal + SafeStore 有优势；否，RAG 通常更便宜。
4. **更新是否高频、并发、多租户或容量超过 512 文件/每文件 104 token？** 任一为是，in-weights 只做缓存或审计副本，主存储放外部系统。
5. **端到端是否比 host 多一次大模型往返还贵？** 若 CPU/GPU 已常驻且 exact get 约45 token，值得；若每次重新加载模型或触发 offload，立即放弃。

因此 in-weights 的胜点是延迟可预测、隐私、离线、可审计和状态随 checkpoint 快照；不是已证明的吞吐优势。Stage 7 的 changed 10/10 不能单独归因于权重介质，因为 host overwrite 已达 90%，且 H1 样本只有 10 例。它还依赖模型主动 put；17/20 普通事实说明漏存和 schema 仍是瓶颈。

memory-disk v1 应沿四层做：**格式层**（magic、version、body sha256、原子写）、**身份层**（模型/分词器/store 配置和 build hash 指纹）、**语义层**（journal 重放、put 覆盖、delete tombstone、NameGuard/SafeStore 36-bit 标签）、**治理层**（权限、来源、TTL、事务、容量告警）。当前快照小：1/10/100/512 文件约 77B/723B/7.2KB/36.7KB；但它不是无限磁盘。512 文件、每文件 104 token、key 前 6 token、墓碑不回收和布局依赖必须在产品 API 中显式返回，而不是静默失败。

## 5. H5 混合检索：最值得做的产品组合

H5 的结构非常清楚：CPU 本地 bge-small 做语义路由，返回 top-k key；模型内 filesystem 只做 exact get 和精确输出。真实数据的 Recall@5=87%，意味着路由仍会漏召回，不能把路由分数当答案；但 exact get 约45 token 对比 RAG 粘贴约12,140 token，候选正确时上下文成本下降约 270 倍（12,140/45≈269.8，算术比例，不代表端到端延迟）。Recall@1=50% 提醒我们必须保留 top-k、候选重排、未命中回退和“找不到”的诚实结果。

部署形状：本地 CPU 常驻 bge-small 与小型索引；GPU/模型进程只接收候选 key 和 exact get；memory-disk 保存正文与 journal；SafeStore 防止同义长名称碰撞；每次回答保留 query、embedding 版本、候选分数、最终 key、快照指纹，形成审计链。若无 GPU，则仍可把 bge 作为离线路由器，但不能假设模型生成速度足够低延迟。模型本体不应被迫承担向量相似度，它只承担确定性读回。

## 6. fp32 chip 与长序列正确性

Stage 8 fp32 chip 的收益集中在正确性：同一 paired harness 下 bf16 3/12 lock divergence，fp32 0/12；代价是约 +0.65GB、速度相较 fp32 结果基线下降约 7–13% 的量级（具体取决于 put/get/Hi 与配置）。对长锁、审计和计算工具，0/12 的方向价值大于少量吞吐损失，所以建议**chip 相关默认使用 fp32 候选**，但必须经过扩大样本和回归门禁后才成为发布默认。

不建议默认全模型 fp32。可行的降本路线是分层/分子层：chip 0–11 的 GDN state update、norm ballast、code/lock 相关 FFN 使用 fp32；Qwen 文本层 12–43 保持 bf16；进一步按敏感度只保留 chip attention 的 q/k/v、alpha/beta 和 exit 子层 fp32，其余 chip MLP 先 bf16。每个局部方案都必须用同一 trace、同一随机种子、同一 GPU 做 paired lock、head 和文本回归。fp32 chip 与 full FLA 的组合不能直接假定正确，因为 FLA 的差异来自累加顺序；应先测 chip reference + Qwen FLA，再测 full reference。

## 7. 三个具体实验（未来两阶段）

| 实验 | 做法 | 接受指标 | 失败动作 |
|---|---|---|---|
| E1：六配置长序列矩阵 | fp32/bf16 chip × no-FLA/selective-FLA/full-FLA；执行长度 1、2、5 的 12 个 lock 会话，另测 Hi/put/get | 目标主档：长序列 lock 0/12；CPU bit-identical；GPU token-level、head mismatch=0；速度不低于 15 tok/s 或记录明确 trade-off | full FLA 若 mismatch，默认回 reference；只保留 selective FLA 速度档 |
| E2：设备 merge 最小闭环 | 将 ADD8 或 SORT4 与 RV32 共置现有 chip 层；跑 owner 分派、回退、连续调用、跨记录 reset | 设备 IR 全量 0 mismatch；HF argmax 0 错；RV32 原有程序 token-level 不变；新增层数 0 或明确≤1；合并峰值 RAM 可复现 | 不满足 owner/exit 或峰值 RAM，保留独立设备，不扩展更多器件 |
| E3：H5 + SafeStore 端到端 | CPU bge top-1/5 路由，memory-disk exact get；100 题按语义/已知名/更新/未命中分层，对比 RAG 粘贴和 exact-only | Recall@5 ≥87%（不得低于已知基线）；命中题 exact 正文 100%；平均输入 ≤500 token（假设目标）；未命中拒答率 100%；快照/指纹/审计日志可恢复 | Recall@5 降低或索引成本超过收益，改成外部 RAG 主存、in-weights 仅缓存 |

## 8. 组合优先级

| 优先级 | 组合 | 未来价值 | 进入条件 |
|---:|---|---|---|
| P0 | fp32 chip + reference/selective FLA 双模式 | 长序列可信，同时保留速度档 | E1 lock/head/text 三重通过 |
| P0 | bge CPU router + exact get + SafeStore | 语义入口与精确存储互补，最接近产品 | E3 top-k、拒答、审计通过 |
| P1 | ADD8/SORT4/DADD 设备树 + RV32 fallback | 计算能力扩展的最强研究证据 | E2 至少一个真实 merge 成功 |
| P1 | memory-disk v1 | 把 demo 状态变成可迁移、可验证组件 | build hash、journal、容量错误码补齐 |
| P2 | in-weights 单介质主张 | 离线/隐私 demo 有吸引力 | 需要更大样本并证明介质相对 host 的增益 |
| P3 | 跨模型共享层、QAT 电路编译、CPU-GPU 混合 | 长期研究但当前阻塞多 | 先有稳定 ABI 和多基座证据 |

## 9. Kill list

1. **杀“in-weights 天然更快/更聪明”叙事**：H1 没有与 host overwrite 形成优势证据，且历史 lock drift 存在。
2. **杀默认全 fp32**：只为修 chip 长序列而牺牲全部 Qwen 层，没有成本收益比。
3. **杀近期 CPU-GPU 混合执行作为性能路线**：已有 offload 结果显示 put 73.6s、get 39.8s，PCIe/WSL 调度税吞噬计算收益。
4. **杀 NF4/无保护 INT8 量化作为默认方案**：NF4 质量崩；head int8 GPU 反而降速并增加约 0.7GB。
5. **杀通用 regex/任意大排序设备**：循环、内存和桥接成本会把小器件优势变成另一台 CPU。
6. **杀跨模型共享 chip 层的近期承诺**：基座 GDN、hidden、tokenizer、kernel 是 ABI，当前没有跨基座证据。
7. **杀把 Recall@1 当 H5 产品指标**：真实 Recall@1 只有 50%；产品必须使用 top-k、重排和拒答。

## 10. “计算能力不得回归”门禁

### CPU 门

- 参考 IR/QEMU/CPU trace 对拍：RV32、ADD8/SORT4/DADD、store 的关键程序逐 token **bit-identical**；任一 mismatch 必须定位到 token、层、IR 节点和舍入。
- 编译器门：设备层 tensor、norm、exit、embedding 行、merged 权重 hash 可复现；旧芯片 12/12 层和既有 168/168 tensor 证据不能下降。
- 状态门：put/get/list/delete、碰撞、满槽、跨会话快照、journal 迁移、NameGuard/SafeStore 全部回归；长序列 1/2/5 的 lock error=0。

### GPU 门

- 不再把 AMD GPU 声称为 bit-exact：严格档报告 CPU bit-identical；GPU 档报告 token-level、head mismatch、lock/head 检查和最大数值差异。
- 每个候选运行同一套 **lock/head checks**：至少 12 个 paired 会话，短/长 prompt、calc、设备调用、跨进程 get；目标 lock divergence 0/12、head mismatch 0、关键答案逐字一致。
- 速度门：发布默认不得低于当前 trim no-FLA 的正确性基线；速度档应以 trim+FLA 的 put/get 约 16.84/13.62 tok/s 为参考，并显式标记 1e-7 微差。
- 回归套件：RV32 7/7 程序、chip gpucheck、store 5 操作、设备边界/随机/重复值/满进位、H1/H5 分层样本、memory-disk 40/40、显存峰值和加载时间。失败时自动回退 reference，不静默降级。

## 11. 给 Lead 的最终建议

**第一，做 fp32 chip 的可局部化默认路径。** 它是目前唯一同时有“长序列正确性收益”和“速度仍在可接受区间”的增强，预期 ROI 最高；但把 fp32 限定在 chip 敏感子层，避免把 +0.65GB 扩大成全模型税。

**第二，做 H5 + SafeStore，而不是继续证明 in-weights 单介质优越。** CPU bge 负责“找到哪个 key”，权重内存储负责“精确读回和审计”；12,140 对 45 token 的输入差距提供了清晰产品价值，但必须用 Recall@5、拒答率和快照审计约束它。

**第三，做一个设备 merge 闭环，优先 SORT4 或 ADD8，再决定设备树规模。** 已有 2 层、千级非零、0 mismatch 的证据足以支持原型；真正未知的是共享 chip 层的 owner/exit、tokenizer 和回退，不是电路数学本身。若 E2 失败，应保留独立 device graph 研究结果，停止继续堆设备。

接下来 2–3 阶段的成功定义不是“增加更多想法”，而是同时证明：能力增加、状态可审计、长序列不漂移、CPU/GPU 行为有明确精度等级，并且任何失败都能回到一个已验证的计算基线。

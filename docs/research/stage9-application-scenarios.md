# Stage 9：落地场景与突破方向（应用/产品向）

> 说明：原计划的 expert subagent 在完成分析后崩溃、未落盘；本文由 Lead 依据其结论摘要、Stage 1–9 证据与 [stage9-technical-breakthroughs.md](stage9-technical-breakthroughs.md) 整理。所有数字引用见 [docs/RESULTS.md](../RESULTS.md)，估计处标注“假设”。

## 0. 先给结论

1. **最强的近期产品路线**：H5 语义路由（CPU bge）+ in-weights exact get + SafeStore/memory-disk 的“可审计记忆盘”。价值来自**语义入口 + 精确读回 + 可审计状态**，不是“权重内存储更快”。
2. **最强的研究突破路线**：**可审计的确定性设备树**（把固定算子编译成 2 层小电路，可合并进现有 12 层、可 IR 验证）。它把“编译器即安装器”从口号变成可复现的工程范式。
3. **fp32 chip 是必须先完成的可信基础设施**：没有它，长序列 lock drift 会让任何产品级叙事失稳。
4. **“in-weights 优于普通存储”目前不成立**：H1 只证明覆盖语义有效（全档 90–100%），不足以支撑介质优势；in-weights 的卖点是离线、隐私、快照携带、固定成本、审计链。

## 1. 场景目录（10 个）

### S1 离线隐私个人记忆盘（**Top 1 产品**）
- 一句话价值：本地模型 + 本地方案内存储 + 本地语义路由，数据不出设备，回答可审计。
- 用户：隐私敏感个人、律师/医生/研究员、离线环境。
- 为什么现在可行：H5 端到端原型已通（Recall@1/3/5 = 58/84/96%，BGE CPU 11ms，SafeStore get 0.5ms）；memory-disk v0 有指纹/journal/迁移；trim runtime 13–15 tok/s 可交互。
- 需要 enabler：4B reader 接入（top-1/3/5 vs RAG 对照）、低分拒答、索引版本门禁、容量错误码。
- 竞争替代：本地 RAG（llama.cpp + 向量库）、加密笔记 + 云端 AI。
- 突破潜力：**4/5**（体验 + 隐私 + 可审计的组合有差异化）。
- 里程碑（6–8 周）：reader 对照实验 → 单机 demo（查询/写入/审计/回滚）→ 小范围用户测试。

### S2 可审计的企业 agent 状态存储（**Top 2 产品**）
- 一句话价值：agent 的长期状态（事实、承诺、权限边界）可精确覆盖更新、可回滚、可证明何时被谁改。
- 用户：合规行业（金融/医疗/法务）的内部 agent 平台。
- 为什么现在可行：H1 覆盖式语义 90–100%；SafeStore 解决长名碰撞；memory-disk 有指纹/journal；exact get 45 token。
- 需要 enabler：并发/权限/TTL/事务、与现有数据库的同步层、审计导出格式。
- 竞争替代：普通 DB + 事件日志、专用 agent memory SaaS。
- 突破潜力：**4/5**（审计是差异化，但介质本身不是护城河；护城河在协议与集成）。
- 里程碑：把 memory-disk v1 做成 API（POST /v1/memory/query），配一个企业场景 POC。

### S3 确定性计算外设（ADD8/SORT4/DADD）作为 agent 工具（**Top 1 研究**）
- 一句话价值：把“可证明的算术/比较/排序”编译成 2 层小电路，作为精确工具，不经过 RV32 的 200–250 token 调度税。
- 用户：需要精确算术/金额/校验的 agent、离线审计、嵌入式规则引擎。
- 为什么现在可行：三设备 IR 全量 exact、HF argmax 0 错；merge_probe 证明三者同时并入仍 **0 额外层**（slots 1899/2029，nz +5626）。
- 需要 enabler：ADD8-only 真实 merge（token namespace、owner/exit dispatcher、RV 状态冻结、embedding 行）、text bridge。
- 竞争替代：普通 CPU 计算、外部计算器 API、符号计算库。
- 突破潜力：**5/5（研究）**，产品 3/5（场景偏窄但可证明性独特）。
- 里程碑：ADD8 rv-tiny 端到端（IR exhaustive 0 mismatch、HF 1000 条 0 错、RV 程序不变）→ SORT4/DADD。

### S4 语义发现 + 精确读回替代 RAG 长上下文
- 价值：用 45 token 的 exact get 替代 12,141 token 的 RAG paste（≈1/270，算例见 RESULTS E 节）。
- 用户：长对话个人助手、客服知识库、代码库问答。
- 为什么可行：LongMemEval Recall@5 87%；bge-small 33M 参数 CPU 可跑。
- 卡点：Recall@1 50–58%，必须 top-k + 重排 + 拒答；索引更新/版本绑定。
- 突破潜力：**4/5**（成本优势明确，但需要真实 reader 端到端证明）。

### S5 编译器即安装器 / 可验证权重供应链
- 价值：从“发布一个 checkpoint”变成“可审计、可重建、可增量安装的软件供应链”（编译器版本、config、tokenizer、权重索引、anti-sink 16KiB、SHA 全部绑定）。
- 用户：模型发布者、审计机构、企业采购。
- 为什么可行：168/168 张量 bit-exact 审计、anti-sink 已定位、无损压缩 10/10 验证。
- 卡点：编译器版本锚定、基座+增量方案未验证（D2 已决定不做）。
- 突破潜力：**3.5/5**（标准/工具价值大于产品收入）。

### S6 端侧/嵌入式 AI IP（chip-in-weights 协处理器）
- 价值：把 2 层、千级非零的电路作为“可 license 的神经外设”，与现有 SoC 的 DSP/NPU 互补。
- 用户：芯片/IP 厂商、边缘设备厂商。
- 为什么可行：ADD8/SORT4/DADD 体积小、可验证；RV32 已有完整工具链证据。
- 卡点：需要与具体 SoC 的 tokenizer/接口绑定；商业模式未验证；竞争（TinyML/FPGA）成熟。
- 突破潜力：**3/5**（长期，需合作方）。

### S7 教育/科研平台（可解释、可验证的权重内计算）
- 价值：教学/论文平台：每个 chip 层可对着 IR 逐 bit 验证；“编译器写进 Transformer”是可演示的范式。
- 用户：高校、研究实验室、模型可解释性团队。
- 为什么可行：审计工具链/报告/benchmark 已具备；页面已公开。
- 卡点：需要打包成 toolkit（compiler-exactness audit benchmark），可能 4–8 周。
- 突破潜力：**3.5/5**（社区/论文杠杆大，收入小）。

### S8 长上下文工具调用 agent（fp32 chip 稳定性）
- 价值：长会话（数百 token 历史）里电路行为稳定，工具调用/状态更新不漂移。
- 用户：长任务 agent、自动化运维、审计型工作流。
- 为什么可行：fp32 chip 在 12 个历史会话 0/12 lock divergence（bf16 3/12）。
- 卡点：样本需扩到 60+ paired；速度代价 −7~−13%。
- 突破潜力：**3.5/5**（可靠性卖点，不是性能卖点）。

### S9 受限部署环境（无外部依赖、单文件模型）
- 价值：一个 checkpoint + 7.951 GiB 无损格式，自带 CPU/文件系统，无外部服务。
- 用户：离线设备、教学演示、嵌入式研究。
- 为什么可行：trim 13.15GB 全驻留 20GB 卡；CPU 端到端可跑。
- 卡点：速度与容量限制（512 文件/104 token）；部署工具链未打包。
- 突破潜力：**3/5**。

### S10 memory-as-checkpoint（持续学习，限定）
- 价值：把记忆随模型快照迁移/回滚；单文件交付 agent 状态。
- 为什么可行：memory-disk journal + 指纹；H1 in-weights 100%。
- 卡点：跨模型/跨 seed 不可移植（实测 0% 读回）；容量硬限。
- 突破潜力：**2.5/5**（需明确定位为“状态快照”而非“通用记忆”）。

## 2. 排序短名单（综合：技术就绪 × 分发优势 × 防御性 × 时间）

| 排名 | 场景 | 就绪度 | 防御性 | 时间到 demo | 判断 |
|---|---|---|---|---|---|
| 1 | S1 离线隐私记忆盘 | 高（H5 原型已通） | 中高（协议+审计） | 6–8 周 | 先做 |
| 2 | S3 确定性计算外设 | 中（IR/emit 已通，端到端缺） | **高**（可证明性独特） | 3–6 周 ADD8 | 研究主攻 |
| 3 | S2 企业 agent 状态存储 | 中高（memory-disk 基础） | 中（集成壁垒） | 8–12 周 | 与 S1 共用底座 |
| 4 | S4 语义发现+精确读回 | 高（检索层已通） | 中 | 4–6 周（reader 对照） | 与 S1 合并 |
| 5 | S5 可验证供应链 | 中高 | 中 | 6–8 周 | 论文/工具 |

## 3. 可证伪的突破命题（每个配最便宜的验证）

| # | 命题 | 验证实验 | 判定 |
|---|---|---|---|
| T1 | “当查询需要精确更新+审计时，exact get 比 RAG 在 token 成本上高一个数量级” | H5 真实 reader：top-1/3/5 vs RAG，测准确率+token | 若 exact get 准确率不低于 RAG 且 token 显著更低 → 成立 |
| T2 | “编译设备（ADD8/SORT4）在真实 merge 后能 token-level 正确且不破坏 RV32” | ADD8 rv-tiny 端到端 + RV 回归 | IR 全量 0 mismatch、HF 1000 条 0 错、RV token 不变 → 成立 |
| T3 | “长序列正确性只需要 fp32 的敏感子层，而非全部 chip 层” | 局部 fp32 消融 + ≥60 paired turns | lock divergence ≤ 全 fp32 且显存/速度更优 → 成立 |
| T4 | “in-weights 记忆的差异化只在离线/审计/快照携带，不在速度或准确率” | 同 harness 对照 host/RAG（3 seeds） | 无显著准确率优势但有审计/离线优势 → 成立 |
| T5 | “compiler-exactness audit 可作为独立 benchmark 被社区采用” | 发布 toolkit + 2 个外部模型接入 | 有外部采用/引用 → 成立 |

## 4. 反炒作清单（证据不足）

- “in-weights 记忆比 RAG 更准/更快”：H1 无介质优势；H5 语义查询 exact-key 0%。
- “0 额外层 = 端到端合并已完成”：目前只是 IR/allocator/emit probe。
- “GPU bit-exact”：AMD 上不成立，验收是 token-level。
- “量化能大幅提速”：INT8/FP8 在 gfx1100 无算力收益；NF4 质量崩。
- “设备树可替代 CPU”：只适合固定形状纯函数，通用解析/循环仍属 RV32。
- “芯片 IP 马上能卖”：缺合作方与接口验证。

## 5. 建议（给 Stage 9/10 的资源分配）

1. **60% 工程**：S1/S4 合成“可审计记忆盘” demo（reader 集成 + 拒答 + audit + 版本门禁），这是最短路径的产品验证。
2. **30% 研究**：S3 ADD8-only rv-tiny 端到端（真实 merge + namespace/owner/exit），这是最强研究叙事。
3. **10% 基础设施**：fp32 局部化 + paired 60 turns + compiler-exactness toolkit；它们让上面两条叙事可信。

> 证据：docs/RESULTS.md、docs/research/stage9-technical-breakthroughs.md、docs/research/stage8-enhancement-portfolio.md、docs/notes/stage5-h5.md、stage7-h1-inweights.md、stage8-h5-e2e.md、stage8-device-merge.md、stage8-fp32chip.md。
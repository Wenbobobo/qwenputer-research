# Stage 9：最值得追求的技术突破方向

> 面向 Lead 的技术战略报告（2026-10-31）。本报告只基于 Stage 1–8、仓库笔记和已保存实验结果；本阶段不运行模型、不使用 GPU。文中的“假设”均以 **假设** 标出，不能当作已验证事实。

## 一、先定边界：项目真正已经证明了什么

Qwenputer 的最强证据不是“把一个模型量化后跑快”，而是证明了一个可审计的组合：编译器能够把确定性计算图写进 Transformer；在 CPU 上，12 个 chip 层、168/168 张量可以与 IR bit-exact；ADD8、SORT4、DADD 等小器件各用 2 层、约 899–3,876 个非零权重，并在 IR 与独立 HF 检查中 0 错。权重地图进一步显示，20.672 GiB 中 61.8% 是精确零，无损 core 压缩到 7.951 GiB，说明部署瓶颈主要是布局和带宽，而不是信息量。

同时，证据也规定了叙事边界。AMD 7900 XT/ROCm 的 bf16 GEMM 存在约 1 ULP 数值差异；CPU 才能称 bit-identical，GPU 应称 token-level 正确。fp32 chip 在同一 harness 的 12 个已知压力会话中把 lock divergence 从 3/12 降为 0/12，但这仍是压力样本结论。in-weights H1 的 10/10 changed 和 27/30 all 没有超过 host 覆盖式的中等规模证据；它证明“可承载更新语义”，没有证明介质独立带来记忆优势。H5 的 50 题 CPU 检索原型达到 Recall@1/3/5=58/84/96%，BGE p50 11.2 ms、SafeStore get p50 0.51 ms，但尚未接入 4B reader。

因此，Stage 9–10 应同时追求两类突破：一类把“编译神经外设”从 ADD8 规模推进到可合并、可调用、可审计的系统；另一类把数值、状态和运行时协议做成可复现的工程科学。所有速度目标都必须与 lock、head、token 和审计记录一起验收。

## 二、突破假设清单

下表的难度为 1–5（5 最难），时间是 CPU 主导的工程估计；不包含大型模型训练。

|编号|突破假设|难度|时间估计|优先判断|
|---|---|---:|---|---|
|B1|编译神经外设可形成通用“hardware-in-weights”设计模式：固定形状算子通过 token family、局部状态和可证明 exit 进入现有 chip 层。|4|6–10 周|最高研究价值|
|B2|设备树可端到端合并：ADD8/SORT4/DADD 共用现有 12 层，新增层数为 0；通过 token namespace、owner/exit 和状态冻结实现安全回退。|4|3–6 周|最高近期闭环|
|B3|in-weights 是一种参数化记忆介质，其优势只在离线、单模型、稳定 key、可审计状态边界内成立；边界可由 decision boundary 与 RAG/host 对照测出。|3|4–8 周|限定性研究|
|B4|fp32/bf16 混合数值设计可把 fp32 只放在 lock-sensitive 的局部（state update、norm、code/exit），在 AMD 上保留稳定性并减少 0.65 GB 成本。|4|3–6 周|高可信度|
|B5|量化感知电路编译可以在目标 dtype/后端上生成“误差预算内可验证”的器件；它不要求全模型 bit-exact，而要求器件 token/lock 不变。|5|8–16 周|长期研究|
|B6|chip layers 可以跨多个兼容模型共享，形成稳定的设备 ABI；共享范围由 hidden/layout/tokenizer/kernel 约束决定，而非名称相似决定。|5|3–6 月|暂缓验证|
|B7|memory-disk v1 可以成为可审计状态协议：magic/version、模型/编译指纹、SHA、journal、tombstone、迁移和容量错误码共同保证可恢复性。|3|3–5 周|工程必做|
|B8|trim + FLA + kernel fusion + native int8/tied-embedding 优化可把常用 put/get 从约 13–15 tok/s 推到 20–30 tok/s，同时不恶化长序列 lock。|5|6–12 周|条件性|
|B9|发布 compiler-exactness audit benchmark/toolkit，能统一报告 IR、CPU、NVIDIA/AMD 后端的 bit、ULP、token、lock 四级精度，从而建立该领域的可比基线。|3|4–8 周|论文/社区杠杆|
|B10|“编译器即安装器”可以扩展为从权重审计、布局生成到增量安装的可复现供应链，而不是一次性 checkpoint 复制脚本。|4|8–14 周|中期平台|

### B1/B2：从小器件到通用模式

**技术命题。** 对固定输入形状、有限位宽、无数据相关循环的纯函数，token family 提供类型，局部 head/conv 提供捕获，MLP 阈值提供比较，统一 exit 提供输出；器件可以与 RV32 回退共存。ADD8 的 65,536/65,536 IR 记录、SORT4 的 14,032 次预测和 DADD 的长进位样本均为强前置证据。

**为什么现在有机会。** merge_probe 已显示三器件同时合并为 12 层、slots 1899/2029，非零增量仅 +5626；容量不是当前 blocker。真正的 blocker 已被定位为 namespace、dispatcher/owner、RV 状态冻结、code-clear/统一 exit 和 embedding plumbing，这些都是可分解的协议工程问题。

**关键实验与验收。** 先只合并 ADD8：定义高位 namespace、ADD8_ENTER/A/B/RESULT，冻结 RV counter，执行连续调用、边界值、非法 token、核心回退和 reset；要求 IR 全量 0 mismatch、HF argmax 0 错、原 RV32 程序 token-level 不变、层数增加 0、merge 峰值 RAM 可复现。随后再加入 SORT4/DADD，并报告 token 成本、embedding 增量和失败回退。

**影响。** 研究上可提出“神经外设”的尺寸—协议—精确性定律；产品上可服务校验、排序、金额和结构化 agent。失败模式是自然语言无法稳定生成专用 token、器件抢占 RV owner、状态泄漏到下一条记录、词表行成本超过计算器件本身。难度 4，首个 ADD8 闭环 3–6 周。

### B3：in-weights 作为参数化记忆的真实边界

**技术命题。** in-weights 的价值不是一般语义记忆，而是让一个单模型快照携带有限、精确、离线、可审计的 KV 状态；语义发现应由 CPU BGE 完成，正文由 exact get 完成。Stage 5 的真实 LongMemEval 结果 exact-key 语义 0%、已知名 100%，而 H5 原型 @5=96%，已经明确了双层结构。

**关键实验。** 同 checkpoint、同 prompt、同 token 预算、至少 3 seeds，对比 in-weights、host overwrite、外部 KV、BGE+exact get、RAG；按 exact key、语义查询、更新、未命中、容量满、跨会话分层。报告 memory write recall、answer accuracy、prompt tokens、wall latency、错误类型和审计恢复率。接受条件应是：不宣称介质优势；只有在某一边界上相对 host 或 RAG 有显著且可复现收益才升级主张。当前事实是 512 文件、每文件 104 token、memory 绑定编译布局，必须显式返回容量和迁移状态。

**影响与失败。** 研究贡献是 decision boundary，而非“权重记忆更聪明”。产品价值是离线隐私、快照可携带、固定成本和审计链。失败模式包括漏存、键冲突、模型未调用 put、embedding 漂移、RAG 误召回、H1 小样本和评分格式偏差。难度 3，4–8 周。

### B4：局部混合精度的稳定性设计

**技术命题。** 对 lock-sensitive 的 chip state/update 路径使用 fp32，中间保持 fp32，出口再回 bf16；Qwen 文本层保留 bf16。下一步不是全模型 fp32，而是做 state update、norm、code/exit、attention q/k/v 的敏感度消融和 error-compensation kernel。

**关键实验。** 在同一 AMD harness 上比较 bf16、全 chip fp32、仅 state fp32、仅 norm/exit fp32、fp32+compensated accumulation，并测 12 个历史压力会话扩展到至少 5 seeds。接受：长序列 lock divergence 不高于 0/60（目标）、head mismatch=0、CPU reference 保持、峰值增量小于全 chip 的 0.65 GB，速度不低于当前 fp32 put/get 的 15.72/14.69 tok/s（目标可调整，标注假设）。

**影响与失败。** 这是最容易形成可靠工程论文的数值设计；失败则回到 fp32 chip correctness-first profile。风险是把相关性误当根因，或局部 fp32 在不同 shape/FLA kernel 上失效。难度 4，3–6 周。

### B5/B6：量化与跨模型共享

量化感知编译应先定义目标：Qwen 12–43 可近似，chip 层要求 token/lock 语义；不能把 NF4 崩溃或 int8 head 负结果包装成成功。实验应在 compile 时注入 W8/W4 误差，输出静态误差预算和 runtime regression；验收包括 token、logit ULP、lock、长序列和显存。当前 ROCm int8 GEMM 不快、无原生 FP8，故 B5 不是 Stage 9 主赌注。

跨模型共享更难：共享必须以 hidden size、GDN 状态布局、token ABI、kernel ABI 和 tokenizer namespace 的 hash 为前提；先做两个同架构小模型的 chip graph 重用，若任何一项不一致即拒绝。没有第二基座证据前，B6 只能做接口设计，不能作性能承诺。

### B7/B8/B9/B10：协议、速度与领域化

memory-disk v1 应补齐权重/编译 build hash、append/delete/tombstone、并发事务、权限/来源/TTL、容量错误码和迁移工具；它的发布价值高于继续扩大“内存介质优势”叙事。B8 的 >20 tok/s 只在逐项实验后成立：先 profile trim 与 FLA 的 kernel fusion，再测 causal-conv1d，最后测原生 int8 GEMV 和 tied embedding；每项都要 lock gate，不能沿用仅 Hi 或短 put 的数字。B9 应提供最小 benchmark：ADD8/SORT4/DADD 全量 IR、CPU bit exact、GPU ULP/token/lock、负对照 seed、不同 batch/shape 和审计 JSON。B10 将 compiler version、config、tokenizer、weights index、anti-sink 16 KiB 和 SHA 绑定，形成可安装、可回滚的 manifest；但不要把未验证的“基座+130 MB 增量”写成现状。

## 三、优先级矩阵与 Stage 9–10 Top 3

评分：影响、可行性、证据强度均为 1–5；成本 1–5，5 表示成本最高。综合分 = 影响×可行性×证据强度÷成本，仅用于排序。

|方向|影响|可行性|证据|成本|综合分|Stage 9–10决定|
|---|---:|---:|---:|---:|---:|---|
|fp32/局部混合数值|5|4|4|2|40|Top 1|
|ADD8→设备 merge 闭环|5|4|3|3|20|Top 2|
|BGE→exact get→SafeStore|5|4|3|3|20|Top 3|
memory-disk v1|4|4|4|2|32|并行基础设施|
compiler-exactness toolkit|5|4|3|3|20|并行发布|
硬件 in-weights 泛化|4|3|3|4|9|由设备闭环证明|
>20 tok/s|4|2|2|5|3.2|条件推进|
量化感知编译|4|2|2|5|3.2|后置|
跨模型共享|5|2|1|5|2|先写 ABI|

**Top 1：fp32/局部混合数值。** 接受测试：至少 60 个长序列 paired turns；bf16 与候选的 lock divergence 显著下降，目标 0/60；head mismatch=0；CPU reference 对拍不退化；候选显存低于全 chip fp32；所有 FLA/no-FLA 条件单独报告。失败即保留 fp32 chip + no-FLA correctness profile，不扩大结论。

**Top 2：ADD8 设备合并。** 接受测试：真实 merge.py 产物而非仅 probe；层数 12 或增加不超过 1；IR exhaustive 0 mismatch；HF 1000 条边界/随机 argmax 0 错；RV32 既有程序 token-level 不变；namespace 冲突、owner、exit、reset、回退各有测试；峰值 RAM 和新增 embedding 行可复现。通过后再合并 SORT4/DADD。

**Top 3：H5 混合检索产品闭环。** 接受测试：预构建/增量索引后完成 100 题真实数据；报告 @1/@3/@5、未命中拒答、更新和容量满；Recall@5 不低于已测 96%（合成）或真实基线 87%（若复用真实集）；命中时 SafeStore exact 正文 100%；审计日志含 query hash、embedding 版本、snapshot fingerprint、journal id；接 reader 前不得把 retrieval accuracy 叫生成 accuracy。

## 四、可发表论文与当前可辩护主张

|论文/Artifact|当前可辩护的 claim|已有证据|强论文还缺什么|
|---|---|---|---|
|Hardware-in-weights / neural peripherals|固定形状有限状态器件可在 Transformer chip 中被编译，并在 IR/CPU 上逐位验证；ADD8、SORT4、DADD 远小于 RV32。|设备树估计、全量 IR、HF 0 错；编译器 12/12 层 bit-exact。|真实 merge、统一协议、与 RV32 同边界的端到端 latency、更多算子和负例。|
|AMD mixed precision correctness|在选定 AMD 长序列压力样本中，chip fp32 将 lock 3/12 降至 0/12，代价约 +0.65GB。|严格 paired 12 会话、head 全 0、原失败点 calc 完整。|至少 5 seeds/更长 trace、定位根因、局部消融、NVIDIA/CPU 对照和统计区间。|
|Parametric memory boundary|in-weights 能承载覆盖式状态，但现有结果不证明其相对 host 的独立优势；语义入口需要 BGE。|H1 n=10/30、host 90%、H5 exact 0% semantic、BGE @5 与 token 对照。|同 prompt/checkpoint 多 seed 介质等价实验；真实 reader；完整 LongMemEval；成本和拒答分析。|
|Auditable memory-disk|快照可小、可校验、可拒绝错误指纹、可 journal 迁移，SafeStore 可防长名静默覆盖。|v0 40/40；seed 迁移；collision fix。|v1 并发、crash consistency、tombstone、版本升级、fuzz、威胁模型和跨平台实现。|
|Compiler-exactness benchmark|精确性必须分为 IR/CPU bit、GPU ULP、token、lock 四级；后端声明不能互换。|CPU bit exact、AMD residual、FLA/no-FLA 对照、负 seed。|公开 harness、更多 GPU/shape/batch、统一 JSON schema、第三方复现。|
|Lossless deployment runtime|零填充 trim 与 fused GDN 比简单量化更能改善该 merged 模型的部署带宽；速度可到约 13–17 tok/s。|61.8% zero、7.951 GiB SHA、trim/FLA measurements、int8 head negative。|同条件 profile、kernel fusion、CPU/AMD/NVIDIA 对照、长序列 correctness gate。|

论文表述应使用“在本机、给定 checkpoint、给定 harness 和样本范围内”。不能写成“GPU bit-exact”“in-weights 比 RAG 更聪明”“H5 生成准确率 96%”或“0 extra layers 已端到端完成”。

## 五、评审人最可能攻击的点与补强

|攻击点|为什么成立|补强方法|
|---|---|---|
|fp32 结论只有 12 个压力会话|3/12→0/12 很有方向性，但样本由已知失败点选择。|扩展未知 seed、随机长 trace、不同序列长度；预注册 lock/head/token 指标与停止规则。|
|AMD 非 bit-exact|Stage 4–7 已发现 ROCm bf16 残差，不能沿用作者 NVIDIA L4 结论。|展示 CPU bit、AMD ULP 分布、token/lock 分层；提供 deterministic reference profile 和后端声明。|
|in-weights 没有 medium advantage|H1 host overwrite 已 90%，in-weights n=10 的 100% 可能是小样本/提示格式。|严格 matched multi-seed ablation；将介质优势降级为未证实，聚焦离线/隐私/审计边界。|
|H5 benchmark 很小且是 retrieval|quick.json n=50，真实全量曾超时，未接 4B reader。|批处理完成 100 题；区分 retrieval、reader answer、拒答；报告 top-k 和置信度，不把候选命中当答案。|
|设备 token 成本比较不公平|DADD 的 RV32 参考含解析/调度，器件输入假设已是 LSB 配对。|统一输入输出协议，分别计 capture、compute、print；同时给最坏和理想边界。|
|速度数字受 kernel/线程/JIT 影响|FLA、ROCm、OMP 和首次编译会改变结果；int8 head 已负收益。|记录环境、warm/cold、JIT 状态；固定线程；每配置重复；使用 lock gate；不以单个 Hi 作为主指标。|
|0 extra layers 只是 allocator probe|merge_probe 未加载 Qwen 权重，真实 owner/exit 未接线。|把“IR/allocator feasible”与“production merge”分开；Top 2 的验收必须是真实 merge 产物。|
|编译器复现不够可审计|anti-sink 16 KiB、版本和 build hash 尚未完全绑定。|compiler commit、torch、tokenizer、config、weights index、SHA 和 anti-sink 纳入 manifest；负 seed 作为持续门禁。|

## 六、六周执行计划

|周|交付|验收|
|---|---|---|
|第 1 周|冻结 benchmark schema；完成 fp32/bf16 paired harness 扩展；定义 ADD8 namespace/dispatcher 状态机；memory-disk v1 manifest 草案。|所有实验输入、seed、dtype、FLA、环境和失败码可复现；不写未经证实的宣传句。|
|第 2 周|ADD8 merged tiny 原型：owner、state freeze、统一 exit、reset、RV fallback；fp32 局部层候选消融。|ADD8 IR 与 tiny exhaustive 通过；局部候选有逐层误差/lock 报告。|
|第 3 周|ADD8 接入 text/core bridge；实现 v1 journal、tombstone、容量错误和 fingerprint；H5 增量索引。|真实合并模型既能跑旧 RV 程序又能跑 ADD8；快照 crash/迁移测试通过。|
|第 4 周|H5 100 题 retrieval 全量；BGE top-k、低分拒答、SafeStore 审计；fp32 长序列扩展至至少 3 seeds。|Recall、拒答、exact-get 和审计四项分开报告；lock 不因 H5 入口改变。|
|第 5 周|SORT4 或 DADD 第二器件；compiler-exactness audit CLI；速度 profiling（trim、FLA、conv、head）。|至少一个第二器件复用 dispatcher；CLI 输出四级 precision schema。|
|第 6 周|Stage 9 论文包和 artifact release candidate；复盘 Top 3 gate，决定是否推进 >20 tok/s/QAT。|可公开复现的 JSON、命令、负对照、限制说明；未通过的方向进入 kill list。|

## 七、六个月路线

**0–6 周：证明闭环。** 交付 Top 3，优先得到真实 ADD8 merge、扩大 fp32 统计和完整 H5 retrieval。memory-disk v1 与 audit toolkit 作为共用基础，不依赖模型训练。

**第 2–3 个月：形成论文系统。** 扩展器件族到 CRC、有限 DFA、查表和小排序；统一 token ABI/owner/exit；用 CPU exhaustive 加 AMD token/ULP/lock 回归。完成 in-weights vs host/RAG 的 matched 多 seed boundary，若没有介质优势就正式把主张收窄为“可审计离线参数化状态”。

**第 4 个月：运行时性能。** 只在正确性 gate 通过的分支测 causal-conv1d、kernel fusion、native int8 GEMV、tied embedding；目标是 put/get >20 tok/s，同时长序列 lock 不回归。量化感知编译以误差预算和器件语义为中心，避免先做 NF4 这种已表现为质量崩的路径。

**第 5 个月：可移植性和安装器。** 做第二个兼容模型或小型同架构基座，验证 chip ABI 是否可共享；把编译器版本、配置、权重索引、anti-sink 和 SHA 纳入 memory-disk/安装 manifest；跨模型不兼容时必须可解释拒绝。

**第 6 个月：发布与止损。** 发布 benchmark、代码、审计数据、论文预印本和限制清单。只有在至少两个后端/多 seed/真实 merge 证据成立时，才把“通用硬件 in weights”升级为强主张；否则保留为“可证明的固定形状神经外设 + 可审计状态协议”。若 >20 tok/s 需要牺牲 lock，停止追逐吞吐；若跨模型 ABI 成本超过单模型编译收益，停止共享层路线。

## 给 Lead 的最终决策

1. **Top 1：局部混合精度（先 fp32 chip，再做局部化）**——已有 3/12→0/12 的最强直接证据；验收是 60 个 paired 长序列、head=0、显存和速度门禁。
2. **Top 2：ADD8 真正设备合并**——merge_probe 已把容量风险降为协议风险；验收是 12 层真实 merge、exhaustive IR/HF、旧 RV 不变、namespace/owner/exit/reset 全通过。
3. **Top 3：BGE 语义路由 + exact get + SafeStore**——最接近可用产品；验收是 100 题、@5 不退化、命中正文 100%、未命中诚实拒答、完整审计链。

可发表主张是：编译器可把小型确定性器件写进 Transformer 并在 CPU/IR 上证明；AMD 后端需要四级精度契约，局部 fp32 能修复选定长序列 lock；in-weights 是受限参数化状态而非普遍优于 RAG 的记忆；memory-disk/SafeStore 能把状态变成可校验协议。最大 gaps 是真实 merge、扩大统计、完整 H5 reader、跨模型证据和公平的端到端速度比较。

# Qcompute · Qwenputer 复现与超越 — ROADMAP

> 目标：复现并验证 Qwenputer（把 32 位 RISC-V CPU 与文件系统编译进 Qwen3.5-4B 权重）的核心能力；
> 做功能/性能优化；产出精美 HTML 原理报告；研究「in-weights 文件系统 / DeltaNet 状态」作为
> 参数化记忆与持续学习基础设施的潜力，并与 RAG / 知识图谱 / Engram 等方案对比。
>
> **状态提示（2026-10-31）**：本文件是追加式历史规划；当前状态以 [STATUS.md](STATUS.md) 与 [docs/RESULTS.md](docs/RESULTS.md) 为准，完成情况见各阶段报告与 [docs/INDEX.md](docs/INDEX.md)。

## 硬性约束
- 硬件：Ryzen 7 5700X（8C/16T）/ 47.9 GB RAM / RX 7900 XT 20GB + Tesla T10 16GB（本轮全程 CPU）
- GPU 静默时段：北京 00:30–09:30、12:30–15:30 —— GPU 工作排窗口外
- 发布 checkpoint：20.73 GiB 磁盘 / 22.20 GB 十进制 / ~18GB 显存门槛；T10 无 bf16 不可用
- 所有结论可复现：命令、日志、指标在 docs/notes 或 docs/reports；证据路径写进 STATUS.md

## 阶段评估（2026-10-29 02:35）
**目标成立；三阶段均已完成主体，可进入收尾与外延。**
1. 复现：机制级（IR/tiny/文件系统/GDN-RAM）+ 发布 checkpoint 端到端（smoke/put/get）全部成功。
2. benchmark 数字（83.3/90/97.5）本机不可复现（需 24GB bf16 GPU + 数小时），按作者声明标注。
3. 记忆假设：**不成立为普遍结论**——只在窄区间更省 token；已给出定量边界与可证伪实验。
4. 优化：无损压缩实测 22.20GB→7.951GiB；CPU 端到端可行（0.21 tok/s）。
5. 创意：A（编译器即安装器，审计进行中）与 D（记忆盘，已实测）最值得推进。

## Stage 0 侦察（✅）
- [x] 上游 README / 博客 / 设计文档；双分支 clone；上游地图 docs/notes/upstream-map.md
- [x] 硬件/环境盘点；22.3GB checkpoint 完整下载（20.73 GiB）

## Stage 1 复现（✅ 机制级全部完成）
- [x] S1.1 环境：uv + torch 2.14.1+cpu + transformers 5.19.0；pytest 全绿
- [x] S1.2 参考/电路：test_rv_ref+fuzz+bridge 143 passed/4 skipped；test_rv_core 94 passed/10 skipped；test_rv_hf 3/3
- [x] S1.3 rv-tiny CPU：构建 2.13B；4 个程序 free-run 与参考逐 token 一致（signext 44、rv32ui-simple 3、memcpy 159、call 474）；residual==IR 118,800 cells
- [x] S1.4 文件系统：test_rv_store 10/10（5 op、哈希、碰撞、满容量、4B 全尺寸、跨会话快照）；store_in_hf prefill+decode PASSED
- [x] S1.5 发布 checkpoint 端到端（CPU）：smoke（自发 list→empty）、put（→ok）、get（空上下文→4471-AZ）全部成功，0 mismatch
- [x] S1.6 独立 GDN-as-RAM 验证 PASS（2492 写/2508 读 exact；3 负对照）
- [x] S1.7 声明核对表 docs/notes/claims-audit.md

## Stage 1b 优化（✅ 主体完成）
- [x] O1 零填充压缩：全量扫描 + 无损压缩 22.20GB→7.951 GiB（10/10 bit-exact + SHA-256）
- [x] O2 CPU 推理验证：加载 ~20s、单 token ~4.7s；list/put/get 可行；长回答/benchmark 不可行
- [x] O3 文件系统容量/延迟：H2 名字别名/探测、H3 token 交叉点 n*≈119–141
- [ ] O4 桥接协议优化：方案分析（C dispatch 200–250 token/调用）——仅记录，未实现
- [ ] O5 agent 记忆 API：设计稿在 stage2/stage3 报告中

## Stage 1c / 2 / 3 报告（✅ 三份 HTML）
- [x] docs/reports/qwenputer-stage1.html（原理/前史/可行性/复现/成本/记忆/优化/场景/参考）
- [x] docs/reports/qwenputer-stage2-memory.html（记忆分类/成本模型/90vs30 打假/Engram/持续学习/H2H3/端到端）
- [x] docs/reports/qwenputer-stage3.html（5 创意 + D 原型结果 + 判据）

## Stage 2 研究（✅）
- [x] R1 文献与机制对比（prior-work.md + parametric-memory.md）
- [x] R2 成本模型：C_qp ≈ τ(t_call+t_print+t_wrap+λ·4·n_names)+(τ−1)S/Q；1k/10k/100k/1M 表
- [x] R3 实验：H2/H3 完成；H1/H4/H5 设计与判据已写（受 GPU/基座缺位限制）
- [x] R4 自我迭代/自我进化：四层判定 + 零干扰上界 + 瓶颈在写/读决策
- [x] R5 阶段二报告；R6 文件夹整理（docs/INDEX.md、冗余合并、口径统一）

## Stage 3 创意（🟡 A 进行中，D 完成）
- [x] D 记忆盘：快照 1/10/100/512 文件 = 77B/723B/7.2KB/36.7KB；换 seed 0% 读回；长名字别名实测
- [x] A 编译器即安装器：**12/12 层、168/168 张量 bit-exact**；final norm / chip embedding / 文本行 chip 特征均 bit-exact（74.6s CPU）
- [x] B 设备树：ADD8/SORT4/DADD 已真实构建（IR exact；见 docs/notes/device-tree-estimate.md）；并入 chip 层的 IR/allocator/emit 可行性见 docs/notes/stage8-device-merge.md
- [ ] C 监控器：评审/证伪清单（依赖作者资源）
- [x] E 自举/OS 量级边界分析

## 后续外延（进行中）
- [x] 基座 Qwen3.5-4B 下载 → **Qwen 侧审计完成**：426 张量 0 无法解释；338 base+widen bit-exact / 80 mute-only / 8 anti-sink；137,752 差异元素全在 mu 列；不可推导残留 16,384B
- [ ] CPU 文本保真小样本（merged vs base）——未做（需同时加载 31GB 双模型；GPU 窗口更合适）
- [ ] 7900 XT ROCm（09:30 后）：压缩布局（~8GiB）加载 + 执行器吞吐验证
- [ ] 下载基座 Qwen3.5-4B：验证「基座+130MB 增量」发布方案
- [ ] H1/H5 真实记忆实验（需基座/GPU）
- [ ] 把 memory-disk 格式写成规范 + 原型读写器
## Stage 4 后续推进（2026-10-29 10:15 启动）
用户确认：推进 1–4；**不做第 5 项（基座+增量安装器 / SHA-256 清单）**，避免复杂度。

### 约束
- GPU 窗口：WSL Ubuntu-24.04 + ROCm 7.2.1 + RX 7900 XT 19.96GB（bf16 OK）；允许时段 09:30–12:30、15:30–00:30，窗口内做完或 checkpoint
- 工具：`wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh <script.py>`（已导出 HSA/PYTORCH_ROCM_ARCH/PATH，venv=~/src/rocm-smoke/.venv，transformers 5.19.0 已装）
- 显存：rv-tiny 4.3GB 与基座 8.7GB 可共存，但需串行启动（A 完成后写 gpu/A_DONE 标志，B 轮询后启动）

### 任务
- [x] S4.1 GPU 路径验证（A）：rv-tiny 7/7 程序 free-run token-exact；HF 路径 43-45 tok/s（CPU 的 ~40×）；Triton 执行器 1,334/4,345/6,993 tok/s（b1/b8/b32）且 IR-exact；发现 ROCm residual 非 bit-exact（第一个 FFN gate/up GEMM 残差，token 不受影响）；发布模型稠密 20.67GiB > 19.96GiB，需 split/压缩
- [x] S4.2 H1 记忆实验（B）：**C4 覆盖式主机工具 27/30=90%**；**C2 append-only 7/30=23.3%**（p=2.4e-7）；C1 pooled strict 42/90=46.7% / lenient 90/90=100%；C3 pooled strict 14/90=15.6% / lenient 48/90=53.3%；**brevity 变体 C3s 56/60=93.3%、C1s 82/82=100%** → 90vs30 主要是回答格式/评分，介质无优势，覆盖语义仅在工具路径有效
- [x] S4.3 memory-disk v0（C）：memory_disk.py（fingerprint/save/load/verify/journal/NameGuard），40/40 测试 PASS；stage3-d v1.1；docs/notes/memory-disk-v0.md
- [x] S4.4 设备树小原型（D）：真实构建 ADD8/SORT4/DADD（均 2 层、899/3876/1454 非零、IR 全量 exact、HF argmax 0 错）；对比 RV32 11-12 层/1.04M 非零
- [x] S4.5 阶段四报告：docs/reports/qwenputer-stage4.html（19.2KB，GPU/H1/memory-disk/设备树 + 证据索引）

### Stage 5 候选（后续轮次）
- [x] S5.1 压缩布局执行端：trim runtime 已在 7900XT 跑通 put/get（见 stage6/7/8）
- [x] S5.2 设备合并：IR/allocator/emit 验证 0 额外层（stage8-device-merge.md）；端到端待做（Stage 9）
- [x] S5.3 H5：LongMemEval-100 完成（stage5-h5.md）；端到端原型见 stage8-h5-e2e.md
- [x] S5.4 名字键修复：SafeStore（36-bit tag + salt probing）38/38 测试通过（stage5-keyfix.md）

### 已完成前置
- WSL ROCm 环境验证：torch 2.9.1+rocm7.2.1、GPU RX 7900 XT、bf16 OK、transformers 5.19.0、qwen3_5 import OK、mem 19.96GB
## Stage 5（2026-10-29 18:00 启动，GPU 窗口 15:30–00:30）

### 目标
- S5.1 发布模型真跑：22.3GB dense 放不下 19.96GB → 用 accelerate device_map（embedding/lm_head 放 CPU，layers 放 GPU）或 trim 布局，跑通 list/put/get，记录 tok/s 与正确性
- S5.2 H5-lite：LongMemEval 风格语义查询（100 题），精确 key vs bge-small 向量检索
- S5.3 6-token 名字键协议修复（扩 key 或显式冲突），在参考 store + memory-disk 上验证
- S5.4（可选）设备合并：ADD8/SORT4 并入 chip 层，验证 0 额外层假设
- S5.5 阶段五报告（docs/reports/，≤1 份）

### 风险与约束
- GPU 静默 00:30–09:30 / 12:30–15:30；单任务串行；卡在 12:30 前收尾或 checkpoint
- device_map offload 可能改变数值/答案（文档记录 HF offloaded cache 曾给出错误答案）；需 IR replay/参考机对照
- 设备合并需要重新 merge（峰值 44.7GB RAM，48GB 机器危险）→ 可能只做 IR 级/小规模验证
## Stage 6（2026-10-29 19:40 启动）：性能与量化
用户目标：先解决性能问题；调研该模型是否有/能否做量化版本（FP8 等）；调研 7900 XT 的 16/8/4-bit 算力性价比与迁移路径（含 W4A8/W4A4）；做轻微适配后继续推进。

### 关键前提（来自阶段一/三结论）
- chip 编译层必须保持 bf16 精确（q35-size.md 明确 chip 层不能 e4m3；本机审计也证明 bit-exact）
- 可量化的只有 Qwen 32 层 + embedding/lm_head；文本行为可接受近似
- 发布模型无官方/社区量化版（HF 上 Qwen3.5-4B 有 FP8/GGUF，但都是 base 模型；merged 44 层扩展词表需自行量化）
- 当前瓶颈：11 层 Qwen 在 CPU（~1 tok/s）+ dispatch load 150–264s；目标让整模型进 20GB 显存

### 任务
- [x] S6.1 GPU 精度微基准（S1）：BF16 75.8–78.3 TFLOPS、INT8 70 TOPS、FP8 无原生（Triton 降级 FP16）、INT4 有 WMMA+decode GEMV 1.4–1.5×
- [x] S6.2 量化适配：bnb8 + PreserveColumns 全 GPU 正确（11.49GB，2.38 tok/s），NF4 质量崩 → 量化仅作显存保底
- [x] S6.3 生态/迁移调研：无 Filesystem 量化版；bnb ✅/torchao ❌/GPTQ/AWQ 可能；W4A8 需自研内核，W4A4 长期
- [x] S6.4 无损 Trim：13.15GB 全进显存；CPU bit-identical；GPU put 6.79/get 9.0；+FLA 后 **put 16.84 / get 13.62 tok/s**（基线 18×）
- [x] S6.5 阶段六报告：docs/reports/qwenputer-stage6.html；STATUS/ROADMAP 已更新

### Stage 7 候选
- [ ] FLA 回归门禁（gpucheck --decode）+ causal_conv1d ROCm 编译
- [ ] lm_head int8/embedding int8（再省 2-3GB/token）
- [ ] W4A8 自定义 Triton 内核（长期）
- [ ] trim+FLA 运行时接入 H1 in-weights 对照臂 / memory-disk v1
- [ ] batch/continuous batching
## Stage 7（2026-10-29 22:00 启动；用户指令：继续 1–4，5 不做自定义内核；并调研 7900XT 社区公认最佳方案 + 使用 C:\Projects\ROCM 知识库）

### 任务
- [x] S7.1 FLA 精确化：selective 路由实现（chip ref + Qwen FLA）put 12.44/get 12.15 tok/s；**gpucheck 无法通过（AMD bf16 本身非 bit-exact，屏蔽 FLA/0.5.2 同样失败）→ 作者 bit-exact 是 NVIDIA L4 结论**
- [~] S7.2 causal_conv1d：未做（A7 agent 失败 + 时间优先给 H1/int8）；列为 Stage 8
- [x] S7.3 int8 head：CPU 检查通过（675MB、circuit 探针 exact），**GPU 实测负结果**（Hi 13.48→11.21、put 13.60→11.73，+0.7GB）；HIPBLASLt 10.58、TunableOp 4.58 均负收益
- [x] S7.4 H1 in-weights：n=10，**changed 10/10=100%、facts 17/20、all 27/30**；3 次历史 lock drift、30 问题轮干净 → 与 host 覆盖式同档，介质无优势
- [x] S7.5 社区最佳实践调研（R1）：llama.cpp HIP/Vulkan、vLLM ROCm、torch.compile/Triton、hipBLASLt TunableOp、bitsandbytes/Quark/AWQ、AOTriton……→ docs/notes/stage7-community.md（llama.cpp Vulkan decode 比 HIP 快 14–30%、vLLM gfx1100 可用但缺 GDN、bnb0.50 ROCm Stable、FP8 无、expandable_segments 禁用；Top5 措施 + 3 实验）
- [x] S7.6 阶段七报告：docs/reports/qwenputer-stage7.html；STATUS/ROADMAP 已更新

### Stage 8 候选
- [x] fp32 chip 层试验：paired 对照 **bf16 3/12 → fp32 0/12 lock divergences**（同 harness/同日/同 GPU），put 15.72/get 14.69/Hi warm 18.87 tok/s，峰值 13.94GB；待扩到更多输入 + gpucheck fp32 门禁
- [ ] 共享 tied 张量 int8（embedding 行 int8 + chip 维稀疏 bf16）+ 更粗 dequant
- [ ] causal_conv1d → FLA Triton 短卷积（剩余 2 个 fallback）
- [ ] memory-disk v1（trim + SafeStore + in-weights）
- [ ] H1 结论整合进阶段二/四报告

### 已知起点
- trim no-FLA：put 6.79 / get 9.00 tok/s，bit-exact；trim+FLA：put 16.84 / get 13.62，但 gpucheck 5 测试 MISMATCH（6e-8..9.4e-7）
- ROCM KB 已确认：稳定栈 ROCm 7.2.1 + torch 2.9.1 + Triton 3.5.1；gfx1100 有原生 IU4 WMMA 但无 turnkey 库；Triton 无 tl.int4；WSL 缺完整 profiler
## Stage 8（2026-10-30 15:45 启动）：发布、计算增强评估、推理提速
用户指令：把解释性 HTML 打包上 GitHub Pages（可分阶段切换，每阶段一页）；之后每阶段报告也上传；继续推进两条主线：① 增强计算部分（评估 brainstorm 方案性价比/落地场景）；② 正常提高推理速度；保持计算能力不倒退，同意与专家深入讨论后选最佳方案。

### 任务
- [x] S8.1 GitHub Pages：公开仓库 github.com/Wenbobobo/qwenputer-research，站点 wenbobobo.github.io/qwenputer-research/（hub + stage1–8，全部 HTTP 200；Actions 自动部署；更新流程 docs/notes/stage8-pages.md）
- [x] S8.2 推理提速：同 session 矩阵 Hi bf16 no-FLA 8.51 / selective 14.59 / full FLA 12.04 / fp32 no-FLA 11.81 / fp32 selective 13.64 tok/s；长序列 bf16 3/12 → fp32 0/12 lock divergence；推荐 correctness-first = fp32 no-FLA, speed-first = bf16 selective（长序列待验证）
- [x] S8.3 计算增强 portfolio：docs/research/stage8-enhancement-portfolio.md（Top3 + kill list + E1/E2/E3 + gates）
- [x] S8.4 增强原型：E2 设备合并 IR/allocator/emit 成功（0 额外层、slots 1899/2029、nz+5626）；E3 H5 端到端 CPU 原型（Recall@1/3/5 = 58/84/96%、BGE 11.2ms、SafeStore 0.5ms、碰撞隔离通过）
- [x] S8.5 阶段八报告：docs/reports/qwenputer-stage8.html 已发布到 Pages

### Stage 9 候选（按 ROI）
- [ ] ADD8-only rv-tiny 端到端原型（真实 merge + tokenizer/namespace + owner/exit dispatcher）
- [ ] H5 reader 集成（4B reader 上 top-1/3/5 vs RAG 对照 + 低分拒答 + 索引版本门禁）
- [ ] fp32 chip 局部化（只 fp32 敏感子层，压缩 +0.65GB 与速度代价）
- [ ] 速度 >20 tok/s（native int8 GEMV / tied embedding 量化 / causal_conv1d Triton 短卷积）
- [ ] memory-disk v1（trim runtime + SafeStore + journal 的可用记忆盘 API）
## Stage 9（2026-10-31）：落地场景专家评估、文档整理与交接
- [x] 应用场景与突破方向：docs/research/stage9-application-scenarios.md、stage9-technical-breakthroughs.md、stage9-synthesis.md（Top1 产品=可审计记忆盘；Top1 研究=确定性设备树）
- [x] 文档系统整理：docs/RESULTS.md（唯一数字台账）、docs/GPU-DISCIPLINE.md、docs/GLOSSARY.md、docs/START-HERE.md（交接 prompt）、docs/DOC-AUDIT.md（审计+归档映射）、README/INDEX/DELIVERABLES 重写、归档 4 篇过时文档 + 清理 %SystemDrive% 垃圾目录
- [x] 阶段九报告：docs/reports/qwenputer-stage9.html 已发布
- [ ] Stage 10 候选：ADD8-only rv-tiny 端到端；H5 reader 集成；fp32 chip 局部化；>20 tok/s；memory-disk v1
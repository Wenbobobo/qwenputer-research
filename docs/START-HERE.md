# START HERE —— Qwenputer 项目交接与启动 Prompt

> 面向：接手的 AI agent / 人类研究者。读完这一份就可以独立开工；更细的证据在 docs/notes 与 docs/reports。
> 最后更新：2026-10-31（Stage 9 收尾）· 仓库：https://github.com/Wenbobobo/qwenputer-research · 在线报告：https://wenbobobo.github.io/qwenputer-research/

## 0. 一句话

复现、审计并扩展 **Qwenputer**：一个把 RISC-V CPU 与完整文件系统"编译进 Qwen3.5-4B 权重"的可运行系统；在此基础上研究"权重内计算/记忆"的边界、性能与落地场景。

## 1. 30 秒速览（已经确立的事实）

| 结论 | 证据/数字 |
|---|---|
| 编译器审计通过：chip 侧 168/168 张量 bit-exact；整模型不可推导残留仅 16,384 B（anti-sink） | docs/notes/stage3-a-rebuild.md、stage3-a2-base-audit.md |
| 22.20 GB → 7.951 GiB 无损压缩（10/10 SHA-256 一致） | docs/notes/weights-map.md |
| 发布模型可在 CPU 端到端 put/get（4471-AZ，0 mismatch） | docs/notes/repro-log.md、demo/ |
| GPU：rv-tiny token-exact；HF 路径 43–45 tok/s；Triton 执行器 1,334–6,993 tok/s（b1–b32） | docs/notes/stage4-gpu.md |
| H1 记忆实验：覆盖式 host 工具 changed 90%，append-only 23%，上下文 15.6–46.7%（strict），加"简短回答"后上下文 93–100% → **90 vs 30 主要是覆盖语义+回答格式，不是介质优势** | docs/notes/stage4-h1.md |
| H1 缺失的 in-weights 一臂已补：发布模型自己的 store changed **10/10=100%**，与 host 同档（n=10, 95% CI 72–100） | docs/notes/stage7-h1-inweights.md |
| H5：精确键语义查询 0%，bge-small 语义 Recall@5 87%（LongMemEval）；exact get 45 token vs RAG ~12,141 token | docs/notes/stage5-h5.md、stage8-h5-e2e.md |
| AMD 精确性边界：7900 XT 上 bf16/fp32 GPU 均非 bit-exact（屏蔽 FLA 也一样，−5.96e-8、0 wrong tokens）；作者的 bit-exact 是 NVIDIA L4 结论；**CPU 路径 bit-identical** | docs/notes/stage7-selectfla.md、stage8-fp32chip.md |
| **fp32 chip 层修复长序列正确性**：同样 12 个历史会话 bf16 3/12 → fp32 0/12 lock divergence（+0.65GB） | docs/notes/stage8-fp32chip.md |
| 设备树 IR 可行且 **0 额外层**：ADD8+SORT4+DADD 同时并入仍 12 层，slots 1899/2029，nz +5,626 | docs/notes/stage8-device-merge.md、analysis/device/merge_probe.json |
| 量化：bnb8 可跑但只有 ~1–2 tok/s；NF4 质量崩；int8 head 负收益；**量化只省显存不省算力** | docs/notes/stage6-quant-*.md、stage7-int8-head.md |
| 当前最快正确配置：trim + fp32 chip + no-FLA（put/get clean，长序列 0/12）；速度优先：trim + bf16 + selective FLA（Hi 14.59 tok/s） | docs/notes/stage8-speed.md |

## 1b. 场景与突破方向（Stage 9 专家结论，详见 docs/research/stage9-synthesis.md）

- **最强近期产品**：H5 语义路由（CPU bge）+ in-weights exact get + SafeStore/memory-disk 的"可审计记忆盘"（离线/隐私/审计；45 token vs RAG 12k）。
- **最强研究突破**：可审计的确定性设备树（ADD8→SORT4/DADD 真 merge；IR 可验证、0 额外层已 probe）。
- **必要基础设施**：fp32 chip 长序列正确性 + 局部混合精度消融 + compiler-exactness audit toolkit。
- **不成立**："in-weights 介质更优"（H1 无介质优势）；"GPU bit-exact"（AMD 只 token-level）；"量化提速"（只省显存）。

## 2. 阶段索引（全部 HTML 已上线）

| 阶段 | 主题 | 报告 |
|---|---|---|
| 1 | 原理、前史、复现、成本、记忆、优化、场景 | docs/reports/qwenputer-stage1.html |
| 2 | 参数化记忆 vs RAG/KG/Engram；90 vs 30 统计拆解 | stage2-memory.html |
| 3 | 5 个创意 + 编译器审计 + memory-disk 边界 | stage3.html |
| 4 | GPU 路径、H1 记忆实验、memory-disk v0、设备树 | stage4.html |
| 5 | 发布模型 GPU 真跑、H5（LongMemEval）、SafeStore 键修复 | stage5.html |
| 6 | 性能与量化（trim+FLA、bnb、硬件实测） | stage6.html |
| 7 | 精确性边界、selective FLA、H1 in-weights、int8、社区调研 | stage7.html |
| 8 | GitHub Pages、速度矩阵、计算增强组合、设备合并、H5 e2e | stage8.html |
| 9 | 落地场景与突破方向（专家）、文档整理、交接 | 本目录 stage9-* 与 docs/research/stage9-* |

## 3. 环境与运行

- **Windows 11 + WSL2 Ubuntu-24.04**；GPU：AMD RX 7900 XT 19.96GB（gfx1100，ROCm 7.2.1）；另有 Tesla T10 16GB（备用，不要与 7900XT 并行满载）。
- WSL Python venv：`~/src/rocm-smoke/.venv`（torch 2.9.1+rocm、triton 3.5.1、transformers 5.19、accelerate、bitsandbytes 0.50.2、flash-linear-attention 0.6.0、einops）。
- 统一入口：`wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh <script.py> [args]`（内部设置 HSA_ENABLE_DXG_DETECTION=1、PYTORCH_ROCM_ARCH=gfx1100、cwd=/mnt/c/Projects/Qcompute）。
- 只读区：`.data/`（发布 checkpoint 20.73GiB、base、bge-small、longmemeval）、`repro/`（上游源码，可读可 patch 但不要改）、`.research/`（上游克隆）。
- **GPU 纪律**：窗口 15:30–00:30；静默 00:30–09:30、12:30–15:30；同一时间只跑一个模型任务（并发两个 14GB 模型会把 20GB 卡打到 thrash）；跑前查 `gpu/run_py.sh gpu/mem.py`；不要用 `HSA_OVERRIDE_GFX_VERSION`，不要用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`（ROCm/RDNA3/WSL 已知崩溃/死锁）。

## 4. 文件地图

| 路径 | 内容 |
|---|---|
| `docs/reports/` | 阶段 HTML 报告（stage1–9） |
| `docs/notes/` | 证据笔记：repro-log、weights-map、stage3-*、stage4-*、stage5-*、stage6-*、stage7-*、stage8-*、memory-disk-v0、device-tree-estimate |
| `docs/research/` | 深度研究/专家报告（stage8-enhancement-portfolio、stage9-application-scenarios、stage9-technical-breakthroughs 等） |
| `analysis/weights/` | scan.py/compact.py 与结果（无损压缩证据） |
| `analysis/device/` | device_graphs.py、device_estimate.*、merge_probe.* |
| `analysis/memory_disk/` | memory_disk.py、safe_store.py、测试与日志 |
| `analysis/experiments/h1/`、`h1_inweights/`、`h5/`、`h5_e2e/` | 记忆/检索实验代码与原始 JSON/JSONL |
| `gpu/` | GPU 实验脚本：tiny_*、triton_*、quant/、trim/（trim_model、fp32_chip、selective_fla、int8_head、trim_chat、矩阵脚本）、kernels/ |
| `gpu/out/` | 原始 JSON/日志（提交时注意体积） |
| `demo/` | 端到端 put/get demo 与日志 |
| `site/` | GitHub Pages 站点（index + stages/ + assets/nav.js） |
| `.github/workflows/pages.yml` | Pages 自动部署 |
| `ROADMAP.md` / `STATUS.md` / `DELIVERABLES.md` / `docs/INDEX.md` | 规划/状态/交付/索引（先读 STATUS 与 INDEX） |

## 5. 关键事实与坑（务必遵守）

1. **`.data/` 与 `repro/` 不要改**；所有实验代码放自己的目录，monkeypatch 优先。
2. **命名键**：in-weights store 只取名字前 6 个 token + 长度；长名会静默覆盖。必须用 SafeStore（36-bit 头部 tag + salt probing）或显式 NameGuard。
3. **性能数字要同 session 同条件比较**：GPU 热/冷、CPU 争用会让 tok/s 差 2–3 倍；写报告时标注测量条件。
4. **GPU 并发禁令**：同一时间一个模型任务；两个 13–14GB 模型并发会导致显存 thrash/挂起。
5. **精确性分级**：CPU = bit-identical 参考；NVIDIA L4 = 作者 bit-exact 结论；AMD GPU = token-level（head_mismatch 0、lock_error None）为验收标准，不要宣称 dim-bit-exact。
6. **int8 head / NF4 / full-FLA 长序列** 都有负结果或未验证；默认不要开启。
7. **页面发布**：报告写好后 copy 到 `site/stages/stageN.html`，在 `site/assets/nav.js` 与 `site/index.html` 加链接，commit+push 自动发布（流程见 docs/notes/stage8-pages.md）。

## 6. 验收 gate（每次提交/发布前）

- CPU：`max_abs_logit_diff = 0.0`（trim/原模型对拍）；put/get transcript 与基线逐字一致。
- GPU：`head_mismatch = 0`、`lock_error = None`、put/get 答案正确；长序列回归用 `gpu/trim/fp32_chip_bench.py --cases 1,2,5`（bf16 应复现 3/12，fp32 应 0/12）。
- 回归用例集：Hi（9 tok）、put（69+3）、fresh get（54+2）、长序列 12 会话、H5 quick（50 题）。

## 7. Stage 9/10 路线（专家收敛后的优先级）

| 优先级 | 方向 | 验收 |
|---|---|---|
| P0 | H5 reader 集成 + 可审计记忆盘 demo | 4B reader 上 top-1/3/5 vs RAG 四组；拒答；审计链 |
| P0 | ADD8-only rv-tiny 端到端 merge | IR 全量 0 mismatch；HF 1000 条 0 错；RV32 不变；namespace/owner/exit/reset |
| P1 | fp32 局部混合精度 | ≥60 paired 长序列，lock 0/60；head=0；显存低于全 chip fp32 |
| P1 | compiler-exactness toolkit | IR/CPU/NVIDIA/AMD 四级精度统一报告 |
| P2 | >20 tok/s（int8 GEMV/tied embed/causal_conv1d） | 逐项 lock gate |
| P2 | memory-disk v1 | 指纹/journal/tombstone/容量错误码 |

原始候选清单如下（按 ROI 排序，可直接执行）：

1. **ADD8-only rv-tiny 端到端原型**（设备合并闭环）：真实 merge + tokenizer/namespace + 互斥 owner/exit dispatcher + RV 状态冻结；验收 = 设备计算 token-level 正确且 RV32 回退可用。
2. **H5 reader 集成**：4B reader 上跑 top-1/3/5 vs RAG 四组对照；低分拒答、索引版本/指纹门禁；优化 LongMemEval 批处理（预构建 corpus/incremental index）。
3. **fp32 chip 局部化**：只 fp32 敏感子层（chip attention/qkv/alpha/beta/exit），把 +0.65GB 与速度代价压到最小；paired lock/head 回归。
4. **>20 tok/s**：native INT8 GEMV、tied embedding 量化（按行 int8 + chip 维稀疏 bf16）、causal_conv1d Triton 短卷积；每项单独过 gate。
5. **memory-disk v1**：trim runtime + SafeStore + journal + 指纹，形成可用记忆盘 API（POST /v1/memory/query 形态见 stage8-h5-e2e.md）。
6. **可选研究**：编译设备扩展（SORT4/DADD 进 rv-tiny）、量化感知电路编译、多模型共享 chip 层。

## 8. 工作方式（本项目的默认纪律）

- 目标与规划：ROADMAP.md 只增不改历史（勾回完成项）；STATUS.md 增量写；每阶段评估"目标是否还成立"；阶段报告 ≤1 份/10 轮。
- 分工：简单任务 `flash-model`；多数事务 `middle-model`；关键难题 `expert-model`；用 subagent/subsession 并行，写清约束（只读区、输出路径、GPU 独占、硬截止）。
- 实验记录：脚本放 gpu/ 或 analysis/，原始 JSON/日志放 gpu/out/ 或 analysis/experiments/，报告写 docs/notes/；阶段 HTML 落 docs/reports/ 并发布。
- 不要在没有证据时宣称"bit-exact / 优于 host / 生产可用"；所有结论给出处（文件+数字+条件）。

## 9. 必读顺序

1. `STATUS.md` → 2. `docs/INDEX.md` → 3. 本文件 → 4. `docs/notes/stage8-fp32chip.md` + `stage8-speed.md` → 5. `docs/reports/qwenputer-stage{6,7,8}.html` 的结论段 → 6. `docs/research/stage8-enhancement-portfolio.md` → 7. 具体领域笔记（H1/H5/quant/device）。

## 10. 联系方式与产物

- 在线报告：https://wenbobobo.github.io/qwenputer-research/（每阶段一页，顶部可切换）
- 仓库：https://github.com/Wenbobobo/qwenputer-research（public；`.data/`、`.research/` 被 .gitignore 排除）
- 本 session 的关键决策与数字均可在 docs/notes 与 gpu/out 中复核；若发现冲突，以"最近日期 + 有原始 JSON 证据"的文档为准。

---

**接手后的第一件事**：读 STATUS 与 INDEX，确认 GPU 窗口，跑一遍 `gpu/trim/fp32_chip_bench.py --cases 1,2,5` 与 put/get 回归；然后从第 7 节选一个方向开工。
# Qcompute · 交付清单（2026-10-29）

本项目完成了对 [Qwenputer](https://github.com/divagr18/qwenputer) 的深度复现、测量、优化与研究。

## 1. 先看什么（10 分钟版）
1. `docs/reports/qwenputer-stage1.html` — **阶段一主报告**：原理（GDN-as-RAM / 编译器 / RISC-V / 桥 / 文件系统）、前史与失败原因、可行性、**本机复现证据**、成本、优化、场景。
2. `docs/reports/qwenputer-stage2-memory.html` — **阶段二报告**：in-weights 文件系统作为参数化记忆 vs RAG/KG/Engram 的定量边界、90vs30 打假、持续学习四层判定、H2/H3 实验。
3. `docs/reports/qwenputer-stage3.html` — **阶段三报告**：5 个创意 + 2 个原型（编译器审计 bit-exact、记忆盘实测）+ 规模边界。

## 2. 核心结论（TL;DR）
- Qwenputer 真实存在且可复现：发布 checkpoint 在 **纯 CPU** 上加载（~20s）并完成 list/put/get 端到端会话；跨会话空上下文正确召回 4471-AZ，0 mismatch。
- 发布权重 = **基座 Qwen3.5-4B + 编译器生成的 chip 侧 + 可推导变换 + 16KiB 拟合常量**：chip 层 **168/168 张量 bit-exact**（12/12 层）、chip embedding 行与文本行 chip 特征 bit-exact；Qwen 侧 426 张量 **0 个无法解释**（全部差异元素只在 mu 列 2732；相对基座总 delta ~0.27MB）；整模型不可推导残留仅 **16,384 B（6,144 非零，anti-sink）**，重建 ~80s CPU。
- 体积：20.73 GiB 磁盘中 **13.72 GiB 是精确零（61.8%）**；无损压缩实测 **22.20GB → 7.951 GiB**（10/10 分片 SHA-256 一致）。
- 记忆：它是**精确 KV 外设**，不是联想记忆；只在「≤512 文件、键名可约定、τ≈1、跨会话无外部依赖」时比 RAG 省 token；端到端比 host 工具慢（5–6s vs 2.3s）；90 vs 30 与 host 工具无显著差异（p≈0.71）。
- 记忆盘：快照很小（1 文件 11.5KB；512 文件逻辑 36.7KB），但**绑定 hash seed/布局**（换 seed 读回 0%），长名字（前 6 token 相同）会静默覆盖。
- 自我进化：事实写入状态、人工程序编译已实现；自动生成电路 1–2 年可做；元学习自举近期是幻想。

## 2b. Stage 4（2026-10-29 续作）新增结论
- **GPU 路径**：rv-tiny 在 AMD RX 7900 XT（ROCm）上 7/7 程序 token-exact，HF 路径 43–45 tok/s（CPU 的 ~40×）；上游 Triton 执行器 IR-exact，1,334 / 4,345 / 6,993 tok/s（batch 1/8/32）。
- **条件性 bit-exact**：同一 checkpoint CPU residual 0 差异；ROCm 上 ~12% cell 差（max 1.4e-3，第一个 FFN 的 gate/up GEMM 残差），但 token 仍全部正确——"exact"依赖后端。
- **设备树**：ADD8/SORT4/DADD 均 2 层、899/3,876/1,454 非零、IR 全量 exact；相对 RV32（11–12 层/1,040,595 非零）小 270–1150×；可并行塞进现有 chip 层。
- **memory-disk v0**：fingerprint/save/load/verify/journal/NameGuard，40/40 测试 PASS；快照 roundtrip、指纹拒绝、跨 seed 逻辑迁移、长名字覆盖拦截均验证。
- **H1（pooled seeds 0–2 + brevity）**：覆盖式主机工具 **27/30=90%**；append-only **7/30=23.3%**（p=2.4e-7）；时间戳上下文 strict 46.7%/lenient **100%**；普通上下文 strict 15.6%/lenient 53.3%；**brevity 变体：普通上下文 93.3%、时间戳 100%** → **90 vs 30 主要是回答格式/评分产物，介质无优势；覆盖语义仅在工具路径有效**。

## 2c. Stage 5（2026-10-29 晚）新增结论
- **发布模型 GPU 真跑**：accelerate device_map 拆分（layers 0–32 GPU / 33–43 CPU），put/get 正确；Lead 复核 put 73.6s、get 39.8s（CPU 2.8–4.3×），0 mismatch；dispatch load 152–264s。
- **H5**：合成 100 语义题 bge Recall@5 94–96%；**真实 LongMemEval 100 题** bge Recall@1/5 = 50%/87%；exact-key 语义 0%、已知名 100%；RAG 粘贴 12,140 token vs 精确 get 45 token → 两阶段混合（bge 路由 → 精确 get）。
- **SafeStore**：36-bit 头部标签 + salt probing，38/38 测试；512 对抗长名 0 碰撞（原始 511 覆写）；映射侧车持久化。

## 2d. Stage 6（2026-10-29 晚）新增结论
- **性能（最终）**：trim + FLA 全 GPU、无 offload；**put 4.10s（16.84 tok/s）、get 3.96s（13.62 tok/s）**，对比 Stage 5 的 73.6s / 39.8s（18× / 10×）；显存 13.15GB；head mismatch 0、lock None。
- **trim（+head-trim+FLA）**：Hi 0.73s（12.39 tok/s，14.7GB）；put 4.38s（15.75）、get 3.86s（14.0）。
- **无损性**：Qwen 裁剪只丢精确零（248 矩阵验证），CPU 对拍 max_abs_logit_diff 0.0、argmax 15/15；GPU 文本与基线逐字一致。
- **量化（负结果/保底）**：bnb8 需 PreserveColumns（mute 列 −15.06 修复）才正确；全 GPU 11.49GB、put/get 正确但仅 1.1–2.4 tok/s；NF4 质量崩不可用。
- **硬件事实**：7900 XT BF16 75.8–78.3 TFLOPS、INT8 70 TOPS（不更快）、**无原生 FP8**、INT4 有原生 WMMA（decode GEMV 1.4–1.5×）。
- **FLA**：flash-linear-attention 0.6.0 可用；GDN 不再走 reference（bnb8 put 1.55×、get 3.3×）；repro 设计预期 FLA。

## 2e. Stage 7（2026-10-29 深夜）新增结论
- **AMD 精确性边界**：gpucheck 在 7900 XT 上即使屏蔽 FLA（纯 torch）也 MISMATCH（−5.96e-8、0 wrong tokens），FLA 0.5.2 同样失败 → 作者 bit-exact 是 NVIDIA L4 结论；AMD 上 CPU bit-identical、GPU token-level。
- **selective FLA**：chip 层 0–11 reference + Qwen 12–43 FLA，put 12.44 / get 12.15 tok/s，正确。
- **干净复测**：trim no-FLA 13.48–14.71 tok/s（此前 6.79/9.0 受 CPU 争用影响）；full FLA put 16.84。
- **H1 in-weights（n=10）**：changed 10/10=100%、facts 17/20、all 27/30；与 host 覆盖式工具同档 → 覆盖语义是关键、介质无优势。
- **int8 head**：CPU 检查通过（675MB、circuit exact），GPU 负结果（11.21–11.96 tok/s、+0.7GB）；HIPBLASLt 10.58、TunableOp 4.58 均负收益。

## 2f. Stage 8 预研（2026-10-29 深夜）新增结论
- **fp32 chip 层修复长序列 lock drift**：chip 0–11 fp32 + Qwen bf16（TRIM_FP32_CHIP=1，参数 +0.65GB）；严格 paired 对照 bf16 **3/12** → fp32 **0/12** lock divergences（同 harness/同日/同 GPU），原失败点 calc 完整执行（4077/2841/2704 chip tokens）、head 全 0；速度 put 15.72/get 14.69/Hi warm 18.87 tok/s，峰值 13.94GB。
- 证据：gpu/trim/fp32_chip.py、fp32_chip_bench.py、gpu/out/c8_fp32/、c8_bf16_pair/、docs/notes/stage8-fp32chip.md。

## 3. 证据文件
| 类别 | 文件 |
|---|---|
| 复现日志 | `docs/notes/repro-log.md`、`repro/logs/*.log` |
| CPU 端到端演示 | `docs/notes/cpu-checkpoint-demo.md`、`demo/run-smoke.log`、`demo/run-a.log`、`demo/run-b.log`、`demo/store.npz` |
| 编译器审计（chip 侧） | `docs/notes/stage3-a-rebuild.md`、`analysis/rebuild/report.json`、负对照 `report_negcontrol_seed2036.json` |
| 整模型审计（Qwen 侧） | `analysis/rebuild/base_report.json`、`docs/notes/stage3-a2-base-audit.md` |
| 权重地图/压缩 | `docs/notes/weights-map.md`、`analysis/weights/scan_results.json`、`compact_results.json` |
| 记忆实验 | `docs/notes/h2h3-experiments.md`、`docs/notes/stage3-d-snapshot.md`、`analysis/experiments/*.json` |
| 研究 | `docs/research/prior-work.md`、`parametric-memory.md`、`expert-assessment.md` |
| 声明核对 | `docs/notes/claims-audit.md` |
| 记忆盘格式设计 | `docs/notes/memory-disk-format.md` |
| 创意草案 | `docs/notes/stage3-ideas.md` |

## 4. 未完成/受资源限制
- 官方 benchmark（83.3 / 90.0 / 97.5）与文本 KL：需 24GB bf16 GPU + 数小时；本机 T10 无 bf16，7900 XT 20GB 需压缩布局（09:30 后的 GPU 窗口）。
- H1/H4/H5 真实记忆实验：需要基座 + GPU。
- 设备树（创意 B）与监控器（创意 C）：仅设计/评审，未做原型。

## 5. 如何复现
```
# 环境
cd repro\qwenputer\experimental\riscv && uv sync --extra hf
# IR/测试
uv run pytest tests -q
# rv-tiny CPU（已构建：dist/rv-tiny）
uv run python -X faulthandler ..\..\..\run_tiny_cpu.py dist/rv-tiny rv32ui-simple memcpy call
# 发布 checkpoint CPU 会话（需 free RAM ≥25GB）
pwsh -File demo\run-demo.ps1 -Step smoke   # 然后 -Step a / -Step b
# 编译器审计
set PYTHONPATH=repro\qwenputer\experimental\riscv
repro\qwenputer\experimental\riscv\.venv\Scripts\python.exe analysis\rebuild\rebuild_chip.py --model .data\Qwen3.5-4B-Qwenputer-Filesystem --out analysis\rebuild\report.json
```
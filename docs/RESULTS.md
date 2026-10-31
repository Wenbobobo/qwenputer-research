# RESULTS —— Qwenputer 关键结果台账（唯一权威数字）

> 目的：所有报告/笔记只引用本表，不再各自维护数字。每条都带测量条件与证据路径；新实验完成后在此追加一行。
> 最后更新：2026-10-31 · 维护人：每阶段收尾的 agent（先改这里，再改报告）。

## A. 模型与审计（硬件无关）

| 指标 | 值 | 条件 | 证据 |
|---|---|---|---|
| 发布 checkpoint 大小 | 20.672 GiB（22.20 GB），594 张量，11.098B 参数 | — | docs/notes/weights-map.md |
| 非零权重 | 7.893 GiB；精确零 12.779 GiB（61.8%） | — | analysis/weights/scan_results.json |
| 无损压缩 | 20.672 GiB → **7.951 GiB**；10/10 SHA-256 一致 | 编码 196s / 解码 40s，CPU | docs/notes/weights-map.md |
| chip 侧审计 | **168/168 张量 bit-exact** | 重建 vs 发布权重 | docs/notes/stage3-a-rebuild.md |
| Qwen 侧审计 | 426 张量：338 base+widen bit-exact；80 mute-only；8 anti-sink（16,384 B） | — | docs/notes/stage3-a2-base-audit.md |
| 整模型可解释性 | 除 16,384 B anti-sink 外全部可推导 | — | 同上 |
| 结构 | 44 层 = 12 chip（GDN）+ 32 Qwen；hidden 4608 = 2560 + 2048；词表 484,800 | — | config/chipformer.json |

## B. CPU 端到端（参考路径，bit-identical 基准）

| 指标 | 值 | 条件 | 证据 |
|---|---|---|---|
| put/get | put(locker_code, 4471-AZ)→ok；get→4471-AZ，0 mismatch | 发布模型，CPU | demo/run-a.log、run-b.log |
| 单 token 延迟 | ~4.7 s（长 prompt 时 ~0.3 tok/s） | CPU 44 层 | docs/notes/repro-log.md |
| trim 对拍 | max_abs_logit_diff = **0.0**；argmax 15/15 + 66/66 | CPU，plain_hi + store_hi(803 tok) | docs/notes/stage6-trim-runtime.md §9 |

## C. GPU 运行时（RX 7900 XT 19.96GB，WSL ROCm 7.2.1）

| 配置 | put | get | Hi | 峰值显存 | 正确性 | 证据 |
|---|---|---|---|---|---|---|
| Stage 5 split（embed+layers0–32 GPU，33–43 CPU） | 73.6 s / 0.98 tok/s | 39.8 s | 9 tok / 7.7 s | 16.5GB + CPU | head 0 / lock None | gpu/out/put_gpu2.log、get_gpu2.log |
| bnb8 全 GPU（PreserveColumns） | 62.6 s / 1.10 | 74.2 s（list→get） | 2.38 | 11.49GB | 正确；需 bf16 旁路列 | docs/notes/stage6-quant-e2e.md |
| trim（no-FLA）干净单进程 | 10.16 s / 6.79* | 6.00 s / 9.00* | 4.47–6.96 | 13.15GB | 正确 | docs/notes/stage6-trim-runtime.md |
| trim（no-FLA）同 session 串行矩阵 | 7.69 | 9.94 | 8.51 | 13.2GB | head 0 / lock None | gpu/out/s8_fp32_*.json（Stage 8） |
| bf16 + selective FLA | 12.44 | 12.15 | 14.59（同 session cold） | 13.22GB | 正确 | docs/notes/stage7-selectfla.md、stage8-speed.md |
| bf16 + full FLA | 16.84 | 13.62 | 12.04–16.8 | 13.15GB | 正确；长序列未验证 | 同上 |
| **fp32 chip + no-FLA（推荐 correctness-first）** | 15.72 | 14.69 | 11.81（同 session） | 13.94GB | **长序列 0/12 lock divergence** | docs/notes/stage8-fp32chip.md |
| fp32 chip + selective FLA | — | — | 13.64（同 session） | 13.91GB | 长序列未验证 | gpu/out/s8_fp32_sel_hi.json |
| int8 head | 11.73 | 11.96 | 11.21 | 13.86GB | put/get 同基线；Hi 首 token 分叉 | docs/notes/stage7-int8-head.md（负结果） |
| NF4 | — | — | 9.26（乱码无效） | 8.64GB | 输出 "!!!!!!!!!!!!" | docs/notes/stage6-quant-e2e.md（负结果） |

\* Stage 6 的 6.79/9.00 是在同机 CPU 检查进程干扰下测的；干净复测见 Stage 8 行。**速度数字只在同 session 内可比。**

## D. 长序列正确性（H1 recall-1/2/5，12 个历史会话）

| chip dtype | lock divergences | 原失败点 chip tokens | head mismatch | 证据 |
|---|---|---|---|---|
| bf16 | **3/12**（recall-1/s1, recall-2/s1, recall-5/s2） | 25（提前分叉） | 0 | gpu/out/c8_bf16_pair/turns.jsonl |
| fp32 | **0/12** | 4077 / 2841 / 2704（calc 完整执行） | 0 | gpu/out/c8_fp32/turns.jsonl |

## E. 记忆实验（H1/H5）

| 条件 | changed strict | facts | all | 证据 |
|---|---|---|---|---|
| C3 普通上下文（verbose） | 14/90 = 15.6% | 100% | 72% | analysis/experiments/h1/h1_results.json |
| C1 时间戳上下文（verbose） | 42/90 = 46.7%（lenient 100%） | 99.4% | 81.9% | 同上 |
| C3s 普通上下文 + 简短回答 | 56/60 = 93.3% | 100% | — | 同上 |
| C1s 时间戳 + 简短回答 | 82/82 = 100% | 100% | — | 同上 |
| C2 append-only 主机工具 | 7/30 = 23.3% | 96.7% | 72.2% | 同上 |
| C4 覆盖式主机工具 | 27/30 = 90.0% | 71.7% | 77.8% | 同上 |
| **C5 in-weights（发布模型自己的 store，n=10）** | **10/10 = 100%**（95% CI 72.2–100） | 17/20 = 85% | 27/30 = 90% | docs/notes/stage7-h1-inweights.md |

H5 检索（CPU bge-small）：合成 Recall@1/3/5 = 58/84/96%（stage8 实测）；LongMemEval-100 Recall@1/5 = 50/87%；exact get 45 token vs RAG ~12,141 token。证据：docs/notes/stage5-h5.md、stage8-h5-e2e.md、analysis/experiments/h5*/。

## F. 设备树与合并

| 指标 | RV baseline | +ADD8 | +SORT4 | +DADD | 三者同时 |
|---|---|---|---|---|---|
| Layers | 12 | 12 | 12 | 12 | **12** |
| Dims | 3753 | 3790 | 3819 | 3767 | 3870 |
| Heads | 164 | 165 | 165 | 165 | 167 |
| FFN units | 47532 | 47614 | 47969 | 47711 | 48230 |
| Slots | 1806 | 1834 | 1859 | 1818 | **1899**（容量 2029） |
| Nonzeros | 1,289,104 | 1,289,793 | 1,292,857 | 1,290,288 | 1,294,730 |

IR check=True；SORT4/DADD CPU emit 通过；阻塞：token namespace/tokenizer（~3,193 embedding rows）、互斥 owner/exit、RV 状态冻结、共享 ONE/ballast。证据：analysis/device/merge_probe.*、docs/notes/stage8-device-merge.md。

## G. 硬件/量化事实（7900 XT）

- BF16 75.8–78.3 TFLOPS；FP16 77.8–78.0；FP32 19.3；INT8 70.4 TOPS（不比 BF16 快）；FP8 无原生 WMMA（Triton 降级 FP16）；INT4 有原生 WMMA，decode 专用 GEMV 1.40–1.51×。证据：docs/notes/stage6-quant-hw.md。
- 工具链：bnb 0.50.2 8bit/NF4 可用；torchao 与 torch 2.9.1 不兼容；ExLlamaV2/Marlin 不支持；FLA 0.6.0 可用（selective 1.2–1.6×，但 chip 非 bit-exact）。
- 禁止：expandable_segments（ROCm/RDNA3/WSL 崩溃）、HSA_OVERRIDE_GFX_VERSION。证据：docs/notes/stage6-quant-ecosystem.md、stage7-community.md。

## H. 发布

- 站点：https://wenbobobo.github.io/qwenputer-research/（hub + stage1–8 全部 200）
- 仓库：https://github.com/Wenbobobo/qwenputer-research（public）
- 更新流程：docs/notes/stage8-pages.md
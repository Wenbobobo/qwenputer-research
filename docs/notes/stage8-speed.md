# Stage 8：速度与长输入正确性（2026-10-30，北京时间）

## 结论

本轮保持默认配置不变；TRIM_SELECTIVE_FLA=1、TRIM_FP32_CHIP=1 都是进程级 opt-in。trim_chat.py 的组合顺序为：selective FLA 先安装路由；模型 CPU 加载、无损 trim、compact 后，fp32 chip 再包裹 0–11 层。TRIM_ALLOW_FLA=1 放开全量 FLA。TRIM_INT8_HEAD 未使用。

## 标准 prompt 矩阵

| 配置 | Hi tok/s（cold/warm） | put tok/s | fresh get tok/s | wall 示例(s) | 峰值显存 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| bf16 + no-FLA | 4.47 / 6.96 | 6.79 | 9.00 | 17.96 | 13.15 GB | clean |
| bf16 + selective FLA | 10.53 / 13.78 | 12.44 | 12.15 | 8.29 | 13.22 GB | clean |
| bf16 + full FLA | 3.33 / 5.77 | 16.84 | 13.62 | 31.10 | 13.05 GB | clean |
| fp32 + no-FLA | 5.75 / 12.35 | 15.72 | 14.69 | 10.47 | 13.94 GB | clean |
| fp32 + selective FLA | 7.30 / 4.79* | pending | pending | 8.80 | 13.91 GB | Hi clean; long run failed before ready |
| fp32 + full FLA | pending | pending | pending | — | — | not run |

所有已完成标准结果 head_mismatch=0、lock_error=None。fp32 selective 补测和 long smoke 均在 ready 前异常退出，未形成可用长锁证据。

## 长序列 smoke

| 配置 | lock divergences | head mismatch |
|---|---:|---:|
| bf16 + no-FLA | 3/12（recall-1/s1、recall-2/s1、recall-5/s2） | 0 |
| fp32 + no-FLA | 0/12 | 0 |
| fp32 + selective FLA | 未完成 | 未知 |

fp32 no-FLA 在三个原失败点完成 4077、2841、2704 chip tokens；bf16 在 token 228/249/232 分叉。

## fp32 gpucheck

gpucheck_fp32chip.py --decode --dtype fp32 使用 torch fallback 完成部分程序：signext、call、memcpy 均 0 wrong tokens；rv32ui-lb prefill 1 wrong token（decode 0）。程序返回非零，因此 gate 未全通过。causal_conv1d 与 flash-linear-attention 未安装，走 reference fallback。

## 推荐

- 正确性优先：fp32 + no-FLA；已验证长输入 0/12 lock divergences，标准 put/get clean；峰值约 13.94 GB。
- 速度优先：bf16 + selective FLA；标准 put/get 12.44/12.15 tok/s，Hi warm 13.78 tok/s；长输入未验证，不能作为 correctness-first 默认。
- full FLA put 最高约 16.84 tok/s，但未通过长输入门禁，不推荐默认。

## >20 tok/s 的剩余工作

1. native int8 GEMV；
2. tied embedding quantization；
3. causal_conv1d Triton/FLA short-conv kernel（单独风险评估后再启用）。

证据：gpu/out/trim_*.json、gpu/out/s8_fp32_selective_hi.json、gpu/out/c8_fp32/turns.jsonl、gpu/out/c8_bf16_pair/turns.jsonl、gpu/out/gpucheck_fp32chip.log。

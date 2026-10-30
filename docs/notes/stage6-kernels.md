# Stage 6 · GDN 内核加速（flash-linear-attention）测量记录

时间：2026-10-29 20:39–20:51 CST（等待 GPU 空闲后串行运行，三个均为独立新进程）。
脚本：[gpu/kernels/run_fla.sh](../../gpu/kernels/run_fla.sh)（等待空闲 → Hi → put → fresh get，00:25 硬截止保护）；状态日志 `gpu/out/fla_status.log`。

## 1. 安装内容
- WSL venv `~/src/rocm-smoke/.venv`：`flash-linear-attention 0.6.0` + `fla-core 0.6.0` + `einops 0.8.2`（安装时间 20:10:56）。
- `python -c "import fla"` 正常，版本 0.6.0；transformers 5.19.0。
- **未**安装 `causal_conv1d`（ROCm 下需源码编译，本阶段未动）。

## 2. transformers 是否切换到 FLA 内核
是。无 FLA 时（`gpu/out/quant_bnb8_hi.log`）有四行回退：

```
[transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. ...
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. ...
[transformers] `causal_conv1d_update` is falling back ...
[transformers] `fused_recurrent_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. ...
```

装 FLA 后（`gpu/out/fla_bnb8_{hi,put,get}.log`）只剩两行 `causal_conv1d_fn` / `causal_conv1d_update` 回退，**`chunk_gated_delta_rule` 与 `fused_recurrent_gated_delta_rule` 的回退行消失**，即 GDN prefill/decode 已走 FLA Triton 内核。gfx1100 上 Triton 编译/运行**无报错**。

## 3. 结果（bnb8 全 GPU，同一 quant_chat.py，仅差 FLA）

| 任务 | 无 FLA 文本 tok/s | FLA 文本 tok/s | 加速 | 无 FLA wall | FLA wall | head_mismatch | lock_error | 转写 |
|---|---|---|---|---|---|---|---|---|
| Hi (12) | 2.38（9 tok/3.79s） | **2.71**（9 tok/3.32s） | 1.14× | 7.8s | 10.2s* | 0 / 0 | None / None | 完全相同（token 级一致） |
| put (120) | 1.10（69 tok/62.6s） | **1.71**（69 tok/40.4s） | 1.55× | 70.5s | 50.2s | 0 / 0 | None / None | 完全相同（73 token 一致，69 text+3 chip） |
| get (120, 新进程) | 1.08（80 tok/74.2s） | **2.44**（54 tok/22.1s） | 2.26× | 86.5s | 26.2s | 0 / 0 | None / None | 见下 |

\* Hi 的 wall 含首次 Triton JIT 编译/autotune；后续进程命中 Triton 缓存后该开销下降（put/get 已体现）。峰值显存 12.34 GB → 12.24 GB。

转写细节（JSON：`gpu/out/fla_bnb8_*.json`）：
- put：`put(locker_code, 4471-AZ)` → `ok` → “I've saved your locker code as 4471-AZ.” —— 与无 FLA 版逐 token 相同，task_sanity=true。
- get（FLA）：直接 `get(locker_code)` → `4471-AZ` → “Your locker code is 4471-AZ.”，**54 text + 2 chip**，与 Stage-5 bf16 GPU 基线 `gpu/out/get_gpu2.log`（54 text + 2 chip，head mismatches 0）**逐字一致**。
- get（无 FLA bnb8）：先 `list()` → `locker_code`，再 `get` → `4471-AZ`（80 text + 3 chip）。两者分歧点是第 5 个 token（函数名 `list` vs `get`），属于 Qwen 侧（非芯片层）bf16 数值微差导致的 argmax 翻转，两者答案都正确、checks_pass=true、task_sanity=true。FLA 版反而与 Stage-5 参考转写一致。get 的 2.26× 有一部分来自少生成 26 个 token 与少一次芯片调用；可比的纯速度以 put（1.55×，同一转写）为准。

## 4. 芯片精确性风险评估
- 12 层芯片层是 GDN，现在同样走 FLA `chunk_gated_delta_rule`（prefill）/`fused_recurrent_gated_delta_rule`（decode）。
- 三次运行 head_mismatch=0、lock_error=None、nonfinite_logits_calls=0，put 的芯片调用（3 chip token）与 get 的芯片读取（2 chip token）均正确 → **本测例中 FLA 未破坏电路精确性**。
- 与 repro 设计一致：`repro/qwenputer/experimental/riscv/chipformer/rv/gpucheck.py` 本身就是“stock HF 模块 + fla 内核”的 RV 核精确性检查；`chipformer/rv/core.py:492` 注释说明商数头只用 one-hot 读取，正是为了规避 “fla's chunk kernel 的 fp32 残差”。即电路是按 FLA 数值设计的。
- 残余风险：仅覆盖 locker put/get 一个场景；FLA Triton autotune 可能对不同序列长度选不同 kernel 配置（数值可能略变）；未跑 gpucheck 全程序集。store npz 字节与无 FLA 版不同（12152 vs 12364 B），但与 Stage-5 `store_gpu2.npz`（12154 B）尺寸接近，且 fresh get 读回正确。

## 5. 附带发现（影响其它阶段）
- FLA 装上（20:10:56）后，quant 矩阵中后续的新进程（bnb8 hf_hi、bnb4 hi/put…）以及 trim bench 在该时刻后启动的运行**都已静默使用 FLA**，不再是无 FLA 基线；判断方法：日志里有无 `chunk_gated_delta_rule ... falling back` 行。
- quant_bnb4_hi（FLA 下）文本为 `!!!!!!!!!!!!`（乱码），head_mismatch 0；是 NF4 本身问题还是与 FLA 交互尚未区分（bnb8+FLA 正常，倾向 NF4 问题）。
- quant_bnb8_hf_hi 失败与 FLA 无关：`--hf` 路径 `torch.tensor([ids])` 收到 `tokenizers.Encoding`（apply_chat_template 返回类型变化）。

## 6. 建议
**保留 FLA（推荐，条件性）**：decode 1.55×（put 同转写）、get 端到端 wall 86s→26s，芯片精确性在 E2E 上保持（head_mismatch 0）。条件：
1. 发布/验收前在 FLA 环境下跑一次 `python -m chipformer.rv.gpucheck --decode`（bf16）全程序集，作为芯片层逐位精确的回归门；
2. 记录“基线是否含 FLA”，所有速度对比必须同条件（日志回退行即标记）；
3. 若需严格复现旧 no-FLA 数值，可用独立 venv，而不是卸载。

## 7. 下一步
- 安装/编译 `causal-conv1d`（ROCm/HIP 需源码构建，或用 FLA 自带的 `fla.modules.convolution` Triton 短卷积替代）——剩余两条回退都在 conv1d 上，decode 每层每 token 都调用 `causal_conv1d_update`，预期还有可观收益；同样需 head_mismatch 回归验证。
- 预热 Triton 缓存（首次 JIT 约数秒），或固定 autotune 配置以消除数值随 shape 变化的风险。
- 在 FLA 下重跑 Stage-5 split / trim(S6) 基线以得到同条件对比；区分 bnb4 乱码根因。
## Lead 复核补充（21:40）

用 `python -m chipformer.rv.gpucheck --decode` 在 FLA 0.6.0 下做了 S7 建议的回归门禁：5 个测试（signext / call / memcpy / rv32ui-lb / rv32ui-sh）的 prefill+decode **全部 MISMATCH**，差异 6e-8 ~ 9.4e-7（fp32 1 ULP 级，参考值为 IR 精确值）。
结论修正：**FLA 下 chip 层并非 bit-exact**；put/get 的正确性来自电路对 1e-7 级扰动的容错，而不是精确性保持。
推荐：保留 FLA 作为可选速度模式（+1.55–3.3×），默认精确路径用 trim（no FLA）；任何依赖逐位精确的程序都应关闭 FLA 或先跑 gpucheck 门禁。证据：gpu/out/gpucheck_fla.log。
## Lead 二次修正（22:15）
后续对照证明：**屏蔽 FLA 的纯 torch fallback 在 7900 XT 上也同样 gpucheck 失败**（残留 −5.96e-8、0 wrong tokens），FLA 0.5.2 亦然。
因此结论应从"FLA 破坏 bit-exact"修正为：**AMD gfx1100/ROCm 的 bf16 GEMM 本身非 bit-exact**（作者 bit-exact 是 NVIDIA L4 结论）；FLA 只是同一量级的数值差异。
AMD 上的正确表述：CPU 路径 bit-identical；GPU 路径 token-level 正确（head_mismatch 0、lock None）。FLA 仍是有效速度手段。证据：gpu/out/nofla_probe.log、gpu/out/fla052_probe.log。
# Stage 4：GPU 路径（ROCm / RX 7900 XT）

> 负责人：GPU-path 子 agent（A）；日期 2026-10-29 10:19–10:40（北京时间）。脚本与日志均在 `gpu/`。上游 `repro/` 源码未改动。

## 1. 环境
- WSL Ubuntu-24.04，ROCm 7.2.1，AMD Radeon RX 7900 XT（gfx1100，19.96 GiB），bf16 正常。
- venv `~/src/rocm-smoke/.venv`：torch 2.9.1+rocm7.2.1，triton 3.5.1+rocm7.2.1，transformers 5.19.0，safetensors 0.8.0，numpy 1.26.4。
- **没有** flash-linear-attention（fla）和 causal-conv1d，GDN 和 conv 都走 transformers 的 torch 参考实现（日志里有 fallback 警告）。**没有** accelerate（不能用 `device_map='auto'`）。
- 入口：`wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh <script> [args]`。

## 2. rv-tiny 在 GPU 上的自由运行（`gpu/tiny_gpu_free.py`）
从保存好的 checkpoint `dist/rv-tiny` 加载（bf16，2130M 参数，显存 3.99 GiB，峰值 4.22 GiB），调用上游 `firmware.run`（注入快照后做普通 greedy `generate`），再与 `ref.Machine.from_program(...).run(10**6)` 的参考 trace 比较。

| 程序 | 生成/期望 token | 逐 token 一致 | 耗时 | tok/s（GPU） | CPU 基线 tok/s |
|---|---|---|---|---|---|
| rv32ui-simple | 3 / 3 | ✅ | 1.9 s | 1.5（启动开销为主） | 0.80 |
| signext | 44 / 44 | ✅ | 3.2 s | 13.6 | — |
| memcpy | 159 / 159 | ✅ | 6.0 s | 26.6 | 1.43 |
| call | 474 / 474 | ✅ | 10.4 s | 45.4 | 0.72 |
| rv32ui-add | 501 / 501 | ✅ | 11.1 s | 45.3 | — |
| c-strops | 1073 / 1073 | ✅ | 24.5 s | 43.7 | — |
| c-crc32 | 4165 / 4165 | ✅ | 95.9 s | 43.4 | （CPU 约需 1 h，未跑） |

日志：`gpu/tiny_gpu_free.log`、`gpu/tiny_gpu_free_ext.log`；JSON 在 `gpu/out/`。

**结论：7 个程序在 ROCm bf16 上自由运行，全部与参考 trace 逐 token 一致**，包括 4,165 token 的 crc32。

## 3. GPU 与 CPU 速度对比
- 稳态（几百 token 以上）约 **43–45 tok/s**，CPU 约 0.7–1.4 tok/s，**快约 30–60 倍**（call：45.4 vs 0.72，约 63 倍；memcpy：26.6 vs 1.43，约 19 倍，短程序被固定开销稀释）。
- 上游 NVIDIA L4 的数字：HF 路径约 250 tok/s（fla kernel），compile+StaticCache 后 112 tok/s，compact 后 609 tok/s。我们用的是 torch fallback，没有 fla，也没有 compile，所以慢一些，属于预期。
- **Triton 执行器**（上游 `tritonexec.Executor2`，HIP graph，`gpu/tritonexec_rocm.py`）：

| 路径 | ms/step | tok/s |
|---|---|---|
| 执行器 batch 1 | 0.749 | 1,334 |
| 执行器 batch 8 | 1.841 | 4,345 |
| 执行器 batch 32 | 4.576 | 6,993 |

  同时 `tritonexec.check` 在 signext / memcpy / call / rv32ui-add / c-crc32（前 1500 位置）上**每维每位置都与 IR 完全相等**。作为对照，L4 上 batch 1 是 2,706 tok/s，batch 32 是 37,743 tok/s。

## 4. 残差 == IR 严格检查（tinycheck）：ROCm 上不是 bit-exact
- `python -m chipformer.rv.tinycheck --device cuda --progs signext,memcpy,call`：模型构建 71 s，随后 signext 的残差检查报 AssertionError（第一个不一致在 pos 0 GO，`j>*` 维，hf −5.96e-8 vs ir 0）。日志：`gpu/tinycheck_cuda.log`。
- 诊断（`gpu/tiny_gpu_residual.py`）：
  - ROCm 上，signext 的 118,800 个 cell 中有 14,192 个不一致（663 维），memcpy 的 422,400 个中有 56,272 个，max|d| 约 1.2–1.4e-3。prefill 和 decode 的结果完全相同。关闭 bf16 reduced-precision reduction 没有变化，`TORCH_BLAS_PREFER_HIPBLASLT=0`（rocBLAS）也没有变化。
  - **同一 checkpoint 在 CPU 上：0 个 cell 不一致**（与 repro-log 一致）。
  - 第一个出错的层级是 level 2（第一个 FFN）。
- 根因定位（`gpu/tiny_gpu_mlp_diff.py`，layer 0 MLP，输入相同）：SiLU、act*up、down_proj GEMM 在 GPU 和 CPU 上 **bit-identical**。差异来自 **gate_proj / up_proj 的 bf16 GEMM**：电路依赖大数相消得到精确 0，ROCm 的 GEMM 累加顺序会留下微小残差，例如 up 输出 9.3e-10，而 CPU 是精确 0；GPU 上非零元素 13,770，CPU 上 10,182。这些残差逐层传播，最终表现为 1e-7～1e-3 的偏差。
- **影响**：电路的 AND/阈值裕量吸收了这些误差，所以 **token 层面完全正确**（第 2 节，7/7 exact）。只是"残差逐位等于 IR"这一更强的命题在 ROCm 上不成立。上游在 L4 + fla 上验证过 bf16 bit-exact，那是 cuBLAS 的累加行为。
- 可能的修复方向（未做）：用 fp32 或确定性的累加顺序做 GEMM，再舍入到 bf16；或者直接走 Triton 执行器（已经 IR-exact）。

## 5. 完整发布模型（Qwen3.5-4B-Qwenputer-Filesystem）可行性
- 实测权重：**22.20 GB = 20.67 GiB**（bf16，11.098B 参数，10 个分片，44 层，hidden 4608，vocab 484,800，词嵌入与 lm_head 共享）。
- 显卡总显存 19.96 GiB（< 20.67 GiB），**整模型 dense 放不进显存**，还要再加 KV/状态缓存和激活。
- 把共享的 embedding/lm_head 拆到 CPU：484,800 × 4,608 × 2 B = 4.47 GB。剩下的层 + norm 为 **16.51 GiB**。在**空闲**的 20 GiB 卡上理论可行，余量约 3 GiB 给缓存和激活，但每个 token 都要在 CPU 上做一次 4.47 GB 的 lm_head GEMV，会成为瓶颈。
- 实际尝试（`gpu/full_model_split.py`，手工切分，因为没有 accelerate）：开始时 GPU 只剩 **8.34 GiB 空闲**，因为 H1 实验已经在用 GPU（Lead 已写 `gpu/A_DONE` 把 GPU 交给 H1）。模型加载到 CPU 成功（1 s，mmap），之后进程没有再输出就结束了，日志止于 "loaded to CPU"，推测是在把层搬到 GPU 时因显存争用而失败。为了不干扰 H1，**没有重试**。
- 后续方案：(a) 在 GPU 空闲时重跑 `gpu/full_model_split.py`（预期显存约 16.5 GiB + 缓存）；(b) 安装 accelerate，用 `device_map='auto', max_memory={0:'18GiB','cpu':'30GiB'}` 自动 offload 约 3 GiB 到 CPU；(c) 只取芯片层 / rv-tiny 路径；(d) 量化（int8 约 11 GB）。不过量化会破坏电路的精确算术，**不适用于芯片部分**。

## 6. Triton on ROCm
- `gpu/triton_sanity.py`：后端 `hip / gfx1100 / warp 32`。vector add 结果精确（首次调用含编译 0.92 s；16M fp32 约 0.36 ms，约 559 GB/s）。bf16 行求和与 torch 对比，误差 7.6e-6，正常。
- 上游 Triton 执行器（`tritonexec`）在 ROCm 上可以原样运行，HIP graph 捕获正常，且与 IR 完全一致（第 3 节）。**今后执行器路径可以直接在本机使用。**

## 7. 注意事项
- 没有 fla / causal-conv1d：HF 路径速度只是下限。装上 fla 的 ROCm 版本（Triton 实现）应该能明显提速，但需要重新验证 exactness。
- ROCm 的 bf16 GEMM 不保证"残差 == IR"，只保证 token 正确。如果下游要做残差级比较（例如 H1 的探针），应该用 CPU 或 Triton 执行器作为参考。
- 时间测量包含 HF generate 的 Python 开销。rv32ui-simple 太短，它的 tok/s 没有参考意义。
- 完整模型这次没有完成实测，只有字节算术和 CPU 加载成功两条结论。

## 8. 文件清单
`gpu/tiny_gpu_free.py`（任务 1），`gpu/tiny_gpu_residual.py`、`gpu/tiny_gpu_mlp_diff.py`（残差诊断），`gpu/tritonexec_rocm.py`（执行器），`gpu/triton_sanity.py`，`gpu/full_model_split.py`，以及对应的 `*.log` 和 `gpu/out/*.json`。

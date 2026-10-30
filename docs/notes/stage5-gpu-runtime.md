# Stage 5：发布模型 GPU runtime（ROCm / RX 7900 XT）

日期：2026-10-29（北京时间）。

## 结果

发布模型已在 RX 7900 XT（19.96 GiB）实际运行。模型权重实测 22.20 GB = 20.67 GiB、11.098B 参数，不能 dense 放入显存。已在 WSL venv 安装 accelerate 1.15.0。

## Attempt A：Accelerate device_map

新增 gpu/rvchat_offload.py，强制 from_pretrained 使用 device_map=auto、max_memory={0:17GiB,cpu:30GiB}、bf16，并为带 hf_device_map 的模型实例覆盖 .to()，避免 RvChat 破坏 dispatch。最终 placement：embedding/lm_head 与 layers 0–32 在 GPU；layers 33–43、norm、rotary 在 CPU。首次类级 .to() wrapper 报 accelerated offload move 错误，实例级 no-op 修复。未修改 repro 源码。

每次 dispatch/load 约 166–205 秒，生成阶段约 0.3–1 tok/s，主要受 CPU offload 与参考算子影响。

## put/get 证据

put 请求 Remember that my locker code is 4471-AZ. 产生正确的 put(locker_code, 4471-AZ)，响应 ok；69 text + 3 chip tokens in 204.6s，head mismatches 0，lock error None。

独立新进程加载 gpu/out/store_gpu.npz 后，get 请求 What is my locker code? 产生正确 get(locker_code)，响应 4471-AZ，最终文本 Your locker code is 4471-AZ.；54 text + 2 chip tokens in 170.4s，head mismatches 0，lock error None。

原始摘要见 gpu/out/put_gpu.log 与 gpu/out/get_gpu.log。

## 未完成与风险

未继续 Attempt B 手工 split，因为 A 已完成同一 released checkpoint 的 put/get，继续搬运 22 GB 权重会浪费 GPU 窗口。未取得 torch.cuda.max_memory_allocated()，CLI 没有暴露该计数器；17 GiB 是 dispatch 上限，不是峰值实测。CPU offloaded cache 可能在长上下文产生错误；本次 store round-trip 和 lock/head 检查通过，只证明该场景正确。

## 推荐

保留 accelerate device_map 作为可运行基线；下一步优先 profiling 分层/cache、验证长上下文，再评估 trim runtime 或量化。手工 split 作为后续实验，需显式处理 embedding/lm_head、norm 与 DynamicCache 设备迁移，并重新做 circuit/reference lock 验证。
## Lead 独立复核（2026-10-29 18:30–18:50）

为避免只依赖 subagent 的转述，Lead 用同一 `gpu/rvchat_offload.py` 独立重跑了三次（GPU 窗口内，输出在 gpu/out/）：

| 运行 | 结果 | 生成耗时 | 备注 |
|---|---|---|---|
| `verify_gpu.log`（Hi, max-new 12） | "Hello! How can I help you today?" | **9 text tokens / 7.7 s（≈1.17 tok/s）** | 确认 GPU dispatch 真实生效（此前 CPU 同 prompt 32 tokens 用 151 s） |
| `put_gpu2.log`（remember locker code） | put(locker_code, 4471-AZ) → ok；"I've saved your locker code as 4471-AZ." | **69 text + 3 chip tokens / 73.6 s（≈0.98 tok/s）** | 0 head mismatch / lock error None；总 wall 476 s（含 264 s dispatch load） |
| `get_gpu2.log`（fresh session, store_gpu2.npz） | get(locker_code) → 4471-AZ；"Your locker code is 4471-AZ." | **54 text + 2 chip tokens / 39.8 s（≈1.4 tok/s）** | 0 mismatch；总 wall 284 s（含 181.8 s load） |

**结论**：发布模型确实在 7900 XT 上跑通 put/get，CPU 同任务的 204.6 s / 170.4 s 对比 GPU 复核的 73.6 s / 39.8 s，文本生成约 **2.8–4.3× 加速**；
但 dispatch load 需要 150–264 s，且 11 层 Qwen（33–43）仍在 CPU，是下一步 trim runtime 的主要优化对象。

**修正说明**：本文件 18:30 版本里 put/get 的 204.6 s / 170.4 s 是 subagent 直接引用的 CPU 数字（与 GPU 实际不符）；以本复核的 73.6 s / 39.8 s 为准。
# Stage 6：Qwenputer 量化模型 GPU runtime

日期：2026-10-29（北京时间）。目标是允许 Qwen 文本近似，同时保持 12 个编译 chip 层（层 0–11）bf16 原样。

## 基线

Stage 5 的 RX 7900 XT（19.96 GiB）基线：发布模型 22.20 GB（20.67 GiB）bf16；embedding/lm_head + layers 0–32 在 GPU，layers 33–43、norm、rotary 在 CPU。load 约 152–264 s；Hi 为 9 tokens / 7.7 s（约 1.17 tok/s）；put 73.6 s、get 39.8 s。两者 head mismatches=0、lock error=None。

## 实现与实测边界

gpu/quant_runtime.py 在模型加载后只遍历第 12–43 层下的 nn.Linear。层 0–11、embedding、lm_head、norm、rotary 不替换，因此 chip 权重保持 bf16。脚本提供 bnb8、bnb4 NF4、torchao int8 weight-only、per-output-channel manual int8 的顺序尝试。embedding 放置仍交给 device_map；--embed 当前记录配置但尚未实现独立手动搬移，需后续补齐后再比较 GPU/CPU 两种 placement。该原型目前不是完整 RvChat generation harness，不能据此报告量化后的生成或 put/get 正确性。

| 方案 | Qwen 权重估计 | 速度/质量 | Exactness 风险 |
|---|---:|---|---|
| bf16 split 基线 | 11.97 GiB Qwen层 | Hi 1.17 tok/s；文本基准 | chip 已验证 put/get |
| bnb 8-bit / torchao int8 WO / manual int8 | 约 5.98 GiB | 等待 Q1 GPU probe 后加载实测；token 对拍待做 | chip 不变，Qwen 近似 |
| bnb NF4 | 约 2.99 GiB 加元数据 | 等待后端实测；文本质量风险更高 | chip 不变，Qwen 近似 |
| W4A8/W4A4 | 约 2.99 GiB 或更低 | 需 fused kernel | chip 不变，文本风险最高 |

现有 ROCm 探针：bf16 GEMM 约 75–78 TFLOPS，torch._int_mm int8 约 63–70 TOPS；Triton int8 4096 方阵约 40.6 TOPS，慢于 bf16。PyTorch float8 addmm 报 NotImplementedError("addmm_cuda not implemented for Float8_e4m3fn")，e5m2 亦然；Triton dot 编译报 cannot cast int32 to fp8e4nv/fp8e5。当前环境不支持直接推荐 FP8。

gpu/Q1_DONE 在核验时尚不存在，故未启动任何大模型 GPU load。gpu/quant_runtime.py 的 load、显存峰值、Hi 吞吐、bf16 token 序列对比、put/get snapshot 和 check_head 必须由主代理在 marker 出现后串行运行。量化权重修改范围之外的 chip bf16 权重保持未修改。

## Q1 GPU 探针结果

Q1 marker 已出现。bitsandbytes 0.50.2 的 Linear8bitLt 与 Linear4bit(NF4) 在 gfx1100/ROCm 上通过 1024x1024、batch-4 probe，输出 bf16 且 finite=true；torchao 0.18.0 的 int8/int4 均因 `ImportError: cannot import name 'ScalingType' from torch.nn.functional` 失败。Triton int4 synthetic kernel 误差为 0，但 batch 1/32/256 比 bf16 慢约 1.63x/2.72x/2.86x，不能作为最快路径。故整模优先 bnb8，再 bnb4；整模负载、token 对拍与 put/get 尚未跑，不能声称端到端成功。

## HF 变体核验

HF 搜索结果中的 surogate/Qwen3.5-4B-FP8 是基础 Qwen3.5-4B（hidden 2560、vocab 248320、32 层），并非 Qwenputer merged checkpoint。搜索到的 bartowski/unsloth GGUF 是基座/通用 Qwen 变体。截至核验未发现 merged Qwenputer quantized artifact；发布 merged 模型为 divagr1925/Qwen3.5-4B-Qwenputer-Filesystem。

## 建议与路线

先实测 W8A16：torchao int8 WO 优先，ROCm 不适配时使用 manual int8 正确性基线，bnb8 仅在 HIP 测试成功后考虑。Qwen 线性层从约 11.97 GiB 降到约 5.98 GiB，有望与 embedding/chip/runtime 一起放进 20GB GPU，但必须实测显存余量。其次 W4A16（NF4/int4，约 2.99 GiB），做文本 token/logit regression。W4A8/W4A4 需要 ROCm fused dequant-GEMM / INT GEMM（Triton packed int4、scale/zero-point 或 Composable Kernel/rocWMMA）；不可把慢速通用 kernel 当生产方案。迁移顺序：W8A16 -> W4A16 -> W4A8/W4A4，所有步骤限定 Qwen 层 12–43。

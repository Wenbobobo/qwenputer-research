# Stage 7: AMD RX 7900 XT (gfx1100 / RDNA3 / 20GB) 社区最佳实践调研报告与 Qwenputer 加速建议

- **调研时间跨度**：2025-09 ～ 2026-10（兼顾本机知识库 2026-07/08 审计与最新社区前沿）
- **目标平台**：AMD Radeon RX 7900 XT (gfx1100, RDNA3, 20GB VRAM), Windows 11 + WSL2 (ROCDXG 1.2.0), ROCm 7.2.1, PyTorch 2.9.1, Triton 3.5.1
- **目标模型对象**：Qwenputer 合并模型（44层混合架构：12层编译型 RISC-V GDN 电路芯片层 + 32层基座 Qwen3.5-4B 扩展层，词表 484,800，全精度 BF16 Dense 约 22.2GB；当前运行于 RvChat 驱动 + Transformers + Trim 运行时 + 可选 FLA）
- **交付目标**：总结本地知识库已确立基线、对比社区最新主流推理与编译优化路线、提出针对 Qwenputer 定制架构的落地推荐与实验优先级。

---

## 一、 本地知识库基线总结与引用（Local KB Summary）

基于本地核心知识库文件（`C:\Projects\ROCM`）的确立结论：

### 1. 软硬件稳定元组与环境基线
- **稳定生产栈**：Windows 11 + WSL 2.7.x + Ubuntu 24.04 + Adrenalin 26.2.2 + ROCDXG 1.2.0 + ROCm 7.2.1 + Python 3.12 + PyTorch 2.9.1 + Triton 3.5.1（引用自 [ROCM_RX7900XT_RESEARCH.md](C:/Projects/ROCM/docs/ROCM_RX7900XT_RESEARCH.md#L15-L23)、[INSTALLED_STATE.md](C:/Projects/ROCM/docs/INSTALLED_STATE.md#L7-L26)）。
- **ROCm 7.14 / 10.0 功能轨评估**：ROCm 7.14（2026-07-15 发布）引入 TheRock 模块化体系，但因其 WSL 工具链及上游 vLLM 兼容矩阵在 2026-07/08 尚存缺口，本机仍以 7.2.1 为稳定基线（[README.md](C:/Projects/ROCM/README.md#L11)、[ROCM_RX7900XT_RESEARCH.md](C:/Projects/ROCM/docs/ROCM_RX7900XT_RESEARCH.md#L227-L230)）。
- **环境隔离原则**：禁止在同一 venv 中混装 ROCm 与 CUDA wheel。双卡（RX 7900 XT + Tesla T10）必须分别通过 `ROCR_VISIBLE_DEVICES` 与 `CUDA_VISIBLE_DEVICES` 单卡隔离，显存无法跨架构自动合并（[ROCM_RX7900XT_RESEARCH.md](C:/Projects/ROCM/docs/ROCM_RX7900XT_RESEARCH.md#L100-L118)）。

### 2. 硬件边界与 20GB 显存红线
- **计算参数**：RX 7900 XT 为 84 CU、Wave32 架构，官方标称 FP32 51.6 TFLOPS、FP16/BF16 103 TFLOPS、INT8 103 TOPS、INT4 206 TOPS（[ROCM_RX7900XT_RESEARCH.md](C:/Projects/ROCM/docs/ROCM_RX7900XT_RESEARCH.md#L28-L37)）。
- **显存硬边界**：FP16/BF16 极限舒适区为 7B/8B 模型。对于 Qwenputer（参数量 11.1B，44 层 Dense BF16 占 22.2GB），**无法全尺寸直接塞入 20GB VRAM**。在 Stage 5 走 `accelerate` 卸载时导致严重的 CPU offload 延迟（put 73.6s，get 39.8s）；必须通过无损 Trim 架构（13.15GB）或混合量化才能实现全 GPU 常驻（[ROCM_RX7900XT_RESEARCH.md](C:/Projects/ROCM/docs/ROCM_RX7900XT_RESEARCH.md#L38-L51)、[Qcompute STATUS.md](C:/Projects/Qcompute/STATUS.md#L16-L19)）。

### 3. 原生 INT4 硬件事实与软件库缺口
- **硬件存在，软件断层**：gfx1100 具备硬件指令 `V_WMMA_I32_16X16X16_IU4` 与 HIP builtin，但公共软件栈（rocWMMA、hipBLASLt、Triton 3.5.1、Composable Kernel）均未提供开箱即用（turnkey）的整数 INT4 Dense GEMM 库。Triton 3.5.1 明确缺少 `tl.int4` 与 `tl.uint4` 支持（引用自 [RDNA3_INT4_GFX1100_RESEARCH_2026-08.md](C:/Projects/ROCM/docs/RDNA3_INT4_GFX1100_RESEARCH_2026-08.md#L10-L24,L70-L89)）。
- **禁止静默降级**：严禁把假量化（fake-QAT）或解包 INT8 当作原生 INT4，必须通过 HSACO 反汇编与 bit-exact oracle 验证（[RDNA3_INT4_GFX1100_RESEARCH_2026-08.md](C:/Projects/ROCM/docs/RDNA3_INT4_GFX1100_RESEARCH_2026-08.md#L115-L130)）。

### 4. WSL2 / ROCDXG 特有缺陷与排障教训
- **空闲 CPU 忙轮询**：ROCDXG #60 导致进程在初始化 GPU context 后，空闲时仍持续霸占 ~2.1 个 CPU 核忙轮询。服务级或常驻进程需妥善管理生命周期（[INSTALLED_STATE.md](C:/Projects/ROCM/docs/INSTALLED_STATE.md#L73-L85)、[TROUBLESHOOTING.md](C:/Projects/ROCM/docs/TROUBLESHOOTING.md#L193-L208)）。
- **调试器与 Profiler 受限**：WSL 环境下 `rocm-smi` 会报 amdgpu 未初始化错误（属预期现象，因其基于 `/dev/dxg` 而非 Linux 原生驱动），且缺少完整的硬件性能计数器（PMC）与 roofline 工具，必须以端到端 wall-clock 与事件计时为准（[TROUBLESHOOTING.md](C:/Projects/ROCM/docs/TROUBLESHOOTING.md#L209-L216)）。
- **环境变量红线**：
  - **严禁全局设置 `HSA_OVERRIDE_GFX_VERSION`**：gfx1100 属官方正式支持目标，滥用 override 会掩盖二进制构建缺失问题（[TROUBLESHOOTING.md](C:/Projects/ROCM/docs/TROUBLESHOOTING.md#L187-L192)）。
  - **严禁开启 `expandable_segments`**：在 WSL2 + ROCm 7.2.x 上，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 会导致严重的虚拟内存映射死锁或 `INTERNAL ASSERT FAILED (!handles_.at(i))` 崩溃（[TROUBLESHOOTING.md](C:/Projects/ROCM/docs/TROUBLESHOOTING.md)；后文详述社区证据）。

---

## 二、 社区主流方案调研与共识矩阵（2025-09 ～ 2026-10）

针对 RX 7900 XT / RDNA3 (gfx1100) 的社区最佳实践，按推理引擎、算子优化、量化生态与运行时调优四个维度进行全面审计与对比：

### 1. 推理引擎生态（Inference Engines on gfx1100）

| 引擎 / 框架 | 社区现状与 gfx1100 表现 | 优势与瓶颈 | 对 Qwenputer 的适用性裁定 | 证据来源与日期 |
|---|---|---|---|---|
| **llama.cpp (HIP 后端)** | **稳定可用但非极速**。支持 GGUF 各种量化格式。Prompt 处理（prefill）极强；但受限于 HIP 小分派开销，长文本解码不如 Vulkan。 | **优势**：显存管理成熟，支持大型上下文切片；<br>**陷阱**：`-DGGML_HIP_ROCWMMA_FATTN=ON` 在长上下文（>16k）下曾发生严重的 prefill 衰减 bug。 | ❌ **无法直接加载**。Qwenputer 为 44 层自定义架构（含 12 层 GDN 循环电路与专有逻辑），GGUF 目前不支持其计算图。 | [llama.cpp Issue #20934](https://github.com/ggml-org/llama.cpp/issues/20934) (2026-03)<br>[llama.cpp PR #25507](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/25507/overview) (2026-05) |
| **llama.cpp (Vulkan / RADV 后端)** | **社区一致公认 Decode 最快**。在 Linux (Mesa RADV) 下，Vulkan 后端比 ROCm HIP 解码吞吐高 **14%～30%**（例如 RX 7900 XTX 解码测试达到 167–177 tok/s，而 HIP 仅 110–144 tok/s）。 | **优势**：RADV/ACO 编译器的微内核发射开销极低；<br>**劣势**：Prefill（提示词处理）速度通常慢于原生 ROCm BLAS。 | ❌ **无法直接加载**。与 HIP 相同，缺乏 GDN 和自定义编译层的算子支持。 | [dev.to ROCm vs Vulkan 2026](https://dev.to/rosgluk/rocm-vs-vulkan-for-amd-local-llm-hosting-2026-guide-5c70) (2026-09)<br>[rx-7900-xtx-llm-benchmark](https://github.com/hcgdw/rx-7900-xtx-llm-benchmark) (2026-07) |
| **vLLM (ROCm 官方/社区)** | **企业级高并发标杆，支持 gfx1100**。ROCm 7.2 / 10.0 已提供对应轮子。2026 年中社区贡献了 gfx1100 专用的 Fused W4A16 HIP MoE/Dense 内核（PR #44075）。 | **优势**：PagedAttention、Continuous Batching 吞吐极高；<br>**瓶颈**：启动加载耗时极长，对自定义非 Transformer 混合架构极难扩展。 | ❌ **无法直接使用**。Qwenputer 依赖专有 RvChat 驱动协议以及带状态校验的逐 token 自回归循环状态更新。 | [vLLM GPU Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) (2026-08)<br>[vLLM PR #44075](https://github.com/vllm-project/vllm/pull/44075) (2026-05) |
| **SGLang (ROCm)** | **发展迅速**。2026 年 8–10 月正式合并针对 RDNA3（gfx1100/gfx1101）的消费级 Radeon 专用运行时支持（PR #21697 及 `sglang-radeon-rdna3`）。 | **优势**：RadixAttention 缓存复用能力极强；<br>**劣势**：与 vLLM 类似，仅面向标准 HuggingFace 骨干网络（Llama/Qwen/Mistral）。 | ❌ **无法直接使用**。不适用于当前自定义 GDN-as-RAM 编译器执行图。 | [SGLang PR #21697](https://github.com/sgl-project/sglang/pull/21697) (2026-10)<br>[sglang-radeon-rdna3](https://pypi.org/project/sglang-radeon-rdna3/) (2026-08) |
| **Ollama (ROCm)** | **开箱即用工具箱**。Linux/WSL 下支持通过 ROCm 自动识别 gfx1100，内部主要调度 llama.cpp。 | **优势**：极简 CLI；<br>**劣势**：对 RDNA3 优化较为粗糙，内部偶见驱动匹配问题。 | ❌ **不适用**。模型格式受限。 | [my-ai-workspace local-llm](https://github.com/hhellbusch/my-ai-workspace/blob/e19f9f42f283d04b6ab8edf1c6928807fbde40a1/research/ai-tooling/local-llm-experiment-journal.md) (2026-02) |
| **hipEngine (原生专有引擎)** | **RDNA3 原生免 PyTorch 极速推理引擎**。由社区专门针对 gfx1100 与 gfx1151（Strix Halo）纯手工编写 HIP 内核，Qwen3.6/3.8 推理速度超越 llama.cpp。 | **优势**：无 PyTorch 运行时开销，直接调用 WMMA 和自定义 GEMV；<br>**劣势**：硬编码支持常见标准大模型架构。 | 💡 **工程参考价值极大**。其揭示了“小分派调度 + 手工 Wave32 HIP 内核”是在 gfx1100 取得极致吞吐的根源。 | [hipEngine GitHub Repository](https://github.com/shisa-ai/hipEngine) (2026-08)<br>[Framework Community Discussion](https://community.frame.work/t/hipengine-fast-native-qwen-3-6-inference-for-rdna3-including-strix-halo/82803) (2026-08) |

---

### 2. 内核与编译层最佳实践（Kernel & Compiler Optimizations on gfx1100）

#### (1) `torch.compile` + Triton 在 RDNA3 上的深度调优规则
社区与 AMD 官方架构拆解报告（[AMD Community RDNA3 Teardown](https://devcommunity.amd.com/t/rdna3-architecture-teardown-isa-hazards-kernel-violations-and-verified-ml-failure-modes/588), 2026-06）揭示了 gfx1100 上的硬件级陷阱与优化准则：
- **强制 Wave32 模式**：RDNA3 引入的双发射指令（VOPD）仅在 Wave32 下生效；若 Triton 生成 Wave64 算子，VOPD 将隐式退化为空操作（no-op），算力直接腰斩（50% 性能损失）。
- **WMMA 依赖冲突（Hazard）**：RDNA3 的 ISA 严格要求在连续的 WMMA 运算之间正确插入 `s_waitcnt` 或交织非依赖 ALU 指令。早期 Triton 3.x 后端存在发射过密引发流水线等待挂起的缺陷，需配合最新 Triton 补丁。
- **Autotune 调优空间**：gfx1100 的 LDS（本地共享内存）仅 64KB/CU，照搬 CDNA（MI300）的 `BLOCK_SIZE_M/N = 128/256` 配置会导致占用率（occupancy）崩塌。在 gfx1100 上，通常 `BLOCK_M=32/64`, `BLOCK_N=32/64`, `num_warps=2/4`（即 64/128 线程，对应 Wave32）才是最佳拐点。

#### (2) hipBLASLt 与 PyTorch TunableOp 调优
- **TunableOp 机制**：PyTorch ROCm 官方提供了针对 GEMM 的运行时启发式搜索。通过设置：
  ```bash
  export PYTORCH_TUNABLEOP_ENABLED=1
  export PYTORCH_TUNABLEOP_TUNING=1
  export PYTORCH_TUNABLEOP_FILENAME=/path/to/tunableop_gfx1100.csv
  ```
  在预热运行中遍历 hipBLASLt 与 rocBLAS 的可选算法并持久化最佳实现，后续设为 `TUNING=0` 直接加载，社区实测在非标尺寸矩阵乘上可带来 **5%～15%** 的纯 GEMM 加速（[PyTorch TunableOp Docs](https://android.xilinx.googlesource.com/platform/external/pytorch/+/4fa46d3bda295a90368f781f09cbac062d09f049/aten/src/ATen/cuda/tunable/README.md)）。
- **优先启用 hipBLASLt**：环境变量 `export TORCH_BLAS_PREFER_HIPBLASLT=1` 在 ROCm 7.x 上能将常规 Linear 算子优先路由至经过 RDNA3 WMMA 优化的 BLASLt 库（[ROCm Troubleshooting](C:/Projects/ROCM/docs/TROUBLESHOOTING.md#L309-L315)）。

#### (3) Flash-Linear-Attention (FLA) 与 Gated DeltaNet (GDN) 的 ROCm 支持
- **官方与社区对齐**：AMD 官方技术指南（[Primus GDN ⇄ FLA Parity](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/hybrid-models/gdn-fla-parity.html)）明确验证了 300M GDN 在 ROCm 上的可行性，FLA 的 Triton 实现（`chunk_gated_delta_rule` 与 `fused_recurrent_gated_delta_rule`）在 ROCm 上与 PyTorch 纯参考实现具有数学一致性。
- **Qwenputer 本机实测结果**：在 Qcompute Stage 6 中实测引入 `flash-linear-attention 0.6.0`，put 端到端由 73.6s 暴降至 4.10s（**16.84 tok/s**），get 降至 3.96s（**13.62 tok/s**），提速 **1.55×～3.3×**（[stage6-kernels.md](C:/Projects/Qcompute/docs/notes/stage6-kernels.md#L23-L37)）。
- **数值精度警示**：但回归门禁（`gpucheck --decode`）发现，Triton 浮点累加重排导致 chip 层产生 1e-7（1 ULP 级别）的浮点微差，导致 5 个标准测试无法达到 100% bit-exact。因此社区与本机形成一致共识：**FLA 适合高吞吐生产模式，而严格逐位电路验证需提供精准回退通道**（[stage6-kernels.md](C:/Projects/Qcompute/docs/notes/stage6-kernels.md#L61-L64)）。

#### (4) Causal-Conv1d 构建情况
- 纯 C++/HIP 版本的 `causal-conv1d`（Dao-AILab）在 ROCm 6.1+ 上已无需打旧的 `amd_hip_bf16.h` 补丁，但源码编译在 WSL 环境下仍存在 ABI 与 clang 兼容性门槛；社区广泛采用 FLA 自带的纯 Triton 短卷积模块（`fla.modules.convolution`）作为零外部依赖的等价无缝替换方案（[causal-conv1d README](https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/main/README.md)）。

---

### 3. 量化生态分析（Quantization Ecosystem on gfx1100）

| 量化方案 | gfx1100 适配状态 | 性能/显存表现 | 落地评估与注意事项 | 证据来源与日期 |
|---|---|---|---|---|
| **bitsandbytes (BNB)** | **0.50.0 正式转正为 Stable**。发布专为 RDNA3 (gfx1100) 优化的 Fused 4-bit SIMT GEMM 与 INT8 内核。 | 8-bit 与 4-bit (NF4) 在 PyTorch 中均能直接通过。但在 Qcompute 本机实测中，4-bit 文本产生乱码，而 8-bit（LLM.int8()）完全正常且端到端稳定。 | 推荐在特定巨型权重（如词表达 484,800 的 embedding/lm_head）采用 INT8 量化，压缩显存并降低内存搬运。 | [bitsandbytes Release 0.50.0](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.0) (2026-07-24)<br>[Habr ROCm 7 Tutorial](https://habr.com/ru/amp/publications/987672/) (2026-02) |
| **AWQ / GPTQ (W4A16)** | **Linux 专属 HIP 优化**。vLLM 引入了基于 `v_dot2_f32_bf16` 的定制反量化 GEMM 内核。 | 在标准自回归 Transformer 解码时，W4A16 相比纯 BF16 带来约 **1.2×～1.5×** 速度提升，并节省大量显存。 | ⚠️ **需离线量化与打包**。Qwenputer 的权重经编译器重排与注入，无法使用通用 AWQ 预训练转换脚本，需定制导出。 | [vLLM PR #44075](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/44075/overview) (2026-05) |
| **AMD Quark** | **AMD 官方量化工具包**（0.8.x）。全面支持 AWQ, GPTQ, SmoothQuant 等。 | 主要输出面向 vLLM 与 ROCm 部署的量化格式。 | 对通用开源 LLM 友好，但同样缺乏对 GDN 状态层自适应的量化保护策略。 | [AMD Quark PyPI](https://pypi.org/project/amd-quark/0.8.2/) (2024-06)<br>[vLLM Quark Integration](https://docs.vllm.com.cn/en/latest/features/quantization/quark/) (2026-05) |
| **FP8 (E4M3 / E5M2)** | **硬件不支持**。RDNA3（gfx1100）架构无原生 FP8 矩阵指令（该特性仅在 CDNA3 MI300 与 RDNA4 gfx120x 上提供）。 | 在 gfx1100 上请求 FP8 会被软件转码回退为 FP16 WMMA 执行，既无吞吐加速，又面临转换开销与溢出风险。 | ❌ **绝对不可用**。严禁在 7900 XT 上采用 FP8 量化。 | [Habr ROCm 7 Guide](https://habr.com/en/articles/987672/) (2026-02)<br>[ROCM_RX7900XT_RESEARCH.md](C:/Projects/ROCM/docs/ROCM_RX7900XT_RESEARCH.md#L35) |
| **Native INT4 (W4A4)** | **有硬件指令，无通用软件栈**。标称 206 TOPS，但缺少公开的 GEMM 库支持。 | 社区目前仅有学术/探针级实验，在 Dense 训练或通用自回归场景中尚无法落地。 | ❌ **暂缓跟进**。工程性价比极低，不可列入近期计划。 | [RDNA3_INT4_GFX1100_RESEARCH_2026-08.md](C:/Projects/ROCM/docs/RDNA3_INT4_GFX1100_RESEARCH_2026-08.md) (2026-08) |

---

### 4. 社区公认的 RX 7900 XT 核心提速与避坑经验（Biggest Wins & Gotchas）

综合 Reddit、GitHub Discussions、AMD 开发者社区和学术报告，总结 7900 XT 运行 LLM 时**效果最显著的优化与最致命的雷区**：

1. **绝对避免 CPU Offload（“全进显存即胜利”）**：
   - 这是 RDNA3 用户最普遍的共识。由于 PCIe 与 WSL2 D2H/H2D 数据搬运的高昂开销，一旦模型某几层被切到 CPU，推理速度会从几十 tok/s 直接暴跌到 0.5～1.5 tok/s。
   - **证明**：Qwenputer 在 Stage 5 卸载 11 层到 CPU 时，put 耗时 73.6s（0.98 tok/s）；而在 Stage 6 采用 Trim 布局（将权重由 22.2GB 压缩至 13.15GB 全部装入 20GB 显存）后，put 降至 10.16s（6.79 tok/s），提速近 7 倍！
2. **警惕 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 陷阱**：
   - 社区常见经验帖常把该开关当作“防止显存碎片、挽救 OOM 的神器”。
   - **但在 ROCm 7.2.x + RDNA3 (gfx1100/gfx1101) 下，该开关是致命的**。PyTorch Issue [#200608](https://github.com/pytorch/pytorch/issues/200608)（2026-10）与 [#192259](https://github.com/pytorch/pytorch/issues/192259)（2026-08）反复证实，由于 ROCm 虚拟内存管理驱动漏洞，开启后会在张量转移时发生 `hsaKmtMemoryVaMap` 永久死锁挂起，或触发 `!handles_.at(i) INTERNAL ASSERT FAILED` 致命崩溃。**必须保持禁用**。
3. **消除 Host/Device 逐步同步（Graph Capture / Fused Step）**：
   - WSL2 下 HIP 启动小内核的固有时延（launch latency）相对裸机更大（hipEngine 测定小 dispatch 下 Vulkan 比 HIP 快 2.4×～10×）。逐层、逐算子的 Python/CPU 交互是吞吐杀手。
   - 在自回归单 token 生成循环中，将采样、状态更新、Conv1d 状态推进尽可能融合进单个执行图或 GPU 端缓冲，能彻底消除 CPU 轮询卡顿。
4. **长上下文下慎用 `GGML_HIP_ROCWMMA_FATTN=ON`（若使用 llama.cpp）**：
   - 在 llama.cpp PR #25507 与 Issue 讨论中证实，在 RDNA3 上无脑强制开启 rocWMMA 处理 FlashAttention 会导致超长 prompt 处理出现极端性能崩塌；反而默认的 MMA/Vec 路由在 head size <= 128 时更稳定。
5. **WSL2 与原生 Linux 的性能差主要在调度与监控**：
   - 纯 GEMM 算力上，WSL2 与原生 Linux 几乎无损（损耗 < 3%）；但 WSL2 目前欠缺 PMC（性能计数器）和完整的 `rocprofv3`，且存在空闲 2 核忙轮询（ROCDXG #60）。

---

## 三、 对 Qwenputer 项目的定制加速建议（Apply to Qwenputer）

### 1. 我们的架构特征与约束
- **定制复合模型**：44 层中前 12 层为编译型 RISC-V 循环电路层（GDN 结构充当精确 RAM），后 32 层为 Qwen3.5 变体；词表极宽（484,800）；
- **专用驱动**：需 RvChat 驱动程序维系与芯片状态的 lockstep 交互；
- **现状成就**：Trim 布局将模型精简至 13.15GB 实现全显存加载；引入 FLA 后解码达到 13.6～16.8 tok/s；但 FLA 导致 chip 出现微小浮点扰动（1 ULP 级别，gpucheck 报 mismatch）。

### 2. 社区实践赋能：性价比与工作量评估排序表

| 措施编号 | 优化方案 | 预期加速 / 显存收益 | 工程复杂度 | 对 Qwenputer 的适用度与收益评估 | 优先级 |
|:---:|---|---|:---:|---|:---:|
| **M1** | **S7.1 混合调度：Chip 层精确回退 + Qwen 深度层走 FLA** | **兼得 100% 精度与 >10 tok/s 高速** | **低**（仅在模型 forward 循环做层路由判断） | **最高**。当前最大痛点是 FLA 导致 chip 存在 1e-7 微差。前 12 层走 PyTorch 原生 reference GDN（严格保证 bit-exact），后 32 层走 FLA Triton 极速内核，完美平衡可靠性与速度。 | 🥇 **P0** |
| **M2** | **S7.3 巨型词表 Embedding / LM_Head INT8 量化** | **立减 2.0～2.5 GB 显存；单步 decode 加速 15%～25%** | **中**（借助 bitsandbytes 0.50.0 或自写 Triton GEMV） | **极高**。词表 484,800 × hidden 4608 的权重高达 4.46GB（占 13.15GB 模型的 1/3 以上！），解码每 step 需做该巨型矩阵向量乘，INT8 可直接减半该算子访存耗时。 | 🥈 **P0** |
| **M3** | **Causal-Conv1d 的 Triton 化（消除最后两条 fallback 警告）** | **预估全流程再提速 8%～15%** | **低～中**（接入 FLA 内置的 `fla.modules.convolution`） | **高**。目前日志仍频繁打印 `causal_conv1d_update` 退回 PyTorch 的警告。解码阶段每层每 token 均执行短卷积状态推进，消除其 Python 开销收益确定。 | 🥉 **P1** |
| **M4** | **ROCm 环境变量与 TunableOp 稳态固化** | **整体 GEMM 算子净收益 3%～8%** | **极低**（仅需配置 env 变量并在预热期生成 tuning CSV） | **高（低垂果实）**。启用 `TORCH_BLAS_PREFER_HIPBLASLT=1`，配合 `PYTORCH_TUNABLEOP` 针对固定 M/N/K 离线校准；确保禁用 `expandable_segments`。 | 🏅 **P1** |
| **M5** | **RvChat 驱动层 CUDA Graph / HIP Graph 静态图捕获** | **解码单 token 延迟降低 10%～20%** | **中～高**（需锁定 KV 步长与状态缓冲区） | **中**。消除 WSL2 下 host 到 device 的小分派开销，让自回归循环直接在 GPU 内部连续回放。 | **P2** |
| **M6** | **移植 vLLM / llama.cpp 核心 RDNA3 W4A16 汇编内核** | 理论提速 1.2×～1.5× | 极高（需将 C++/HIP 源码适配进自定义 PyTorch 扩展） | **低**。工作量巨大，且当前显存已足够全放入，收益不如优先优化词表与调度。 | **P3** |

---

## 四、 关键建议汇报与下一步实验路线

### 1. 汇报核心：面向 RX 7900 XT 的 Top 5 最佳实践
1. **显存优先全卡驻留（零 CPU Offload）**：针对 20GB 显存，Trim 布局（13.15GB）是保证 >10 tok/s 的根本前提，任何层级回退 CPU 都会使吞吐崩溃。
2. **核心层混合精度调度**：由于 RDNA3 上 Triton 内核浮点结合律微差，对“逻辑电路级”敏感层采用精确 PyTorch 参考算子，对普通注意力/深度层采用高吞吐 Triton 内核。
3. **针对大词表（Vocab 484,800）进行极致访存压缩**：LM_Head 是该模型特有的显存与带宽大头，8-bit 量化是最有效杠杆。
4. **严防 WSL2 / ROCm 驱动级陷阱**：坚决不碰 `expandable_segments`（死锁/崩溃）；坚决不设 `HSA_OVERRIDE_GFX_VERSION`；及时关闭空闲进程避免 ROCDXG #60 占核。
5. **深挖 RDNA3 矩阵库潜力**：利用 `hipBLASLt` 与 `TunableOp` 离线搜索最佳 GEMM 实现，补全 Causal-Conv1d Triton 状态推进。

### 2. 下一步建议执行的最优 3 项实验（Top 3 Experiments by Gain/Effort）

```text
[实验 1: S7.1 混合精度层路由]
目标：实现 chip 0–11 层走 reference GDN，Qwen 12–43 层走 FLA
预期收益：彻底消除 gpucheck 5/5 MISMATCH（重回 bit-exact），且保持 >12 tok/s 的高吞吐
工作量：仅修改 RvChat/模型前向中对 FLA 的层级判断（~0.5 天）

[实验 2: S7.3 LM_Head / Embedding INT8 量化]
目标：利用 bitsandbytes 0.50.0 的 stable ROCm 内核对 vocab 投影层做 INT8 量化
预期收益：显存占用由 13.15GB 降至约 10.8GB（释放 >2.3GB），单步 decode 提速 15%~20%
工作量：在 trim 运行时为 embedding 和 lm_head 加载 int8 权重与 dequant/gemv（~1 天）

[实验 3: ROCm 环境 TunableOp 校准与 Causal-Conv1d Triton 消除警告]
目标：导出 PYTORCH_TUNABLEOP 配置文件，并引入 FLA 自带的 Triton short-conv 消除 fallback
预期收益：消除终端两行回退警告，全模型 GEMM 与 Conv1d 再获 5%~10% 净加速，且完全无外部编译依赖
工作量：环境配置与现有成熟模块引入（~0.5 天）
```


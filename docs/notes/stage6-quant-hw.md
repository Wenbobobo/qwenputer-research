# Stage 6：RX 7900 XT 精度与量化硬件实测

日期：2026-10-29（北京时间）。Q1 已完成，GPU 已释放，交接标志见 [Q1_DONE](../../gpu/Q1_DONE)。未读取或修改 22GB checkpoint。

## 结论

当前环境最划算的通用计算精度是 **BF16**：大 GEMM 约 76–78 TFLOPS，与 FP16 相当，数值范围更适合 LLM。INT8 能运行但本次并不比 BF16 快；FP8 只是可存储、可软件转换，gfx1100 没有原生 FP8 WMMA。Triton 接受 FP8 后实际发出 FP16 WMMA，本次不值得迁移成 FP8 计算。

**四位权重有 decode 带宽收益，但需要专用 kernel。** 通用“解包 INT4→BF16→WMMA”核速度只有 BF16 的 0.23–0.62 倍；单 token 专用融合 GEMV 相比合理的 BF16 Linear 基线达到 **1.40–1.51 倍**。因此先采用可工作的 bnb NF4/W4A16 做显存与模型质量实验；prefill 与 decode 分开选 kernel。W4A8 是后续工程选项，W4A4 需要原生 INT4 WMMA kernel、激活校准与精度验证，不能只修改 dtype。

## 环境与方法

WSL Ubuntu-24.04，ROCm 7.2.1；RX 7900 XT gfx1100，torch 报告总显存 20440 MiB；torch 2.9.1+rocm7.2.1.gitff65f5bc，Triton 3.5.1+rocm7.2.1.gita272dfa8。全部 Python 通过项目 [run_py.sh](../../gpu/run_py.sh) 进入原 WSL venv。GPU 测试串行，各用例清理局部 tensor、gc、empty_cache、同步，进程退出后再启动下一项。

脚本：[bench.py](../../gpu/quant/bench.py)、[gemv.py](../../gpu/quant/gemv.py)、[tooling.py](../../gpu/quant/tooling.py)、[fp8_probe.py](../../gpu/quant/fp8_probe.py)、[inspect_codegen.py](../../gpu/quant/inspect_codegen.py)。GEMM 用 3 次 warmup、10 次 CUDA/HIP Event 计时；INT4 tiled 用 20 次，专用 GEMV 用 50 次。编译时间不计入。吞吐为 2MNK/time；INT8 的单位为 TOPS，浮点为 TFLOPS。短时微基准、热缓存、无锁频，无多轮统计置信区间，不代表模型 tokens/s。

## Torch GEMM 实测

原始数据：[torch.json](../../gpu/quant/out/torch.json)。A/B/C 为同精度方阵；FP32 未启用任何显式降精度路径。

| M=N=K | BF16 ms / TFLOPS | FP16 ms / TFLOPS | FP32 ms / TFLOPS | torch._int_mm ms / TOPS |
|---|---:|---:|---:|---:|
|4096|1.812 / 75.84|1.766 / 77.83|7.139 / 19.25|1.952 / 70.39|
|8192|14.051 / 78.25|14.089 / 78.04|58.884 / 18.67|17.498 / 62.84|

8192 能装入显存。INT8 数据为 [-8,7]，输出 INT32；表中不含 activation quantize、scale、bias、requantize 等实际模型成本。简单 Triton INT8 4096 核为 **3.108 ms / 44.22 TOPS**，与 FP32 参考最大绝对误差 0。它不是高度调优的 rocBLAS 替代品。

## FP8 支持边界

原始数据：[triton.json](../../gpu/quant/out/triton.json)、[fp8_direct.json](../../gpu/quant/out/fp8_direct.json)。

|路径|E4M3FN|E5M2|
|---|---|---|
|torch GPU allocation、FP32→FP8→FP16 cast|成功|成功|
|torch.mm，512 方阵|失败：addmm_cuda not implemented for Float8_e4m3fn|失败：addmm_cuda not implemented for Float8_e5m2|
|Triton BF16→FP8→BF16，1,048,576 元素|0.02068 ms，最大量化误差 0.25|0.02480 ms，最大量化误差 0.5|
|Triton tl.dot，512³，FP32 输出|0.3873 ms / 0.693 TFLOPS|0.04852 ms / 5.533 TFLOPS|
|相对已量化输入 FP32 参考的 max abs error|7.63e-6|1.14e-5|
|实际生成 WMMA 指令输入|FP16|FP16|

机器码保存在 [E4M3 ISA](../../gpu/quant/out/float8_e4m3fn.amdgcn) 与 [E5M2 ISA](../../gpu/quant/out/float8_e5m2.amdgcn)，均观察到 v_wmma_f32_16x16x16_f16。**这是 FP8 存储经转换进入 FP16 WMMA，不是原生 FP8 算力。** 初版 masked tl.load 的整数 other=0 会触发 cannot cast int32 to fp8 编译错误；改用对齐尺寸的无 mask load 后成功。该脚本 mode=0 无 mask 仅适用于当前整除 tile 的测试尺寸。

FP8 的 512³ 与 BF16 的 4096³不能直接作同形状峰值比较；已有 ISA 足以排除本环境原生 FP8 WMMA 路线。转换误差也不代表模型精度。

## INT4 weight-only：可行但必须按场景选核

布局为 W[N,K]、X[M,K]；两权重一字节，均匀有符号值 (nibble−8)×0.125，BF16 activation。这里没有 NF4 codebook、逐组 scale/zero 元数据，也没有真实模型量化误差；比较 BF16 基线使用同一解量化后权重，测的是执行成本。

原始数据：[int4.json](../../gpu/quant/out/int4.json)。通用 Triton 核在每个 tile 内融合解包后 BF16 tl.dot；tile M/N/K=32/64/64。

|M|N|K|融合 INT4 ms|torch BF16 Linear ms|速度比 BF16/INT4|有效 TFLOPS|
|---:|---:|---:|---:|---:|---:|---:|
|1|9216|4608|0.2935|0.1806|0.615|0.289|
|32|9216|4608|0.3173|0.1166|0.368|8.57|
|256|9216|4608|0.9939|0.3486|0.351|21.88|
|1|4608|4608|0.2617|0.0900|0.344|0.162|
|32|4608|4608|0.2695|0.0831|0.308|5.04|
|256|4608|4608|0.6715|0.1535|0.229|16.19|

上述六项 max abs error 均为 0，但性能不合格。M=1 时仍分配 32 行 tile，浪费计算；每 tile 解包、长 K 串行循环也有成本。不能据此推断“四位硬件没有价值”。

补测专用单 token kernel：每 program 处理 4 个输出，融合解包、FP32 multiply/reduce，BF16 输出，**没有使用 INT4 WMMA，也没有使用 BF16 WMMA**。原始数据：[gemv.json](../../gpu/quant/out/gemv.json)。基线是 X[1,K] @ W.T；曾尝试 torch.mv，但该路径比 Linear 慢很多，最终表格只使用合理的 torch.mm Linear 基线。

|N×K|INT4 GEMV ms|BF16 Linear ms|速度比|权重字节/时间（GB/s）|max abs error|
|---|---:|---:|---:|---:|---:|
|9216×4608|0.1160|0.1622|1.40|183.1|0.015625|
|4608×4608|0.04044|0.06093|1.51|262.5|0|

纯权重从约 84.93/42.47 MB 降到 21.23/10.62 MB，少读约 4 倍；实测只快约 1.5 倍，说明解包、reduction、调度与缓存成本不可忽略。这里的 GB/s 是 packed 权重有效字节量除耗时，**不是硬件计数器测得的 DRAM 带宽**；未用 profiler 证明严格 memory-bound。热缓存小层也可能落入 Infinity Cache，不能外推全模型。decode 的低算术强度使 weight-only 有潜在收益；prefill 应保留高吞吐 BF16 或另调优量化 GEMM。

## 工具支持矩阵

安装记录：[install.log](../../gpu/quant/out/install.log)；GPU 结果及完整异常：[tooling.json](../../gpu/quant/out/tooling.json)。执行 pip install bitsandbytes torchao 后原 torch 未被升级。

|工具/路径|本机版本|实测|边界|
|---|---|---|---|
|bitsandbytes Linear8bitLt|0.50.2|GPU 成功，4×1024 输入、1024×1024 层、BF16 输出 finite|has_fp16_weights=False；提示 BF16 输入在量化时转 FP16；只验证 smoke，没有证明快于 BF16|
|bitsandbytes Linear4bit NF4|0.50.2|GPU 成功，同尺寸，BF16 compute/output finite|可作为 W4A16 显存实验起点；不等于 native INT4 WMMA|
|torchao int8 weight-only|0.18.0|失败，导入阶段|ImportError: cannot import name 'ScalingType' from torch.nn.functional|
|torchao int4 weight-only|0.18.0|同上|扩展提示要求 torch>=2.11.0，当前2.9.1；尚未到 GPU quant kernel 阶段|

上游 [bitsandbytes 安装文档](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/docs/source/installation.mdx) 已列 ROCm、RDNA 和 gfx1100。不能套用旧版“只有 CUDA/Instinct”的结论，也不能由 all features 推导某个原生 WMMA kernel 已被使用。[较旧 ROCm 6.4.3 文档](https://rocm.docs.amd.com/en/docs-6.4.3/how-to/rocm-for-ai/inference-optimization/model-quantization.html) 的 Instinct 限定属于其版本范围。

[torchao workflow 支持表](https://github.com/pytorch/ao/blob/d1e0e2d8ee59cb2ccee19d303b20931e58820d8/docs/source/workflows/index.md) 未给 gfx1100 保证，ROCm 条目包含 MI350x 原型 MX 工作流；[Int4WeightOnlyConfig API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int4WeightOnlyConfig.html) 的 ROCm packing 参数也不等于7900 XT性能承诺。可未来在隔离环境匹配 torch/torchao 版本再测，本次不为此升级工作环境。

## RDNA3 硬件事实与官方峰值

[AMD RX7900 XT 产品页](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html) 当前列值：

|类型|官方峰值|本次大矩阵表现|
|---|---:|---:|
|FP32 vector|51.6 TFLOPS|18.67–19.25 TFLOPS|
|FP16 vector|51.6 TFLOPS|本次测的是 GEMM，非 vector 单独测试|
|FP16 matrix|103 TFLOPS|77.83–78.04 TFLOPS|
|BF16 matrix|产品页未单列；按相同每 CU WMMA吞吐推算约103 TFLOPS|75.84–78.25 TFLOPS|
|INT8 matrix|103 TOPS|62.84–70.39 TOPS|
|INT4 matrix|206 TOPS|本次未实现/测量原生 INT4 WMMA|

[AMD GPUOpen RDNA3 WMMA 说明](https://gpuopen.com/learn/wmma_on_rdna3/) 列 FP16/BF16=512 operations/clock/CU，INT8=512，INT4=1024，并明确输入 f16、bf16、iu8、iu4；所以 BF16 约103是架构等吞吐推算，**不是产品页直接公布的 BF16 数值**。产品页峰值与运行频率/实现有关，不能把未达到峰值视为不支持。

[RDNA3 dense WMMA builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html) 覆盖 gfx1100–1103 与 RDNA3.5 gfx1150–1153，原生支持 FP16、BF16、INT8、INT4，整数累加 INT32。**gfx1100 有原生 INT4 WMMA，无原生 FP8 WMMA。** RDNA3.5同一类型集合；[RDNA4 gfx1200/1201 文档](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html) 明确新增 FP8 E4M3/BF8 E5M2，以及更深 K 的 INT4 变体。[rocWMMA 类型支持表](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/data-type-support.html) 也区分硬件支持与库类型支持。

## 建议与 W4A8/W4A4 迁移路径

1. **现在：BF16 通用计算 + bnb NF4/W4A16 显存实验。** BF16/FP16 大算力相同，优先 BF16；bnb smoke已过，先测模型质量、峰值显存、prefill/decode latency 与 tokens/s。不要先追 FP8，也不要用仅 weight size 推定速度。22GB checkpoint 留给后续独立模型实验。
2. **decode：先做有 scale/zero 的 W4A16 专核。** 现有 toy GEMV已证实1.4–1.5x局部收益；下一步加入逐组 scale/zero/NF4查表、bias，调 tile、并行 K reduction/split-K 与 batch 分派；尽量不把整层 BF16 权重写回显存。用真实 layer 及全模型验证，避免热缓存小层偏差。
3. **W4A8：可行，优先“INT4存储→寄存器INT8 + A8 + INT8 WMMA”。** native IU8 WMMA要求双方8位；四位权重应融合解包到 INT8，激活动态 per-token/per-group 量化，INT32 累加后按 scale dequant/requant，处理 zero-point 和 outlier。也可分块缓存解包，但应计算额外带宽。现测 INT8 TOPS不高于BF16，主要价值是带宽/容量，不应承诺计算翻倍。需要定制 Triton 支持的 INT8 kernel 或 HIP/LLVM builtin，并包含校准与端到端性能预算。
4. **W4A4：硬件存在原生 IU4 WMMA，软件工程成本高。** 对称/非对称 A4/W4 packing 必须满足 wave fragment布局，以 HIP/LLVM __builtin_amdgcn_wmma_i32_16x16x16_iu4 或 inline ISA实现，或扩展 Triton lowering 后使用；本次 tl.dot(INT8)不等于 IU4。需激活校准/QAT或等价策略控制四位激活误差、分组缩放与zero校正、INT32累加和输出融合。理论206 TOPS不能作为已达性能。NF4是非均匀码本，不能直接喂给整数乘法 WMMA；若走 native W4A4，应采用整数兼容量化或额外解码映射，重做质量评估。
5. **优先级：W4A16可运行与模型质量 → decode专核 → W4A8实测收益 → W4A4研究。** Prefill与decode分别决策。现阶段无需为硬件不具备的FP8 WMMA投入，也无需为torchao最新版升级整个可用ROCm环境。

本次所有GPU进程已退出；[Q1_DONE](../../gpu/Q1_DONE)用于释放交接。后续GPU工作由Lead调度。

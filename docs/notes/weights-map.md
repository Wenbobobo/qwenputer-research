# 权重地图：20.7 GiB 里装的是什么，以及怎样无损压缩

对象：`.data/Qwen3.5-4B-Qwenputer-Filesystem`（只读）。全部数字都是在本机 CPU 上实测的：
`analysis/weights/scan.py` 扫描了全部 10 个分片，耗时 3.5 分钟；`analysis/weights/compact.py`
在 10 个分片上做了完整的编码、解码和逐字节校验。

## 0. 结论先行

- **总计 20.672 GiB，其中非零只有 7.893 GiB，精确零占 12.779 GiB（61.8%）**。没有非有限值（inf/nan 为 0），也没有 -0.0。
- 三块大头：
  - Qwen 32 层：11.97 GiB，非零 6.65 GiB。就是 Qwen 原本 2560 宽的矩阵补零到 4608 宽。
  - 嵌入：4.16 GiB，非零 1.24 GiB。
  - 12 个 chip 层：4.54 GiB，**非零只有 129 万个值，约 2.5 MB**。
- **无损压缩已经实测**：
  - 用 `auto` 编码（稠密核心 + CSR 残差），**20.672 GiB → 7.951 GiB（38.5%）**。
  - 10 个分片都能逐字节重建原文件，SHA-256 全部一致。
  - 全量编码 196 s，解码 40 s，纯 CPU。
- 推理真正需要的只有约 7.9 GiB 的非零权重：
  - 20 GB 的 ROCm 卡（7900 XT）放不下稠密的 20.67 GiB，压缩后的约 8 GiB 能放下。
  - 48 GB 内存的 CPU 机可以直接 mmap 稠密原件，用来做正确性复现。

## 1. 结构事实（来自 config.json / chipformer.json / 扫描）

- hidden 4608 = Qwen 2560 + chip 维 2048（`chip_dims_offset` 2560，ballast 18，slots 1806）。
- 共 44 层：第 0–11 层是 chip（全部 linear_attention），第 12–43 层是 Qwen 原本的 32 层。
- 词表 484800 = Qwen 248320 + chip token 236469 + 对齐填充。tie_word_embeddings=true，所以没有单独的 lm_head 张量。
- 已经是纯文本模型（`Qwen3_5ForCausalLM`），没有视觉塔。
- dtype：bf16，另有 48 个 F32 小张量（Qwen 层的 `A_log` 和 `linear_attn.norm`，共 15 KB）。
- **Qwen 层只读一个 chip 维**：第 2732 维（`mu_dim`，mute 标记）。
  - 涉及 88 个张量：`in_proj_a/b/z`、`q_proj`、`k_proj`。
  - Qwen 层的输出行从不写 chip 维（`nz_rows_beyond_2560 = 0`）。
  - Qwen 层的 norm 在 chip 维上为 0，按 (1+w) 约定增益为 1，但这些维没有人读。
- 嵌入的四个象限（行按 248320 切，列按 2560 切），非零计数：
  - Qwen 词 × Qwen 维：**635,699,200 = 248320 × 2560，完全稠密**，就是基座嵌入。
  - Qwen 词 × chip 维：18,674,331（ballast、字节位、id 位、token 检测器等）。
  - chip token × Qwen 维：**0**。
  - chip token × chip 维：11,782,899。
- 最终 norm 把 chip 维的增益设为 0，只保留 21 个 code 维、18 个 id 维和 tau。所以绑定的 lm_head 实际只读 2560 + 40 维。

## 2. 20.7 GiB 去了哪里

| 部分 | 张量数 | 磁盘 GiB | 非零 GiB | 零 GiB | 非零占比 | 无损最优（实测/估计） |
|---|---|---|---|---|---|---|
| Qwen 32 层（矩阵） | 320 | 11.968 | 6.650 | 5.318 | 55.6% | ≈6.65（trim/core） |
| chip 12 层（矩阵） | 132 | 4.542 | 0.0024 | 4.540 | 0.05% | ≈0.006（CSR；zlib 后约 1 MB） |
| 嵌入（= lm_head） | 1 | 4.161 | 1.241 | 2.920 | 29.8% | 1.281（core，实测 1311.9 MiB） |
| 各类 norm | 141 | 0.001 | — | — | — | 原样 |
| **合计** | 594 | **20.672** | **7.893** | **12.779** | 38.2% | **7.951（实测）** |

逐层情况（`scan_results.json` → `summary.named_layers`；每层稠密存储都是 0.379 GiB，full-attention 层是 0.360 GiB）：

| 层 | 非零个数 | 非零 128×128 块 | CSR 大小 |
|---|---|---|---|
| chip0 / 1 / 2 / 3 | 20,419 / 99,702 / 40,915 / 57,885 | 161 / 148 / 167 / 304 | 0.23 / 0.53 / 0.31 / 0.38 MiB |
| chip4 / 5 / 6 / 7 | 158,507 / 124,487 / 164,046 / 302,964 | 595 / 589 / 482 / 902 | 0.76 / 0.63 / 0.79 / 1.32 MiB |
| chip8 / 9 / 10 / 11 | 143,916 / 59,118 / 46,543 / 70,602 | 536 / 457 / 255 / 475 | 0.71 / 0.39 / 0.34 / 0.43 MiB |
| qwen0（第 12 层，GDN） | 112,928,000（55.6%） | 6954 | trim 215.6 MiB |
| qwen15（第 27 层，full attn） | 107,489,536（55.6%） | 6600 | trim 205.3 MiB |
| qwen31（第 43 层，full attn） | 107,489,536（55.6%） | 6600 | trim 205.3 MiB |

- chip 层合计 1,289,104 个非零值。chip 矩阵里单个张量最多只有 484 种不同的值，大多数只有 1–18 种，所以对字典编码和 zlib 极其友好。
- 二维权重的块稀疏度（全零块的比例）：

  | 部分 | 64×64 全零块 | 128×128 全零块 | 128×128 非零块数 |
  |---|---|---|---|
  | chip | 98.2% | 96.6% | 5071 |
  | Qwen | 44.3% | 44.2% | — |
  | 嵌入 | 65.4% | 63.1% | — |

  chip 层的非零值是"散点"，不是块：128×128 块存储比 CSR 大 20 倍以上。

各分片情况（chip 层散布在多个分片里，不是连续的）：

| 分片 | 内容 | GiB | 非零 GiB | auto 实测 |
|---|---|---|---|---|
| 00001 | 嵌入 | 4.161 | 1.241 | 1311.9 MiB |
| 00002 | chip 0,1,10,11 + Qwen 12,13（部分） | 2.034 | 0.289 | 298.0 MiB |
| 00003 | Qwen 13–18 | 2.033 | 1.129 | 1158.8 MiB |
| 00004 | chip 2 + Qwen 18–23 | 2.033 | 0.919 | 943.3 MiB |
| 00005 | Qwen 23–28 | 2.077 | 1.154 | 1183.9 MiB |
| 00006 | chip 3 + Qwen 28–33 | 2.025 | 0.915 | 938.8 MiB |
| 00007 | Qwen 33–38 | 2.024 | 1.125 | 1153.8 MiB |
| 00008 | chip 4 + Qwen 38–42 | 2.033 | 0.919 | 943.8 MiB |
| 00009 | chip 5–9 + Qwen 42,43 | 2.016 | 0.202 | 209.2 MiB |
| 00010 | chip 9（剩余部分）+ 最终 norm | 0.237 | 0.000 | 0.2 MiB |

## 3. 推理真正需要什么

| 用途 | 必需的权重 | bf16 大小 |
|---|---|---|
| 只要 Qwen 文本（不要计算机） | 基座 Qwen3.5-4B 的文本部分 | ≈7.83 GiB（6.65 + 1.18）；直接用基座即可 |
| 完整 Qwenputer（稠密，原生 transformers） | 全部 | 20.67 GiB 显存/内存，每个 token 都要读 20.67 GiB 权重 |
| 完整 Qwenputer（只算非零） | Qwen 层 2560 宽 + mu 列；chip 层稀疏；lm_head = 248320×2560 稠密 + 约 40 维稀疏 | ≈7.9–8.0 GiB |

稠密推理把 61.8% 的带宽和存储花在乘零上。所以：
- 嵌入的 lm_head 部分每个 token 要做 484800×4608 的乘法，其中真正有用的是 248320×2560 加上 40 维的稀疏部分。
- chip 层每层 0.38 GiB 的 GEMM 里，有效 MAC 不到万分之 15。

只算非零的执行方式：
- Qwen 层：对 `x[:, :2560]` 做原尺寸 GEMM，再加 mu 列的秩 1 修正，输出散回前 2560 维。
- chip 层：SpMM，或者按块的 gather。

这是 trim/core 布局能直接对上的计算形状。上游的 `fastexec.py`/`tritonexec.py` 紧凑执行器可能已经部分实现了这一点，要对照代码确认。

## 4. 压缩方案（按收益排序）

全部方案都要求逐位无损；"实测"指通过 `compact.py` 验证过，"估计"来自 `scan_results.json` 的 `est` 字段。

| # | 方案 | 结果大小 | 状态 | 代价 / 阻塞 |
|---|---|---|---|---|
| 1 | **core（稠密核心 + CSR 残差，按张量取最小）** | **7.951 GiB（实测，SHA 全一致）** | 已实现 `compact.py --policy auto` | 自定义格式，原生 transformers 不能直接加载；解码 40 s 后才能还原成标准 safetensors |
| 2 | 基座 + 增量 | 约 130 MB 增量（chip CSR 约 7 MiB + 嵌入残差约 120 MB + 锚汇列 16 KiB）；zlib 后约 6 MB；另需下载基座约 8 GB | 设计阶段，**未验证** | 假设是 Qwen 核心与基座 Qwen3.5-4B 逐位相同（merge.py 只做了拷贝补零），本机没有基座所以无法验证。norm 经过 `(1+w)·√(2560/4608)−1` 重缩放，需要从基座确定性地重算，或者直接存这 104 个 norm（约 1 MB） |
| 3 | 从编译器重新生成 chip 部分 | chip 层 4.54 GiB 中 100% 理论上可以重新生成，嵌入的 chip 特征（约 3046 万项）也可以 | 未做 | 见下文"再生成分析" |
| 4 | 块稀疏 bs64 / bs128 | Qwen 分片 7：bs64 1157 MiB，bs128 1162 MiB（trim 1153 MiB）；chip 部分（分片 9 的 L5–9）：bs64 47.5 MiB，bs128 88.1 MiB，CSR 3.7 MiB | 已实测 | 存储上比 core 差；好处是 GPU 块稀疏 kernel 好写（chip 层 5071 个非零 128 块，约 166 MB） |
| 5 | 纯 CSR | chip 层很好（约 7 MiB），Qwen 层是稠密的 1.11 倍（分片 7 实测 2304 MiB） | 已实测 | 不能用于 Qwen |
| 6 | 纯文本导出 | 这个 checkpoint 已经是 `Qwen3_5ForCausalLM`，没有视觉塔，**再导出收益为 0**；去掉 chip 就等于基座 Qwen | — | 会丢掉计算机 |
| 7 | 通用压缩（zlib/zstd） | 能把 core 的残差部分再压 10–25 倍（嵌入残差约 116 MB → 4.7 MB；chip CSR 3.7 MiB → 0.39 MiB）；稠密核心是真实 bf16 权重，按 q35-size.md 只能再压到约 72% | 部分实测 | 只影响下载和存储 |

### 再生成分析（选项 c）

`chipformer.json` 记录了：
- 布局（`rv1-store`）
- circuit 统计（12 层，3753 dims，164 heads，47532 units）
- slots 1806
- 全部维度编号
- `store` 配置（`slot_bits` 10，`n_chunks` 512，`n_names` 512，seed 2035）

对应的构建命令是 `python -m chipformer.rv.merge BASE --out DIR --store`（rv-store.md）。

能重新生成的部分：
- chip 层 = `emit_layers(core.build(Target, Text(tokenizer), store.Config()))`。先用 fp32 发射再转 bf16，不涉及前向计算，是确定性的。
- 嵌入的 chip 特征列和 chip token 行 = `text_rows` 和 `embedding_matrix`，只依赖 tokenizer 和编译器。
- Qwen 层 = 基座拷贝 + 补零 + norm 重缩放 + mute 常数（−1024/normed_ballast）。

被阻塞的部分：
1. **需要基座**：要下载 Qwen/Qwen3.5-4B（含视觉约 9.3 GB）。
2. **anti-sink 列不可复现**：`mute.fit_anti_sink` 对 8 个 full-attention 层的 `k_proj[:, 2732]` 跑了一遍前向拟合，结果依赖设备和数值。解决办法是把这 8×1024 个 bf16 值（16 KiB）直接存下来。
3. **编译器版本没有锚定**：chipformer.json 和模型卡只给了 `rv-store` 分支，没有 commit 和 torch 版本。逐位复现必须先用 `scan.py`/`compact.py` 对比 SHA 才能确认。
4. **内存**：merge 的峰值内存是 44.7 GB（rv-store.md），在 48 GB 的机器上很危险。可以改成逐张量生成并直接对比，不落盘整个模型。

因此选项 3 相比选项 2 唯一的好处，是省掉约 120 MB 的嵌入残差和 7 MiB 的 chip CSR。收益太小，风险高，**不推荐**作为存储方案。它适合用来做"权重 == 编译器输出"的审计。

## 5. 针对我们的硬件的建议

- **48 GB 内存的 CPU 机**：
  - 正确性复现直接 mmap 原始 safetensors（20.67 GiB，放得下）。
  - 不要在这台机器上跑 merge（峰值 44.7 GB）。
  - 要提速就把 Qwen 层裁成 2560 宽，chip 层用稀疏乘法：每个 token 的权重流量从 20.7 GiB 降到约 7.9 GiB，大约 2.6 倍。
  - 嵌入/lm_head 拆成"248320×2560 稠密 + 40 维稀疏"，可以省掉每个 token 约 3 GiB 无用读取。
- **20 GB 的 ROCm（7900 XT）**：
  - 稠密 20.67 GiB 加上激活和 KV **放不下**。
  - 必须用压缩布局（约 8 GiB）：Qwen 层裁剪成稠密 GEMM，chip 层用稀疏或块 gather。显存余量约 11 GiB，可以留给 KV 和批处理。
  - 落地方式有两种：用 `compact.py` 的 core 格式加一个自定义加载器；或者把 chip 层留在 CPU、只把 Qwen 部分放到 GPU（约 6.65 + 1.3 GiB）。
- **16 GB 的 T10（没有 bf16）**：
  - 不能转成 fp16。bf16 转 fp16 会损失范围，而电路的精确性是在 bf16 舍入下验证的。
  - fp32 稠密需要 41 GiB，不可行；fp32 裁剪后约 15.9 GiB，也放不下。
  - 可行的做法：按 bf16 存约 8 GiB 的压缩权重，按层即时升到 fp32 计算（bf16 到 fp32 的转换是精确的）。激活必须在 HF 原本的位置舍入回 bf16，才能保住电路的精确性。
  - 这一点**没有验证**，要用 IR replay（上游 `replay.py`/`execcheck.py`）逐 token 对拍。
- **存储和分发**：保留原件作为权威副本，压缩副本用 `compact.py --policy auto`（7.95 GiB，SHA 可证）。拿到基座后可以再评估选项 2（增量约 130 MB）。

## 6. 证据与复现

- `analysis/weights/scan.py` → `analysis/weights/scan_results.json`
  - 594 个张量逐个列出：dtype、shape、numel、bytes、非零个数、非零占比、类别、非零行列、64/128 块、象限计数、不同值个数、各编码的估计大小。
  - summary 里有 by_class、by_group_class、by_shard、block_sparsity、named_layers、checks。
  - 运行日志：`analysis/weights/scan_log.txt`。
  - 运行命令：`uv run --with numpy --with safetensors python analysis/weights/scan.py`
- `analysis/weights/compact.py` → `analysis/weights/compact_results.json`
  - 每次运行都会：逐张量做 `np.array_equal` 位比较；对整个分片，用"原始 header + 解码后的数据"重建字节流，与原文件比对 SHA-256。
  - 运行日志：`analysis/weights/compact_log.txt`。
  - 运行命令：`uv run --with numpy --with safetensors python analysis/weights/compact.py 1 2 ... 10 --policy auto`（另有 csr / bs128 / bs64 / trim / core 策略，以及 `--tensors` 正则筛选）。
- 保留的压缩样品：`analysis/weights/compact/00009-of-00010.auto.qcz.safetensors`（209 MiB，chip 5–9 + Qwen 42/43）和 `00010-of-00010.auto.qcz.safetensors`（155 KiB）。
- 方法和背景参考：
  - `.research/upstream/qwenputer-main/docs/q35-size.md`
  - `.research/upstream/qwenputer-rv/experimental/riscv/docs/rv-store.md`
  - `.research/upstream/qwenputer-rv/experimental/riscv/chipformer/rv/merge.py`
  - `.research/upstream/qwenputer-rv/experimental/riscv/chipformer/q35/mute.py`

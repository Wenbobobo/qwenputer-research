# Stage-3 A2 · Qwen 侧审计：发布权重 = 基座 Qwen3.5-4B + 确定性合并 + 16 KiB anti-sink

> 脚本：analysis/rebuild/audit_base.py（逐张量 bit 级对比）、analysis/rebuild/audit_base_sink.py（anti-sink CPU 重拟合，可选）
> 原始数据：analysis/rebuild/base_report.json · 日志：analysis/rebuild/audit_base.log、audit_base_sink.log
> 环境：纯 CPU，torch 2.14.1+cpu，transformers 5.19.0（仅 sink 重拟合用到）；主审计 24 s（safetensors 按张量读取 + 嵌入按 16384 行分块，不实例化模型）；sink 重拟合 146 s（加载 bf16 基座约 9 GB）。
> 前置：chip 侧已在 stage3-a-rebuild.md 证明为编译器输出（168 张量 + chip 维全部 bit-exact）。

## 0. 结论
**发布 checkpoint 的 594 个张量中，没有任何一个无法解释。** Qwen 侧 426 个对象（32 层 × 424 张量 + final norm 的 Qwen 部分 + embedding 的 Qwen 块）全部可由基座 `Qwen/Qwen3.5-4B` 经上游 `rv/merge.py + q35/merge.py + q35/mute.py` 的规则复现：

- 338 个与「基座 + 加宽规则」逐位一致（bf16/fp32 位模式）；
- 80 个仅在 mu 维（列 2732）上与之不同，且差异 **逐位等于** mute 常数 bf16(-1024/√4608) = **-15.0625**；
- 8 个（8 个 full-attention 层的 k_proj）仅在列 2732 上不同，即 anti-sink 列：结构完全符合 `fit_anti_sink`，CPU 重拟合与之余弦 ≥ 0.999995，但非 bit-exact（设备相关）。

**不可推导残差 = 8 × 1024 bf16 = 16,384 B（16 KiB；其中非零 6,144 个元素 = 12 KiB）。** 可推导但非零的 mute 增量为 132,608 个元素 = 265,216 B（约 259 KiB，单一常数）。
因此："published = base + compiler + 约 0.27 MB 规则化增量，其中只有 16 KiB 必须随配方存储"。比任务预期的"~MB delta"更小。

## 1. 方法
1. 读取上游规则（只读，不导入被审计代码，审计脚本内独立重写）：
   - 键名：`model.language_model.*` → `model.*`，层号 +12（n_chip=12）；vision tower、mtp.* 丢弃（基座 738 个张量中 426 个是 text，其余为 `model.visual`、`mtp.*`）。
   - `IN_COLS`（in_proj_qkv/z/a/b, q/k/v_proj, gate/up_proj）：最后一维 0 填充 2560→4608。
   - `OUT_ROWS`（out_proj, o_proj, down_proj）：第 0 维 0 填充。
   - `WIDE_NORMS`（input/post_attention_layernorm, final norm）：g = (1+w)·√(2560/4608) − 1（fp32 计算后转 bf16），再 0 填充。
   - 其它（conv1d, dt_bias, A_log(fp32), linear_attn.norm, q_norm, k_norm）：原样复制。
   - mute（mu = 2732，normed_ballast = √4608）：线性注意力层 in_proj_a/b/z[:, mu] = -1024/√4608；全注意力层 q_proj 每个头的 gate 半部（行 h·512+256 … h·512+511）[:, mu] = 同一常数；k_proj[:, mu] = anti_sink/√4608。
2. 对每个发布张量计算期望值，**按位模式（int16/int32 视图）比较**；不一致时定位行/列，检查是否只落在列 2732，并逐位验证 mute 公式或 anti-sink 结构（RoPE 维 0..63 为 0、每个 kv 头 |key[64:]| = 16）。
3. embedding 只比较 Qwen 块 [0:248320, 0:2560]，另查 chip token 行的 Qwen 列与对齐 pad 行；embedding 与 norm 的 chip 部分已在 stage3-a 审计。
4. 可选：用上游 `fit_anti_sink(base, "cpu")` 在本机重拟合，与发布列比较。

## 2. 键集合
| 项 | 数量 |
|---|---|
| 发布张量总数 | 594 |
| chip 层张量（0..11 层，stage3-a 已证 bit-exact） | 168 |
| Qwen 层张量（12..43 层） | 424（24 线性层 × 14 + 8 全注意力层 × 11） |
| 其它 | embed_tokens、norm |
| 基座 text 张量映射后缺失 / 发布中多出 | 0 / 0 |
| config.layer_types[12:] 与基座 layer_types | 完全一致（全注意力层 = 15,19,…,43） |

## 3. 分组结果（Qwen 层）
| 层类型 | 张量 | 规则 | 数量 | bit-exact | 仅 mute | anti-sink | 不一致元素 |
|---|---|---|---|---|---|---|---|
| linear | input_layernorm / post_attention_layernorm | 范数重缩放+填充 | 24+24 | 48 | 0 | 0 | 0 |
| linear | in_proj_qkv | 列填充 | 24 | 24 | 0 | 0 | 0 |
| linear | in_proj_a | 列填充 + mute | 24 | 0 | 24 | 0 | 768（=24×32） |
| linear | in_proj_b | 列填充 + mute | 24 | 0 | 24 | 0 | 768 |
| linear | in_proj_z | 列填充 + mute | 24 | 0 | 24 | 0 | 98,304（=24×4096） |
| linear | conv1d / dt_bias / A_log / norm | 复制 | 4×24 | 96 | 0 | 0 | 0 |
| linear | out_proj / down_proj | 行填充 | 2×24 | 48 | 0 | 0 | 0 |
| linear | gate_proj / up_proj | 列填充 | 2×24 | 48 | 0 | 0 | 0 |
| full | input/post_attention_layernorm | 范数重缩放+填充 | 8+8 | 16 | 0 | 0 | 0 |
| full | q_proj | 列填充 + mute(gate 行) | 8 | 0 | 8 | 0 | 32,768（=8×16×256） |
| full | k_proj | 列填充 + anti-sink | 8 | 0 | 0 | 8 | 6,144（=8×4×192 非零） |
| full | v_proj / gate_proj / up_proj | 列填充 | 3×8 | 24 | 0 | 0 | 0 |
| full | o_proj / down_proj | 行填充 | 2×8 | 16 | 0 | 0 | 0 |
| full | q_norm / k_norm | 复制 | 2×8 | 16 | 0 | 0 | 0 |
| **合计** | | | **424** | **336** | **80** | **8** | **138,752** |

| 非层张量 | 比较范围 | 结果 |
|---|---|---|
| final norm | [0:2560] vs (1+w)·√(2560/4608)−1 | ✅ 2560/2560 bit-exact（对照：未重缩放的基座值会 2560 处全不一致，证明重缩放确被应用） |
| embed_tokens | [0:248320, 0:2560] vs 基座 embed（6.36 亿元素） | ✅ 0 处不一致 |
| embed_tokens | chip token 行 [248320:484789, 0:2560] | ✅ 全 0（chip token 无 Qwen 部分） |
| embed_tokens | 对齐 pad 行 [484789:484800, 全列]（11 行） | ✅ 全 0 |

## 4. 不一致定位
- **全部 138,752 个不一致元素都在列 2732（mu_dim）**；每个张量 `mismatch_outside_mu = 0`。
- in_proj_a/b/z：列 2732 的所有行 = bf16(-15.0849…) = -15.0625，基座填充值为 0 —— 逐位符合 mute 公式。
- q_proj（8192×4608，16 头 × [query 256 | gate 256]）：仅 gate 行（每头后 256 行，共 4096 行）的列 2732 = -15.0625；query 行该列仍为 0 —— 逐位符合。
- k_proj（1024×4608，4 个 kv 头 × 256）：仅列 2732；每个头的 RoPE 维 0..63 为 0，非零 192×4 = 768 个；乘回 √4608 后每头 |key| ∈ [15.984, 16.015]（SINK_NORM = 16，偏差来自 bf16 舍入）。**只出现在 8 个全注意力层（15,19,23,27,31,35,39,43）的 k_proj**，线性层、v_proj、MLP 等均无。
- 上述发布列的 bf16 位模式与 fp32 值已存入 base_report.json（`anti_sink_columns_bf16_hex`、`anti_sink_keys_fp32`），可直接作为配方的 16 KiB 附件。

## 5. anti-sink CPU 重拟合（audit_base_sink.py）
| 层 | 1024 元素中 bf16 位相等数 | 每 kv 头余弦 | 最大绝对差 |
|---|---|---|---|
| 15 | 794 | 1.000000 / 0.999999 ×3 | 2.4e-4 |
| 19 | 717 | 0.999999 ×4 | 2.4e-4 |
| 23 | 633 | ≥0.999998 | 2.4e-4 |
| 27 | 665 | 0.999999 ×4 | 1.2e-4 |
| 31 | 675 | ≥0.999997 | 2.4e-4 |
| 35 | 639 | 0.999997 ×4 | 2.4e-4 |
| 39 | 598 | ≥0.999995 | 2.4e-4 |
| 43 | 600 | 0.999996 ×4 | 2.4e-4 |

整体位相等 64.95%（含 2048 个 RoPE 零），最小余弦 0.999995，最大相对差（对列最大值）≈ 0.2%。说明发布 anti-sink 确为 `fit_anti_sink` 在相同 prompts 上的输出（发布方当时大概率在 CUDA 上运行），只是 bf16 前向的累加顺序/内核差异导致末位不同 —— **必须存储，不能本地重算后期待 bit-exact**。

## 6. 判定
- ✅ Qwen 侧没有任何训练/微调痕迹：除 mu 维的 mute 常数与 anti-sink 外，所有 Qwen 权重与基座逐位一致（或为确定性加宽/重缩放结果）。
- ✅ 结合 stage3-a：594 个张量中 586 个可从「基座 + 编译器 + 规则」完全确定性重建；剩下 8 个 k_proj 中除 1 列（1024 元素）外也逐位一致。
- ✅ 残差：不可推导 16 KiB（anti-sink），可推导 mute 增量 ≈259 KiB；**无未解释的不一致**。
- 结论：**published = base(Qwen3.5-4B) + compiler(chip) + 16 KiB anti-sink**，可以用"基座 + 编译器 + 配方 + 16 KiB"分发并在本地逐位重建整个 22 GB checkpoint。

## 7. 注意事项 / 未验证项
- **anti-sink 设备相关**：CPU 重拟合与发布值约 35% 元素在 bf16 末位不同（最大绝对差 2.4e-4）；分发配方时必须携带 base_report.json 中的 8×1024 bf16 原值。
- **编译器 commit 未锚定**：chip 侧对比依赖当前 rv-store 分支代码；发布方未记录 commit / torch / transformers 版本。若要作为正式格式，应在配方中记录 commit hash 与发布张量 hash。
- **基座版本**：本审计使用本地 .data/Qwen3.5-4B（HF 缓存 trees 记录 851bf6e8…，推测为下载 revision）；其它 revision 的基座需重跑本脚本。
- mute 常数在 bf16 中为 -15.0625（≠ 精确值 -15.0849），这是上游 `t[:, d] = w` 写入 bf16 张量时的就近舍入，复现时须同样在 bf16 中赋值。
- 本审计只检查权重；config/tokenizer 的一致性见 stage3-a（tokenizer 扩展部分）。基座的 vision tower、MTP 头按规则被丢弃，未检查。

## 8. 复现
```
repro\qwenputer\experimental\riscv\.venv\Scripts\python.exe analysis\rebuild\audit_base.py ^
  --base .data\Qwen3.5-4B --model .data\Qwen3.5-4B-Qwenputer-Filesystem --out analysis\rebuild\base_report.json
rem 可选：anti-sink CPU 重拟合（需上游包，约 2.5 分钟，需加载约 9 GB bf16 基座，未精确测峰值）
set PYTHONPATH=repro\qwenputer\experimental\riscv
repro\qwenputer\experimental\riscv\.venv\Scripts\python.exe analysis\rebuild\audit_base_sink.py
```

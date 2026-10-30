# Stage-3 A：编译器即安装器 —— 芯片层逐位重建审计

> 脚本 analysis/rebuild/rebuild_chip.py · 报告 analysis/rebuild/report.json · 负对照 analysis/rebuild/report_negcontrol_seed2036.json · 日志 analysis/rebuild/run.log
> 环境：纯 CPU、48 GB RAM、Python 3.13、torch 2.14.1+cpu、numpy 2.5.4；单次约 76–90 s。不实例化 Qwen，只读 safetensors。

## 0. 结论
**发布 checkpoint 的 12 个芯片层（168 个张量）、最终 norm 的芯片维度、全部 236,469 行芯片 token 嵌入、以及 248,320 行文本 token 嵌入的芯片维度 [2560:4608]，与上游编译器输出按 bf16 位模式逐位一致，不匹配元素数为 0。**
"芯片是算出来的，不是训练出来的"——对芯片部分**已证实**。没有不匹配需要诊断。

## 1. 方法（照搬 rv/merge.py::build_merged，不碰 Qwen）

| merge.py 步骤 | 本复现 |
|---|---|
| `BR.Text(src/tokenizer.json)`（原始 Qwen 分词器） | 读取发布版扩展分词器（484,789 项），在第一个 `<|reserved_0|>` 处截断（T.n=248,070，T.bytes 同步截断）恢复原始词表视图；TCO/TCC=[248058,248059]、260 个 printable、236,469 个芯片 token 名均校验一致 |
| `core.build(Target(16,32,9216), T, store)` | `store = Config(**chipformer.json['store'])`（slot_bits 10, n_chunks 512, n_names 512, hash_bits None, seed 2035） |
| `allocate(c, None, keep, 9216)` | 同（clearable=None），slots=1806 |
| `Layout(4608, off=2560, m=18, slots)` | 同 |
| `emit_layers(c, lay, tc, 0, 'model.', alloc, keep)` → bf16 | 同；tc 取发布版 config.json（emit 只读 hidden/nk/nv/intermediate/conv_kernel，与合并时一致） |
| final norm 芯片维度：-1；code=CODE_GAIN；id=4·CODE_GAIN；tau=2^60 | 同 |
| 芯片行 `rv.tiny.embedding_matrix` | 同；另用精确路径 `emit.embedding_rows` 抽 1,991 行交叉验证 |
| 文本行 `text_rows(c, lay, T, base_emb, v0)` | base_emb 取发布嵌入前 248,320 行（只用到 Qwen 维 [0:2560]），只比较芯片维 [2560:] |

比较：bf16 按 int16 位模式逐元素比较（可区分 ±0、NaN 载荷），同时记录 torch.equal、非零数、每个发布张量的 SHA-256。

**判别力对照：**
1. 比较器自检：单个 bf16 最低位翻转、写入 -0.0 均被检出。
2. 负对照：store seed 2035→2036。电路 ffn_units 47532→47602（stats 不再匹配）；**第 5 层 MLP 的 gate/up/down 不匹配**（68,390 / 112 / 919 个元素，集中在 MLP 单元行 7309–8493，即 store 的哈希/名称检测单元），其余 11 层仍一致。审计对单个配置字段敏感，且能定位到层/张量/行列。

## 2. 复现命令
```powershell
cd C:\Projects\Qcompute
$env:PYTHONPATH = 'C:\Projects\Qcompute\repro\qwenputer\experimental\riscv'
$env:HF_HUB_OFFLINE = '1'
$py = 'C:\Projects\Qcompute\repro\qwenputer\experimental\riscv\.venv\Scripts\python.exe'
& $py -u analysis/rebuild/rebuild_chip.py
& $py -u analysis/rebuild/rebuild_chip.py --skip-emb --store-seed 2036 --out analysis/rebuild/report_negcontrol_seed2036.json
```
注意：仓库根 `.venv`（Python 3.12）没有 numpy/torch，必须用 riscv 子目录的 `.venv`。
内存：emit_layers 一次返回 12 层 fp32（每层约 2.03 亿 float ≈ 0.81 GB，共约 9.8 GB），随后逐层转 bf16、比较、释放；文本行阶段约 2×2.3 GB。估计峰值约 12 GB（< 20 GB）。（Windows 峰值探针未成功，report.json 中 memory 为空，此为按张量尺寸估算。）

## 3. 逐层结果（每层 14 张量，发布版无多余键）

| 层 | 逐位一致 | 一致张量 | 非零权重 |
|---|---|---|---|
| 0 | ✅ | 14/14 | 20,419 |
| 1 | ✅ | 14/14 | 99,702 |
| 2 | ✅ | 14/14 | 40,915 |
| 3 | ✅ | 14/14 | 57,885 |
| 4 | ✅ | 14/14 | 158,507 |
| 5 | ✅ | 14/14 | 124,487 |
| 6 | ✅ | 14/14 | 164,046 |
| 7 | ✅ | 14/14 | 302,964 |
| 8 | ✅ | 14/14 | 143,916 |
| 9 | ✅ | 14/14 | 59,118 |
| 10 | ✅ | 14/14 | 46,543 |
| 11 | ✅ | 14/14 | 70,602 |

张量：input/post_attention_layernorm [4608]（全 0）、A_log [32]（0）、dt_bias [32]（−110）、linear_attn.norm [128]（1）、conv1d [8192,1,4]、in_proj_qkv [8192,4608]、in_proj_z [4096,4608]、in_proj_a/b [32,4608]、out_proj [4608,4096]、gate/up_proj [9216,4608]、down_proj [4608,9216]。合计约 129 万个非零值，其余全为零。

| 其他检查 | 结果 |
|---|---|
| circuit stats (12 layers / 3753 dims / 164 heads / 47532 ffn units / max_units 9216 / max_heads 16) | ✅ 与 chipformer.json 一致 |
| slots 1806、code_dims 2710–2730、tau 2733、mu 2732、id_code_dims 3146–3163 | ✅ |
| model.norm.weight[2560:]（2048 维） | ✅ 逐位一致 |
| 芯片 token 行 [248320:484789]（236,469×4608，非零 11,782,899） | ✅ 逐位一致 |
| embedding_rows 精确路径抽样 1,991 行 | ✅（快速路径 = 精确路径） |
| 对齐填充行 [484789:484800] | ✅ 全零 |
| 文本行芯片维 [0:248320, 2560:]（非零 18,674,331） | ✅ 逐位一致 |

## 4. 不匹配诊断
**无不匹配。** 所有候选原因均被排除：
- 布局/分配差异：slots、关键维度位置、全部张量一致；
- 浮点发射差异：emit 在 fp32 构造，`check_bf16` 保证每个权重本身就是 bf16 值，fp32→bf16 无损，与 CPU/GPU、torch 版本无关（本次 CPU 重建，原发布多半在 GPU 机上构建，仍一致）；
- 分支代码漂移：rv-store 副本即发布所用代码（对芯片部分而言）；
- 训练/修改：任何梯度更新都会改动 bf16 位，不存在。

唯一的非确定成分 `fit_anti_sink`（q35/mute.py，前向拟合）只写 Qwen 层（12–43）里受 mu 维（2732）门控的静音常数，不影响芯片层，也不在本次审计范围内。

## 5. 范围与局限
- 已证明：芯片层 0–11、norm 芯片维、所有芯片行、文本行芯片维 == 编译器输出。
- 未证明（需要原版 Qwen3.5-4B 基座）：Qwen 层 12–43 是否 = 基座加宽 + norm 重缩放 + mute；嵌入 Qwen 维 [0:2560]（本次取自发布张量，自我引用）；norm 的 Qwen 维。下一步：下载基座后逐张量做 widen 比较，mute 涉及的行列单独标为"拟合常数"。
- 上游 repro 目录不是 git 仓库，无 commit，用源码哈希代替。

## 6. 生产级"编译器即安装器"需要什么
1. **固定编译器**：pin commit，或至少源码哈希。本次 SHA-256 前缀：dsl/graph.py 3B15E44C40B1AA56、q35/emit.py 8228710971822FA9、q35/ir.py B45494FFC0C92CB7、q35/merge.py 34C2D4C7B02C145A、q35/tiny.py 8673EED6D8484307、rv/bridge.py 4F48324DBCF265A8、rv/core.py 6DC467F0AD9189AD、rv/merge.py A092813899DB9321、rv/spec.py 1B11CF800538DB85、rv/store.py 8FCDB0329195AAE6、rv/store_rv.py 3141F682023DA5F0、rv/tiny.py A4A8B60EA7D8B5A9；chipformer/**/*.py 全树哈希 47B113EE1928B1557FBE08AB128F0170。
2. **完整配置**：Target(16,32,9216)、store.Config（尤其 seed：负对照表明一个种子就会改第 5 层）、hidden/off/m、clearable=None、keep。chipformer.json 已基本足够，建议补充 Target 三元组、clearable 策略、编译器版本/哈希。
3. **分词器**：编译依赖原始 Qwen 词表字节（文本行字节位、TCO/TCC、printable、换行集合、store 检测器）。需 pin 原始 tokenizer.json 哈希，或像本次一样从扩展分词器截断恢复。
4. **torch 版本**：对芯片部分不敏感（全是精确 bf16 常数、无归约运算），但仍应记录 torch/numpy 版本。
5. **anti-sink/mute 常数**：唯一的拟合成分，应作为单独、带哈希的数据补丁随配方发布（或固定拟合设备/种子），不能声称由编译器复现。
6. **安装流程**：原版 Qwen + 编译器 → 本地编译芯片层/嵌入芯片维/norm 芯片维 → 应用 widen + mute 补丁 → 对照逐张量 SHA-256 清单校验（report.json 已为每个芯片张量记录 sha256_pub，可直接作清单）。这样无需分发约 4.5 GB 的芯片权重（其中 99.9%+ 是零），第三方也能独立核验。

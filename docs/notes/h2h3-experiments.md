# H2 / H3 实验记录：name-key 别名统计与 token 交叉点

> 状态：完成（只用参考实现 `store.py` 的协议与 Qwen3.5 tokenizer，未加载 22 GB 检查点，未构建或合并模型）。
> 原始数字：`analysis/experiments/h2h3_results.json`（键 `H2`、`H3`、`capacity`）。

## 0. 结论摘要

- **别名（H2）**：真实风格的名字（snake_case、`s1..s200`、常用短名）在 1600 个名字中**别名率为 0**；只有刻意构造的“长公共前缀”组（6 token 以上共享前缀，400 个）全部别名，占总语料 20%。别名会让 `put` 覆盖旧值，使 `get` 返回他人文本。
- **探测数（H2）**：真实名字在 α=0.1/0.25/0.5 下，命中平均探测 1.05/1.17/1.49，未命中 1.11/1.39/2.48，与线性探测解析值（1.055/1.167/1.5；1.117/1.389/2.5）吻合。
- **get 的 chip token 开销（H2）**：1 次 SP 探测 + 1 次 SG（单块）= 2.05（α=0.1）～ 2.49（α=0.5）chip token；加上 Qwen 写的调用与回显文本见 H3。
- **容量耗尽（H2）**：512 名字上限在“按不同键创建”512 次后即达到；把 2000 名字全部随机插入时，平均 512 个被创建、350 个因别名被覆盖、1138 个被拒（full）。真实场景需要 ≥512 个不同键的名字才会耗尽。
- **Token 成本（H3）**：`get('user_city')` 的 Qwen 调用 24 token，回显包装 11 token，总上下文 = 24 + 11 + L，L=10/20/40 时分别 45/55/75 token。
- **list 与 RAG 的交叉点 n\***：仅 list 时 n\* ≈ 141；list + 一次 get（目标 L=20）时 n\* ≈ 125（L=10 为 128，L=40 为 119）。落在文档预测的 100–150 区间内。每个名字的 list 输出平均 3.55 token（文档假设 4）。RAG 基线 = 20（q 假设）+ 5×100 = 520 token。
- **容量（capacity）**：参考实现的逻辑状态 = 1,104,896 bit = **138,112 B（约 135 KiB）**；同样 512×104=53,248 token 的文本约 160,787 B（约 157 KiB，按 3.02 B/token 实测）；同样 token 数的 KV cache = **1,744,830,464 B（1.625 GiB）**，即 KV 约为 store 逻辑状态的 **12,600 倍**。
- **注意**：`docs/research/parametric-memory.md` 中“8 层全注意力、KV 32 KB/token”与 config.json 一致（8 层 full_attention，36 层 linear_attention，kv_heads=4，head_dim=256，bf16），无需修正。

## 1. 方法与来源

| 项 | 来源 |
|---|---|
| 键（6 token + 长度）、`name_key` | `repro/qwenputer/experimental/riscv/chipformer/rv/store.py` 的 `StoreScan.name_key`、`key_bits` |
| 哈希（10 位 GF(2) 矩阵，`RandomState(2035)`） | `store.py` 的 `Config.hash_matrix`、`Config.slot_of` |
| 线性探测、墓碑 | `store.py` 的 `respond()` 中的 SP 循环 |
| 调用 token 序列 | `store.py` 的 `StoreSession.call_tokens`（list/get 的 `respond` 布局） |
| tokenizer | `.data/Qwen3.5-4B-Qwenputer-Filesystem/tokenizer.json`（`tokenizers` 库） |
| 模型配置 | `.data/Qwen3.5-4B-Qwenputer-Filesystem/config.json`（只读） |

纯 Python 副本在 `analysis/experiments/common.py`，并断言 `<tool_call>` 等特殊 id 与 tokenizer 一致。副本没有 import `store` 包（它依赖 torch 与 bridge），因此不需要加载检查点。

### 文件

| 文件 | 作用 |
|---|---|
| `analysis/experiments/common.py` | 协议副本：常量、`key_bits`、`Config.slot_of`、`name_key_from_tokens`、`Directory`（线性探测）、`call_tokens`、回显布局 |
| `analysis/experiments/h2_names.py` | 2000 名字语料（snake / s1..s200 / 长前缀组 / 常用短名） |
| `analysis/experiments/h2_alias.py` | H2：别名、探测、get 芯片 token、耗尽、常用名安全性 |
| `analysis/experiments/h3_cost.py` | H3：get/list 的精确 token 数、RAG 基线、n\* |
| `analysis/experiments/capacity.py` | 容量：store 逻辑状态、文本字节、KV cache、GDN 固定状态 |
| `analysis/experiments/h2h3_results.json` | 所有原始数字 |

### 语料（2000 名）

| 类别 | 数量 | 说明 |
|---|---|---|
| snake | 770 | 2–3 个词用 `_` 连接（如 `user_city`、`dog_name`），与 bench 不重复 |
| short_sN | 200 | `s1`..`s200` |
| adversarial_prefix | 400 | 40 组，每组 10 个，共享 >6 token 的前缀，只在末尾不同 |
| bench_short | 630 | 63 个常用词（city、job、name、age…）× 10 种形式（`{x}`、`my_{x}`、`user_{x}`、`{x}_2`…） |

实际语料去重后正好 2000 个（`assert` 通过）。realistic 子集 = 除 adversarial 外的 1600 个。

## 2. H2：name-key 别名与探测

### 2.1 语料层面

| 类别 | 名字数 | 与他人共享键的名字 | 别名率 |
|---|---|---|---|
| snake | 770 | 0 | 0% |
| short_sN | 200 | 0 | 0% |
| bench_short | 630 | 0 | 0% |
| adversarial_prefix | 400 | 400 | 100% |
| **合计** | 2000 | 400 | **20%**（全部来自刻意构造的组） |

- 不同键数：1640；最大键组大小 10（即一组 10 个名字全部别名）。
- token 长度直方图：1–5 token 为真实名字（bench 平均 2.19 token），12–15 token 为长前缀组。
- 630 个常用短名：与语料中任何其他名字都**没有键冲突**，全部 ≤5 token，因此 `list` 不会截断它们。Qwen 常用的 `city`、`job` 等在此语料中是安全的。（哈希**槽**仍会与其他名字共享，约 82% 的常用名与某个名字共享一个槽，属于正常的线性探测，不是别名。）

### 2.2 负载因子实验（50 次随机子集，每次按随机顺序插入）

N = α·1024；α = 0.1 / 0.25 / 0.5 → N = 102 / 256 / 512。

**(a) 全部 2000 名（含长前缀组）**

| α | N | 子集别名率 | 错误 get 率（返回他人文本） | 命中探测 | 未命中探测 | 未命中假命中率 | 覆盖次数/次 |
|---|---|---|---|---|---|---|---|
| 0.1 | 102 | 7.3% | 3.9% | 1.048 | 1.100 | 7.6% | 4.0 |
| 0.25 | 256 | 14.6% | 8.6% | 1.127 | 1.272 | 14.5% | 22.1 |
| 0.5 | 512 | 18.5% | 12.6% | 1.310 | 1.824 | 18.8% | 64.6 |

**(b) realistic：去掉长前缀组（1600 名）——这是“真实名字”的答案**

| α | N | 别名率 | 错误 get 率 | 命中探测 | 解析命中 | 未命中探测 | 解析未命中 | 假命中率 | get 芯片 token（命中，= 探测 + 1 SG） |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 102 | 0 | 0 | 1.054 | 1.055 | 1.111 | 1.117 | 0 | 2.05 |
| 0.25 | 256 | 0 | 0 | 1.167 | 1.167 | 1.385 | 1.389 | 0 | 2.17 |
| 0.5 | 512 | 0 | 0 | 1.490 | 1.500 | 2.481 | 2.500 | 0 | 2.49 |

- 未命中 get 的芯片 token = 未命中探测数（不输出 SG）。
- 解析值来自线性探测公式：命中 ½(1+1/(1−α))，未命中 ½(1+1/(1−α)²)。

### 2.3 耗尽：512 名字上限

- 把 2000 名字随机顺序全部 `put`（50 次）：平均 **512 个被创建**（达到 `n_names=512` 上限），**350 个因别名覆盖**，**1138 个被拒（full）**。
- 结论：耗尽需要 **≥512 个不同键**的名字。在真实名字下这意味着 α=0.5 的 1024 槽表恰好在上限处满；别名越多，越早被覆盖而不是创建。
- 对 Qwen 实际用法（几十条事实）远低于上限。

## 3. H3：get / list 的 token 成本与 RAG 交叉点

### 3.1 get(name) 的精确 token 数

名字 `user_city`（2 token）。调用序列由 `call_tokens("get", name)` 生成，回显由 `get_response_tokens` 生成。

| 部分 | token |
|---|---|
| Qwen 写的调用（`<tool_call>`、`<function=get>`、`<parameter=name>`…） | **24**（其中包装与 op 22，名字 2） |
| 回显包装（`\n<tool_response>\n`、`</tool_response>\n<think>…`） | 11 |
| 打印的内容 | L |

| 内容长度 L | 调用 | 回显包装 | 回显总计 | **get 总上下文 token** | chip token（命中，LF 0.1 / 0.5） |
|---|---|---|---|---|---|
| 10 | 24 | 11 | 21 | **45** | 2.05 / 2.49 |
| 20 | 24 | 11 | 31 | **55** | 2.05 / 2.49 |
| 40 | 24 | 11 | 51 | **75** | 2.05 / 2.49 |

chip token 只包括 SP 探测与 SG，不计入 Qwen 的文本 token。

### 3.2 list() 与读 n 个名字

list 调用 12 token；回显包装 11 token；每个名字打印 **min(length, 6)** 个键 token（含换行），realistic 名字平均 **3.55 token/名**。

| n 个名字 | list 调用+回显 | list + get(目标, L=10) | list + get(目标, L=20) | list + get(目标, L=40) | RAG（520） |
|---|---|---|---|---|---|
| 1 | 26.6 | 72.1 | 82.1 | 102.1 | 520 |
| 5 | 40.9 | 86.5 | 96.5 | 116.5 | 520 |
| 10 | 58.6 | 104.1 | 114.1 | 134.1 | 520 |
| 20 | 94.4 | 140.0 | 150.0 | 170.0 | 520 |
| 50 | 200.6 | 246.2 | 256.2 | 276.2 | 520 |
| 100 | 379.1 | 424.7 | 434.7 | 454.7 | 520 |
| 200 | 732.8 | 778.3 | 788.3 | 808.3 | 520 |
| 400 | 1442.6 | 1488.2 | 1498.2 | 1518.2 | 520 |

（每个 n 为 200 次随机抽样的均值；list + get 的目标取抽样中的第一个名字。）

### 3.3 RAG 基线

- k=5，c=100：5×100 = 500 token 粘贴进提示；查询 q 按文档假设取 **20**，合计 **520**。
- 一个典型问题（“What city does the user live in, and what is their dog's name?”）实测 16 token，因此 q=20 是保守取值。
- 不计回答 token，两边都不计。

### 3.4 交叉点 n\*（最小的 n，使 list 方案 ≥ RAG）

| 方案 | n\*（采样） | 解析近似 | 文档预测 |
|---|---|---|---|
| 仅 list | **141** | 140 | 100–150 ✓ |
| list + get，L=10 | 128 | — | 100–150 ✓ |
| list + get，L=20 | **125** | 124.5 | 100–150 ✓ |
| list + get，L=40 | 119 | — | 100–150 ✓ |

- 文档的斜率假设是 4 token/名字；实测 **3.55 token/名字**（打印的键 token），交叉点因此略靠后。
- 因此 `parametric-memory.md` 中“n_names≳120”的判断在这组测量下成立（实测 119–141）。

## 4. 容量（逻辑状态 vs 文本 vs KV）

来源：`config.json`（8 层 full_attention，36 层 linear_attention，num_key_value_heads=4，head_dim=256，bf16）；store 的 Device/Entry 字段（`store.py`）。

| 项 | 字节 | 说明 |
|---|---|---|
| store 逻辑状态（内容 18 bit × 512×104） | 119,808 | 958,464 bit |
| 目录（1024 槽 × (键 111 + alive 1 + head 9 + tail 9)） | 16,640 | 133,120 bit |
| order（512 × 10） | 640 | |
| 长度（512 × 7）+ 指针（512 × 9） | 448 + 576 | |
| **store 逻辑状态合计** | **138,112**（约 134.9 KiB） | 1,104,896 bit |
| 文本（53,248 token × 3.02 B/token，实测于 parametric-memory.md） | 160,787（约 157 KiB） | |
| KV cache（53,248 token × 32,768 B/token） | **1,744,830,464**（约 1.625 GiB） | 8 层 × 2 × 4 × 256 × 2 B |
| GDN 固定循环状态（36 线性层 × 32 头 × 128×128 × fp32） | 75,497,472（约 72 MiB） | 与文件数无关；store 只用其中一部分头，store.py 未给出物理布局 |

- KV 与 store 逻辑状态之比 ≈ **12,600×**；KV 与文本之比 ≈ **10,850×**；文本与 store 逻辑状态之比 ≈ 1.16。
- 注意：store 逻辑状态是参考字典中实际存的 bit 数，不等于 GDN 中的物理占用（打包、fp32 与 bf16 的差异都未测）。这里的数字是**逻辑下界**。

## 5. 局限与假设

1. 名字单独编码（`encode(name)`），没有前导空格或上下文；Qwen 实际写出的 token 可能略有不同。
2. 哈希矩阵由 `numpy.random.RandomState(2035)` 生成（与 `store.py` 的 `Config.hash_matrix` 相同的调用）。本次只使用了 `common.py` 中的副本，未与 `store.py` 的实际 `Config` 对象逐项比对 key→slot 映射；这是一项待补的核对。
3. chip token 数只按协议（SP/SG）计数，未经 RV32 电路验证。
4. list 假设列出全部存活名字；get 假设内容 ≤ 104 token（单块，CMAX）。
5. RAG 的 q=20 为假设，不计模板开销。
6. 长前缀名字（≥6 token）在 `list` 中被截断为前 6 个键 token，且它们与其他名字别名——这是协议的已知限制，不是测量误差。

## 6. 复现命令

在仓库根目录 `C:\Projects\Qcompute` 下运行（PowerShell / cmd）：

```bat
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PY=repro\qwenputer\experimental\riscv\.venv\Scripts\python.exe
%PY% analysis\experiments\h2_alias.py
%PY% analysis\experiments\h3_cost.py
%PY% analysis\experiments\capacity.py
```

顺序：先 H2（写入 `H2`），H3 和 capacity 各自写入自己的键。`h3_cost.py` 依赖 `h2_names.py`（同目录导入）。脚本未使用 `-I`，因为要导入同目录的 `common`；它们是本仓库自己的代码，不是不可信输入。

最小冒烟测试（应输出 `(2, (8656, 198, 510, 0, 0, 0))`）：

```bat
%PY% -c "import sys; sys.path.insert(0,'analysis/experiments'); import common as c; t=c.load_tokenizer(); print(c.name_key_from_tokens(c.encode(t,'city')))"
```

运行时间与峰值内存：本次未测量（脚本只加载 tokenizer 与小的 numpy 矩阵，未加载检查点）。

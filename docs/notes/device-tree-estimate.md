# Device-tree 原型估算（Stage-3 idea B）

> 结论先行：**完整器件已实际构建并验证**（不是纯估算）。三个小型专用器件（ADD8 / SORT4 / DADD）各自只需 **2 层、1 个 head、82–437 个 MLP 单元、0.9k–3.9k 非零权重**，在 IR（bf16 精确性检查）与 HF 独立 Qwen3.5 checkpoint（argmax）上均 **0 错误**。相对 RV32 核心（11–12 层、1,040,595 非零），器件约 **小 270–1150 倍（非零）、浅 5–6 倍（层）**；对固定形状、位宽有界的纯函数运算，每次操作的 trace token 降低 1–2 个数量级。标注「估算」的数字是解析推算，未构建。

代码：[device_graphs.py](<../../analysis/device/device_graphs.py>)（图构建），[device_estimate.py](<../../analysis/device/device_estimate.py>)（测量脚本），结果 [device_estimate.json](<../../analysis/device/device_estimate.json>)。

复现（CPU，约 60 s）：

    repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe analysis/device/device_estimate.py --hf

## 1. 方法

- IR 使用 `repro/qwenputer/experimental/riscv/chipformer/q35/`（ir.py / prims.py / emit.py / tiny.py），未修改任何 repro 源码。
- 约定沿用 chip8_q35 / rv core：
  - 每个 token 带输入 code 特征 `CODE_AMP·(2bit−1)`（tied lm_head 读回）；
  - 最后一个 MLP 子层是 **exit**：写入下一个 token code 的 `KAPPA·(2bit−1)`（只在器件「拥有」的位置），emit 时清掉其余 chip 维和 ballast；
  - 输入 code 必须在**更早一个** MLP 子层清零（`code.clear`）：`−e` 与 `±KAPPA` 无法在一个 bf16 MLP 输出中精确相加（第一次尝试 1 层 ADD8 正是因此被 `ir.evaluate` 拒绝）。所以「计算」1 层即够的器件，按 I/O 约定也是 **2 层**；若并入已有宿主（宿主已在早层清 code），计算部分可只占 1 层。
- 非零权重计数与 `chipformer.rv.measure.nonzeros` 完全相同：`emit_layers` + `tiny.pick_layout`，统计所有 `!= 0` 的元素（包含每层 144 个常数：dt_bias 16 + norm.weight 128），与 RV32 的 1,040,595 口径一致。另给出按图项的解析计数（graph_term_estimate）作交叉检查。
- 精确性：
  1. `ir.evaluate(check=True)`：teacher-forced，检查每个中间量是 bf16、silu 在精确区、key one-hot 等；每个拥有位置解码出的下一个 token 必须等于记录中的 token；
  2. 每个器件前 8 条记录用 `kernel=True` 重放 HF torch Gated DeltaNet kernel 的读法；
  3. `tiny.build(..., share=True)` 生成独立 Qwen3.5 模型（slot 共享 + exit 清除），bf16 前向，检查 argmax。

## 2. 器件设计

| 器件 | 记录（trace） | L1 attn | L1 mlp | L2 mlp (exit) |
|---|---|---|---|---|
| **ADD8** 8+8→9 位 | `A(a) B(b) S(a+b)`；token A 0–255、B 256–511、S 512–1023 | 上一 token head（conv lag 1 把 a 的位带到 B 位置） | code clear | 超前进位加法器直接折叠进 exit：`c_i=[a mod 2^i + b mod 2^i ≥ 2^i]` 是线性式上的一个阈值（2 单元），`s_i=a_i+b_i+c_i−2c_{i+1}` 线性 |
| **SORT4** 4×8 位稳定排序 | `X0 X1 X2 X3 Y0 Y1 Y2 Y3`（Xk/Yk 为位置化 token 族，各 256） | 快照 head：在 X3 处用 conv lag 0..3 一次写入 4 个输入（32 数据列） | 6 个比较器 `d_ij=[x_i≤x_j]`（各 2 单元）；rank 线性；code clear | 对槽 k 选 `rank_i==k` 的 x_i（`t_eq` 带 lits） |
| **DADD** 十进制逐位大数加法 | `G P(a0,b0) S(z0) P(a1,b1) S(z1) …`，LSB 先；P 0–99，S 100–119（z=a+b+cin，打印 z mod 10，进位 z≥10），G=120 记录起点 | 上一 token head（取上一个 S 的进位） | code clear | z 的各位（`t_eq`×20） |

注：把 RV32 的不同 token 族用作输入槽位（Xk、Yk）的手法，即用词表换深度；ADD8 的 a、b、和直接作为单 token，因此只适用于小位宽（见 §6）。

## 3. 测量结果（实测）

| 指标 | ADD8 | SORT4 | DADD | RV32 核心（参考） |
|---|---|---|---|---|
| 层数 | **2** | **2** | **2** | 11（P2）/ 12（P3） |
| IR dims | 37 | 66 | 14 | 2,679 |
| heads | 1 | 1 | 1 | 99 |
| FFN 单元 | 82（22 + 60） | 437（34 + 403） | 179（16 + 163） | 42,012 |
| 单子层最大单元 | 60 | 403 | 163 | 8,149 / 9,216 |
| **非零权重（层）** | **899** | **3,876** | **1,454** | **1,040,595** |
| 其中常数（dt_bias / norm） | 288 | 288 | 288 | — |
| 图项解析估计 | 607 | 3,588 | 1,166 | — |
| 独立模型 hidden | 64 | 128（IR 布局）/ 64（slot 共享后） | 64 | 2,816 |
| 独立 HF 模型总非零（含 embedding） | 18,714 | 40,978 | 3,287 | — |
| 相对 RV32 非零 | 1/1157 | 1/268 | 1/716 | 1 |

ADD8 位宽扫描（同一结构，非零含 288 个常数）：1 位 370、2 位 421、4 位 547、8 位 899 → 每位约 **+66 非零、+9 单元**，层数恒为 2。

## 4. 精确性检查

| 器件 | IR teacher-forced（check=True） | kernel 重放 | HF 独立模型 argmax |
|---|---|---|---|
| ADD8 | **穷举 65,536/65,536 记录，0 错** | 8 条通过 | 40 条，0 错 |
| SORT4 | 3,508 条（8 条边界：全等、升/降序、0/255 交替、重复值；3,000 随机；500 条值 0–3 的高重复），14,032 次预测，0 错 | 8 条通过 | 40 条、160 位置，0 错 |
| DADD | 11,039 条（全部 2 位×2 位 10,000 种；4/8/12/24/40 位随机各 200；`10^n−1 + 1` 长进位链 n≤39），38,261 次预测，0 错 | 8 条通过 | 60 条（含 40 位）、1,619 位置，0 错 |

记录：DADD 第一版没有记录起点 token，连续拼接的记录会把上一条最后的进位带进下一条（5,553/38,261 错；单条独立的 HF 检查是干净的）。加入 G 起始 token（进位特征为 0）后为 0 错。这说明串行器件**需要显式的记录边界或复位**。

## 5. 每次操作的 trace token 成本

| 运算 | 专用器件（实测记录格式） | RV32 核心 | CHIP-8 v15 |
|---|---|---|---|
| 8 位加法（操作数已在 trace 中） | **3 token**（器件生成 1 个） | 1 条 ADD ≈ 1.3–1.8 token；从 trace 取数/写回另需数条指令（估算 5–10 token） | 3–5 token/指令，同量级 |
| 4 元素排序（8 位） | **8 token**（器件生成 4 个） | 估算约 30–40 条指令（6 次比较交换 ≈ 20 + 8 次 load/store + 调用）≈ **45–70 token**（估算） | 估算 100+ token |
| 十进制加法 4 位 + 3 位 | **9 token**（2n+1；器件生成 4 个） | calc 调用 254 token | 约 54（含 CALL） |
| 十进制加法 12 位 + 12 位 | **25 token**（器件生成 12 个） | 303 token | 约 113 |
| 十进制加法 40 位 | 81 token | 估算 600+ | — |

对 DADD 这一点必须加注：RV32 的 254–303 token 从「RAM 里的文本」算到「打印完」，包含解析、调度（约 200–250 token 的 C 调度开销）和 MSB 优先的打印；DADD 的记录假设有人已把操作数变成 **LSB 优先的逐位配对 token P(a_k,b_k)**。Qwen 本身难以可靠地产生倒序配对，所以真实接入需要一个捕获缓冲 + 按位对齐读取（类似 RV32 的 0x8000 文本缓冲或 CHIP-8 的参数捕获）：**估算再加 1–2 层、1–3 个 head、数千非零**，token 成本约为 操作数文本 + n 个输出 token（MSB 优先输出还需对全部位做进位前缀：并行前缀 ≈ log2(n) 级 MLP，估算 40 位内 +2–3 层）。即使如此，估算的端到端成本仍约为 **n 位：~2n–3n token，层数 ~5–7**，远低于 RV32 的 254–684 token/11–12 层。

## 6. 何时专用器件胜过 RV32 核心（verdict）

**胜出（建议做成器件）：**
1. **固定形状、无数据相关循环的纯函数**：加/减、比较、小排序网络、查表、位操作、CRC/校验和的单步、BCD 转换的单步。深度 = 依赖链长度（本次 1 级阈值 + 1 级 exit），与 RV32 的 11 层无关。
2. **可写成逐 token 串行的 digit/limb 运算**（DADD 型）：每个 token 的工作只依赖少量最近 token（conv 窗口 ≤4 或一个 head 槽），状态只有进位之类的少数位。大数加/减、逐位比较、乘法的单行（RV32 已有的 BD 行操作本质上就是这种器件）都属此类。代价随位宽线性，层数恒定。
3. **高频热点**：器件把 RV32 每次调用 200–250 token 的调度/寄存器簿记成本直接去掉；在 calc 场景里这才是大头（数字工作本身 RV32 已是约 1 token/位）。

**不胜出（留在 RV32 核心）：**
1. **数据相关控制流 / 无界循环**：gcd、pow、除法的商位搜索链、通用解析、任意程序。器件需要自己实现 PC/调度，即退化为 CPU。
2. **需要大内存或随机访问**：每 16 个 head 一层子层，ROM/RAM 是 RV32 的主要层成本，器件不会更便宜。
3. **宽位单 token 操作数**：ADD8 用单 token 携带 8 位操作数，门限小工具的范围限制是 256 步（`t_ge` 的 hi−theta+1≤256），且每族 token 词表 = 2^位宽。32 位需 nibble/byte 串行（回到 DADD 型）或二级超前进位（估算 +1 层、~4–5k 非零）。
4. **种类繁多、低频**：每个器件都要新的 token 族（ADD8 1,024 行、SORT4 2,048 行、DADD 121 行 embedding；在 4B 合并模型中每行是 hidden 3,968 宽，词表成本可能超过器件本身的非零），并需要桥接/调度逻辑决定何时由器件「拥有」位置。

**合并方式估算**：器件最大子层只用 1 head + ≤403 单元；RV32 在 4B 目标上每子层上限 16 key head / 9,216 单元、最满子层用到 8,149 单元，所以一个器件可以**与核心并行驻留在同一批 chip 层中，边际层数 ≈ 0**（估算；前提是在相应子层有剩余单元/head，且 owner/exit 与核心的 exit 合并成同一个输出 code 多路选择）。「device tree」= 共享 chip 层、按 token 族分派的若干小器件 + RV32 作为通用回退，是可行的。

## 7. 注意事项

- 层数 2 中有 1 层只为 I/O 约定（输入 code 清零 + exit）；并入宿主后可与宿主的 exit 共享。
- 非零口径与 rv/measure 一致（含常数）；独立 HF 模型的「含 embedding」非零主要来自词表行（ADD8 1,024 个 token × ~17）。
- token 成本中 RV32 对 8 位加法与 4 元素排序的数字是**估算**（未在 RV32 上实测这两个程序）；DADD 与 RV32 calc 的比较不是同一端到端边界（见 §5 加注），端到端的器件数字（含文本捕获/对齐/MSB 输出）为**估算**。
- 仅 CPU：HF 前向使用 transformers 5.19 的参考 PyTorch kernel（无 fla / causal_conv1d），只检查了每器件 40–60 条记录；IR 侧检查覆盖更全。
- 未做：与 rv core 的实际合并（merge.py）、Triton 执行器、Qwen 文本侧的桥接。

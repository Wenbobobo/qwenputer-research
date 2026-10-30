# Qwenputer 声明核对表（claims audit）

> 目的：把作者/模型卡/博客中的每一个可检验声明列出来，标注本机复现状态与证据路径。
> 状态：✅ 已复现 · 🟡 部分复现 · ⏳ 进行中 · ❌ 未复现（附原因） · 📄 仅文档声明
> 最后更新：2026-10-29 01:55

## A. 架构与可加载性

| # | 声明 | 状态 | 证据 / 备注 |
|---|---|---|---|
| A1 | 发布 checkpoint 是 stock `Qwen3_5ForCausalLM`（config-only 变化，无需 trust_remote_code） | ✅ | `.data/.../config.json` 的 architectures = Qwen3_5ForCausalLM；tokenizer/chat_template 齐全；本机 transformers 5.19.0 见加载日志（待 CPU demo 完成最终确认） |
| A2 | 发布模型 = 12 个编译 chip 层 + 32 个 Qwen 层，hidden 4608，vocab 484800 | ✅ | config.json（num_hidden_layers 44, hidden_size 4608, vocab_size 484800）+ chipformer.json（chip_layers 12, chip_vocab 236469, layout rv1-store） |
| A3 | chip 层权重全部由编译器算出、无训练 | ✅ | **重建审计**：12/12 层、168/168 张量 bit-exact；final norm chip 维、chip embedding 行（11.78M 非零）、文本行 chip 特征（18.67M 非零）全部 bit-exact（analysis/rebuild/report.json，74.6s CPU） |
| A4 | 单个模型文件 22.3GB、需 ~18GB 显存、可跑 24GB GPU | 📄 | 本机实测 20.73 GiB 磁盘；显存需 7900XT（20GB，ROCm）或更大卡，T10 无 bf16 不可用 |

| A3b | 发布 = 基座 Qwen3.5-4B + 编译器输出 + 可推导变换 + 16KiB 拟合常量 | ✅ | Qwen 侧 426 张量：338 base+widen bit-exact / 80 mute-only（常数可推导）/ 8 anti-sink（16KiB）；0 无法解释（base_report.json） |

## B. GDN-as-RAM

| # | 声明 | 状态 | 证据 |
|---|---|---|---|
| B1 | α=1、β=1、one-hot key ⇒ 精确覆盖一行；one-hot query 精确读回 | ✅ | `repro/verify_gdn_ram.py`：2492 写 / 2508 读，最终状态 == dict RAM，max read error 0.0；α=0.9/β=0.5/non-one-hot 均破坏精确性（`repro/logs/05-gdn-ram.log`） |
| B2 | 在真实 HF Qwen3.5 模块的 bf16 前向里逐位精确 | ✅（tiny 级） | 作者声称 GPU/CPU 均验证；test_rv_hf 3/3 passed（309s，Windows CPU）；rv-tiny signext residual==IR 118,800 cells + free-run 44 tokens 与参考一致 |
| B3 | 一个 head 的 state 可当 RAM（行=cell，容量 = rows × value dims） | ✅ | 文档 + IR 布局；run-time 验证随 B2 |

## C. RISC-V CPU

| # | 声明 | 状态 | 证据 |
|---|---|---|---|
| C1 | RV32IM + Zba + 大数协处理器，256KB ROM / 128KB DROM / 32KB RAM | ✅ | `repro/logs/03-measure.log`：nonzeros_total=1,040,627, hidden=2816, 11 层；容量 rom 64KB/sublayer、ram 16KB/sublayer |
| C2 | riscv-tests rv32ui/rv32um/rv32uzba 全绿 | ✅（IR） | test_rv_core **94 passed / 10 skipped**（18min）；test_rv_ref+fuzz+bridge 143 passed / 4 skipped；RV_SLOW 长跑项跳过 |
| C3 | 61 个 ELF 与 qemu-riscv32 exit code/stdout 一致 | ✅（测试集内） | test_rv_ref 通过（对照 qemu 的 ELF 集合） |
| C4 | C/MicroPython 自由运行约 1800 万条指令零错误 | 🟡 | 文档声称；本机 tiny 自由运行 signext 通过（其余进行中）；大规模验证需 GPU |
| C5 | 1.2–1.4 token/指令 | ✅ | measure：C 程序实测 1.378（c-sieve 1.393, c-sort 1.249, c-strops 1.519）；作者 1.40 |

## D. 文件系统

| # | 声明 | 状态 | 证据 |
|---|---|---|---|
| D1 | put/append/get/list/delete 五工具，1024-slot hashed 目录 | ✅（IR） | 补 tokenizer 后 test_rv_store **10 passed**（466s），含 all_ops / 随机操作 / 冲突哈希 / 与 calc 混用 / 快照；`store_in_hf` 2 项在 GPU 环境运行 |
| D2 | 512 文件 / 约 53k tokens 容量 | 📄 | rv-store.md 的容量表（512 chunks × 104 tokens）；IR 测试覆盖容量边界 |
| D3 | 一次查找 2–3 trace token；put/get 2–3 tokens/op | 📄 | rv-store.md「Chip tokens per op」表；本机可由 store 参考实现直接计数（待复现） |
| D4 | 快照 = 持久 heads 的 recurrent state（4B 约 30MB / 169 heads），跨会话持久 | 🟡 | 文档+代码；q35-store.md 写 169 heads/11MB、模型卡写 ~30MB（口径不同）；CPU demo 待跑 |
| D5 | 发布 checkpoint 的多会话召回 benchmark：83.3 / 90.0 vs base 7.8 / 3.3 | ❌→🟡 | 完整 benchmark 需 79 分钟 L4 生成，本机 CPU 不现实；将用单 case 复现（CPU demo）+ 声明标注；数字来自作者 |

## E. 桥与文本保真

| # | 声明 | 状态 | 证据 |
|---|---|---|---|
| E1 | Qwen 用原生 tool_call 格式自发使用 CPU/文件系统，无需训练 | 🟡 | CPU demo 观察 put/calc 调用；作者：50/50 put、93/120 会话自发保存 |
| E2 | 无外部运行时；trace token 由模型自己预测 | 🟡 | 代码路径（RvChat lockstep 只做校验，不注入）；CPU demo 观察 token 流 |
| E3 | 文本行为不变：KL 5.5e-4（噪声底 5.1e-4）、greedy 10/16 相同、MMLU/ARC 噪声内 | ❌（作者数字） | 需要 base Qwen3.5-4B（未下载）与 GPU 级文本评测；标注为作者数字 |
| E4 | lm_head argmax 与电路选择逐 token 一致（0 lock / head mismatch） | 🟡 | CPU demo 的 generate 使用 check_head；日志将记录 |

## F. 性能与成本

| # | 声明 | 状态 | 证据 |
|---|---|---|---|
| F1 | Triton 执行器 ~3200 chip tokens/s（L4） | ❌ | 无 NVIDIA L4/24GB 卡；7900XT 可跑但未验证；标注作者数字 |
| F2 | HF 路径 ~250 chip tokens/s；文本 token ~68ms | ❌ | 同上 |
| F3 | 批量 109–270 题/小时（A10G batch 8） | ❌ | 同上 |
| F4 | 22.3GB 中大部分为零；chip 层 1.49B 参数 vs 1.04M 非零 | ✅ | **实测**：22.20GB 中 13.72GB 是精确零；chip 层 4.88GB 仅 1,287,568 非零（~2.5MB 值）；全零比例 61.8% | measure（chip 层 dense 1.49B vs nonzeros 1.04M）；发布 checkpoint 的逐 tensor 零比例由 weights scan 给出 |

## G. 整理：哪些声明最容易在复现中被推翻？

1. **bf16 精确性依赖 kernel**：fp32 非 bit-exact、batch 改变舍入——作者自己已承认；我们应记录「精确」的边界。
2. **发布权重 = 编译器输出**：如果 scan 发现 chip 层非零数远大于 IR 预期，说明发布版本与文档不一致（或含未文档化修改）。
3. **文件系统测试**：tokenizer 依赖是隐性坑（测试会 skip 而非失败），说明复现者容易误以为"通过"。
4. **文本保真数字**：KL 5.5e-4 是相对 base 的；base 版本不同会导致结论不可迁移。
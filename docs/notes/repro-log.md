# Qwenputer CPU 复现日志（rv-store 分支，repro/qwenputer）

> 执行时间：2026-10-29 01:35–02:20（Asia/Shanghai）。仅 CPU，无 GPU。上游源码未修改；新增脚本在 `repro/`，日志在 `repro/logs/`。
> 只记录实际跑过的命令与日志行。工作目录（除特别说明）：`repro/qwenputer/experimental/riscv`。

## 1. 环境与版本

| 项 | 值 |
|---|---|
| 主机 | Windows 11 Pro 10.0.26200，16 逻辑核，48 GB RAM，无 GPU 使用 |
| uv | 0.11.29 |
| 系统 Python | 3.12.13（无 numpy；未使用） |
| 项目 venv（uv 依 lock 自选） | **CPython 3.13.14**（`riscv/.venv`） |
| torch | **2.14.1+cpu**（Windows PyPI wheel 本身就是 CPU 版，118 MB，无需额外 CPU index；`cuda False`，threads 8） |
| transformers | 5.19.0（含 qwen3_5；无 fla / causal_conv1d → 走 PyTorch 参考 kernel） |
| numpy / pytest | 2.5.4 / 9.1.1 |

- 命令：`uv sync --extra hf` → 41 包，44 s（`repro/logs/01-env.log`）。
- **Tokenizer**：`test_rv_store` / `test_rv_bridge` 部分用例需要 HF cache 里的 Qwen3.5 `tokenizer.json`，否则整体 skip（首次跑 `test_rv_store` 12/12 skipped，见 `04-pytest-store.log`）。处理：仅下载 `Qwen/Qwen3.5-4B` 的 tokenizer.json / tokenizer_config.json / config.json 到 `repro/hf_home`，之后以 `HF_HOME=repro\hf_home HF_HUB_OFFLINE=1` 运行（`04-tokenizer-download.log`）。未碰 `.data/`。
- 副产物（均在 repro/** 内）：`riscv/.venv`、`riscv/dist/rv-tiny`（4.26 GB safetensors）、`repro/hf_home`、`.pytest_cache`。

## 2. 测试结果表

| 测试 | 命令 | 结果 | 耗时 | 日志 |
|---|---|---|---|---|
| test_rv_ref + fuzz + bridge（无 tokenizer） | `uv run pytest tests/test_rv_ref.py tests/test_rv_fuzz.py tests/test_rv_bridge.py -q --durations=5` | 135 passed, 12 skipped | 100.6 s | 02-pytest-ref-fuzz-bridge.log |
| 同上（有 tokenizer，`-rs`） | 同上 + `-rs`，HF_HOME=repro\hf_home | **143 passed, 4 skipped**（skip 原因：`test_rv_ref.py:30: long run (RV_SLOW=1)`） | 385.6 s | 02-pytest-ref-fuzz-bridge-tok.log |
| test_rv_core | `uv run pytest tests/test_rv_core.py -q --durations=5` | **94 passed, 10 skipped**（skip 为上游设定：`checked free-running by the executor` / `long trace (RV_SLOW=1)`） | 1090 s（与其它作业并行） | 02-pytest-core.log |
| test_rv_hf | `uv run pytest tests/test_rv_hf.py -q` | **3 passed**（signext prefill/decode、call prefill） | 309.5 s | 02-pytest-hf.log |
| test_rv_store（IR，10 项） | `uv run pytest tests/test_rv_store.py -x -v --durations=0 -k "not store_in_hf"` | **10 passed** | 466.3 s | 04-pytest-store-ir.log |
| test_rv_store::test_store_in_hf[prefill] | `... -k store_in_hf` | **PASSED** | — | 04-pytest-store-hf.log |
| test_rv_store::test_store_in_hf[decode] | 同上；单独重跑加 `-X faulthandler` | 第一次进程无 traceback 退出（exit=-1，400 s）；**重跑 PASSED** | 433.1 s | 04-pytest-store-hf.log, 04-pytest-store-hf-decode-retry.log |

**与上游文档的差异**：rv-store.md 记录 `test_rv_hf` 在 Windows torch 2.14.1+cpu 上 access violation 崩溃；**本机（venv 为 Python 3.13.14）3/3 通过，未复现该崩溃**。但另有两次 HF 长时 decode 进程静默退出（exit=-1、无 Python traceback、Windows 事件日志无 python 崩溃记录），见 §7。

## 3. IR 电路规模（`03-measure.log`）

命令：`uv run python -m chipformer.rv.measure --nonzeros`（36.6 s）

- stats：**11 layers，2,679 dims，99 heads，42,014 MLP units**（单子层最多 8,149 / 9,216；最多 16 heads）
- **nonzeros_total = 1,040,627**，standalone hidden = 2816
- 每层非零：L0 3,791 / L1 97,596 / L2 40,832 / L3 57,803 / L4 147,273 / L5 51,319 / L6 163,502 / L7 301,805 / L8 142,290 / L9 10,478 / L10 23,938
- heads：L3/L4 ROM 16+16，L7/L8 RAM 16+16，L9 DROM 16，L5 regs 3 + quo 4，L6 bn 8，L1 pc/count/j/vcnt 4
- capacity：ROM 65,536 B/子层（×4 = 256 KB），RAM 16,384 B/子层（×2 = 32 KB）
- C 程序实测 tokens/instr = **1.378**（crc32 1.563，sieve 1.393，sort 1.249，strops 1.519）
- 与 rv32.md 对比：文档 "1,040,595 nonzero / 2,679 dims / 99 heads / 42,012 units"；本次 1,040,627 / 2,679 / 99 / 42,014 —— 规模一致，非零数差 32、单元数差 2（rv-store 分支在 P2 文档之后有少量电路改动，L11 单元 800 vs 文档 798）。

## 4. 文件系统（store）IR 级结果（`04-pytest-store-ir.log`）

10/10 通过，总 466 s。对应 rv-store.md "Tests" 一节，逐项：

| 用例 | 验证内容 | 耗时 |
|---|---|---|
| test_token_ids | 记录 token id 与 Qwen tokenizer 对齐；call_tokens 与 chat template 一致 | 0.46 s |
| test_reference_vs_dict | 参考实现 vs 普通 dict：随机 op（含 2-bit hash 碰撞、写满、全碰撞） | 1.18 s |
| test_hash_spreads | 哈希目录分布 | 0.03 s |
| test_circuit_all_ops | 电路 vs 参考逐 token + 每位置 owner：**put/append/get/list/delete 五个 op**、空文本、复活 tombstone | 17.0 s |
| test_circuit_with_calc | store 调用与 calc 调用交错 | 78.5 s |
| test_circuit_random[cfg0-3]、[cfg1-4] | 随机调用：小 hash（**强制碰撞**）、覆盖、append 链、删除再 put、前 6 token 相同的名字、超长文本、**写满（容量）** | 106.1 s / 95.2 s |
| test_circuit_full_size | 4B 尺寸配置，HF kernel 的 head 读回放（kernel=True） | 69.1 s |
| test_long_text_between | 调用间夹数千 text token | 71.5 s |
| test_snapshot_restore | **新会话的 store heads 从会话 1 的状态启动（跨会话快照）** | 21.4 s |

另：HF 模块 vs IR（chip-only 模型，每维每位置）`test_store_in_hf[prefill]` 和 `[decode]` 都在 CPU 上 PASSED（decode 是第二次运行才通过的，见 §7）。所以 test_rv_store 的 12 项在本机全部跑通。

## 5. rv-tiny（HF 标准 Qwen3_5ForCausalLM）结果

**构建**（`06-tinycheck.log`）：
`uv run python -m chipformer.rv.tinycheck --device cpu --save dist/rv-tiny --progs signext,call,rv32ui-add`
- `built rv-tiny: hidden 2816, 12 layers, 2130M parameters (90 s)`，保存 `dist/rv-tiny`（model.safetensors 4,260,660,712 B，bf16）。
- 代码中没有缩小 ROM/RAM head 的开关（`tiny.build` 直接用 `core.build()`，ROM_WORDS/RAM_WORDS 为 spec 常量且 core 内有 assert），故未做缩小配置；2.1B 在 CPU 上可跑，只是慢。
- `signext        residual == IR (118800 cells); generate == reference: True (44 tokens; 131.4 s)`
- 第二个程序 call 期间进程静默退出（`exit=-1 elapsed_s=769.4`，无 traceback）。

**自由运行**（新脚本 `repro/run_tiny_cpu.py`，从保存的 checkpoint 载入，调用上游 `firmware.run`；`07-tiny-freerun.log`）：
`uv run python -X faulthandler ..\..\..\run_tiny_cpu.py dist/rv-tiny rv32ui-simple memcpy call`

| 程序 | 生成 token / 期望 | 与参考 trace 一致 | 墙钟 | 速度 |
|---|---|---|---|---|
| signext（tinycheck，含 residual==IR） | 44 / 44 | True | 131.4 s（含残差检查） | — |
| rv32ui-simple（riscv-tests ELF） | 3 / 3 | True | 3.7 s | 0.80 tok/s |
| memcpy（汇编，lbu/sb/lw/sw 循环） | 159 / 159 | True | 110.9 s | 1.43 tok/s |
| call（递归 fact + sum，sw/lw 栈） | 474 / 474 | True | 655.8 s | 0.72 tok/s（与另一 22 GB 进程争 CPU） |

结论：**4 个程序在 CPU 上 free-running greedy generate 与参考 trace 逐 token 相同**（signext 还验证了 exit 前残差与 IR 每维每位置相等，118,800 cells）。CPU 速度约 0.7–1.4 tok/s（torch 参考 kernel、bf16、8 线程），比文档 L4 GPU 慢约 2–3 个数量级，长程序（crc32 4,166 token）不现实。

## 6. GDN-RAM 验证（`05-gdn-ram.log`）

脚本 `repro/verify_gdn_ram.py`（numpy，<100 行）：`S = αS + β k (v − Sᵀk)ᵀ`，读 `o = Sᵀq`。
命令：`uv run python ..\..\..\verify_gdn_ram.py`（0.9 s）

```
d_k=128 d_v=128: 2492 writes, 2508 reads, final state == dict RAM: True, max read error 0.0
alpha=0.9: row 3 exact True, other rows kept False
beta=0.5 : row 3 exact False (half-way: [3.5 3.5])
non-one-hot k: rows 3,4 disturbed, row 3 exact False
PASS
```

one-hot k、β=1、α=1 时更新恰好"替换第 i 行、其他行不动"，读回与 dict RAM 比特级一致；反例说明 α<1 会衰减其他行、β<1 只写一半、非 one-hot k 会串行。

## 7. 未完成 / 异常及原因

1. **HF 长时进程静默退出**（2 次）：tinycheck 在 call（769 s）、`test_store_in_hf[decode]`（400 s）均以 exit=-1 结束，无 Python traceback，Windows Application 日志无对应崩溃事件。同时段有其它大进程（analysis/weights/compact.py、一个 22 GB 的 python 进程，非本人启动）争用 CPU/内存。之后用 `-X faulthandler` 单独重跑：call free-run 成功；store decode 重跑结果见下行。原因未定（可能是上游所述 Windows torch CPU 问题的另一种表现，或外部终止），**未修改上游**。
   - store_in_hf[decode] 重跑（`uv run python -X faulthandler -m pytest tests/test_rv_store.py -v -rs --durations=0 -k "store_in_hf and decode"`）：**PASSED，433.1 s**（`04-pytest-store-hf-decode-retry.log`）。即两次静默退出在单独重跑时都没有复现。
2. 未跑：RV_SLOW 长用例（上游默认 skip）、c-crc32 等长 trace 的 tiny 自由运行（CPU 约 1 tok/s，4k token 需 ~1 h）、发布 checkpoint（.data/ 归 Lead）。
3. tinycheck 的 rv32ui-add 因进程中止未跑到。

## 8. 下一步

- 在空闲 CPU 上（无并发大进程）重跑 `tinycheck --progs call,rv32ui-add` 以确认静默退出是否与资源争用相关；若复现，用 `-X faulthandler` + WER LocalDumps 抓 dump。
- 用 `fastexec`/`fast.compact`（仅非零块）在 CPU 上测速，评估长 trace（crc32、calc）可行性。
- 发布 checkpoint 下载完成后，用 RvChat 在 CPU 上跑一次短 put/get 会话 + 快照跨会话。
- GPU 时段：tritonexec / storecheck / 官方 benchmark 子集。
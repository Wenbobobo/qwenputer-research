# Stage 6：无损裁剪运行时（trim runtime）

## 结论（摘要）

- **裁剪后的 RvChat 完整跑在 GPU 上，无 CPU offload**：加载后 12.37 GB，峰值 `max_memory_allocated` 13.15 GB（卡 19.96 GiB）。
- **速度**：Hi 冷启动 4.47 tok/s、热态 6.96 tok/s；put 72 token 10.2 s（基线 73.6 s，7.2×）；get 56 token 6.0 s（基线 39.8 s，6.6×）。
- **正确性**：put/get 文本与 Stage-5 基线**逐字相同**，`head_mismatch=0`、`lock_error=None`，跨进程 get 正确取回 `4471-AZ`。
- **CPU 等价**：真实 checkpoint 上被裁掉的权重项全部是精确 0（248 个矩阵，0 个非零）；plain_hi（13 token）与 RvChat 实际 store_hi prompt（803 token）上，原模型 vs 裁剪模型全词表 logits **逐 bit 相同**（max_abs_diff=0.0，15/15 与 66/66 位置一致）。
- bnb8 对照：GPU 可跑、2.4 tok/s，但文本是乱码；trim 更快且无损。

## 设计

1. `gpu/trim/trim_model.py`（前一代理）：`SlimLinear` 把第 12--43 层 Qwen 线性层输入列打包为 `0:2560 + mu(2732)`、输出行只保留 `0:2560` 再散回 4608 维 residual；RMSNorm、GDN recurrent cache、full-attention KV cache 全部沿用 HF 实现。
2. `gpu/trim/common.py`（本轮）：
   - `check_dropped_zero`：在真实 checkpoint 上证明被丢弃的列（2560:2732、2733:4608）和行（2560:）全为 0——这是“无损”的前提，每次加载都会断言。
   - 屏蔽 `fla`：20:10 有人往 venv 装了 flash-linear-attention 0.6.0，transformers 在 import 时为**所有** GDN 层（含 chip 0--11 层）绑定 fla kernel，chip 精确性未经验证；默认 `sys.modules['fla']=None` 回到参考 torch 路径（与 Stage-5 基线一致），`TRIM_ALLOW_FLA=1` 可放开。
   - `SlimHead`：可选 lm_head 变体，只算文本词表 248320 行、非零列 hull，chip 行填 -inf。
3. `gpu/trim/trim_chat.py`：仿 `gpu/quant/quant_chat.py` 的 monkeypatch：`Qwen3_5ForCausalLM.from_pretrained` → CPU bf16 加载 → 零检查 → `trim_model` → 上游 `chipformer.q35.fast.compact(model, qwen=False, n_chip=12)`（chip 层 bf16 精确）→ `model.to('cuda')` → 实例/`nn.Module.to` 守卫。之后直接构造 `chat.RvChat(..., device='cuda', fast=True)`，走 fast（CUDA graph chip 视图）路径，带 `StoreSession` lock 和 `check_head`。
4. 未修改 `.data` 与 `repro/`。

参数量：Qwen 稠密 6.424 G → 3.570 G（省 2.854 G，约 5.3 GiB bf16）；CPU 侧参数总量 12.26 GB（含 embedding/lm_head tied 4.47 GB）。

## CPU 等价（`gpu/trim/trim_cpu_check.py` + `trim_cpu_resume.py`，`gpu/out/trim_cpu_check.json`）

同一次加载：先跑原模型，再原地 trim，用同一 token 序列（forced）再跑一遍，比较 prompt 最后 64 位置 + 生成位置的全词表 logits。

| prompt | 长度 | 比较位置 | max_abs_diff | argmax 一致 | 逐 bit 相同位置 | 生成 |
|---|---|---|---|---|---|---|
| plain_hi（无 system） | 13 | 15 | **0.0** | 15/15 | 15/15 | `Hello! How` = `Hello! How` |
| store_hi（MEM_SYSTEM+工具，RvChat 实际 prompt） | 803 | 66 | **0.0** | 66/66 | 66/66 | `Hello! How` = `Hello! How` |

store_hi 的裁剪侧因进程两次无日志被杀（原因未定，可能是外部 kill 或内存压力；成功那次 max RSS 28.9 GB），改用 `trim_cpu_resume.py` 读取保存的原模型 logits（`gpu/out/trim_cpu_check.json.store_hi.orig.pt`）单独跑裁剪模型完成。

说明：CPU 上补零累加恰好逐 bit 相同（fp32 累加 + 加 0 精确）。之前合成测试的 0.0625 来自不同 K 长度选择了不同 GEMM 分块；真实 checkpoint 的 CPU 路径没有出现。GPU（rocBLAS）上 reduction order 仍可能不同，但 put/get 的 token 序列与基线逐字相同。

## GPU 结果（fresh process，fla 屏蔽，`gpu/out/trim_{hi,put,get}.json`）

| 运行 | 加载 load_s / init_s | 峰值显存 | token | 生成耗时 | tok/s | head_mismatch / lock_error |
|---|---|---|---|---|---|---|
| Hi（max-new 12） | 152.6 / 178.1 s | 13.12 GB | 9 text | 2.01 s（热 1.29 s） | 4.47（热 6.96） | 0 / None |
| put | 113.4 / 145.9 s | 13.15 GB | 69 text + 3 chip | 10.16 s | 6.79 | 0 / None |
| get（新进程，载入 npz） | 59.5 / 87.2 s | 13.15 GB | 54 text + 2 chip | 6.00 s | 9.00 | 0 / None |

加载时间主要是 CPU 读 checkpoint（/mnt/c 9p）+ `.to('cuda')` 22--64 s，随页缓存变热而缩短。Hi 热态第二次生成 token 与第一次相同（`same_tokens=true`）。

put 转录：

```
<tool_call>
<function=put>
<parameter=name>
locker_code
</parameter>
<parameter=text>
4471-AZ
</parameter>
</function>
</tool_call>
<tool_response>
ok
</tool_response>
<think>

</think>

I've saved your locker code as 4471-AZ.<|im_end|>
```

get 转录（新进程）：

```
<tool_call>
<function=get>
<parameter=name>
locker_code
</parameter>
</function>
</tool_call>
<tool_response>
4471-AZ
</tool_response>
<think>

</think>

Your locker code is 4471-AZ.<|im_end|>
```

## 对比

| 方案 | 设备 | Hi | put | get | 文本 |
|---|---|---|---|---|---|
| Stage-5 基线（offload 33--43 层到 CPU） | GPU+CPU | 9 tok / 7.7 s | 72 tok / 73.6 s | 56 tok / 39.8 s | 正确 |
| bnb8（Qwen 层量化） | GPU，峰值 16.5 GB | 2.4 tok/s | — | — | 乱码 |
| **trim（本阶段）** | **GPU，峰值 13.15 GB** | **9 tok / 2.0 s（热 1.3 s）** | **72 tok / 10.2 s** | **56 tok / 6.0 s** | **与基线逐字相同** |

## lm_head 裁剪变体

`--head-trim`（`SlimHead`）：lm_head 只算文本词表 248320 行、文本行非零列 hull `0:3184`（文本 embedding 行在 chip 维也有非零，故不是 2560+40），权重 1.58 GB（原 tied 4.47 GB）。chip token 由电路决定，chip 行填 -inf，不影响解码；但 `check_head` 诊断无意义（该变体关闭 check_head）。数学上无损（被丢弃列全为 0）。

| 运行 | 峰值显存 | 生成 | tok/s |
|---|---|---|---|
| Hi + head-trim（Lead 的 `headtrim_fla.sh`） | 14.80 GB（embedding 仍保留供输入） | 9 tok / 0.89 s（热 0.56 s） | **10.1（热 16.2）**，文本相同 |

只测了 Hi；put/get 未在 head-trim 下跑（不影响电路 token，预计可行，但未验证）。显存增加可通过让输入 embedding 与 SlimHead 共享存储消除。

### 放开 FLA（`TRIM_ALLOW_FLA=1`，Lead 跑）

| 运行 | 峰值显存 | 生成 | tok/s | head_mismatch / lock_error |
|---|---|---|---|---|
| Hi | 13.12 GB | 9 tok / 2.71 s（热 1.56 s） | 3.3（热 5.8） | 0 / None |
| put | 13.05 GB | 69+3 tok / 4.10 s | **16.8** | 0 / None |
| get（新进程） | 13.05 GB | 54+2 tok / 3.96 s | **13.6** | 0 / None |

put/get 文本与基线逐字相同、lock 全程无误，说明 fla kernel 下 chip 路径在这两个用例上仍精确（仅经验证据，未做 `--check` replay）。

## 限制

1. **逐 bit**：数学上无损（丢弃项全为 0，已在真实权重上断言）；CPU 上实测逐 bit 相同。GPU 上 rocBLAS 对 2561-K 与 4608-K 可能选不同 kernel，reduction order 不同时文本层 logits 可能有 bf16 级差异（chip 层不受影响，仍是上游 compact）。验收若要求 GPU dense-kernel bitwise，需要补零到原 K 或自定义累加。
2. fla：默认屏蔽。放开后 put/get 更快且仍正确（见上），但 chip 层精确性只有 put/get lock 的经验证据，需 `--check` replay 或长对话回归后才能作为默认。
3. 加载仍需先在 CPU 上载入完整 22 GB checkpoint（RAM 峰值高）；`load_trimmed_state_dict` 可逐 tensor 裁剪以降低峰值，尚未接入。
4. 每 token 仍读 embedding/lm_head 全 bf16（tied，4.47 GB）；decode 时 lm_head 是主要带宽开销之一。
5. CPU 等价只做了 2 个 prompt × 3 token；未做 CPU put/get（太慢）。

## 下一步

0. 组合 trim + head-trim + fla，跑 put/get（预计 put/get 再快 1.5--2×）。
1. 用 `load_trimmed_state_dict` 直接构建裁剪模型，降低 CPU RAM 峰值与加载时间。
2. lm_head 只算文本词表（见上）并评估 `head_mismatch` 诊断的替代。
3. 在 GPU 上对原模型（offload）与 trim 做 logits 对拍，量化 reduction-order 差异。
4. 长对话 / calc 工具调用回归（`--check` replay）。
## 9. Lead 补充（2026-10-29 21:30–22:40）

### 9.1 head-trim + FLA 组合（用本项目脚本在 Lead 的 follow-up 里跑）
| 配置 | Hi | put | get | 显存峰值 |
|---|---|---|---|---|
| trim + FLA（无 head-trim） | 首跑含 JIT 3.3 tok/s | **4.10 s / 16.84 tok/s** | **3.96 s / 13.62 tok/s** | 13.15 GB |
| **trim + head-trim + FLA** | **0.73 s / 12.39 tok/s** | 4.38 s / 15.75 tok/s | 3.86 s / 14.00 tok/s | 14.7 GB |

两者 put/get 都正确（head_mismatch 0、lock_error None，get 返回 4471-AZ）。head-trim 把 Hi 延迟从 2.0 s 降到 0.73 s，但对 put 的改善在噪声内；短问答场景用 head-trim，长任务用无 head-trim（显存低 1.6GB）。

### 9.2 重要：FLA 不是 bit-exact（回归门禁实测）
`python -m chipformer.rv.gpucheck --decode`（在 FLA 0.6.0 下）对 5 个测试（signext / call / memcpy / rv32ui-lb / rv32ui-sh）的 prefill+decode 全部报 MISMATCH：
- 差异量级：6e-8 ~ 9.4e-7（fp32 1 ULP 级）；
- 参考值是 IR 精确值，FLA 的 chunk kernel 累加顺序不同导致残差；
- 这解释了为什么 put/get 仍然正确（电路对 1e-7 级扰动有容错），但**不能再说 chip 层在 FLA 下 bit-exact**。

### 9.3 最终推荐
- **默认（要精确性）**：trim，不加 FLA —— GPU 上 put 10.16 s / get 6.0 s / Hi 2.0 s（6.8–9 tok/s），文本与 Stage-5 逐字一致，CPU 对拍 bit-identical。
- **速度模式**：trim + FLA（或 + head-trim）—— put 4.1 s / get 4.0 s / Hi 0.73–2.0 s（13.6–16.8 tok/s），但 chip 为"功能正确、非 bit-exact"，用于允许 1e-7 级数值扰动的负载，并保留 lock/head 检查。
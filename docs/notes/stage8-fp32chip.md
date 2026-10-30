# Stage 8：fp32 chip 层（C8）

日期：2026-10-29，北京时间；AMD Radeon RX 7900 XT / ROCm / WSL Ubuntu-24.04。GPU hard stop 2026-10-30 00:25。

## 当前结论

fp32 chip 已通过 Hi、put、保存 npz 后 fresh get 的混合精度验证，均 head_mismatch=0、lock_error=None，取回 4471-AZ。已完成 H1 原先失败的 recall-1、2、5 共 12 个历史会话，**fp32 0/12 lock divergences**（三个原失败点均 clean）。同 harness bf16 对照正在执行；最终因果判断以 paired 结果为准。

## 思路与实现

新增 [fp32_chip.py](../../gpu/trim/fp32_chip.py)，在 CPU bf16 权重加载、Qwen 无损 trim 与 upstream fast.compact 之后，把 chip 层 0–11 参数与浮点 buffers 转 fp32，再上 CUDA。仅对 [trim_chat.py](../../gpu/trim/trim_chat.py) 做 TRIM_FP32_CHIP=1 的小型 opt-in patch；缺省行为不变。没有改 repro/ 或 .data。

包装层在 chip 入口把 hidden states 转 fp32；chip 层间保持 fp32，层 11 出口转回 bf16。这样不会在每个 chip 层之间反复 bf16 舍入；Qwen 12–43、embedding、lm_head 仍为 bf16。RvChat fast view 与主模型共享这些模块，普通 HF prefill、fast chip prefill、CUDA graph decode 都覆盖；wrapper 保留原 decoder metadata 属性访问。

[runtime.json](../../gpu/out/c8_fp32/runtime.json) 逐层 dtype 审计确认 0–11 为 torch.float32，12–43/embed/head 为 torch.bfloat16。真实 checkpoint 248 个被丢弃矩阵零项检查通过。合成 CPU wrapper 检查确认中间输出 fp32、最终输出 bf16。

本次 no-FLA（reference torch GDN），无 int8/head trim。不能把改善解释成 FLA 开关效果。Stage7 H1 笔记明确指出 calc lock 的根因此前并未独立定位；本轮检验 fp32 假设。

## 内存与速度

实际 compact chip fp32 参数/缓冲 **1,300,628,248 bytes**，相对 bf16 增量约 **650 MB**，大于任务开始时估计的 0.2–0.3GB。模型载入 GPU 分配 13,020,154,880 bytes；压力测试峰值 **13,936,472,064 bytes（13.94GB）**。

fp32 no-FLA 当前测量（text_tokens / generate.seconds，wall 包含 prefill 另记 JSONL）：

| turn | fp32 tok/s | text tokens | lock / head |
|---|---:|---:|---|
| Hi cold | 9.76 | 9 | None / 0 |
| Hi warm | 18.87 | 9 | None / 0 |
| put | 15.72 | 74 | None / 0 |
| fresh get | 14.69 | 54 | None / 0 |

本次固定 torch/OMP/MKL 8 CPU threads，旧 Stage6 默认线程数与时间不同，不能直接把旧 6.79/9.00 tok/s 与这里相除作为 fp32 加速。put/fresh-get 使用显式 locker_code 提示（完整文本在 JSONL），与旧 standard prompt 语义相同但非逐 token 相同。

## 长序列设计与结果

[fp32_chip_bench.py](../../gpu/trim/fp32_chip_bench.py) 直接复用 h1_gen.recall_cases(10,0)，选取先前出现失败的三个案例 1、2、5，各 4 个会话。每会话新对话 [MEM_SYSTEM,user]，CALC_TOOL + ST.TOOLS，256 text-token budget、max_chip=200000、check_head=True。跨会话携带 save_store() snapshot 与深拷贝的参考 StoreSession.dev，完整工具执行由模型 chip/store heads 驱动。共 12 历史轮，prompt 831–864 tokens，单轮生成最多 256 text tokens。

| 先前失败点 | fp32 chip tokens | fp32 lock |
|---|---:|---|
| recall-1/s1 | 4077 | None |
| recall-2/s1 | 2841 | None |
| recall-5/s2 | 2704 | None |

所有 12 轮 head_mismatch=0，lock_error=None。三个原失败点均经过额外 calc，而非因为删掉 calc 规避验证。许多会话仍打满 text budget、出现重复 put/get，此结果只说明本测试的 circuit lock 正确性，不证明保存策略/答案评分已解决。

## 命令与证据

完整命令清单：[c8_commands.txt](../../gpu/out/c8_commands.txt)。核心命令（WSL）：

```bash
export PYTHONPATH=/mnt/c/Projects/Qcompute:/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv:/mnt/c/Projects/Qcompute/repro/qwenputer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
TRIM_FP32_CHIP=1 timeout 3000 bash gpu/run_py.sh gpu/trim/fp32_chip_bench.py \
  --out gpu/out/c8_fp32 --cases 1,2,5 --deadline 1793291100
TRIM_FP32_CHIP=0 timeout 2400 bash gpu/run_py.sh gpu/trim/fp32_chip_bench.py \
  --out gpu/out/c8_bf16 --cases 1,2,5 --deadline 1793291100
python gpu/trim/analyze_fp32_chip.py
```

- [fp32 log](../../gpu/out/c8_fp32.log)、[fp32 turns](../../gpu/out/c8_fp32/turns.jsonl)、[runtime](../../gpu/out/c8_fp32/runtime.json)。
- [bf16 log](../../gpu/out/c8_bf16.log)、[summary](../../gpu/out/c8_summary.json)（paired 完成后更新）。
- 初次直接 upstream gpucheck --decode --dtype fp32 走 FLA，约 6 分钟没有单程序结果，主动终止；不能当作 PASS。之后使用 [gpucheck_fp32chip.py](../../gpu/trim/gpucheck_fp32chip.py) 的 reference torch launcher。upstream fp32 模式本来允许非 bit-exact dims，只以 wrong tokens 判失败。

## 推荐与边界

等待 paired bf16 与独立 gpucheck 完成后填写最终建议。即使 12/12 clean，也只能称“在此次选定的长序列压力样本消除了已重现的 lock 分歧”，不能证明 AMD 全部输入 bit-exact；fp32 普通浮点运算仍可能有残差。
## 9. 严格 paired 对照完成（Lead 补跑，23:46）
同一 harness、同一天、同一 GPU，`fp32_chip_bench.py --cases 1,2,5` 两种 dtype 各 16 轮（含 Hi/put/get 4 轮 + 12 个历史会话）：

| 指标 | bf16 chip | fp32 chip |
|---|---|---|
| 历史会话 lock divergences | **3/12（recall-1/s1、recall-2/s1、recall-5/s2）** | **0/12** |
| 原失败点 chip tokens | 25（calc 未执行完，提前分叉） | 4077 / 2841 / 2704（calc 完整执行） |
| head mismatch | 全 0 | 全 0 |
| Hi cold / warm tok/s | 12.42 / 16.56 | 9.76 / 18.87 |
| put tok/s | 16.88 | 15.72 |
| fresh get tok/s | 15.52 | 14.69 |
| 峰值显存 | ~13.3GB | 13.94GB |

bf16 的分叉发生在 token 228/249/232（got V(0), want F0(32716)），与 Stage 7 H1 观察完全一致；fp32 在同位置继续执行 calc 并返回正确结果。
**结论：在这 12 个曾失败的长序列输入上，fp32 chip 层消除了 bf16 的 circuit lock divergence，且速度代价很小（put/get 约 −7%~−13%，Hi warm 反而更快）。** 边界：只证明这组压力样本；普通 fp32 浮点仍可能有残差，不宣称全输入 bit-exact。
证据：gpu/out/c8_fp32/turns.jsonl、gpu/out/c8_bf16_pair/turns.jsonl、c8_bf16_pair.log。
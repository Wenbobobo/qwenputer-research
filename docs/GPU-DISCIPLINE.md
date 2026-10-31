# GPU 纪律与环境（唯一权威）

> 适用于本机 WSL2 + RX 7900 XT（gfx1100）。违反这些会导致崩溃、数据损坏或几小时的浪费。

## 1. 时间窗
- 允许：**15:30–00:30（Asia/Shanghai）**；静默：**00:30–09:30、12:30–15:30**，静默期间不得启动 GPU 任务；跑不完的任务在 00:25 前收尾/保存。

## 2. 硬件与并发
- GPU：AMD RX 7900 XT 19.96GB。Tesla T10 仅备用；**不要与 7900 XT 并行满载**（历史事故：0x3B + WHEA，见 C:\Projects\ROCM 知识库）。
- **同一时间只跑一个模型任务**。两个 13–14GB 模型并发会让 20GB 卡进入显存 thrash/挂起（Stage 8 实测）。
- 跑前检查：`wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh /mnt/c/Projects/Qcompute/gpu/mem.py`（free >17GB 再启动）。

## 3. 环境
- WSL：Ubuntu-24.04，ROCm 7.2.1，venv `~/src/rocm-smoke/.venv`（torch 2.9.1+rocm、triton 3.5.1、transformers 5.19、accelerate、bitsandbytes 0.50.2、fla 0.6.0）。
- 统一入口：`wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh <script.py> [args]`。
- 需要 PYTHONPATH 时：`repro/qwenputer/experimental/riscv`（注意：`repro/qwenputer/chipformer` 是另一个包，会 shadow 真正的包，不要把它放进 PYTHONPATH）。

## 4. 禁止事项
- ❌ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（ROCm/RDNA3/WSL 已知崩溃/死锁，pytorch #200608/#192259）。
- ❌ `HSA_OVERRIDE_GFX_VERSION`。
- ❌ 修改 `.data/` 与 `repro/`；实验代码放 `gpu/` 或 `analysis/`，用 monkeypatch。
- ❌ 把 `.data/`（20GB）、`.research/`、venv、大日志提交到 git。

## 5. 数据与输出
- 原始结果一律落 JSON/日志到 `gpu/out/` 或 `analysis/experiments/*/`；报告只引用文件+数字+条件。
- 重要实验写 `gpu/<FLAG>_DONE` 标记，方便串行调度。
- 后台任务用 background job，轮询间隔 >90s；GPU 任务收尾约 60–90s，不要提前判死。

## 6. 精确性验收（AMD）
- CPU：bit-identical（trim 对拍 max_abs_logit_diff 0.0）。
- GPU：`head_mismatch = 0`、`lock_error = None`、答案正确 = token-level 通过；**不要宣称 dim-bit-exact**（AMD bf16/fp32 自有 1-ULP 差异）。
- 长序列回归：`gpu/trim/fp32_chip_bench.py --cases 1,2,5`（bf16 3/12、fp32 0/12）。
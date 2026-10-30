# Stage 7：INT8 lm_head 与 ROCm 调优

## 实现

新增 TRIM_INT8_HEAD=1 opt-in head，默认 trim 行为不变。文本词表 248320 行的前 2560 列采用逐行对称 INT8，保留 float32 scale；final RMSNorm 有效 chip 列共 40 列，所有行保持 BF16。chip 行的 Qwen 列全量零断言。输出按 16384 行块反量化为 BF16 后 GEMM，默认返回全词表以保留 head_mismatch；和 --head-trim 同用时 chip 行为 -inf。输入 embedding 保持 BF16，因此这是实验性 weight-only INT8，不减少 tied embedding 的常驻显存。

真实 head 检查结果（gpu/out/int8_head_cpu.json）：INT8 buffer 675476800 bytes，单块 BF16 dequant 83886080 bytes；248320x2560 量化，40 chip 列精确。Qwen 随机探针 max_abs_diff=0.0493164，argmax 一致；circuit 探针 max_abs_diff=0、chip 行逐值相同。CPU plain Hi 6 token 为 Hello! How can I help，INT8 与 BF16 token 完全一致。实现还包含 tied checkpoint 断言、chip 精确断言和 activation dtype guard。

## GPU 结果

所有运行均未启用 expandable_segments 或 HSA_OVERRIDE_GFX_VERSION；等待 D7_DONE 后在空闲显存 17.74 GiB 时运行。

| 配置 | Hi 冷/热 tok/s | put kernel tok/s / wall | get kernel tok/s / wall | 峰值 | 正确性 |
|---|---:|---:|---:|---:|---|
| 同轮 trim | 13.48 / 18.73 | 13.60 / 8.25s | 14.71 / 7.46s | 13.22GB | mismatch=0, lock=None, 4471-AZ |
| INT8 head | 11.21 / 8.73 | 11.73 / 6.85s | 11.96 / 6.68s | 13.86GB | put/get 逐 token 相同；Hi 首 token 分叉 |
| hipBLASLt | 10.58 / 16.47 | 未测 | 未测 | 13.39GB | Hi token 相同 |
| TunableOp tuning iter=2 | 4.58 / 13.17 | 未测 | 未测 | 17.47GB | Hi token 与 INT8 一致；校准约 108s |

INT8 put/get 完整文本和 token 与基线一致，均返回 locker code 4471-AZ；Hi 首 token 分叉为 list tool-call，故不能宣称全场景 bit-exact。INT8 相对同轮基线冷端 wall tok/s：Hi -16.9%，put -13.8%，get -18.7%，且峰值显存增加约 0.64GB。hipBLASLt 冷端约 -21.5%，warm 约 -12.1%；TunableOp 当前配置也没有收益。

## Causal-conv 与建议

已确认 FLA causal_conv1d API 可用，但它使用 [B,T,D] 并返回 output/final_state，Transformers reference 使用 [B,D,T] 且原地更新 [B,D,W] cache；本轮不接入，避免 chip 精度风险。

当前建议保持 INT8 head 默认关闭。现有实现的 chunked dequant BF16 GEMM 增加显存并降低速度；只有改成原生 INT8 GEMV、同时量化或压缩 tied embedding，并重新证明 Hi/put/get 稳定后，才值得继续。

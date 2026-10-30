# Stage 7 · FLA 精确化：选择性路由 + AMD 精确性边界

状态：完成（2026-10-29 22:40）。执行者 A7 agent 在写完结果后崩溃，marker 与本文由 Lead 根据其产物补写。

## 1. 做了什么
- 发现 transformers 5.19 把 GDN kernel 在 `modeling_qwen3_5` 里全局绑定；作者文档的 bit-exact 验证环境是 **FLA 0.5.2 + NVIDIA L4**（不是依赖锁定）。
- 实现 selective FLA：按 `layer_idx` 路由，**chip 层 0–11 走 reference torch GDN，Qwen 层 12–43 走 FLA**；dispatch 探针确认生效（ref=96 fast=480 次调用）。
- 代码：gpu/trim/selective_fla.py、gpu/trim/gpucheck_selective.py、gpu/trim/trim_chat.py（`TRIM_SELECTIVE_FLA=1`）。

## 2. 性能（trim runtime，7900 XT，13.15GB 峰值）
| 模式 | Hi | put | get |
|---|---|---|---|
| no-FLA（全 reference） | 4.47–6.96 tok/s | 6.79 | 9.00 |
| **selective FLA** | **10.53（cold）/13.78（warm）** | **12.44** | **12.15** |
| full FLA | 12.39–16.8 | 16.84 | 13.62 |

selective 的 put/get 全部正确（head 0、lock None、4471-AZ），文本与基线一致。

## 3. 精确性门禁（关键修正）
| 配置 | gpucheck --decode |
|---|---|
| FLA 0.6.0 全量 | MISMATCH（−5.96e-8，0 wrong tokens） |
| **屏蔽 FLA 的纯 torch fallback** | **同样 MISMATCH（残差完全相同）** |
| FLA 0.5.2（作者版本） | 同样失败（677 dims 不 bit-exact，0 wrong tokens） |

结论：**AMD gfx1100/ROCm 的 bf16 GEMM 本身非 bit-exact**，与 FLA 无关；作者的 bit-exact 是 NVIDIA L4 结论。
AMD 上正确的精确性表述：**CPU 路径 bit-identical（max_abs_logit_diff 0.0）；GPU 路径 token-level 正确**（head_mismatch 0、lock None）。
FLA 不引入新的错误类别，因此可以放心作为 GPU 速度手段。

## 4. 推荐
- GPU 默认：**selective FLA（chip 参考 + Qwen FLA）** 作为精确优先的加速档（12.2–12.4 tok/s）；
- 追求极限速度：full FLA（16.8 tok/s put）；
- 需要 CPU 参考/审计：trim no-FLA（6.8–9 tok/s，CPU bit-identical）；
- gpucheck 不能再作为 AMD 的 bit-exact 门禁；改为检查 head_mismatch/lock/token 一致性。

证据：gpu/out/trim_selfla_{hi,put,get}.json、gpu/out/gpucheck_selfla.log、gpu/out/nofla_probe.log、gpu/out/fla052_probe.log、gpu/out/dispatch_probe*。
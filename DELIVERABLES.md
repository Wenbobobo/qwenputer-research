# Qcompute · 交付清单（2026-10-31 更新）

本项目完成了对 [Qwenputer](https://github.com/divagr18/qwenputer) 的深度复现、审计、优化、扩展与落地评估。

**在线报告**：https://wenbobobo.github.io/qwenputer-research/  ·  **仓库**：https://github.com/Wenbobobo/qwenputer-research
**接手先读**：[docs/START-HERE.md](docs/START-HERE.md)  ·  **权威数字**：[docs/RESULTS.md](docs/RESULTS.md)

## 1. 10 分钟版
1. 打开在线站点，依次看 Stage 1 → 8（顶部可切换）。
2. 读 [docs/RESULTS.md](docs/RESULTS.md) 的 A–H 表（所有关键数字与证据路径）。
3. 读 [docs/START-HERE.md](docs/START-HERE.md)（环境、纪律、路线）。

## 2. 核心结论（均以 RESULTS 为准）
- **可复现**：发布 checkpoint 在 CPU 上端到端 put/get（4471-AZ，0 mismatch）；GPU rv-tiny token-exact。
- **可审计**：chip 侧 168/168 张量 bit-exact；整模型不可推导残留仅 16,384 B；22.20GB → **7.951 GiB** 无损压缩。
- **记忆边界**：in-weights 是精确 KV 外设而非联想记忆；H1 显示 90 vs 30 主要来自覆盖语义+回答格式；H1 in-weights 100%（n=10）与 host 覆盖式同档 → **介质无优势**；H5 语义路由 Recall@5 87%、exact get 45 token vs RAG ~12,141。
- **性能**：trim runtime 13–15 tok/s（RX 7900 XT）；**fp32 chip 修复长序列**（bf16 3/12 → fp32 0/12 lock divergence）；selective FLA 可到 14.6 tok/s（长序列待验证）。
- **增强可行性**：ADD8/SORT4/DADD 同时并入 chip 层仍 0 额外层（slots 1899/2029），IR/emit 已验证；H5 两阶段检索原型由 CPU 路由 11ms + exact get 0.5ms 支撑。
- **量化**：只省显存不省算力；bnb8 可用但慢，NF4/int8 head 负结果。

## 3. 分阶段一句话
| 阶段 | 主题 | 报告 |
|---|---|---|
| 1 | 原理、前史、复现、成本、记忆、优化、场景 | docs/reports/qwenputer-stage1.html |
| 2 | 参数化记忆 vs RAG/KG/Engram；90 vs 30 统计拆解 | stage2-memory.html |
| 3 | 5 个创意 + 编译器审计 + 记忆盘边界 | stage3.html |
| 4 | GPU 路径、H1、memory-disk v0、设备树 | stage4.html |
| 5 | 发布模型 GPU 真跑、H5、SafeStore | stage5.html |
| 6 | 性能与量化（trim、bnb、硬件实测） | stage6.html |
| 7 | AMD 精确性边界、selective FLA、H1 in-weights、int8、社区 | stage7.html |
| 8 | Pages 上线、速度矩阵、计算增强组合、设备合并、H5 e2e | stage8.html |
| 9 | 落地场景/突破方向（专家+综合）、文档整理、交接 | [stage9.html](docs/reports/qwenputer-stage9.html)、docs/research/stage9-*、docs/START-HERE.md |

## 4. 负结果与边界（非常重要）
- **GPU bit-exact 不成立**（AMD）；验收用 token-level + head/lock；作者 bit-exact 是 NVIDIA L4 结论。
- **长序列 bf16 会 lock drift**（3/12）；默认正确性配置用 fp32 chip。
- **FLA 会让 chip 非 bit-exact**；只作速度模式并保留 head/lock 检查。
- **量化不省算力**：INT8/FP8 无收益；NF4 质量崩；int8 head 更慢更占显存。
- **in-weights 介质无速度/准确率优势**；优势在离线、隐私、可审计、状态随 checkpoint。
- **H5 Recall@1 仅 58%**：top-1 不能当答案；需 top-3/5 + 重排 + 拒答。
- **设备合并尚未端到端**：merge_probe 是 IR/allocator/emit 级验证，生产 merge 仍缺 tokenizer/owner/exit 工作。

## 5. 证据与索引
- 文档总索引：docs/INDEX.md；文档审计与归档映射：docs/DOC-AUDIT.md；归档：docs/archive/2026-10-31/。
- 术语：docs/GLOSSARY.md；GPU 纪律：docs/GPU-DISCIPLINE.md；环境：docs/START-HERE.md §3。
- 原始数据：analysis/weights、analysis/device、analysis/memory_disk、analysis/experiments/{h1,h1_inweights,h5,h5_e2e}、gpu/out。

## 6. 未完成 / 下一阶段（Stage 9+）
- ADD8-only rv-tiny 端到端（真实 merge + tokenizer/namespace + owner/exit）。
- H5 reader 集成（4B reader 上 top-1/3/5 vs RAG 对照 + 拒答/版本门禁）。
- fp32 chip 局部化（只敏感子层）与 paired 长序列扩样（目标 0/60）。
- >20 tok/s：native INT8 GEMV、tied embedding 量化、causal_conv1d。
- memory-disk v1：trim + SafeStore + journal 的可用记忆盘 API。
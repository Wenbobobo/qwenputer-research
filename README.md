# Qcompute

复现、审计、优化并扩展 [Qwenputer](https://github.com/divagr18/qwenputer)——把 RISC-V CPU 与文件系统编译进 Qwen3.5-4B 权重；并研究「权重内计算/记忆」的边界与落地场景。

**在线报告**：https://wenbobobo.github.io/qwenputer-research/ （每个阶段一页，顶部可切换）
**仓库**：https://github.com/Wenbobobo/qwenputer-research

## 先读什么

| 文件 | 用途 |
|---|---|
| [docs/START-HERE.md](docs/START-HERE.md) | **接手必读**：项目结论、环境、纪律、文件地图、验收 gate、路线图 |
| [STATUS.md](STATUS.md) | 当前状态与下一步（最新） |
| [docs/RESULTS.md](docs/RESULTS.md) | **唯一权威数字台账**（所有报告引用这里） |
| [docs/GPU-DISCIPLINE.md](docs/GPU-DISCIPLINE.md) | GPU 窗口/并发/禁止事项/验收 |
| [ROADMAP.md](ROADMAP.md) | 阶段计划与完成项 |
| [docs/INDEX.md](docs/INDEX.md) | 文档索引（报告/研究/笔记/归档） |
| [DELIVERABLES.md](DELIVERABLES.md) | 交付清单与结论摘要 |

## 目录

- `docs/reports/` — 阶段 HTML 报告（stage1–9，已发布 Pages）
- `docs/notes/` — 证据笔记（复现、审计、性能、记忆、设备树…）
- `docs/research/` — 深度研究/专家报告
- `analysis/` — 权重、设备、记忆盘、H1/H5 实验代码与原始数据
- `gpu/` — GPU 实验脚本（trim runtime、fp32 chip、selective FLA、量化）与原始结果
- `demo/` — 发布模型 CPU 端到端演示
- `site/` — GitHub Pages 站点；`.github/workflows/pages.yml` 自动部署
- `.data/` — 模型权重/数据集（**不提交**，20GB+）；`repro/` — 上游工作副本（只读，不改）

## 30 秒核心数字（详见 docs/RESULTS.md）

- 编译器审计：chip 侧 **168/168 张量 bit-exact**；整模型不可推导残留仅 16,384 B。
- 无损压缩：22.20 GB → **7.951 GiB**（10/10 SHA-256 一致）。
- GPU：trim runtime 13–15 tok/s（RX 7900 XT），**fp32 chip 修复长序列**（bf16 3/12 → fp32 0/12 lock divergence）。
- 记忆：H1 in-weights changed **10/10=100%**，与 host 覆盖式工具同档 → **覆盖语义是关键、介质无优势**；H5 语义路由 Recall@5 87%，exact get 45 token vs RAG ~12,141。
- 设备合并：ADD8+SORT4+DADD 同时并入仍 **12 层、0 额外层**（slots 1899/2029）。

## 注意

- 具体环境、GPU 纪律与禁止事项见 [docs/GPU-DISCIPLINE.md](docs/GPU-DISCIPLINE.md)。
- `.data/` 与 `repro/` 不要修改；所有实验代码放 `gpu/` 或 `analysis/`，用 monkeypatch。
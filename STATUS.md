# Qcompute · STATUS

更新时间：2026-10-31 10:45 (Asia/Shanghai) · **Stage 8 完成**

## Stage 8 结果
| 项 | 状态 | 结果 |
|---|---|---|
| GitHub Pages | ✅ | 仓库 github.com/Wenbobobo/qwenputer-research；站点 wenbobobo.github.io/qwenputer-research/（hub + stage1–8 全部 200）；Actions 自动部署；每阶段 HTML 可按同一流程上线 |
| 速度/正确性矩阵 | ✅ | Hi：bf16 no-FLA 8.51、bf16 selective 14.59、bf16 full-FLA 12.04、fp32 no-FLA 11.81、fp32 selective 13.64 tok/s；长序列 lock：bf16 3/12 → fp32 0/12；推荐 correctness-first = fp32 no-FLA，speed-first = bf16 selective（长序列待验证） |
| 计算增强 portfolio | ✅ | docs/research/stage8-enhancement-portfolio.md：Top3 = fp32 chip / H5 混合检索 / 设备树合并；含 kill list 与不倒退 gate |
| E2 设备合并 | ✅ | IR/allocator/emit 成功：三设备同时 12 层、3870 dims、167 heads、48230 units、slots 1899/2029、nz +5626；IR check=True，private code 与独立设备一致 |
| E3 H5 端到端 | ✅ | CPU 原型：Recall@1/3/5 = 58/84/96%、BGE 11.2ms p50、SafeStore get 0.51ms、长名碰撞隔离通过；LongMemEval 全量因批处理效率超时（待优化） |
| 阶段八报告 | ✅ | docs/reports/qwenputer-stage8.html 已发布到 Pages |

## 当前最优配置
- **正确性优先**：trim + fp32 chip + no-FLA（长序列 0/12 lock divergence，put/get clean，13.94GB）
- **速度优先**：trim + bf16 + selective FLA（Hi 14.59 tok/s）
- **不建议默认**：full FLA（长序列未验证）、int8 head（负结果）、NF4（质量崩）

## 交付
- 报告：docs/reports/qwenputer-stage1..8.html（全部已发布）
- 笔记：stage8-pages / speed / device-merge / h5-e2e / fp32chip / community / selectfla / h1-inweights / int8-head
- 研究：docs/research/stage8-enhancement-portfolio.md
- 数据：analysis/device/merge_probe.*、analysis/experiments/h5_e2e/、gpu/out/s8_*

## Stage 9 完成（2026-10-31）
- 专家/综合分析：docs/research/stage9-application-scenarios.md（产品/场景向）、stage9-technical-breakthroughs.md（技术/研究向）、stage9-synthesis.md（综合决策）
- 文档整理：RESULTS/GPU-DISCIPLINE/GLOSSARY/DECISIONS/START-HERE/DOC-AUDIT 新建；README/INDEX/DELIVERABLES 重写；归档 4 篇过时笔记；清理垃圾目录
- 报告：docs/reports/qwenputer-stage9.html 已发布到 Pages
- 交接：docs/START-HERE.md（完整接手 prompt，含场景结论/P0-P2 路线/验收 gate）

## Stage 9 候选
ADD8-only rv-tiny 端到端；H5 reader 集成；fp32 chip 局部化；>20 tok/s（native int8/tied embed/causal_conv1d）；memory-disk v1

## GPU 纪律
- 窗口 15:30–00:30；静默 00:30–09:30/12:30–15:30；当前 10:45 已过静默，GPU 可用（空闲 16.38GB）
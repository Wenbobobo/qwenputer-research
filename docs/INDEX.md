# Qcompute 文档索引

> 最后更新：2026-10-31 · Stage 9 收尾 · 权威数字见 [RESULTS.md](RESULTS.md)；接手先读 [START-HERE.md](START-HERE.md)。

## 0. 入口
| 文件 | 内容 |
|---|---|
| ../README.md | 项目入口（在线报告/仓库/30 秒数字） |
| START-HERE.md | **接手启动 prompt**（结论/环境/纪律/地图/gate/路线） |
| ../STATUS.md | 当前状态与下一步 |
| ../ROADMAP.md | 阶段计划与完成勾选 |
| ../DELIVERABLES.md | 交付清单与结论摘要 |
| RESULTS.md | **唯一权威结果台账** |
| GPU-DISCIPLINE.md | GPU 窗口/并发/禁止事项/验收 gate |

## 1. 阶段报告（docs/reports/，已全部发布到 Pages）
| 文件 | 内容 | 状态 |
|---|---|---|
| qwenputer-stage1.html | 原理 / 前史 / 可行性 / 复现证据 / 成本 / 记忆 / 优化 / 场景 | ✅ |
| qwenputer-stage2-memory.html | 参数化记忆 vs RAG/KG/Engram；90 vs 30 统计拆解 | ✅ |
| qwenputer-stage3.html | 5 个创意 + 编译器审计 bit-exact + 记忆盘实测 | ✅ |
| qwenputer-stage4.html | GPU 路径 / H1 记忆实验 / memory-disk v0 / 设备树 | ✅ |
| qwenputer-stage5.html | 发布模型 GPU 真跑 / H5（LongMemEval）/ SafeStore 键修复 | ✅ |
| qwenputer-stage6.html | 性能与量化（trim+FLA、bnb、硬件实测） | ✅ |
| qwenputer-stage7.html | AMD 精确性边界 / selective FLA / H1 in-weights / int8 / 社区调研 | ✅ |
| qwenputer-stage8.html | Pages 上线 / 速度矩阵 / 计算增强 portfolio / 设备合并 / H5 e2e | ✅ |
| qwenputer-stage9.html | 落地场景与突破方向 / 文档整理 / 交接 | ✅（已发布） |

## 2. 研究（docs/research/）
| 文件 | 内容 |
|---|---|
| prior-work.md | 前史调研：fast weights / NTM / Tracr / PAW / ROME / Engram / RAG，及"为什么没成功" |
| parametric-memory.md | in-weights FS vs RAG/KG/Engram 的定量对比、成本模型、可证伪实验 H1–H5（实测更新见 notes） |
| expert-assessment.md | 早期专家评估（历史文档，路线图部分已过时） |
| stage8-enhancement-portfolio.md | 计算增强组合：Top3 / kill list / E1–E3 / 不倒退 gate |
| stage9-application-scenarios.md | 落地场景与突破方向（产品/场景向） |
| stage9-technical-breakthroughs.md | 技术突破方向（研究向专家报告） |
| stage9-synthesis.md | 两份专家报告的综合决策摘要（P0/P1/P2 路线、可证伪命题、kill list） |

## 3. 笔记（docs/notes/）——按主题
### 3.1 复现、审计、权重
| 文件 | 内容 |
|---|---|
| upstream-map.md | 上游仓库地图与阅读顺序 |
| claims-audit.md | 作者声明逐条核对（部分已被 Stage 4–8 更新，见文内） |
| repro-log.md | CPU 复现记录（环境、测试、IR、tiny、store） |
| weights-map.md | checkpoint 逐层/逐张量结构与压缩分析 |
| cpu-checkpoint-demo.md | 发布 checkpoint CPU 演示 + put/get |
| stage3-a-rebuild.md / stage3-a2-base-audit.md | 编译器审计（chip 侧 / Qwen 侧） |
| stage3-d-snapshot.md | 记忆盘快照/可移植性/长名实测 |
| h2h3-experiments.md | H2 名字别名/探测统计、H3 token 成本 |

### 3.2 GPU、性能、量化、精确性
| 文件 | 内容 |
|---|---|
| stage4-gpu.md / stage5-gpu-runtime.md | GPU 路径与发布模型 GPU 真跑 |
| stage6-quant-hw.md | 7900 XT 精度算力实测（BF16/INT8/FP8/INT4） |
| stage6-quant-ecosystem.md | 量化生态与 W4A8/W4A4 迁移路径 |
| stage6-quant-e2e.md | bnb8/NF4 端到端（bnb8 可用但慢；NF4 质量崩） |
| stage6-trim-runtime.md | 无损 trim runtime（含 §9 Lead 复核/修正） |
| stage6-kernels.md | FLA 内核（含两次 Lead 修正） |
| stage7-selectfla.md | selective FLA + AMD 非 bit-exact 定论 |
| stage7-int8-head.md | int8 head 负结果 |
| stage8-fp32chip.md | **fp32 chip：长序列 lock 3/12 → 0/12（paired）** |
| stage8-speed.md | 六配置速度矩阵与长序列门禁（当前速度权威） |

### 3.3 记忆、H1/H5、memory-disk
| 文件 | 内容 |
|---|---|
| stage4-h1.md | H1 主机工具/上下文实验 |
| stage7-h1-inweights.md | **H1 in-weights 对照臂（changed 10/10）** |
| stage5-h5.md | H5-lite + 真实 LongMemEval 检索 |
| stage8-h5-e2e.md | H5 端到端原型（router + SafeStore） |
| memory-disk-v0.md | memory-disk v0 API/指纹/journal/NameGuard |
| stage5-keyfix.md | SafeStore 键协议修复（512 对抗长名 0 碰撞） |

### 3.4 设备树、发布、流程
| 文件 | 内容 |
|---|---|
| device-tree-estimate.md | ADD8/SORT4/DADD 构建与对比（实测） |
| stage8-device-merge.md | 设备合并 IR 可行性（0 额外层） |
| stage8-pages.md | GitHub Pages 发布与更新流程 |

## 4. 归档（docs/archive/2026-10-31/）
被取代/重复文档按 `DOC-AUDIT.md` 的映射复制归档（非破坏性）；活跃目录不再保留过时副本。归档清单见 [DOC-AUDIT.md](DOC-AUDIT.md)。

## 5. 代码与数据目录
| 路径 | 内容 |
|---|---|
| analysis/weights/ | 扫描/无损压缩脚本与结果 |
| analysis/device/ | 设备图/估算/合并探针 |
| analysis/memory_disk/ | memory-disk v0 + SafeStore + 测试 |
| analysis/experiments/{h1,h1_inweights,h5,h5_e2e}/ | 记忆/检索实验 |
| gpu/ | GPU 脚本（tiny/triton/quant/trim/kernels）与 out/ 原始结果 |
| demo/ | CPU 端到端 demo |
| site/ | Pages 站点（源：docs/reports；生成：site/stages） |

## 6. 维护规则
- 数字只写进 [RESULTS.md](RESULTS.md)；报告引用它，不再各自维护。
- 阶段报告写 docs/reports/ 后按 stage8-pages.md 发布。
- 过时文档先复制到 docs/archive/YYYY-MM-DD/ 再移出活跃目录，并在 DOC-AUDIT.md 记录映射。
# Qcompute 文档索引

> 最后更新：2026-10-29 · Stage 1 收尾 / Stage 2 启动

## 顶层
| 文件 | 内容 |
|---|---|
| ROADMAP.md | 阶段目标、验收项、当前关键路径 |
| STATUS.md | 工作流状态、已确认事实、阻塞、下一步 |
| README.md | 项目入口与目录说明 |
| docs/INDEX.md | 本索引 |

## 报告（docs/reports/）
| 文件 | 内容 | 状态 |
|---|---|---|
| qwenputer-stage1.html | **阶段一：Qwenputer 深度解析**（原理 / 前史 / 可行性 / 复现证据 / 成本 / 记忆 / 优化 / 场景） | 主体完成，待复现数字收尾 |
| qwenputer-stage2-memory.html | 阶段二：参数化记忆与持续学习研究 | ✅ |
| qwenputer-stage3.html | 阶段三：5 个创意 + 2 个原型（A 审计 bit-exact / D 记忆盘） | ✅ |
| qwenputer-stage4.html | 阶段四：GPU 路径 / H1 / memory-disk v0 / 设备树 | ✅ |
| qwenputer-stage5.html | 阶段五：GPU 真跑发布模型 / H5（LongMemEval）/ SafeStore 键修复 | ✅ |
| qwenputer-stage6.html | 阶段六：性能与量化 —— trim+FLA 把 put 从 73.6s 降到 4.1s（18×） | ✅ |
| qwenputer-stage7.html | 阶段七：AMD 精确性边界 / selective FLA / H1 in-weights / int8 / 社区调研 | ✅ |

## 研究（docs/research/）
| 文件 | 内容 |
|---|---|
| prior-work.md | 前史调研：fast weights / NTM / Tracr / PAW / ROME / Engram / RAG，及"为什么没成功" |
| parametric-memory.md | in-weights FS vs RAG/KG/Engram 的定量对比、成本模型、可证伪实验 H1–H5 |
| expert-assessment.md | 专家评估：可行性、脆弱性、记忆性价比、自我进化边界、6 个月路线图 |

## 笔记与证据（docs/notes/）
| 文件 | 内容 |
|---|---|
| upstream-map.md | 上游仓库地图与阅读顺序 |
| claims-audit.md | 作者声明逐条核对（✅/🟡/⏳/❌） |
| repro-log.md | 本机 CPU 复现记录（环境、测试、IR、tiny、store） |
| weights-map.md | 发布 checkpoint 的逐层/逐张量结构与压缩分析 |
| cpu-demo.md | 发布 checkpoint CPU 实机演示记录（当前 BLOCKED：内存竞争） |
| h2h3-experiments.md | H2 名字别名/探测统计、H3 token 成本交叉点（已完成） |
| cpu-checkpoint-demo.md | 发布 checkpoint CPU 实机演示：smoke 成功 + put/get 记录 |
| stage3-ideas.md | Stage 3 创意草案（编译器即安装器 / 设备树 / 监控器 / 记忆盘 / 自举） |
| stage3-a-rebuild.md | **编译器审计**：发布 chip 权重 168/168 张量 bit-exact 重建 |
| stage3-d-snapshot.md | 记忆盘：快照大小/可移植性/长名字别名实测（v1.1 全量） |
| memory-disk-v0.md | memory-disk v0：API/指纹/迁移/NameGuard/40 测试 |
| device-tree-estimate.md | 设备树：ADD8/SORT4/DADD 真实构建与对比 |
| stage4-gpu.md | GPU 路径：rv-tiny 7/7 exact、43–45 tok/s、executor 1.3–7k tok/s、ROCm residual 条件性 |
| stage4-h1.md | H1：覆盖式 90% vs append-only 23%；brevity 变体 93–100% → 90vs30 主要是回答格式/评分 |
| memory-disk-v0.md | memory-disk v0（fingerprint/journal/NameGuard，40/40） |
| device-tree-estimate.md | 设备树：ADD8/SORT4/DADD 真实构建 |

## 代码与数据
| 路径 | 内容 |
|---|---|
| repro/qwenputer/ | 上游 rv-store 分支工作副本（未改源码） |
| repro/verify_gdn_ram.py | 自写 GDN-as-RAM 独立验证 |
| analysis/weights/ | scan.py（全量零值扫描）、compact.py（无损压缩原型）、scan_results.json、compact_results.json |
| analysis/experiments/ | H2/H3 实验脚本与原始数据 |
| demo/ | 发布 checkpoint CPU 演示脚本与日志 |
| .data/ | 模型权重（20.7 GiB，不入库） |
| .research/upstream/ | 上游源码与文档快照（不入库） |

## 冗余/合并说明（Stage 2 整理）
- `docs/notes/upstream-map.md` 与 README 的上游介绍有重叠 → 保留 upstream-map 作为唯一入口，README 只放链接。
- `docs/research/prior-work.md` 的"作者后续文章"与 `docs/research/expert-assessment.md` 的安全部分有重叠 → 报告第 7、12 节统一引用，笔记不重复。
- 体积口径统一：磁盘 20.73 GiB（本机实测）≈ 22.3 GB（模型卡）；博客 18.8GB 是旧 P3 数字，不再引用。
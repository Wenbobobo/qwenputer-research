# Qcompute 文档审计与整合提案（DOC-AUDIT）

> 审计时间：2026-10-31 11:10（Asia/Shanghai）· 审计人：文档管理子 agent · 范围：根 *.md、docs/**（.md/.html）、site/、analysis|gpu|demo 下零散 .md
> 原则：**只读 + 复制**。本次未删除、未移动、未改写任何既有文件；仅新增本文件与 `docs/archive/2026-10-31/**` 副本。
> 未纳入：`docs/research/stage9-*.md`（其它 agent 正在写，审计时尚不存在）、`docs/reports/*.html` 内容不改动。

---

## (a) 全量清单

状态图例：**current** 现行 · **outdated** 内容过时 · **duplicate** 重复 · **superseded** 被取代 · **redundant** 冗余（信息已在别处） · **evidence** 历史证据（冻结，保留原位）

### 根目录
| 路径 | 大小 | 最后更新 | 用途 | 状态 | 问题 |
|---|---:|---|---|---|---|
| README.md | 0.5KB | 10-29 01:31 | 项目入口 | **outdated** | 未提 gpu/、analysis/、site/、Pages 站点、Stage 4–8；目录说明停在 Stage 1 |
| ROADMAP.md | 14.5KB | 10-31 10:40 | 阶段计划+勾选 | current / 部分 outdated | 追加式堆叠 8 个阶段；Stage 3「B 设备树」未勾但 S4.4 已完成；「后续外延」中 ROCm、基座下载、H1/H5、memory-disk 规范均已完成仍未勾；Stage 5/7 候选未回勾；头部「阶段评估」停在 10-29 02:35 |
| STATUS.md | 2.2KB | 10-31 10:45 | 当前状态 | current | 只含 Stage 8（合理），需与 DELIVERABLES 分工清楚 |
| DELIVERABLES.md | 9.4KB | 10-31 10:40 | 交付清单+结论 | **outdated（部分）** | 标题日期仍为 2026-10-29；§1「先看什么」只列 stage1–3；§3 证据表停在 Stage 3；§4「未完成」写 GPU/H1/H5/设备树未做，已被 Stage 4–8 推翻；§2b–2g 与 STATUS/各报告重复 |

### docs/ 顶层
| 路径 | 大小 | 最后更新 | 用途 | 状态 | 问题 |
|---|---:|---|---|---|---|
| docs/INDEX.md | 4.4KB | 10-31 10:40 | 文档索引 | **outdated** | 头注「最后更新 2026-10-29 · Stage 1 收尾」；stage1 报告状态仍写「待复现数字收尾」；**缺 stage5–8 共 18 篇 notes 与 stage8-enhancement-portfolio（合计 19 篇）**；memory-disk-v0、device-tree-estimate **重复列两次**；cpu-demo 仍标 BLOCKED；代码表缺 gpu/、site/、analysis/{memory_disk,device,rebuild} |

### docs/reports/（HTML，不改动）
| 路径 | 大小 | 最后更新 | 状态 |
|---|---:|---|---|
| qwenputer-stage1.html | 103KB | 10-29 02:49 | current（阶段报告，冻结） |
| qwenputer-stage2-memory.html | 29KB | 10-29 12:06 | current（冻结） |
| qwenputer-stage3.html | 19KB | 10-29 10:19 | current（冻结） |
| qwenputer-stage4.html | 25KB | 10-29 12:14 | current（冻结） |
| qwenputer-stage5.html | 15KB | 10-29 18:50 | current（冻结） |
| qwenputer-stage6.html | 19KB | 10-29 23:13 | current（冻结） |
| qwenputer-stage7.html | 19KB | 10-29 23:46 | current（冻结） |
| qwenputer-stage8.html | 16KB | 10-31 10:40 | current |

### site/（GitHub Pages 发布产物）
| 路径 | 状态 | 说明 |
|---|---|---|
| site/stages/stage1–8.html | **duplicate（生成物）** | 与 docs/reports/*.html 仅差 +43B 导航注入；属部署产物，保留，但应声明「docs/reports 为源、site/ 为生成」 |
| site/index.html、assets/nav.js、assets/site.css | current | Pages hub |
| .github/workflows/pages.yml | current | 自动部署 |

### docs/research/
| 路径 | 大小 | 最后更新 | 用途 | 状态 |
|---|---:|---|---|---|
| prior-work.md | 15KB | 10-29 01:38 | 前史调研 | current（冻结） |
| parametric-memory.md | 19KB | 10-29 01:40 | 记忆对比/成本模型/H1–H5 设计 | current（H1/H5 结论已被实测更新，索引需注明实测见 stage4-h1 / stage7-h1-inweights / stage5-h5 / stage8-h5-e2e） |
| expert-assessment.md | 9.6KB | 10-29 01:43 | 专家评估+6 月路线图 | current（路线图已过时，冻结为历史） |
| stage8-enhancement-portfolio.md | 18KB | 10-30 15:53 | 计算增强组合 Top3/kill list | current |

### docs/notes/（34 篇）
| 路径 | 大小 | 最后更新 | 主题 | 状态 | 说明 |
|---|---:|---|---|---|---|
| upstream-map.md | 2.6KB | 10-29 01:31 | 上游 | current | 唯一上游入口 |
| claims-audit.md | 6.5KB | 10-29 02:47 | 声明核对 | **outdated** | 文内「最后更新 01:55」；GPU/bit-exact/benchmark 等多项状态已被 Stage 4–7 改变（AMD 上仅 token-level exact 等） |
| repro-log.md | 10KB | 10-29 02:21 | CPU 复现 | evidence | §7–8「未完成/下一步」过时，日志冻结 |
| cpu-checkpoint-demo.md | 3.6KB | 10-29 02:30 | CPU 演示 | evidence | 现行版 |
| cpu-demo.md | 0.7KB | 10-29 02:21 | CPU 演示 | **superseded** | 自述「已合并」到 cpu-checkpoint-demo.md 的跳转桩 |
| weights-map.md | 13.6KB | 10-29 01:54 | 权重/压缩 | current | |
| h2h3-experiments.md | 12.8KB | 10-29 01:54 | 记忆 H2/H3 | evidence | |
| stage3-ideas.md | 5.3KB | 10-29 02:07 | 创意草案 | **superseded** | 被 stage3 报告 + stage8-enhancement-portfolio 取代；B/C 优先级已过时 |
| stage3-a-rebuild.md | 8.0KB | 10-29 02:44 | 审计 chip 侧 | evidence | |
| stage3-a2-base-audit.md | 9.5KB | 10-29 02:48 | 审计 Qwen 侧 | evidence | 与 a-rebuild 同主题 |
| stage3-d-snapshot.md | 10.3KB | 10-29 10:30 | 记忆盘实验 | evidence | |
| memory-disk-format.md | 2.5KB | 10-29 02:38 | 记忆盘格式草案 | **superseded** | 被 memory-disk-v0.md（实现+格式）取代 |
| memory-disk-v0.md | 7.5KB | 10-29 10:30 | 记忆盘 v0 | current | |
| device-tree-estimate.md | 10.8KB | 10-29 10:30 | 设备树 | current | 文件名含 estimate，但内容已是实测构建 |
| stage4-gpu.md | 7.3KB | 10-29 10:38 | GPU 路径 | evidence | |
| stage4-h1.md | 5.3KB | 10-29 12:15 | H1 host | evidence | |
| stage5-gpu-runtime.md | 3.6KB | 10-29 18:50 | 发布模型 GPU | evidence | 性能数字已被 stage6 取代 |
| stage5-h5.md | 8.2KB | 10-29 18:45 | H5-lite | evidence | |
| stage5-keyfix.md | 10.3KB | 10-29 18:30 | SafeStore | current | |
| stage6-quant-ecosystem.md | 19.8KB | 10-29 19:46 | 量化调研 | current | 调研性质，宜归 research/；与 stage7-community 的 ROCm 工具矩阵重叠 |
| stage6-quant-hw.md | 11.9KB | 10-29 19:47 | 硬件实测 | current | |
| stage6-quant-model.md | 3.9KB | 10-29 19:47 | 量化 runtime 原型 | **superseded** | 文内「等待 Q1 marker / 待测」，结果已由 stage6-quant-e2e.md 给出 |
| stage6-quant-e2e.md | 8.4KB | 10-29 20:27 | bnb8/NF4 实测 | current | |
| stage6-trim-runtime.md | 9.9KB | 10-29 21:30 | trim runtime | current | 文首摘要数字（put 10.2s 等）已被文末 §9 Lead 补充及 stage7/8 修正，易误读 |
| stage6-kernels.md | 7.5KB | 10-29 22:18 | FLA 内核 | current | 含两次 Lead 修正；与 stage7-selectfla 主题重叠 |
| stage7-community.md | 23.8KB | 10-29 22:03 | 社区调研 | current | 调研性质，宜归 research/ |
| stage7-selectfla.md | 2.3KB | 10-29 22:25 | selective FLA / AMD 精确性 | current | |
| stage7-h1-inweights.md | 9.9KB | 10-29 22:48 | H1 in-weights | evidence | |
| stage7-int8-head.md | 2.6KB | 10-29 23:14 | int8 head（负结果） | evidence | |
| stage8-fp32chip.md | 6.4KB | 10-29 23:46 | fp32 chip | current | |
| stage8-pages.md | 3.2KB | 10-30 15:56 | Pages 发布流程 | current | 运维手册性质 |
| stage8-speed.md | 2.6KB | 10-30 15:59 | 速度矩阵 | current | **当前速度数字唯一权威** |
| stage8-device-merge.md | 4.3KB | 10-30 16:25 | 设备合并 E2 | current | |
| stage8-h5-e2e.md | 6.4KB | 10-30 16:31 | H5 E3 | current | |

### 其它零散 .md（不改动，仅登记）
| 路径 | 说明 |
|---|---|
| demo/README.md | 演示目录说明，current |
| analysis/experiments/h1/h1_examples.md | H1 原始示例，evidence |
| gpu/quant/ledger.md、gpu/quant/e2e_ledger.md | 量化实验台账，evidence |

**统计**：根 4 + docs/INDEX 1 + reports 8 + research 4 + notes 34 = 51 份受审文档（另 site/ 生成物 8、零散 .md 4）。状态：outdated 4（README / DELIVERABLES / INDEX / claims-audit）、superseded 4（cpu-demo / memory-disk-format / stage3-ideas / stage6-quant-model）、ROADMAP 部分过时；其余 current 或 evidence。

---

## (b) 重复 / 近重复（建议合并）

| # | 重复组 | 现象 | 建议 |
|---|---|---|---|
| D1 | DELIVERABLES §2b–2g ↔ STATUS ↔ ROADMAP 勾选行 ↔ 各阶段 HTML 摘要 | 同一组数字（put 4.10s/16.84、3/12→0/12、Recall 58/84/96…）写了 3–4 遍，口径已漂移（如 trim no-FLA 6.79/9.0 vs 干净复测 13.48–14.71） | 建唯一结果台账 docs/RESULTS.md，其余只引用 |
| D2 | INDEX.md 内部 | memory-disk-v0、device-tree-estimate 各列两次 | 重写 INDEX 时去重 |
| D3 | cpu-demo.md ↔ cpu-checkpoint-demo.md | 前者已自述合并 | 归档前者 |
| D4 | memory-disk-format.md ↔ memory-disk-v0.md | 格式草案被 v0 完整覆盖 | 归档前者 |
| D5 | stage6-quant-model.md ↔ stage6-quant-e2e.md（+ gpu/quant/ledger） | 待测稿 vs 实测稿 | 归档前者 |
| D6 | stage6-kernels ↔ stage7-selectfla ↔ stage8-speed（+ stage6-trim-runtime §9） | FLA 精确性/速度结论分散 3–4 篇，含多次「Lead 修正」 | 主题综述 topics/gpu-runtime-and-speed.md（原文留作证据） |
| D7 | stage6-quant-ecosystem ↔ stage7-community ↔ stage6-quant-hw | ROCm/gfx1100 工具支持矩阵重复 | 两篇调研移入 research/；硬件事实留 quant-hw |
| D8 | stage3-d-snapshot ↔ memory-disk-v0 ↔ stage5-keyfix ↔ stage5-h5 ↔ stage8-h5-e2e | 记忆盘/SafeStore/H5 是同一产品线 | 主题综述 topics/memory-disk.md |
| D9 | stage4-h1 ↔ stage7-h1-inweights ↔ parametric-memory §5 | H1 设计与两次实测分散 | 主题综述 topics/h1-overwrite-semantics.md |
| D10 | site/stages/*.html ↔ docs/reports/*.html | 生成副本 | 保留；在 INDEX 与 Pages 手册中声明源/生成关系 |
| D11 | README ↔ upstream-map ↔ INDEX 末节「冗余说明」 | INDEX 承诺 README 只放链接，但 README 未指向 Pages 等新入口 | README 精简为入口+链接 |
| D12 | GPU 窗口规则：ROADMAP 硬性约束 / Stage4 约束 / Stage5 风险 / STATUS「GPU 纪律」/ 各 notes「hard stop 00:25」 | 5 处表述不一 | 抽到 docs/GPU-DISCIPLINE.md |

---

## (c) 过时文档（归档或冻结标注）

| 文件 | 处理 | 理由 |
|---|---|---|
| docs/notes/cpu-demo.md | 归档（已复制） | 跳转桩，已合并 |
| docs/notes/memory-disk-format.md | 归档（已复制） | 被 memory-disk-v0 取代 |
| docs/notes/stage6-quant-model.md | 归档（已复制） | 「等待实测」稿，结果在 quant-e2e |
| docs/notes/stage3-ideas.md | 归档（已复制） | 草案，被 stage3 报告 + stage8 portfolio 取代 |
| docs/INDEX.md | 重写（快照已复制） | 缺 19 篇、重复 2 条、状态错 |
| DELIVERABLES.md | 重写（快照已复制） | §1/§3/§4 停在 Stage 3 |
| README.md | 重写（快照已复制） | 停在 Stage 1 |
| ROADMAP.md | 瘦身+回勾（快照已复制） | 多处已完成未勾 |
| docs/notes/claims-audit.md | 原位追加「v2 截至 Stage 8」状态列 | 多项状态已变 |
| repro-log §7–8、stage6-trim-runtime 摘要、stage5-gpu-runtime 性能数 | **不归档**，文首加一行「历史证据；最新数字见 RESULTS.md」 | 证据保留原位 |

---

## (d) 缺失文档

| 缺失 | 建议路径 | 内容要点 |
|---|---|---|
| 交接文档 | docs/HANDOFF.md | 一页：项目是什么、当前最优配置、5 分钟跑通 CPU/GPU、在途工作（stage9）、谁写哪里、禁区（.data、repro 上游源码不改） |
| 术语表 | docs/GLOSSARY.md | chip 层/Qwen 层、trim、head-trim、FLA/selective FLA、lock divergence、head mismatch、anti-sink、mu 列、GDN-as-RAM、SafeStore、NameGuard、memory-disk、H1–H5、E1–E3、τ、rv-tiny、ADD8/SORT4/DADD、bit-exact vs token-level exact |
| 环境搭建 | docs/ENVIRONMENT.md | CPU：uv / riscv .venv / torch 2.14.1+cpu / transformers 5.19.0；GPU：WSL Ubuntu-24.04 + ROCm 7.2.1 + torch 2.9.1 + `gpu/run_py.sh`；环境变量 TRIM_SELECTIVE_FLA / TRIM_FP32_CHIP / TRIM_ALLOW_FLA / TRIM_INT8_HEAD；内存门槛（CPU 加载需 ≥25GB free） |
| GPU 纪律 | docs/GPU-DISCIPLINE.md | 窗口 09:30–12:30 / 15:30–00:30；静默 00:30–09:30 / 12:30–15:30；00:25 hard stop；单任务串行；*_DONE marker 交接；空闲显存检查；checkpoint 规则 |
| 文件地图 | docs/FILEMAP.md | analysis/{weights,rebuild,experiments,memory_disk,device}、gpu/{trim,kernels,quant,out}、demo/、repro/、site/、.data/、.research/、.local/ 的用途；哪些是证据、哪些是生成物 |
| 结果台账 | docs/RESULTS.md | 关键数字唯一来源：指标、值、日期、配置、证据路径、是否被取代（解决 D1） |
| 决策日志 | docs/DECISIONS.md | 用户/Lead 决策：不做基座+增量安装器（S4）、不做自定义内核（S7）、correctness-first 默认、portfolio kill list 等 |
| Pages 手册 | docs/ops/pages.md（由 stage8-pages.md 提升） | 每阶段 HTML 上线流程 |

另：根目录存在异常目录 `%SystemDrive%`（疑似某工具误展开环境变量产生），建议 Lead 核查后清理（本次未触碰）。

---

## (e) 目标文档树（提案）

```
README.md                  # 入口：简介 + Pages 链接 + 指向 docs/INDEX.md / HANDOFF.md
ROADMAP.md                 # 总目标、阶段表(✅/🟡)、当前阶段任务、下一阶段候选；历史详情下沉
STATUS.md                  # 当前阶段增量 + 当前最优配置 + 阻塞
DELIVERABLES.md            # 先看什么(Pages + stage1–8) / TL;DR 10 条 / 证据入口 → RESULTS.md
docs/
  INDEX.md                 # 全量索引
  HANDOFF.md               # 新
  GLOSSARY.md              # 新
  ENVIRONMENT.md           # 新
  GPU-DISCIPLINE.md        # 新
  FILEMAP.md               # 新
  RESULTS.md               # 新：数字唯一来源
  DECISIONS.md             # 新
  DOC-AUDIT.md             # 本文件
  reports/                 # 阶段 HTML（源）；site/stages 为其生成物
  research/                # prior-work, parametric-memory, expert-assessment,
                           # stage8-enhancement-portfolio, stage9-*（在写）,
                           # quant-ecosystem（自 notes 移入）, rocm-community（自 notes 移入）
  topics/                  # 新：按主题的现行综述（引用 notes 证据）
    audit-bitexact.md
    memory-disk.md
    h1-overwrite-semantics.md
    gpu-runtime-and-speed.md
    quantization.md
    device-tree.md
  notes/                   # 证据日志，冻结（保留 stageN- 原名，不再改写）
  ops/
    pages.md
  archive/
    2026-10-31/            # 本次副本（见 f）
```

---

## (f) 合并 / 归档计划（source → target）

### F1. 本次已执行（仅复制，原文件未动；SHA-256 前 12 位校验一致）
| 源 | 副本 | SHA-256(12) | 原因 |
|---|---|---|---|
| docs/notes/cpu-demo.md | docs/archive/2026-10-31/notes/cpu-demo.md | 7DA0D8E1786B | superseded |
| docs/notes/memory-disk-format.md | docs/archive/2026-10-31/notes/memory-disk-format.md | D0CD46305F2A | superseded |
| docs/notes/stage6-quant-model.md | docs/archive/2026-10-31/notes/stage6-quant-model.md | F2EA6797196C | superseded |
| docs/notes/stage3-ideas.md | docs/archive/2026-10-31/notes/stage3-ideas.md | F3BA8ADA59B3 | superseded |
| docs/INDEX.md | docs/archive/2026-10-31/root-snapshot/INDEX.md | 3A77759553B4 | 重写前快照 |
| README.md | docs/archive/2026-10-31/root-snapshot/README.md | 826E5072BF81 | 重写前快照 |
| DELIVERABLES.md | docs/archive/2026-10-31/root-snapshot/DELIVERABLES.md | 09955050552B | 重写前快照 |
| ROADMAP.md | docs/archive/2026-10-31/root-snapshot/ROADMAP.md | 950A376731DE | 重写前快照 |
| STATUS.md | docs/archive/2026-10-31/root-snapshot/STATUS.md | 5FD501FF784F | 重写前快照 |

> 快照取自 2026-10-31 10:40–10:45 版本。若 Lead 整合前根文件又被修改，应重新拍快照。

### F2. 待 Lead 批准后执行（合并/移动/删除原件）
| 源 | 目标 | 动作 |
|---|---|---|
| notes/cpu-demo, memory-disk-format, stage6-quant-model, stage3-ideas | （已在 archive） | 删除原件，INDEX 指向 archive |
| DELIVERABLES §2/2b–2g、STATUS 数字、ROADMAP 勾选行数字 | docs/RESULTS.md | 抽取为台账，原处改链接 |
| ROADMAP Stage 1b…Stage 8 详细任务段 | ROADMAP 阶段表 + archive 快照 | 折叠；回勾 Stage 3-B、「后续外延」、S5/S7 候选 |
| ROADMAP Stage4/5 约束段 + STATUS「GPU 纪律」 | docs/GPU-DISCIPLINE.md | 抽取统一 |
| ROADMAP Stage4「工具」行 + repro-log §1 + DELIVERABLES §5 | docs/ENVIRONMENT.md | 抽取统一 |
| INDEX「代码与数据」表 + README 目录行 | docs/FILEMAP.md | 扩展 gpu/、site/、analysis 子目录 |
| ROADMAP 用户指令行 + portfolio kill list | docs/DECISIONS.md | 抽取 |
| stage3-a-rebuild + stage3-a2-base-audit + claims-audit | docs/topics/audit-bitexact.md | 综述（原文留 notes） |
| stage3-d-snapshot + memory-disk-v0 + stage5-keyfix + stage5-h5 + stage8-h5-e2e | docs/topics/memory-disk.md | 综述 |
| stage4-h1 + stage7-h1-inweights + parametric-memory §5 | docs/topics/h1-overwrite-semantics.md | 综述 |
| stage4-gpu + stage5-gpu-runtime + stage6-trim-runtime + stage6-kernels + stage7-selectfla + stage8-fp32chip + stage8-speed | docs/topics/gpu-runtime-and-speed.md | 综述（数字以 stage8-speed 为准） |
| stage6-quant-hw + stage6-quant-e2e + stage7-int8-head | docs/topics/quantization.md | 综述 |
| device-tree-estimate + stage8-device-merge | docs/topics/device-tree.md | 综述 |
| notes/stage6-quant-ecosystem.md | docs/research/quant-ecosystem.md | 移动 |
| notes/stage7-community.md | docs/research/rocm-community.md | 移动 |
| notes/stage8-pages.md | docs/ops/pages.md | 移动 |
| notes/claims-audit.md | 原位追加 v2 状态列 | 更新 |

---

## 附：5 个最重要整合动作
1. **建 docs/RESULTS.md 作为数字唯一来源**，DELIVERABLES/STATUS/ROADMAP 改为引用，消除口径漂移。
2. **重写 docs/INDEX.md**：补齐 19 篇缺失（stage5–8 共 18 篇 notes + portfolio）及 stage9 新文、去重、修正状态、说明 site/ 与 gpu/。
3. **瘦身 ROADMAP**：阶段表 + 当前阶段，回勾陈旧未勾项；历史详情以 archive 快照为准。
4. **新增 HANDOFF + ENVIRONMENT + GPU-DISCIPLINE**（GPU 窗口/hard stop/marker 规则散落 5 处）。
5. **重写 DELIVERABLES**：「先看什么」改为 Pages + stage1–8；删除失效的 §4「未完成」；证据表指向 topics/ 与 RESULTS。
---

## (g) Lead 已执行的整合动作（2026-10-31 11:20–11:35）

| # | 动作 | 结果 |
|---|---|---|
| 1 | 新建 docs/RESULTS.md | 唯一权威数字台账（A–H 八类，含条件与证据路径） |
| 2 | 新建 docs/GPU-DISCIPLINE.md | 统一窗口/并发/禁止事项/验收 gate（替代 5 处分散表述） |
| 3 | 新建 docs/GLOSSARY.md、docs/DECISIONS.md | 术语与关键决策（D1–D15） |
| 4 | 新建 docs/START-HERE.md | 交接启动 prompt（结论/环境/地图/纪律/gate/路线/工作方式） |
| 5 | 重写 README.md、docs/INDEX.md、DELIVERABLES.md | 入口指向 Pages/START-HERE/RESULTS；INDEX 覆盖全部 30 notes + 6 research；DELIVERABLES 重写并删除失效 §4 |
| 6 | 移动 4 篇被取代笔记到归档 | cpu-demo / memory-disk-format / stage6-quant-model / stage3-ideas（副本已在 docs/archive/2026-10-31/notes/，原文件移出活跃目录） |
| 7 | 删除根目录 `%SystemDrive%` 垃圾目录 | 内容为 Windows Caches 的误展开副本（约 970KB），已确认无关后删除 |
| 8 | 补勾 ROADMAP 陈旧完成项 + 顶部状态提示 | Stage3-B 设备树、S5.1–S5.4 已回勾；新增 Stage 9 章节 |
| 9 | STATUS 增加 Stage 9 进展 | 文档整合与专家报告进度 |

### 未采纳/延后
- docs/topics/ 六个综述：暂缓（INDEX 已按主题分组，避免再增一层文档）；如后续需要可用 notes 聚合生成。
- docs/ENVIRONMENT.md / FILEMAP.md：并入 START-HERE §3/§4 与 GPU-DISCIPLINE，不再单列。
- claims-audit.md 重写：保留为历史审计（文首已提示部分状态被 Stage 4–8 更新），不重写，避免丢失当时证据。
- 两篇调研（stage6-quant-ecosystem、stage7-community）暂留 notes/；INDEX 已按主题归入 GPU/性能组，移动会破坏既有链接。
# Stage 9 综合决策摘要：场景与突破方向

> 汇总两份专家输入：[stage9-application-scenarios.md](stage9-application-scenarios.md)（产品/场景向）与 [stage9-technical-breakthroughs.md](stage9-technical-breakthroughs.md)（技术/研究向）。数字以 [docs/RESULTS.md](../RESULTS.md) 为准。

## 1. 一句话

**近期靠“可审计记忆盘”（H5 语义路由 + exact get + SafeStore/memory-disk），中期靠“可审计的确定性设备树”（ADD8→SORT4/DADD 真 merge），fp32 局部混合精度与 compiler-exactness toolkit 是让两者可信的基础设施；“in-weights 介质更优”不是卖点。**

## 2. 双专家收敛点（共识）

| 主题 | 共识 |
|---|---|
| 最强产品路线 | H5 语义路由 + 精确 get + SafeStore + memory-disk 的“离线可审计记忆盘” |
| 最强研究路线 | 编译神经外设（设备树）从 ADD8 到真 merge 的闭环 |
| 必须先做的可信基础 | fp32 chip（长序列正确性）+ 局部混合精度消融 + compiler-exactness audit |
| in-weights 记忆定位 | 限定在离线/隐私/快照携带/审计；不宣称准确率或速度优势 |
| 量化的定位 | 只省显存不省算力；bnb8 仅作显存保底，NF4/int8 head 弃用 |
| 速度目标 >20 tok/s | 条件性推进（native int8 GEMV / tied embedding / causal_conv1d），每项过 lock gate |

## 3. 优先级（Stage 9/10）

| 优先级 | 方向 | 验收 | 预计 |
|---|---|---|---|
| P0 | H5 reader 集成 + 可审计记忆盘 demo | 4B reader 上 top-1/3/5 vs RAG 四组；未命中拒答；审计链完整 | 6–8 周 |
| P0 | ADD8-only rv-tiny 端到端 merge | 真实 merge 产物；IR 全量 0 mismatch；HF 1000 条 0 错；RV32 回归不变；namespace/owner/exit/reset 测试 | 3–6 周 |
| P1 | fp32 局部混合精度 | ≥60 paired 长序列，lock 目标 0/60；head=0；显存低于全 chip fp32 | 3–6 周 |
| P1 | compiler-exactness toolkit | 统一报告 IR/CPU/NVIDIA/AMD 的 bit/ULP/token/lock 四级精度 | 4–8 周 |
| P2 | >20 tok/s | 逐项（int8 GEMV / tied embed / causal_conv1d）单独 lock gate | 6–12 周 |
| P2 | memory-disk v1 | 权重/编译 hash、append/delete/tombstone、事务/权限/TTL、容量错误码 | 3–5 周 |

## 4. 可证伪命题（近期验证）

- T1：exact get 在“更新+审计”查询上比 RAG 省一个数量级 token 且准确率不降。
- T2：ADD8 真 merge 后 token-level 正确且不破坏 RV32。
- T3：只需 fp32 敏感子层即可保持长序列 lock（而非全 chip fp32）。
- T4：in-weights 的差异化只在离线/审计/快照，而非速度或准确率。
- T5：compiler-exactness audit 可被社区采用为 benchmark。

## 5. 不做什么（kill list 摘要）

默认全模型 fp32、CPU-GPU 混合执行、NF4/无保护 INT8、通用 regex/大排序设备、跨模型共享 chip 层（无第二基座证据前）、把 in-weights 包装成“更聪明的记忆”、以 Recall@1 作为产品指标、把 IR probe 说成端到端 merge 完成。

## 6. 对“突破”的判断

- **最可能带来真实突破的两条**：① 可审计设备树（研究/论文/工具）；② 语义路由+精确读回（产品/成本）。
- **fp32 局部化**更可能是“必要的工程论文”，不是产品突破。
- **in-weights 介质本身**在现有证据下不构成突破，最多是产品叙事的一部分。
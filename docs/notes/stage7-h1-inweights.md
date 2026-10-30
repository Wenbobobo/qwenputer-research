# Stage 7 H1：发布模型 in-weights store 条件

日期：2026-10-29，北京时间。GPU 工作约 22:26–22:46，早于 00:25 hard stop；先 5 例，再扩展为 10 例。seed=0，recall-0…recall-9。

## 结果与判断

**C5 in-weights：changed 严格 10/10=100%，普通事实 17/20=85%，所有题 27/30=90%；changed new-first 10/10=100%。** Wilson 95% 区间分别为 changed 72.2–100%、facts 64.0–94.8%、all 74.4–96.5%。最初 5 例 checkpoint 为 5/5、8/10、13/15。

结果方向符合 Stage 2 的预测：发布模型自身文件系统能够表现出 host overwrite 一样的高 changed 严格分，未出现 append-only/context 主比较中的低 changed 分。但本轮不是等价性检验，不能声称 in-weights 与 host overwrite 等价或更优；尤其存在 3 次历史会话的 calc lock 分歧，见下文。

## 设计和 harness

运行器：[run_inw.py](../../analysis/experiments/h1_inweights/run_inw.py)。复用 H1 的 h1_gen.recall_cases(n,0)，即 AST 载入 upstream storebench 的相同生成器；每例总共 8 项事实分配到 4 个会话，最后会话更新其中 2 项；3 个问题包含 2 个普通事实、1 个更新事实。未把每句拆开或额外强制保存，以保留与 H1/upstream recall 的可比性。

模型为 released Qwen3.5-4B-Qwenputer-Filesystem，使用 gpu/trim/trim_chat.py.install，lossless Qwen trim + fast.compact(qwen=False,n_chip=12)，整模型 GPU，无 head-trim、无 FLA。两个进程加载时均断言 248 个裁掉矩阵的元素全部为 0。初次/扩展初始化约 124.8/131.7s，加载后 GPU 参数内存约 12.37GB。所有 70 轮 generate wall 合计 909.71s；每例约 79–99s，比预估的 30–60s 慢，主要因历史阶段常输出到 256 token 上限并重复工具调用。

每个历史会话只提供 [system=ST.MEM_SYSTEM,user=原始会话]，工具为 (CALC_TOOL,*ST.TOOLS)，通过 RvChat.prompt_ids 使用原生模板（greedy、enable_thinking=False）。模型自行生成 put/get/list/delete/calc，文件系统由模型的 chip/store heads 执行；Python StoreSession 只作为 lockstep 参考检查，不向模型强制注入 host 工具答案。历史阶段预算 256 text tokens，问答阶段 200，与 H1 host 条件一致；max_chip=200000，check_head=True。

跨历史会话携带 chat.save_store() 的 heads snapshot 与参考 device；会话 4 后写出每例 npz。每个问题重建空对话，并以 chat.load_store_file(npz) 载入同一最终 snapshot；问题之间不携带问答上下文或新写入的 store。这是用户允许的“fresh session with memory loaded”，并非每问重新加载全部权重。两组运行均以 shell timeout 强制限制到 2026-10-30 00:25，另有 case 开始前剩余时间检查。

## 计分

严格规则复用 H1 own_text + score_fact：删除完整 tool_call/tool_response/think 块，gold 大小写不敏感地出现在模型自身答案中；更新题还要求 old 不出现。补充 new-first：gold 出现且在任何 old 之前（old 未出现亦通过）。主分使用完整 own_text，answer 字段仅为展示截断。未闭合的工具调用保留 upstream own_text 的处理方式；三个错题均未出现 gold，因此不会因该细节改变分数。

## 每例结果

| case | Changed 严格 / new-first | Facts | All | runtime 备注 |
|---|---:|---:|---:|---|
| recall-0 | 1/1 / 1/1 | 2/2 | 3/3 | clean |
| recall-1 | 1/1 / 1/1 | 2/2 | 3/3 | s1 calc lock 分歧 |
| recall-2 | 1/1 / 1/1 | 2/2 | 3/3 | s1 calc lock 分歧 |
| recall-3 | 1/1 / 1/1 | 1/2 | 2/3 | partner 未保存，反复 get 到上限 |
| recall-4 | 1/1 / 1/1 | 1/2 | 2/3 | partner 未保存，反复 get 到上限 |
| recall-5 | 1/1 / 1/1 | 2/2 | 3/3 | s2 calc lock 分歧 |
| recall-6 | 1/1 / 1/1 | 2/2 | 3/3 | clean |
| recall-7 | 1/1 / 1/1 | 1/2 | 2/3 | sister 未保存，反复 get 到上限 |
| recall-8 | 1/1 / 1/1 | 2/2 | 3/3 | clean |
| recall-9 | 1/1 / 1/1 | 2/2 | 3/3 | clean |
| 合计 | **10/10 / 10/10** | **17/20** | **27/30** | 70 turns，head mismatch=0 |

全部 30 问答轮无 lock_error。40 个历史轮中 32 轮至少一次 put；3 个普通事实错题都对应相关历史会话没有保存该值。模型有时声称“saved”但没有调用 put，也出现不必要的 delete 和更换 key 名称，故普通事实漏存与 schema 一致性仍是问题。

## 与现有 H1 比较

完整历史结果（C1/C3 pooled seeds0/1/2 各90例；C2/C4 seed0 各30例；C5 seed0 10例）：

| 条件 | Changed 严格 | Changed new-first | Facts | All |
|---|---:|---:|---:|---:|
| C1 timestamp/latest-wins | 42/90=46.7% | 90/90=100% | 179/180=99.4% | 221/270=81.9% |
| C2 append-only host | 7/30=23.3% | 7/30=23.3% | 58/60=96.7% | 65/90=72.2% |
| C3 plain context | 14/90=15.6% | 48/90=53.3% | 180/180=100% | 194/270=71.9% |
| C4 overwrite host | 27/30=90.0% | 27/30=90.0% | 43/60=71.7% | 70/90=77.8% |
| **C5 in-weights trim no-FLA** | **10/10=100%** | **10/10=100%** | **17/20=85%** | **27/30=90%** |

为避免只用不同 n 比较，另读取已有 H1 原始 JSONL 的相同 seed0 recall-0…9，未重新运行模型：

| 同 10 例条件 | Changed 严格 | Facts | All |
|---|---:|---:|---:|
| C1 | 3/10=30% | 20/20=100% | 23/30=76.7% |
| C2 | 4/10=40% | 20/20=100% | 24/30=80% |
| C3 | 3/10=30% | 20/20=100% | 23/30=76.7% |
| C4 | 8/10=80% | 14/20=70% | 22/30=73.3% |
| C5 | 10/10=100% | 17/20=85% | 27/30=90% |

这支持“高 changed 分不要求把工具执行放在 host Python 上”的可行性结论；不支持“存储介质本身造成优势”的因果结论。C5 与 C4 的 prompt 并非逐 token 相同：C5 原生 MEM_SYSTEM 包含 calc，C4 当时工具面去掉 calc，且 base 与 released filesystem checkpoint、解码运行时也不同。

## 精确性、故障与敏感性

1. **不能称此次 no-FLA GPU 运行全程 bit-exact。** A7_DONE 报告 AMD gfx1100 上 no-FLA GPUcheck 同样存在约 5.96e-8 残差。零权重裁剪检查证明数学上不丢非零权重，并不证明 ROCm bf16 整个执行 bitwise 等价。
2. 本轮 recall-1/s1、recall-2/s1、recall-5/s2 在模型生成额外 calc 调用后出现相同类型的 lock 分歧：got V(0)，want F0(32716)，位置分别228、249、232。generate 在首次分歧停止，后续会话使用已经保存的 store snapshot 继续。具体根因未定位；不能仅凭 A7 的浮点残差把此次故障归因于 FLA/ROCm。head_mismatch 始终0，但该诊断与 lock 验证覆盖不同。
3. 全部70轮中的3轮有 lock_error，均为上述历史 calc，全部问答轮 clean。把受影响的整例排除后，仅作敏感性分析：7例 changed **7/7=100%**，facts **11/14=78.6%**，all **18/21=85.7%**。这不是重新定义主样本，也不能修复端到端验证缺口。
4. 原始 store_after 展示因 tuple key 解码 bug 返回 <error>；不影响模型执行、npz 保存或答案评分。最终代码修复了该显示函数，但本次已在运行的进程仍用旧函数。汇总中的 transcript_store_views 为完整已生成工具调用的独立重建，明确标作推断，不是 GPU heads 的解码证明。原始 JSONL 保留错误供审计。不要使用原始 question 的 gold_in_store/old_in_store 字段作结论；这些字段来自失败的显示函数，聚合脚本另行根据 transcript 重算。
5. transcript 显示，10 个 changed 问题新值都被 put；只有4例在某个会话结束时曾保留旧值，其中 recall-1/7/9 同名更新后不再有旧值，recall-5 新旧值因 key 不同并存却仍答对。因此不能把10/10全部归因于“覆盖删除旧值”。保存策略、键选择、检索与最终表达都有贡献。会话内曾保存后删除的值不一定计入“会话末曾保留”。
6. **小 n、单 seed、提示格式敏感。** changed 100% 的95%下界只有72.2%；C1严格/lenient差异已表明表达方式会大幅影响主分。没有要求短答、没有事后改 system 强制 put，也没有针对错误重试。本轮原生 store 提示含 calc，模型有重复工具行为，256/200预算会截断未完成调用。

## 对“90 vs 30”叙事的含义

现在可以补充：在本小样本上，发布模型自己的文件系统也得到高 changed 严格分（100%），与 host overwrite 的方向一致。H1 的90% vs append-only23.3%不是只能在host实现中出现的现象。

但不能宣传“in-weights memory 本身让模型从30%提高到90%”。同一批事实下，时间戳上下文的 lenient 已达100%；短答探针也能大幅提高严格分。C5的10个更新题并非全部实际测试同key覆盖，且普通事实仍有漏存。较准确的解释是：**保存、更新语义、键命名、检索和答案格式共同决定该 benchmark 表现；in-weights实现能够承载这种行为，尚未证明其介质具有独立优势。** 要做介质等价性或因果归因，下一轮需相同checkpoint/prompt/tools/预算的host参考、扩大多seed，并先解决calc lock分歧。

## 原始材料

- [运行器](../../analysis/experiments/h1_inweights/run_inw.py)，[聚合器](../../analysis/experiments/h1_inweights/analyze_inw.py)，[同例基线比较](../../analysis/experiments/h1_inweights/compare.py)。
- [问题原始 JSONL](../../analysis/experiments/h1_inweights/out/inw_questions.jsonl)，[历史会话 JSONL](../../analysis/experiments/h1_inweights/out/inw_sessions.jsonl)，每例 memory 文件在 analysis/experiments/h1_inweights/out/npz/。
- [结果 JSON](../../analysis/experiments/h1_inweights/h1_inweights_results.json)，[同例基线 JSON](../../analysis/experiments/h1_inweights/matched_baselines.json)。
- [首5例日志](../../analysis/experiments/h1_inweights/run5.log)，[扩展日志](../../analysis/experiments/h1_inweights/run10.log)。两次 runtime 初始化保留在各日志中；out/runtime.json 是扩展进程的加载元数据。
- [GPU释放标记](../../gpu/D7_DONE)。

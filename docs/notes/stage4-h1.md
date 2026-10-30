# Stage 4 H1：覆盖语义与上下文基线

## 目的与设计

检验‘changed-fact 优势来自覆盖语义而非记忆介质’。复用 upstream storebench.py 的 recall_cases(n, seed)，通过 AST 仅载入生成器纯 Python 常量/函数；未导入或修改 Chipformer。每例 4 个历史会话、8 项事实，第 4 会话改变其中 2 项，再问 3 题。

C1：四段历史加日期时间，提示最近值为当前值；无工具。C2：Host Python add_entry/list_entries/get_entries，永远 append，同名条目均按追加顺序返回，不可覆盖/删除。C3：upstream context 形式，无时间戳和优先规则，粘贴历史。C4：Host Python put/get/list/delete（含 append），put 覆盖同名值。

同一 Qwen3.5-4B BF16、greedy、thinking disabled；回答最多 200 tokens、历史保存最多 256 tokens。WSL Ubuntu 24.04 按 gpu/run_py.sh 运行。等 A_DONE 后检查约 19GB 空闲显存才加载；所有 GPU 工作在 12:30 北京时间前完成。先做每条件 n=3 pilot 验证，再做正式条件。C1/C3 seed 0、1、2，各 30 例；C2/C4 seed0 各 30 例。C1/C3 各 90 changed +180 unchanged，C2/C4 各 30 changed +60 unchanged。C2/C4 使用完全相同 seed0 case/questions。

## 提示与计分

C1 问题消息粘贴有时间戳的四段历史，并提示‘如果事实随时间改变，最近会话值为当前值’。C3 仅标注 Session 1–4 并粘贴历史。C2 系统提示声明 append-only：更新时对同名 add_entry，查询时 list_entries 后 get_entries。C4 沿用 upstream MEM_SYSTEM 语义：put 新值到同名以覆盖，答题前 list/get。C2/C4 历史阶段逐条运行，工具由 Python 执行；答题阶段空上下文，保留 store state。

严格主分复用 upstream score_fact：模型自身答案中出现 gold（不区分大小写），changed 题还要求旧值不得出现。工具调用/回复和 think 区域不计分。报告 Wilson 95% CI，以及普通事实和所有题准确率。补充诊断 new-first：changed 答案出现新值，且新值先于任何旧值；只用于解释，不能替代主分。

## 结果

| 条件 | Changed 严格 | Changed new-first | 普通事实 | 所有题 |
|---|---:|---:|---:|---:|
| C1 timestamp/latest-wins | 42/90 = 46.7% [36.7–56.9] | 90/90 = 100% [95.9–100] | 179/180 = 99.4% [96.9–99.9] | 221/270 = 81.9% [76.8–86.0] |
| C2 append-only host | 7/30 = 23.3% [11.8–40.9] | 7/30 = 23.3% [11.8–40.9] | 58/60 = 96.7% [88.6–99.1] | 65/90 = 72.2% [62.2–80.4] |
| C3 plain context | 14/90 = 15.6% [9.5–24.4] | 48/90 = 53.3% [43.1–63.3] | 180/180 = 100% [97.9–100] | 194/270 = 71.9% [66.2–76.9] |
| C4 overwrite host | 27/30 = 90.0% [74.4–96.5] | 27/30 = 90.0% [74.4–96.5] | 43/60 = 71.7% [59.2–81.5] | 70/90 = 77.8% [68.2–85.1] |

CI 为二项比例 Wilson 95% 近似。C1 seed0/1/2 changed 分别 12/30 (40.0%), 16/30 (53.3%), 14/30 (46.7%)。C3 分别 9/30 (30%), 3/30 (10%), 2/30 (6.7%)。C1-C3 pooled 差 +31.1pp，two-sided Fisher p=1.02e-5。C4-C2 差 +66.7pp，Fisher p=2.44e-7。

## 诊断与限制

- H1 按严格预注册阈值：C1=46.7%，没有达到 >=80%；C2=23.3%，远低于 C4=90.0%。结果支持在此实现中 overwrite 有助于处理相冲突历史，但并不支持‘时间戳+最新优先的粘贴上下文达到 80%’。
- C1 的 48 个严格 changed 错题全都先把新值明确说成当前值，再提到旧值作历史比较；new-first 实际 90/90。C3 的 76 个严格错题也都同时提新旧值，但仅 48/90 先讲新值。主分忠实采用 upstream 的旧值不得出现，但它把‘比较时提旧值’和‘把旧值当作答案’视为同一种失败。不要仅凭严格百分比判断模型没识别新值。
- C2 120 个历史会话有 112 个至少产生一条 save 调用；90/90 答题轮有读取工具调用。30 个 changed 中，4 个旧值未存、新值已存（4/4 正确）；26 个新旧均存，仅 3/26 严格正确，模型常复述两个值。C4 只有 73/120 历史会话产生 save 调用；changed 中 18 个旧值未存、新值已存（18/18 正确），9 个新旧均曾存、put 覆盖后（9/9 正确），3 个新旧均未存（0/3）。因此 90% 不能单独归因于覆盖操作，保存阶段是否将旧值写入是混杂。C2/C4 工具面和提示也不同。
- 输出格式敏感性探针（非注册条件）：C1/C3 另加 ‘Answer in one short sentence.’，seed0/1 各 30 例。C1s 60/60 changed 正确；C3s 56/60。格式/长度会显著改变严格分，但属后置分析，不纳入主比较；seed2 探针未运行。静默期未继续 GPU。
- 未运行 Qwenputer in-weights store 条件；故不能推断 host overwrite 与 in-weights medium 等价或谁更好。当前实验测试的是同一基础模型下上下文与两种 host 存储语义。

## Verbatim 示例

同一 seed0 case recall-0 的完整逐字文本位于 analysis/experiments/h1/h1_examples.md：C1/C3 prompt 与逐题回答；C2/C4 历史阶段模型输出、host 调用、存储快照及工具问答。

## 文件

生成器适配 h1_gen.py；运行器 h1_run.py；汇总 h1_analyze.py；case 级 JSONL 位于 out/；核心四条件合并 JSONL 为 h1_results.jsonl；汇总 h1_results.json；日志 full.log；pilot 单独保留不纳入结果。

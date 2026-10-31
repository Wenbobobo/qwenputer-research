# DECISIONS —— 关键决策记录（ADR 简版）

> 目的：让接手者理解决策背后的取舍，避免重复试错。新决策追加在末尾，不删除历史。

| # | 决策 | 日期 | 理由/证据 |
|---|---|---|---|
| D1 | 以发布 checkpoint（Filesystem 版）为唯一复现对象，不重建基座 | 10-29 | 重建需要基座+编译器版本锚定，风险高；发布件可审计（stage3-a） |
| D2 | 无损压缩采用 core（稠密核心+CSR），不采用 base+delta | 10-29 | delta 需要基座与 CUTLASS 级验证；core 已 10/10 SHA 验证（weights-map） |
| D3 | 记忆结论：in-weights 是精确 KV 外设，不包装成"更聪明的记忆" | 10-29 | H1 90 vs 30 主要是覆盖语义+回答格式；H1 in-weights 100% 与 host 90% 同档（stage4-h1、stage7-h1-inweights） |
| D4 | 量化只量化 Qwen 12–43 层；chip 层保持 bf16 | 10-29 | chip 是编译状态机，量化即毁；保留 mu 列需 PreserveColumns（stage6-quant-e2e） |
| D5 | 不再把"GPU bit-exact"作为目标；AMD 验收 = token-level + head/lock | 10-30 | 屏蔽 FLA 的纯 torch 也 MISMATCH；作者 bit-exact 是 NVIDIA L4（stage7-selectfla） |
| D6 | 默认正确性配置 = trim + fp32 chip + no-FLA | 10-31 | 长序列 lock bf16 3/12 → fp32 0/12；速度 11.8–15.7/14.7 tok/s（stage8-fp32chip、stage8-speed） |
| D7 | FLA 只作 opt-in 速度模式，需 head/lock 检查 | 10-30 | FLA 让 chip 非 bit-exact；selective 可减少影响但不能消除（stage7-selectfla） |
| D8 | int8 head / NF4 弃用；bnb8 仅作"显存 ≤12GB"保底 | 10-30 | int8 head 更慢更占显存；NF4 质量崩；bnb8 正确但 1–2 tok/s（stage6-quant-e2e、stage7-int8-head） |
| D9 | 长名键问题用 SafeStore 协议层解决，不改电路 | 10-29 | 电路 key 只取前 6 token；SafeStore 36-bit tag + salt 通过 38/38 测试、512 对抗长名 0 碰撞（stage5-keyfix） |
| D10 | 设备树方向继续，但先做 ADD8-only rv-tiny 端到端 | 10-30 | merge_probe 证明 0 额外层；生产阻塞是 namespace/owner/exit/tokenizer（stage8-device-merge） |
| D11 | H5 产品形态 = CPU bge 路由 + exact get + SafeStore，不做 RAG paste | 10-30 | Recall@5 87–96% vs exact get 45 token；RAG 12k token（stage8-h5-e2e） |
| D12 | 以 docs/RESULTS.md 为唯一数字台账 | 10-31 | 同一数字在 3–4 处漂移（6.79 vs 13.48）；DOC-AUDIT 审计结论 |
| D13 | 阶段 HTML 全部发布到 GitHub Pages；site/stages 为生成物，docs/reports 为源 | 10-30 | 用户要求；Pages 自动部署已验证（stage8-pages） |
| D14 | 过时文档 copy 到 docs/archive/YYYY-MM-DD/ 后移出活跃目录，不直接删除 | 10-31 | 保留证据链；DOC-AUDIT 记录映射 |
| D15 | 文档收敛：START-HERE / RESULTS / GPU-DISCIPLINE / GLOSSARY / DECISIONS / INDEX 为入口，其余为证据 | 10-31 | 减少重复叙事，服务交接 |
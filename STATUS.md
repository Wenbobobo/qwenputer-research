# Qcompute · STATUS

更新时间：2026-10-30 15:50 (Asia/Shanghai) · **Stage 8 进行中**

## Stage 8 工作流
| 项 | 负责人 | 状态 | 目标 |
|---|---|---|---|
| S8.1 GitHub Pages | A | 🔄 | 清理 git、site/（hub + stage1–7 可切换）、公开仓库 + Actions Pages、线上 200、后续更新流程文档 |
| S8.2 推理提速 + 不倒退 | B | 🔄 | fp32 chip × FLA 六配置矩阵；长序列 1,2,5 门禁；目标 0/12 lock 且 ≥15 tok/s |
| S8.3 计算增强 portfolio | C（expert） | 🔄 | 设备树/in-weights 记忆/H5/SafeStore/fp32 chip → ROI + kill list + 3 实验 + 不倒退 gate |
| S8.4 增强原型 | 待定 | ⏳ | 依 C 结论选 1–2 个 |
| S8.5 阶段八报告 + 发布 | lead | ⏳ | stage8.html → Pages |

## Stage 7 结论（已完成）
- AMD bf16 非 bit-exact（作者为 NVIDIA L4）；selective FLA 12.44/12.15；H1 in-weights changed 10/10=100%；int8 head 负结果
- Stage 8 预研：fp32 chip 修复长序列（bf16 3/12 → fp32 0/12），put 15.72/get 14.69/Hi warm 18.87 tok/s，峰值 13.94GB

## 基础设施
- gh CLI：已登录 Wenbobobo（repo+workflow scopes）→ 可创建公开仓库并部署 Pages
- 仓库尚无 .git；.data（20GB）/.research/.venv 必须排除

## GPU 纪律
- 窗口 15:30–00:30；静默 00:30–09:30/12:30–15:30；当前 GPU idle（16.87GB free）
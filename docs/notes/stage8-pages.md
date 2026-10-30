# Stage 8: GitHub Pages 部署

## 项目信息

- **仓库**: https://github.com/Wenbobobo/qwenputer-research
- **GitHub Pages**: https://wenbobobo.github.io/qwenputer-research/
- **状态**: ✓ 已部署，所有页面可访问（HTTP 200）

## 站点结构

```
site/
├── index.html              # 主页（研究档案 hub）
├── assets/
│   ├── site.css           # Hub 页面样式（深色主题）
│   └── nav.js             # 顶部导航栏（首页 + Stage 1-7）
└── stages/
    ├── stage1.html        # 把一台计算机编译进 220 亿字节的权重里
    ├── stage2.html        # 权重里的文件系统，是更好的"记忆"吗？
    ├── stage3.html        # 复现之后，还能往哪里走？
    ├── stage4.html        # 从"能跑"到"知道边界"
    ├── stage5.html        # 从"能复现"到"能用"
    ├── stage6.html        # 把 22.3GB 塞进 20GB，然后快 18 倍
    └── stage7.html        # 精确性的边界，和记忆的最后一臂
```

每个 stage 页面：
- 保留原始报告的完整内容和自包含 CSS
- 通过 `<script src="../assets/nav.js"></script>` 注入顶部导航栏
- 独立可读，不依赖外部资源

## 更新工作流

未来添加新阶段报告时：

1. **生成报告**: 在 `docs/reports/` 创建 `qwenputer-stageN.html`
2. **复制到站点**: 
   ```bash
   cp docs/reports/qwenputer-stageN.html site/stages/stageN.html
   # 在 </body> 前注入: <script src="../assets/nav.js"></script>
   ```
3. **更新 hub**:
   - 编辑 `site/index.html`，在 `.stage-timeline` 添加新的 stage 卡片
   - 编辑 `site/assets/nav.js`，在 `.nav-stages` 添加 Stage N 链接
4. **提交推送**:
   ```bash
   git add site/ docs/reports/
   git commit -m "Add Stage N report"
   git push
   ```
5. **自动部署**: GitHub Actions 自动构建并部署（~17 秒）

## 文件排除

`.gitignore` 已配置排除：

- `.data/` — 20GB 数据集（不推送）
- `.venv/`, `.repro-env/`, `.research/` — 虚拟环境
- `analysis/weights/` — 大型权重文件（>100MB）
- `gpu/out/*.pt` — GPU 检查点（>100MB）
- `*.safetensors`, `*.ckpt`, `*.bin` — 模型二进制文件
- `__pycache__/`, `*.pyc`, `*.log` — Python 缓存和日志

## 清理记录

### 删除的无效目录
- `%SystemDrive%/` (0.00 MB) — 环境变量展开错误产生的垃圾目录

### 仓库大小
- 本地 git 仓库: 6.07 MiB（经过 gc 清理）
- GitHub 推送: 公开可见，所有大文件已排除
- 最终工作树: 790 文件（不含 .data 等排除项）

## 部署验证

✓ 所有页面 HTTP 200:
- Hub: https://wenbobobo.github.io/qwenputer-research/
- Stage 1: https://wenbobobo.github.io/qwenputer-research/stages/stage1.html
- Stage 7: https://wenbobobo.github.io/qwenputer-research/stages/stage7.html

## GitHub Actions 工作流

`.github/workflows/pages.yml`:
- **触发**: push to main 分支，或手动触发
- **权限**: pages: write, id-token: write
- **步骤**: Checkout → Setup Pages → Upload site/ artifact → Deploy
- **部署时间**: ~17 秒

---

**完成时间**: 2026-10-30  
**负责人**: Wenbobobo (通过 Kiro subagent)

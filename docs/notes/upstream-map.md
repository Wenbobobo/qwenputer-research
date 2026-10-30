# 上游地图（Qwenputer）

## 两个分支
| 目录 | 分支 | 内容 |
|---|---|---|
| .research/upstream/qwenputer-main | main | CHIP-8 v1（8 位游戏机 + 大数运算）+ experimental/riscv 的早期快照（P3 之前） |
| .research/upstream/qwenputer-rv | rv-store | **文件系统版**：RV32IM+Zba+大数协处理器 + put/append/get/list/delete 五个内建工具 |
| repro/qwenputer | rv-store 副本（去 .git） | 本机复现工作区 |

## 关键文档（先读顺序）
1. README.md — 总览与 benchmark 数字
2. experimental/riscv/docs/rv32.md — 设计全集：门控 DeltaNet 作 RAM、trace 格式、11 层电路布局、P0-P3 历史、测量
3. experimental/riscv/docs/rv-store.md — 文件系统 v2：hashed 目录、记录协议、容量、快照、benchmark
4. docs/q35.md — 编译器（IR/图/发射/overlay/mute/bridge）原理（CHIP-8 版，同构）
5. docs/q35-store.md — 文件系统 v1（8 token/行，旧布局）
6. docs/q35-size.md — 22GB 里零填充在哪、压缩选项（重要：优化切入点）
7. docs/running.md / quirks.md / trace_format.md
8. 博客：.research/upstream/blog.txt（作者叙事、follow-up 链接）

## 代码结构（repro/qwenputer）
- experimental/riscv/chipformer/rv/
  - spec.py trace/token 定义；ref.py 参考 ISA 模拟器；core.py 电路构建；elf.py 加载 ELF
  - hf.py HF 模块 + cache 注入；rfw/chat.py 驱动（RvChat）；bridge.py 工具调用桥
  - store.py / store_rv.py 文件系统参考实现与电路；store_snapshot.py 快照
  - fastexec.py / tritonexec.py 紧凑执行器（torch.compile / Triton）；rvbatch.py 批处理
  - measure.py / replay.py / execcheck.py / gpucheck.py / textcheck.py / evals / evalarith
  - env/ C 程序、crt0、linker script、MicroPython 构建
  - tests/ test_rv_ref/core/fuzz/hf/bridge/store
- chipformer/q35/（同目录内耦合的编译器）：ir.py（图与精确求值）、prims.py、emit.py、merge.py、mute.py、overlay.py
- 模型卡与权重：HF divagr1925/Qwen3.5-4B-Qwenputer-Filesystem（.data/ 下载中）

## 本机复现路线（按性价比）
1. IR 级（纯 CPU，无 torch 也可）：tests/test_rv_core.py、test_rv_ref.py、test_rv_store.py —— 验证电路逻辑与文件系统协议 exact
2. rv-tiny HF 模型（CPU bf16/fp32）：tinycheck 自由运行小程序，验证「权重里跑 CPU」token-exact
3. 发布 checkpoint（CPU bf16，22.3GB/48GB RAM）：短 store 会话（put/get）验证跨会话文件系统
4. GPU（7900 XT ROCm，静默时段外）：RvChat executor 全速 + 官方 benchmark 子集

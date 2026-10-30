# 发布 checkpoint CPU 实机演示（Qwen3.5-4B-Qwenputer-Filesystem）

日期：2026-10-29 · 硬件：Ryzen 7 5700X / 47.9GB RAM · 纯 CPU（torch 2.14.1+cpu, transformers 5.19.0）
代码：repro/qwenputer（rv-store 分支），CLI：`python -m chipformer.rv.chat <model> ...`

## 1. smoke（已成功）— 证据：demo/run-smoke.log

命令：
```
.venv\Scripts\python.exe -u -m chipformer.rv.chat .data/Qwen3.5-4B-Qwenputer-Filesystem --ask "Hi" --max-new 32
```

结果（原始日志尾部）：
```
<tool_call>
<function=list>
</function>
</tool_call>
<tool_response>
empty
</tool_response>
<think>

</think>

Hello! How can I help you
[32 text + 0 chip tokens in 151.4s, head mismatches 0, lock error None]
```

**这说明了什么：**
1. 22.3GB 的合并 checkpoint 在**纯 CPU** 上成功加载（594 个 tensor，约 20s；总内存约 21GB）。
2. 模型**自发调用**了内建文件系统的 `list` 工具（没有任何外部 prompt 强制）——桥接与工具格式兼容性得到端到端验证。
3. 编译电路在模型内部执行了 list 并回答 `empty`（空 store 的正确结果），**没有外部运行时**参与。
4. Qwen 读到 `<tool_response>empty</tool_response>` 后继续正常对话（"Hello! How can I help you"）。
5. `head mismatches 0`：lm_head argmax 与电路输出码逐 token 一致；`lock error None`：与参考 store 机完全一致。
6. 速度：32 个 text token 用了 151.4s ≈ **4.7s/token**（含逐 token 的 44 层前向 + 电路检查）；模型加载约 20s，IR 电路构建+prefill 约占其余时间。

## 2. 跨会话记忆演示（A/B 均成功 ✅）

- step A: `--memory demo/store.npz --ask "Remember that my locker code is 4471-AZ." --max-new 200` → 日志 demo/run-a.log
- step B: `--memory demo/store.npz --ask "What's my locker code?" --max-new 200` → 日志 demo/run-b.log
**step A 结果（已成功，日志 demo/run-a.log 尾部）：**
```
ok
</tool_response>
<think>

</think>

I've saved your locker code as 4471-AZ.<|im_end|>
[69 text + 3 chip tokens in 204.6s, head mismatches 0, lock error None]
```
- Qwen 自发调用 put（写 locker code）；电路在权重内回答 `ok`；Qwen 确认保存。
- **69 text + 3 chip tokens，0 head mismatch，lock error None。**
- 进程结束保存 `demo/store.npz`：**11.5 KB**（只存被写过的 state 行，不是固定 11–30MB——后者是"满容量"的量级）。
**step B 结果（已成功，日志 demo/run-b.log）：**
```
<tool_call>
<function=get>
locker_code
</function>
</tool_call>
<tool_response>
4471-AZ
</tool_response>
<think>

</think>

Your locker code is 4471-AZ.<|im_end|>
[54 text + 2 chip tokens in 170.4s, head mismatches 0, lock error None]
```
- **新会话、空上下文**，只加载 11.5KB 的 store.npz 快照。
- Qwen 自发调用 get("locker_code")；编译电路在权重内打印出 **4471-AZ**；Qwen 回答完全正确。
- **0 head mismatch / lock error None**。这就是作者模型卡里"保存状态→新会话仍能找到"的端到端复现（本机 CPU）。

## 3. 备注
- 早先的 demo/run-demo.ps1 有一个 PowerShell 5.1 的坑：`$ErrorActionPreference='Stop'` 会把 python 的 stderr 警告当成终止错误（17s 就退出）。已改为直接命令行调用（日志用 `*>` 重定向）。
- 速度结论：本机 CPU 单 token ≈ 4.7s（32 token 短回答）；400 token 的回答约 30 分钟。所以完整 benchmark 不可行，但**单 case 演示完全可行**。
- 内存：加载峰值约 21GB + Python/torch ≈ 22GB；48GB 机器可跑，但需要其余任务让出内存。
# GLOSSARY —— Qwenputer 术语表

| 术语 | 含义 |
|---|---|
| chip 层 / chipformer | 被编译进权重的 12 个 GDN 线性注意力层（第 0–11 层），实现 RISC-V CPU + 文件系统的状态机 |
| Qwen 层 | 第 12–43 层，基座 Qwen3.5-4B 的文本层（被补零到 hidden 4608） |
| hidden 4608 | 2560 个 Qwen 维 + 2048 个 chip 维（chip_dims_offset=2560） |
| mu_dim / mute 列 | 第 2732 维（chip 维之一），Qwen 层唯一读取的 chip 维；权重约 −1024/√4608 ≈ −15.06，用于 mute/sink 语义 |
| ballast | 编译时为 bf16 精确性加入的常量（B=2^40 等） |
| anti-sink | 8 个 full-attention 层的 k_proj[:, 2732] 拟合常量，共 16,384 B，是整模型唯一不可推导残差 |
| slot / slot_bits | chip 层的状态行分配单位；发布件 store 配置 slot_bits=10、n_chunks/n_names=512 |
| 设备树 / peripheral | 把固定形状小算法（ADD8/SORT4/DADD 等）编译成 2 层小电路，作为 CPU 外设 |
| IR | 编译器的中间表示；审计时逐张量与发布权重对比 |
| trim / trim runtime | 无损裁剪运行时：Qwen 层只保留 2560+mu 输入列、只写前 2560 行；chip 层用上游 fast.compact |
| fp32 chip | 只把 chip 层 0–11 用 fp32 计算（Qwen 保持 bf16），修复 AMD 长序列 lock drift |
| selective FLA | chip 层走 reference torch GDN、Qwen 层走 flash-linear-attention 融合内核 |
| FLA | flash-linear-attention；在 gfx1100 可用，但会让 chip 层非 bit-exact |
| H1 | 多会话记忆实验（4 会话/8 事实/2 更新/3 问题）：C1 时间戳上下文、C2 append-only host、C3 普通上下文、C4 覆盖式 host、C5 in-weights |
| H5 | 精确键 vs bge-small 语义检索实验（LongMemEval-100） |
| in-weights store | 发布模型自身的文件系统：512 文件、每文件 ≤104 token、名字键 = 前 6 token + 长度 |
| SafeStore | 协议层键修复：36-bit 头部 hash tag + salt probing + 映射侧车，防长名静默覆盖 |
| memory-disk | 快照/指纹/journal/NameGuard 的状态持久化层（v0 已实现） |
| C1–C5 | H1 的五个条件（见上） |
| lock / lock_error | RvChat 的参考机（reference machine）对照；divergence 表示电路行为与参考不一致 |
| head_mismatch | RvChat 检查电路 head 输出与参考是否一致的计数器 |
| logits argmax / lenient / strict | H1 评分：strict=旧值不得出现；lenient=新值先于旧值 |
| bit-exact / token-level | CPU 可 bit-identical；AMD GPU 只能保证 token 级正确（head/lock 干净） |
| bnb8 / NF4 / int8 head | 量化方案；bnb8 需 PreserveColumns 修复 mute 列；NF4 质量崩；int8 head 负收益 |
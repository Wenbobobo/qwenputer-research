# Stage 5.3 协议级键冲突修复与评测（Key-Protocol Fix）

> 状态：完成（2026-10-29）。纯 CPU 运行，无 GPU，未加载 22GB HF 模型权重。
> 交付物：
> - 核心实现：analysis/memory_disk/safe_store.py
> - 测试套件：analysis/memory_disk/test_safe_store.py
> - 测试日志：analysis/memory_disk/test_safe_store_log.txt

---

## 1. 问题背景：未修改电路的静默覆盖缺陷

Qwenputer 的权重内对象存储（In-weights Store，见 repro/qwenputer/experimental/riscv/chipformer/rv/store.py）将文件名键控为：
key = (min(len(tokens) + 1, LEN_MAX=7), tokens[:NMAX=6])
即：
1. 前 6 个 token 的 18 位 vocabulary ID（共 6 * 18 = 108 位）；
2. 3 位的 token 长度编码（上限 LEN_MAX = 7）。

### 缺陷表现
- 当两个不同的长名字（>= 6 个 token）拥有相同的前 6 个 token 时，两者的 store key 完全一致（均为 (7, (tok0, ..., tok5))）。
- 静默覆盖：在底层 StoreSession 中，第二次 put 会命中同一槽位，返回 ok 并静默覆盖前者的正文 chunk；
- 回读混乱：get 任意一个名字都会返回后写入文件的内容；
- 列表截断：底层的 list 仅能按目录状态打印前 6 个 token，长名字被截断且无换行间隔；
- 阶段 3 D 实测证据（analysis/experiments/d_snapshot_results.json）：
  - A: 'the locker code for the safe room in'
  - B: 'the locker code for the safe in room'
  - 两者共享 key (7, (1719, 33531, 1970, 364, 279, 5902))，写入 B 后 A 的内容变为 B。
- 现有 NameGuard 的局限：analysis/memory_disk/memory_disk.py 的 NameGuard 仅在 API 层探测碰撞并抛出 NameConflict 拒绝写入，无法让底层存储同时接纳这两个合法文件。

---

## 2. 协议级修复方案：SafeStore

为了在完全不修改硬件/权重电路的前提下彻底消除碰撞并支持完整 512 容量，我们设计并实现了双层命名空间协议 SafeStore（analysis/memory_disk/safe_store.py）。

### 2.1 双层命名空间与确定性消歧

1. 逻辑名（Logical Name）：调用方（用户、Agent）使用的原始全名（任意字符串）。
2. 规范名（Canonical Name）：底层电路 store 实际接收和比对的名字。
   - 短名与非冲突名零代价透传：若逻辑名的 raw key 在当前 store 中未被其他逻辑名占用，规范名保持不变：
     canonical_name = logical_name
     保证常规短名（如 city, dog_name, pin）及前缀独特的长名零开销、不被改动。
   - 冲突时确定性前缀消歧：当新逻辑名的 key 与 store 内已有活动项碰撞时，SafeStore 在名字头部插入 36-bit 哈希标签：
     canonical_name = f"#{hex36}# {logical_name}"
     - 为什么在头部？因为底层 store 只采样前 6 个 token。尾部追加后缀（如 (1)）在长名中会被直接截断；只有头部前缀能强制改变前 6 个 token 的 key 编码。
     - 为什么是 36-bit？Qwen 词表 ID 为 18 位（ID_BITS = 18），2 个 token 恰好提供 2 * 18 = 36 位状态空间（2^36 approx 6.87 * 10^10）。
     - 防碰撞盐值探测（Salt Probing）：若 primary tag 仍发生罕见键碰撞，自动递增 salt（#<hex36>.<salt>#）探测空闲键，在 512 满容量极限下数学保证 0 碰撞。

### 2.2 映射表持久化与完整语义

SafeStore 维护双向映射表：
- logical_to_canon / canon_to_logical
- key_to_logical / logical_to_key

支持的核心操作：
- put(name, text)：冲突自动消歧，存储并建立双向映射；同名再次 put 保持既有规范名就地覆盖。
- get(name)：通过映射表解析规范名，从 store 精确检索目标文件正文。
- list()：遍历底层 dev.order，逆向映射回完整未截断的逻辑名。
- delete(name)：调用底层 delete，并同步清理映射项；删除冲突项之一不会破坏另一项。
- 快照持久化：在保存 store 快照（.npz / .pkl）时，伴随写入 .map.json 侧车文件（及在 .json header 中内嵌 name_mapping），加载时无缝恢复双向映射。

---

## 3. 碰撞概率与 H2 语料统计分析

### 3.1 生日悖论理论模型

设在空间大小为 M 的集合中随机分配 N 个键，至少发生一次碰撞的概率为：
P_collision approx 1 - exp(- N * (N - 1) / (2 * M))

在 Qwenputer 满容量 N = 512 条件下：
- 无标签（Raw Store）：前缀相同即 100% 碰撞；
- 1-token 标签（18-bit，状态数 M = 248,044）：
  P approx 1 - exp(- 512 * 511 / (2 * 248,044)) approx 1 - e^(-0.527) approx 40.97%
  碰撞风险极高，不可单独作为硬保证。
- 2-token 标签（36-bit，状态数 M approx 6.15 * 10^10）：
  P approx 1 - exp(- 512 * 511 / (2 * 6.1525 * 10^10)) approx 2.13 * 10^(-6)
  单次碰撞概率仅约百万分之二；结合 SafeStore 的 salt probing，实际碰撞率为 0。

### 3.2 H2 真实语料与对抗集实测数据

使用 analysis/experiments/h2_names.py（2000 个真实名字语料，包含 bench_short, short_sN, snake, adversarial_prefix）及 512 对抗长名进行评测（measure_tagging_schemes）：

| 语料集 | 规模 N | 原始 Store 碰撞数（率） | 1-token (18-bit) 平均碰撞 | 2-token (36-bit) 平均碰撞 | SafeStore 协议实测碰撞 |
|---|---:|---:|---:|---:|---:|
| H2 全量语料 | 2000 | 360 (18.00%) | 5.80 | 0.00 | 0 (360 自动消歧) |
| H2 对抗子集（40组长前缀） | 400 | 360 (90.00%) | 0.40 | 0.00 | 0 (360 自动消歧) |
| 512 全对抗长名（同前缀） | 512 | 511 (99.80%) | 0.20 | 0.00 | 0 (511 自动消歧) |

实测证明：原始 store 在 512 对抗长名下只剩 1 个键（丢失 511 个文件）；SafeStore 协议在所有条件下均保持 0 碰撞与 100% 的存取准确率。

---

## 4. 测试套件与验证结果

执行脚本：analysis/memory_disk/test_safe_store.py
日志记录：analysis/memory_disk/test_safe_store_log.txt
运行环境：CPU-only，riscv .venv（Python 3.13.14，numpy 2.5.4）。

```powershell
$env:PYTHONPATH="C:/Projects/Qcompute/repro/qwenputer/experimental/riscv;C:/Projects/Qcompute/analysis/memory_disk;C:/Projects/Qcompute/analysis/experiments"
& repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe analysis/memory_disk/test_safe_store.py
```

### 4.1 详细用例通过情况

| 测试用例组 | 验证内容 | 结果 |
|---|---|---|
| 1. 阶段 3 D 碰撞对独立存取 | A 与 B raw key 相同；A 原名写入，B 自动加上 #[tag]# 规范名；底层 store 呈现 2 个独立文件；get(A) 返回 4471-AZ，get(B) 返回 4472-AZ；list() 返回完整逻辑名。 | PASS |
| 2. 常规短名零开销不变 | city, dog_name, pin, color 写入后 canonical 名与 logical 完全相等，无 tag 插入；更新覆写正常。 | PASS |
| 3. 删除语义隔离 | 删除 A 后，A 返回 None，冲突伴随文件 B 及常规文件 city 读取完全正常，列表正确同步。 | PASS |
| 4. 512 极限对抗长名存取 | 512 个共享相同前 6 token 的极端对抗长名；raw store 仅产生 1 个 key；SafeStore 成功 put 512 个文件（耗时 0.287s），底层 store 达到 100% 满容量；512 次 get 逐一核验 100% 匹配（耗时 0.106s，0 错漏）；list 精确恢复 512 个全名。 | PASS |
| 5. 快照与映射表持久化回读 | 保存快照及 .map.json 侧车文件；在全新 SafeStore 会话中通过快照重建，所有文件与映射表 100% 恢复。 | PASS |
| 6. 语料库碰撞统计基准 | H2 2000 / H2 adv400 / adv512 方案对比测量全部符合理论预期。 | PASS |

综合判定：38 项断言全部 PASS，0 失败。

---

## 5. 电路级修复备选方案及未采纳原因

除了本协议级方案外，架构上存在另一种方案：修改电路参数（Circuit-Level Fix）。

### 5.1 电路级方案设计
若要在硬件/权重电路层扩大键区分度，需调整以下设计（见 repro/qwenputer/experimental/riscv/chipformer/rv/store_rv.py 与 store.py）：
1. 扩大捕获深度：NMAX = 6 -> 8（或更大）；
2. 扩大长度编码：LEN_BITS = 3 -> 4 位（最大长度由 7 增至 9）；
3. 哈希矩阵重塑：
   hash_matrix: hbits * (NMAX * ID_BITS + LEN_BITS) = 10 * (8 * 18 + 4) = 10 * 148
   输入向量从 111 位增至 148 位；
4. 目录状态行（Directory State Row）重排：
   在 Gated DeltaNet 的持久化 head 中，每行存储向量维度为 HD = 128。
   - 原布局：used(1) + alive(1) + NMAX * 18(108) + LEN_BITS(3) + head(9) + tail(9) = 131 位（通过 packed 方式压缩入 128 维）；
   - 若 NMAX = 8：单是名字 token 位即占 8 * 18 = 144 位，严格超过了 HD = 128 的单一状态行物理容量。
   - 必须引入第二个 value head（vhead）或多行跨行目录结构。

### 5.2 为何未采纳电路级方案
1. 编译代价不可承受：
   - 阶段 3 D 探测（d_snapshot_results.json 的 part3_circuit_probe）证实：在最小电路配置（slot_bits=4, n_chunks=16, n_names=8）下调用 core.build 耗时超过 500 秒被系统超时终止，且全量编译需要海量内存；
2. 权重重新合并（Weight Merge）：
   - 修改电路层会变更 persistent heads 与 vheads 的维度和行结构，要求重新编译 GDN 权重，并与 22GB 的 Qwen3.5-4B 基座权重重新做 safetensors 合并；
3. 嵌入特征与词表映射破坏：
   - 输入词表 ID 的位打包与输出解码逻辑需同步重写，破坏当前已固化的 checkpoint 兼容性；
4. 边际收益有限：
   - 即使扩大到 NMAX = 8，对于超过 8 个 token 的长名仍旧存在截断与碰撞风险，无法从根本上消除逻辑冲突；
5. 协议修复完胜：
   - SafeStore 方案保持现有电路与权重 100% 不变，0 编译成本，0 权重破坏，且通过动态 36-bit 标签与盐值探测在数学上彻底根除了 512 容量下的任何碰撞。

---

## 6. 残留局限性（Residual Limitations）

1. 容量天花板：受限于底层硬件电路常数，最大文件数仍受 n_names = 512 与 n_chunks = 512 限制；
2. 文本分块截断：未配合 append 时，单次 put 正文仍遵循电路固有的 CMAX = 104 token 截断；
3. 映射表依赖性：裸 StoreSession / 原始电路状态内部存储的是带标签的规范名（如 #[tag]# ...）；跨系统冷迁移时需一同携带 .map.json 侧车文件，否则只能解析出截断的规范名。

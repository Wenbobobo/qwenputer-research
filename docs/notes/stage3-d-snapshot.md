# Stage-3 D（v1.1）：store 快照的可移植性与"记忆盘"规模（reference 实验）

状态：完成（reference store 部分）；电路（IR）映射未完成，见"未完成与局限"。
脚本：`analysis/experiments/d_snapshot.py`；结果：`analysis/experiments/d_snapshot_results.json`。
环境：riscv 的 `.venv`（Python 3.13.14，numpy 2.5.4），CPU，未加载任何模型权重（不碰 22 GB checkpoint，不跑 merge）。
工作集监控（Windows，1900 MB 看门狗）下，reference 部分峰值约 160 MB。

## 结论（给 Lead 的要点）

- **快照大小近似线性：约 72 字节 / 文件**（打包后的逻辑状态）。1 / 10 / 100 / 512 个文件分别为 77 / 723 / 7180 / 36741 字节；原文 UTF-8 为 30 / 300 / 3000 / 15360 字节，逻辑快照约是原文的 2.4 倍（主要来自 token 内容 34 bit/token 与目录项）。满容量（512 chunks, 1024 slots）的逻辑快照约 0.24 MB（解析估算，未实测）。
- **状态行利用率低**：512 个文件只触碰 2050 / 6660 行（30.8%），每个文件占 4 行（dir、order、len、content 各一行）。满容量的"11 MB / 30 MB"数字在仓库的 .py/.md 中找不到出处，本实验无法核对，此处不采用。
- **同 seed 可复原，换 seed 全部失效**：同 seed 恢复后 10/10 命中；seed 2036/2037/2038 时 0/10 命中（全部 "not found"，没有返回错误文本）。原因是 hash 矩阵由 `RandomState(seed)` 生成，换 seed 后 10 个名字的槽位几乎都对不上（槽位一致 0/10）。
- **slot_bits 也绑定**：slot_bits 从 10 改为 8（seed 不变）时只有 4/10 命中。这与代码一致：同 seed 下 8 位矩阵是 10 位矩阵的前 8 行（`RandomState` 按行填充，属于由代码推断），因此低 8 位相同，而存储槽位 ≥ 256 的名字查不到。
- **名字超过 6 个 token 会静默覆盖**：store 只保留前 6 个 token 与长度（上限 7）作为 key。两个名字前 6 个 token 相同时，第二次 put 返回 "ok" 并覆盖第一个文件（names_in_store 从 2 变成 1）；list 只显示前 6 个 token。这是数据丢失，不是报错。
- **绑定因素**：hash seed 与 hash_bits/slot_bits；n_chunks / n_names（索引位宽与满判定）；tokenizer 的 id 空间（store 存的是 Qwen token id，不是字节）；NMAX=6 / LEN_BITS=3 / CMAX=104 / TPR=14 等布局常数；store 协议（SP/SN/SA/SO/SD/SG/SL 的含义）；以及 IR 层 persistent heads 的个数与 vheads 形状（决定 save_store 的行结构）。
- **跨模型可移植性结论：IR 快照不可移植；逻辑内容（名字 + 文本）可移植**。换 checkpoint、换 tokenizer 或换电路 build 时，应导出逻辑内容，在新环境用 put 重放，而不是搬运 head 状态。

## 1. 快照里有什么

两层结构：

| 层 | 对象 | 内容 | 大小来源 |
|---|---|---|---|
| 逻辑（store.py `Device`） | dir（slot → Entry：6 个 token 的名字 key、长度、alive、head、tail）、order、lens、nexts、content（(chunk,index) → token id）、寄存器 NEXTG/GCUR/LCUR | 本实验测量对象 | 逐字段打包计算 |
| IR（rv/chat.py `save_store`） | 每个 persistent head 的每个 value head 的非零行（行 → 长度 HD=128 的 float64 向量） | 电路里的 GDN 状态，docstring 写明"只保存文件和寄存器，固件状态不在内" | 未实测（需要电路 build） |

逻辑打包规则（脚本 `logical_bits`）：dir 项 = slot(10) + 名字 key(3+6×18=111) + alive(1) + head(9) + tail(9)；order = 10 bit/项；lens = 9+7 bit；nexts = 9+9 bit；content = 位置(16 bit) + token id(18 bit)；寄存器约 5 字节。

## 2. 大小与文件数（v1.1 全量）

| 文件数 | 逻辑快照（字节） | 每文件（字节） | UTF-8 原文（字节） | 逻辑 / 原文 | pickle(Device) | 触碰行（占 6660 行） |
|---|---|---|---|---|---|---|
| 1 | 77 | 77 | 30 | 2.57 | 328 | 6（0.09%） |
| 10 | 723 | 72.3 | 300 | 2.41 | 1738 | 42（0.63%） |
| 100 | 7180 | 71.8 | 3000 | 2.39 | 15814 | 402（6.0%） |
| 512 | 36741 | 71.8 | 15360 | 2.39 | 84154 | 2050（30.8%） |

512 文件的逻辑快照分解：content 26112 字节（71%）、dir 8960（24%）、lens 1024、order 640、寄存器 5。
逻辑快照 / token id 按 u32 存放（4 字节/token）约为 1.2。
"满"的 512 文件时再 put 返回 `full`，符合设计。
满容量的理论上限（解析，未实测）：content 53248 token × 34 bit + dir 1024 × 140 bit + order + lens + nexts ≈ 247 KB。

行数与 IR 大小：逻辑行总数上限为 6660（dir 1024、order 512、len 512、next 512、content 512×8、寄存器 4）。每个逻辑行在 IR 中是 HD=128 维的向量，float64 时约 1 KB，所以"逻辑行 × 128 × 8 字节"约为 6.8 MB（float32 约 3.4 MB）。这只是数量级估计：content 行可能分布在两个 value head 中，而且没有测 vheads 个数。

## 3. 可移植性测试（同一份逻辑内容，10 个文件）

| 恢复配置 | 命中（get 原文） | "not found" | 其他 | 槽位一致（存储槽 = 新 hash） |
|---|---|---|---|---|
| 同 seed 2035 | 10/10（100%） | 0 | 0 | 10/10 |
| seed 2036 | 0/10 | 10 | 0 | 0/10 |
| seed 2037 | 0/10 | 10 | 0 | 0/10 |
| seed 2038 | 0/10 | 10 | 0 | 0/10 |
| slot_bits 8（n_names 128），seed 2035 | 4/10（40%） | 6 | 0 | 4/10 |

- 同 seed 恢复后，还能继续写入：新 put 返回 `ok`，get 返回新内容。
- 换 seed 的失败是干净的 "not found"，没有返回错误文本。但在同一份快照上继续 put 会产生重复项（推断，未测）。
- 槽位一致数与 seed 之间的关系与随机 hash 的预期相符（约 1/1024 的偶然一致，实测 0/10）。

### 长名字（超过 6 个 token）

前 6 个 token 相同、第 7/8 个 token 对调的两个名字（A 为 8 token 的句子）：

- put A → `ok`，get A → `value A`；
- put B（与 A 同前缀）→ `ok`（被当作同一个文件），get A 变成 `value B`；
- names_in_store = 1，list 只显示 `the locker code for the safe`（6 个 token）。

这说明超过 NMAX 的名字共享 key，写入会静默覆盖。上线前必须修复（拒绝、告警，或对整个名字做哈希）。

## 4. 绑定因素与跨模型结论

| 因素 | 为何绑定 | 不匹配时的表现 |
|---|---|---|
| hash seed | 槽位 = GF(2) 矩阵（`RandomState(seed)`）× 名字 bit | 查不到（测得 0%） |
| hash_bits / slot_bits | 决定槽位位数；低位是否一致取决于矩阵生成顺序 | 部分查不到（测得 40%） |
| n_chunks / n_names | 索引位宽、满判定（NEXTG/NORD） | 容量不符或满判定错误（未测） |
| tokenizer（id 空间） | store 存 Qwen token id（18 bit），不是字节 | 文本需重新编码；id 表不同则内容错乱 |
| 布局常数 NMAX=6、LEN_BITS=3、CMAX=104、TPR=14、ROWS_PER_CHUNK=8 | 决定 key、chunk 与行的打包 | 解码错位 |
| 协议（SP/SN/SA/SO/SD/SG/SL） | 记录的含义 | 错误解释 |
| IR 电路 build（persistent heads 与 vheads 形状，HD=128） | save_store 的行结构只对同一 build 有意义 | 无法加载（未测，依赖 build） |

**跨模型可移植性判定：**

- IR 快照（save_store）：**不可移植**。它绑定到特定电路 build、head 布局和 hash；换 checkpoint 或换 build 需要重放。
- 逻辑快照（Device）：**只在同 seed、同 slot_bits、同布局、同 tokenizer 下可移植**；换 seed 要重新 put。
- 逻辑内容（名字 + 原文）：**可移植**。它只有几十字节/文件，重放很便宜（512 个 put 在 reference 上 0.13 s）。

## 5. 产品级"记忆盘"需要什么（v1.1）

1. **版本与清单**：魔数、格式版本、store 协议版本、seed、hash_bits、slot_bits、n_chunks、n_names、NMAX/CMAX/TPR、tokenizer 标识与词表哈希、电路 head 布局摘要、载荷校验和（sha256）。
2. **以逻辑内容为主格式**：导出名字与原文（规范化顺序），IR 状态只作为可丢弃的缓存，由重放生成。
3. **加载时校验**：恢复后检查 slot_agreement，发现 seed 或 slot_bits 不一致时报错，而不是静默 not found。
4. **迁移路径**：换 seed/slot_bits 时重放 put；换 tokenizer 时重新编码文本；换布局或协议时拒绝或显式迁移。
5. **名字策略**：超过 NMAX 的名字必须拒绝或告警；或者对整个名字做哈希作为 key。
6. **容量与碎片**：512 个名字上限包括已删除的名字（tombstone）；导出时压缩；给出剩余容量。
7. **精度与体积**：save_store 用 float64 按行存储。bf16/float16 加稀疏索引应当可行，需要实测。
8. **版本历史与回滚**：快照带序号，支持回滚；快照写入要原子化。

## 未完成与局限（v1.1）

- **电路映射（任务 3）未完成**：`core.build(target=Target(key_heads=2, value_heads=4, intermediate=64), text=..., store=Config(slot_bits=4, n_chunks=16, n_names=8, hash_bits=2))` 在 500 秒内没有返回（node SIGTERM 结束）；看门狗未触发（stderr 为空）。没有尝试满容量配置。因此 vheads 个数、IR 实际字节数未测，只有上面的估算。
- **tokenizer 替代（D 用 Qwen3.6 tokenizer）**：HF 缓存中没有 Qwen3.5 的 tokenizer，D 改用 Qwen3.6-27B 的 tokenizer.json（只读，未加载模型）。store 的 17 个硬编码 token id 全部匹配（词表 248070），但没有在 Qwen3.5 上直接核对；新记忆盘实现改用 `.data/Qwen3.5-4B-Qwenputer-Filesystem/tokenizer.json`（扩展词表），并断言 "\n"→198、"</"→510 与 store 硬编码一致（见 `analysis/memory_disk/test_memory_disk.py`）。
- **11 MB / 30 MB 来源不明**：仓库里的 .py/.md 找不到这两个数字，无法对照。
- **大小扫描只有 put**：没有 append、delete、tombstone；因此 next 行在大小扫描中为 0，append 的增量未测。
- **seed 只测了 2035–2038 与 slot_bits 8 一种**；没有测其他 hash_bits。
- **并行负载**：实验期间另有一个进程（`analysis/rebuild/rebuild_chip.py --store-seed 2036`，约 5.4 GB）在运行，不属于本实验，未触碰。它与 seed 2036 的负对照有关，可能需要对照本文结果。

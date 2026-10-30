# Memory-disk 格式 v0（设计草案）

> 背景：阶段三 D 实验证明快照很小（1/10/100/512 文件 = 77B/723B/7.2KB/36.7KB），但**绑定编译**（hash seed / 布局 / tokenizer），
> 换 seed 读回 0%。所以"记忆盘"需要一个带指纹与迁移能力的格式。

## 1. 文件结构
```
magic = "QPUMEM01"            # 8 bytes
header = {
  format_version: 1,
  model_fingerprint: {
    base_model_id: "Qwen/Qwen3.5-4B",
    base_sha256: "...",                 # 基座权重指纹
    chipformer_config_sha256: "...",    # config.json + chipformer.json 的 hash
    compiler_commit: "...",             # 必须锚定（当前发布未提供）
    tokenizer_sha256: "...",
    store_layout: {slot_bits, n_chunks, n_names, hash_bits, seed},
    head_layout_sha256: "...",          # 164 heads / value-head 映射的 hash
  },
  created_at, tool_version, ops_journal_len,
  checksum: "sha256 of body",
}
body = { heads: {head_name: [[row_index, values...], ...]}, regs: {...} }   # sparse rows only
journal = [put/list/get 的操作序列（可选，用于跨版本重放迁移）]
```

## 2. 加载语义
1. 校验 magic/版本/checksum。
2. 指纹不匹配 → **拒绝加载**（默认），或进入 `journal-replay` 模式：清空 store，按 journal 重放 put。
3. 指纹匹配 → 直接注入持久 heads 的 recurrent state；registers（NEXTG/NORD/GCUR/LCUR）从 body 恢复。

## 3. 迁移（换模型/换编译）
| 场景 | 策略 |
|---|---|
| 同模型、同编译 | 直接加载 body（bit-exact） |
| 同基座、不同 seed/布局 | journal 重放（不保证名字可寻址性以外的东西） |
| 不同基座 | 不支持：状态行语义依赖 GDN 维度与 head 布局，必须重放 + 重新验证 |
| 损坏/部分写入 | 拒绝；需要 journal 或上一版本 |

## 4. 安全与可审计
- 纯数据格式，不含可执行代码；加载器只做 tensor 反序列化与范围校验。
- 每个文件条目附内容 hash，可审计"记忆是否被篡改"。
- 记录 `ops_journal` 可回放，适合事故复盘与隐私删除（删除=从 journal 重建）。

## 5. 最小实现（建议）
- `memory_disk.py`：save/load/verify/export_journal 四个函数，~200 行。
- 测试：roundtrip（本机已有 11.5KB store.npz 可作 fixture）；seed mismatch 拒绝；journal replay 重建后 get 一致。
- 与上游 `RvChat.save_store/load_store` 兼容：npz 作为 body 容器即可起步。
# Memory-disk v0（实现与设计）

> 背景：阶段三 D（`docs/notes/stage3-d-snapshot.md`，v1.1）证明快照很小（1/10/100/512 文件 = 77B/723B/7.2KB/36.7KB，≈72 B/文件），但**绑定编译**（hash seed / 布局 / tokenizer id / 电路 build），换 seed 读回 0%。因此"记忆盘"是一个**带指纹与迁移能力**的格式。

实现：`analysis/memory_disk/memory_disk.py`（纯 Python + numpy，不加载模型；tokenizers 与 chipformer.rv.* 均为惰性导入）。
测试：`analysis/memory_disk/test_memory_disk.py`，日志：`analysis/memory_disk/test_log.txt`。
运行方式（CPU-only，RAM << 4GB，不碰 22GB 模型）：

```powershell
$env:PYTHONPATH="C:/Projects/Qcompute/repro/qwenputer/experimental/riscv"
& repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe analysis/memory_disk/test_memory_disk.py
```

## 1. API

| 函数 | 说明 |
|---|---|
| `fingerprint(model_dir, store_config=None)` | 绑定指纹：sha256（流式）config.json + chipformer.json + tokenizer.json，再加入 store 配置（默认取 `chipformer.json["store"]`）与格式版本；返回含 `fp` 的字典 |
| `save(snapshot, path, fp, journal=None)` | 写 body .npz + sidecar header .json（魔数 QPUMEM01），原子替换；可选 `journal` 记录逻辑内容 |
| `load(path, expected_fp=None, force=False)` | 校验魔数/版本/body sha256；指纹不同抛 `FingerprintMismatch`（`force=True` 仅调试）；返回与 `chipformer.rv.chat.load_store_file` 相同的 snapshot dict |
| `verify(path, expected_fp=None)` | 返回 header + 布尔（magic_ok/version_ok/checksum_ok/fingerprint_match） |
| `parse_snapshot_npz(path)` | 读取无 header 的原始 .npz（兼容 chat.save_store_file 产物），同样返回 snapshot dict |
| `export_journal(names_texts)` | 规范化 [{name,text}]（或 (name,text)）为 journal |
| `replay_into_reference_store(names_texts, *, seed=2035, ...)` | 用 chipformer.rv.store 的 reference StoreSession 把 journal 重放进全新 store，返回 (session, results) |
| `reference_get(ses, name)` | 在 reference session 上 get(name)，返回正文 |
| `NameGuard(tokenizer_path)` | 名字碰撞守卫：`check(name)` / `register(name)` / `put_checked(name, text, store_call=None)` |

## 2. 文件格式

- **body**：`<path>.npz`，与 `chipformer.rv.chat.save_store_file` 逐字节同构：`<head>/n` = value-head 个数；`<head>/<x>/<r>` = 第 x 个 value head 的第 r 行（float64，长度为 HD=128）。因此 body 可被 `RvChat.load_store` 直接消费，反之亦然。
- **header（sidecar）**：`<path>.json`，字段：`magic="QPUMEM01"`、`format_version=1`、`tool_version`、`fingerprint`（完整指纹字典）、`created_at`（ISO-8601 UTC）、`body_bytes`、`body_sha256`、`journal`（可选）。
- 写盘原子化：临时文件 + `os.replace`，避免半写状态。

## 3. 指纹语义

- 指纹 = sha256( config.json + chipformer.json + tokenizer.json（各文件先单独哈希，再按固定顺序并入）+ store_config(规范化 JSON) + format_version )。
- 指纹用于**绑定到编译产物**：同模型目录 + 同 store 配置 ⇒ 同指纹。加载时只有指纹匹配才允许直接注入 IR 状态（bit-exact 恢复）；否则只能走迁移（journal 重放）。
- 注意：指纹**不**覆盖模型权重二进制（safetensors）与电路 build 哈希；当前实现只覆盖阶段 D 实测的三个绑定文件 + store 配置。若需更强绑定，可后续把 `model.safetensors.index.json` 或 head 布局摘要并入。

## 4. 迁移策略

1. `load(path, expected_fp=fp)`：指纹匹配 → 直接加载 body（IR 状态可用）。
2. 指纹不匹配 → **拒绝加载**，抛出 `FingerprintMismatch`；调用方读取 header 中的 `journal`，用 `replay_into_reference_store`（或目标环境内的 put）在新 store 中重放逻辑内容。
3. 无 journal 的旧快照：可导出名字与正文后重新 put（逻辑内容可移植，见测试 3）。
4. 换 tokenizer：文本需重新编码（名字与正文都以 token id 存储）。
5. 换电路布局/协议：不支持直接迁移，重新放。

## 5. NameGuard（API 级防静默覆盖）

- 问题：store 的 key = 前 6 个 token id + 长度（上限 7）；两个不同名字若 key 相同，后 put 会静默覆盖前者（D 例：`'the locker code for the safe room in'` vs `'the locker code for the safe in room'`，key 同为 `(7, (1719, 33531, 1970, 364, 279, 5902))`）。
- 用法：

```python
from memory_disk import NameGuard
guard = NameGuard("C:/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem/tokenizer.json")
guard.register("city")          # 记录已存在的名字
info = guard.check("dog_name")  # {collision: False, ...}
guard.put_checked(name, text, store_call=lambda n, t: ses.call("put", T.encode(n), T.encode(t)))
# 碰撞时抛 NameConflict，并附建议别名（如 '[1] the locker code for the safe in room'）
```

- 建议别名算法：先尝试后缀 `（name） (i)`（对短名有效，长度组件变化）；若 key 不变（长名，长度已封顶），改加前缀 `[i] name`（改变前 6 个 token）。别名 key 保证不与现有注册 key 冲突。
- key 公式与 store 真实捕获语义一致（名字 token + 尾部换行 + 短名的闭合 "</" token，前 6 个 stored id，长度 = min(token 数+1, 7)），测试中与 reference StoreSession 的 `dev.names()` 逐项核对通过。

## 6. 测试结果（2026-10-29，python 3.13.14 / numpy 2.5.4，venv）

| 测试 | 结果 |
|---|---|
| 1. roundtrip：demo/store.npz（49 heads / 9 非零行 / 11750 B）→ save → load → 数组逐一相等 | 通过（body 11750 B，header 1542 B） |
| 2. 指纹不匹配 → `FingerprintMismatch`；force=True 可读；verify 布尔正确；篡改 body → checksum 错误 | 通过 |
| 3. journal 重放：10 个文件在 seed 2035 与 seed 2036 两个全新 store 中 get() 均返回原文（逻辑内容可移植），槽位不同（city: 276 vs 812） | 通过（10 puts + 20 gets ≈ 2.5 s） |
| 4. NameGuard：D 例两名字 key 相同；check(B) 报碰撞；put_checked(B) 抛 NameConflict；get(B) 仍读到 A 的文本（同 key 机制）；A 未被覆盖，store 仍 1 个文件；建议别名写入成功；短名 city/dog_name 无碰撞；同名字重 put 允许 | 通过（40/40 PASS） |

## 7. 局限

- **指纹不含权重字节与电路 build 哈希**：目前只覆盖 config.json/chipformer.json/tokenizer.json + store 配置；若换权重但上述文件不变，指纹不变（语义上合理：逻辑 store 只绑定 tokenizer+布局+seed）。
- **journal 顺序敏感**：export_journal 保留调用方顺序；重放顺序影响 list 顺序（内容不影响）。
- **NameGuard 是 API 级护栏**：只保护走 `put_checked` 的写入；绕过它直接 put 仍会静默覆盖。名字超过 6 token 但前缀不同的名字不受影响。
- **长文本截断**：store 本身对每文件正文截断到 CMAX=104 token；journal 保存的是调用方给的原文，重放时同样截断。
- **未测**：append/delete/tombstone 重放；换 tokenizer（重新编码）场景；IR 快照在不同电路 build 间的加载（依赖 build，见 stage3-d 局限）。
- **磁盘格式向后兼容**：format_version=1 固定；若 store 布局常数（NMAX/CMAX/TPR）变更，指纹会变（store_config 中不含这些常数，目前隐式绑定在 tokenizer+chipformer.json 中；建议后续把布局常数显式纳入 store_config 再哈希）。

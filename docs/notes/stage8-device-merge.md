# Stage-8 E2：小型器件并入 RV 核心可行性

## 结论

IR/allocator 层面可行。ADD8、SORT4、DADD 三个器件同时加入 RV 核心后仍为 12 层，不需要新增 layer；最大子层仍为 16 heads、9,216 FFN units。生产级 merge 仍需要 token namespace、dispatcher/owner、RV 状态冻结、统一 exit 和 embedding/tokenizer plumbing。

## 方法

[merge_probe.py](../../analysis/device/merge_probe.py) 在 CPU 上按发布版构图：`core.build(Target(16,32,9216), Text(tokenizer), Config(**chipformer.json["store"]))`，再复制 [device_graphs.py](../../analysis/device/device_graphs.py) 的三个 Graph，给器件 dims 和 token features 加独立前缀/namespace，共享 host 的 ONE/ballast；随后调用真实 `allocate` 和 `emit_layers`。未加载 Qwen 权重、未使用 GPU、未写入 .data。

结果见 [merge_probe.json](../../analysis/device/merge_probe.json)；运行日志见 [merge_final.log](../../analysis/device/merge_final.log)。

## 数字

| 图 | layers | dims | heads | FFN units | max units/heads | slots | emitted nonzeros |
|---|---:|---:|---:|---:|---:|---:|---:|
| RV baseline | 12 | 3753 | 164 | 47532 | 9216 / 16 | 1806 | 1289104 |
| + ADD8 | 12 | 3790 | 165 | 47614 | 9216 / 16 | 1834 | 1289793 |
| + SORT4 | 12 | 3819 | 165 | 47969 | 9216 / 16 | 1859 | 1292857 |
| + DADD | 12 | 3767 | 165 | 47711 | 9216 / 16 | 1818 | 1290288 |
| 三者同时 | 12 | 3870 | 167 | 48230 | 9216 / 16 | 1899 | 1294730 |

三者同时的 allocator 结果为 success，slots 1899，仍低于 hidden=4608、ballast=18 布局可用的 2029 slots。最紧的原有子层仍是若干 16-head attention 层和 MLP L10=9216；探针把器件 head 放入可用的首层，把器件 MLP 工作安排到有余量的层，统一输出安排到最终 MLP。

非零增量相对 baseline：ADD8 +689，SORT4 +3753，DADD +1184，三者同时 +5626。增量来自真实 CPU emit_layers 计数，包含 dt_bias/norm 常数口径。

## IR/emitter 检查

短记录 ADD8(255,1)、SORT4([255,0,127,127])、DADD(999,1) 的 `ir.evaluate(check=True)` 均通过；合并图的器件私有 code 与独立图相等，host owner 在这些器件 token 上为 0。三者同时图也成功发射，最终非零 1294730。

一个中间错误揭示了实现约束：把器件 ONE 替换为 active gate 会产生无法精确表示的 bf16 阈值常数，`check_bf16` 失败。修正为共享 ballast/ONE、只对器件 token features/head/owner 做门控后，ADD8 及三器件组合均可 emit。

## 阻塞

1. **Token family/embedding**：RV 现有 VOCAB 与 TEXT_TOKEN 数值区间已占用；裸 ADD8 0–1023 等 ID 会碰撞。需要稳定高位 namespace、扩展 tokenizer 的 family()/tok()/decode_code()，并新增 embedding rows。三者共需约 3193 行，4608 hidden、bf16 约 28 MiB 行数据，另有 padding。
2. **Owner/exit 多路选择**：RV owner 是 rv_owner + store.S，器件 owner 不能直接相加；必须做 mutually exclusive dispatcher，并在统一 exit 选择 RV code、device code、store id code。
3. **RV 状态机**：器件 token 若直接流入 RV graph，会推进 j/capture/store 等状态；需要冻结 RV head/counter，或定义 opaque/no-op device token。
4. **Code-clear 顺序**：输入 code 必须在较早 MLP 清除，exit 才能写 KAPPA code；同一 MLP 不能精确叠加 clear 和 exit。器件 exit 需与 host 最终 exit 共用或严格排序。
5. **Ballast sharing**：ONE/ballast 不能被 active detector 替代，否则阈值权重失去 bf16 精确性；真实 merge 必须共享 host ballast。
6. **Compile/RAM**：layer 数不增加，但 12 层 fp32 emit 的固定峰值约 9.1 GiB；真实 rv/merge 还会叠加 Qwen safetensors、embedding 和 shard 写出，不能把本探针当完整权重 merge。

## Verdict 与下一步

**Verdict：IR/allocator feasible now；production merge needs X。** X 是上述 namespace、dispatcher/owner、RV state freeze、unified exit/code-clear 和 tokenizer/embedding 接线；容量本身不是 blocker。

最小原型是只把 ADD8 接入 rv-tiny（无 store、无 Qwen text bridge），定义 ADD8_ENTER/A/B/RESULT namespace，冻结 RV state，统一 exit 输出后恢复 RV；验证 standalone tiny、merged tiny 与 IR 三者 argmax 一致后，再接带 text 的 core.build。

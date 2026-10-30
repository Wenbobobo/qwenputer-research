# Stage 6：量化 RvChat 端到端实测

日期：2026-10-29，北京时间。设备为 WSL ROCm 的 RX 7900 XT（19.96 GiB）。本次 bnb8 已完成跨进程 put/get；NF4 已执行 Hi，质量不通过。

## 实现与关键修复

入口 [quant_chat.py](../../gpu/quant/quant_chat.py) 通过 monkeypatch 模型类的 from_pretrained：先 CPU bf16 加载发布 checkpoint，调用 chipformer.q35.fast.compact(qwen=False,n_chip=12)，只替换 model.model.layers[12:44] 的 248 个 Linear，再一次移到 CUDA。只对这个已准备好的模型屏蔽后续 .to；RvChat 使用 fast=True，原 CUDA graph chip 路径保留。没有修改 repro 源码或 checkpoint，没有重装 torchao。

层 0–11 只做上游精确 nonzero-block 压缩；Qwen 小型 Conv1d、norm、embedding/lm_head 均保持 bf16，全在 GPU，无 CPU offload。整个 generate 使用 no_grad，避免 upstream fast 路径的第二次 chip prefill 保留 autograd 图。

直接量化整个 merged Qwen Linear 的 bnb8 首测虽然能运行，却生成乱码。原因是 merged mute 列约为 −1024/sqrt(4608)=−15.06，按行 INT8 的量化步长约 0.119，普通语言权重可能大量舍入为零。修正版把 residual-input projection 中索引 >=2560 的非零特殊列独立保留为 bf16 小投影，在量化前将这些列置零，再将特殊列投影加回。该操作覆盖 DeltaNet a/b/z、attention q gate 和 k anti-sink。它仅改变运行时副本；chip 权重不量化。

## 测量口径

每个 Hi/put/get/HF 用例都是新 Python 进程；get 从对应 put 的 npz 恢复。Hi max-new=12，put/get=120。load_s 包括 CPU 加载、压缩、量化和 GPU 搬移；init_s 额外包括 tokenizer、IR、fast view 构建。CPU load ~1s 依赖 safetensors mmap/热文件缓存，不等于冷盘实际读完 22GB。

RvChat seconds 是上游 decode 循环计时，排除 prompt prefill、第二次 chip prefill、首次图捕获；wall_s 是同步后的完整 generate 时间，包含这些开销。text_tokens 沿用上游定义，终止 EOS 不计入，电路输出的可打印 token 计为 text；chip token 单独报告。HF wall_s 包括其 prefill 与 decode，不能直接与 RvChat decode seconds 比较。

显存为 torch.cuda.max_memory_allocated，包含模型、fast view、缓存、首轮 workspace 与图捕获，但不等于整卡驱动总占用；reserved 另存 JSON。所有测量都保留原 tokenizer 行为，警告与 Stage-5 一致。环境缺少 causal_conv1d 和 flash-linear-attention，上游仍走参考实现，可能主导 prefill/DeltaNet 开销。

## 实测结果

修正版 bnb8 Hi：load 140.37s，init 155.10s，常驻加载 10.775 GiB，峰值分配 11.492 GiB。回答为 Hello! How can I help you today?，9 text / 0 chip；decode 3.788s（2.376 tok/s），generate wall 7.800s（1.154 text tok/s）；head mismatch=0，lock error=None。

bnb8 put 已通过：69 text + 3 chip，decode 62.608s（1.102 text tok/s），wall 70.492s；head mismatch=0、lock error=None、ended_eos=true。工具写入 locker_code=4471-AZ 后返回 ok，最终回答与 Stage-5 相同：I've saved your locker code as 4471-AZ. 全部 248 个 bnb 权重已为 int8，参数和 buffers 都在 CUDA。

|配置/任务|load / init (s)|加载 / 峰值分配 (GiB)|text+chip|decode (s)|decode tok/s|生成 wall (s)|结果|
|---|---:|---:|---:|---:|---:|---:|---|
|bnb8 Hi|140.37 / 155.10|10.775 / 11.492|9+0|3.788|2.376|7.800|正常问候|
|bnb8 put|144.57 / 165.51|10.775 / 11.491|69+3|62.608|1.102|70.492|正确写入|
|bnb8 get（新进程）|105.38 / 128.61|10.775 / 11.492|80+3|74.231|1.078|86.491|正确取回，先list再get|
|bnb NF4 Hi|217.89 / 238.43|7.867 / 8.635|12+0|1.295|9.264|187.021|质量失败：!!!!!!!!!!!!|

bnb8 三项 head mismatch 均为0、lock error=None，put/get 都 ended_eos=true、task_sanity=true。post-init 分配11.140GiB，peak reserved11.736GiB。NF4 为248个 uint8 packed 权重、全部参数/buffers CUDA；post-init8.230GiB、reserved峰值9.064GiB。NF4未达到EOS，9.26 tok/s没有可用文本质量，不能作为有效加速。NF4 wall大幅高于decode，说明cold prefill/初始化工作非常重；尚未profile定位。

NF4 put 在加载至 layer38 后被中断，没有完成 JSON；矩阵 supervisor exit1，无 Python traceback，不能把它报告成模型 OOM。检查时已存在另一任务的 trim_chat GPU 进程，后续停止大模型启动并交由Lead串行协调；NF4 get未执行。短小的zero-rich NF4 probe对随机/零行/零列/全零矩阵均输出finite，排除了该小尺寸上简单“全零块必NaN”的假说，不能据此解释整模质量失败。

原始数据：[bnb8 Hi](../../gpu/out/quant_bnb8_hi.json)、[put](../../gpu/out/quant_bnb8_put.json)、[get](../../gpu/out/quant_bnb8_get.json)、[NF4 Hi](../../gpu/out/quant_bnb4_hi.json)、[汇总](../../gpu/out/quant_summary.json)。完整日志在 [gpu/out](../../gpu/out/)，汇总工具为 [summarize_e2e.py](../../gpu/quant/summarize_e2e.py)。

## put/get 文本与质量

bnb8 put 的关键 transcript：

```
<tool_call><function=put><parameter=name>locker_code</parameter>
<parameter=text>4471-AZ</parameter></function></tool_call>
<tool_response>ok</tool_response>
I've saved your locker code as 4471-AZ.<|im_end|>
```

bnb8 get 的关键 transcript（独立进程，memory_loaded=true）：

```
<tool_call><function=list></function></tool_call>
<tool_response>locker_code</tool_response>
<tool_call><function=get><parameter=name>locker_code</parameter></function></tool_call>
<tool_response>4471-AZ</tool_response>
Your locker code is 4471-AZ.<|im_end|>
```

put 最终文本与 bf16 相同；get 最终文本也相同，但工具调用轨迹不同。本次未做通用语言质量分数或完整logit/token对拍，不能推广到任意任务。

## HF baseline 与回退

纯 HF baseline使用同一量化加载函数、直接model.generate，不构造RvChat。首次 bnb8 在生成前因tokenizer返回BatchEncoding而报 Could not infer dtype of tokenizers.Encoding；已按RvChat规则取input_ids修复，修复后尚未GPU复测，因此没有可用HF吞吐结果。复测脚本 [run_hf.py](../../gpu/quant/run_hf.py) 需在独占GPU时运行。

bnb与RvChat模块替换和CUDA graph能兼容，故未触发manual8回退实测。脚本保留manual8选项；其每forward反量化整矩阵，属于省存储的bf16计算回退，不是原生INT8性能。

## 建议

推荐 **bnb8 + bf16特殊列保护** 作为本机已验证的无offload路径：约11.49GiB峰值，embedding/lm_head保留bf16且不需要CPU回退。不能推荐直接整行bnb8（已观察乱码），也不能推荐本次NF4配置（Hi质量失败）。bnb8阈值采用Linear8bitLt默认threshold=0，未启用常见threshold6激活outlier路径；未来可另测，但它不能解决mute权重列污染。

相对Stage-5，Hi decode约2.03倍，put decode约1.18倍；get因为多一次list反而慢约1.86倍。优先价值是显存余量与全GPU驻留，下一步应在稳定独占GPU环境补HF baseline和NF4数值诊断，并针对参考DeltaNet算子、重复chip prefill与reference lock主机成本做profile。

复现：通过WSL原环境运行 bash gpu/run_py.sh gpu/quant/run_e2e.py；脚本串行启动每任务新进程，并设置2026-10-30 00:25北京时间截止。不可与其他大模型任务并发。

## 基线与精确性边界

Stage-5 bf16 CPU split：Hi 9 tokens/7.7s（1.17 tok/s），put 73.6s，get 39.8s；put/get head mismatch=0、lock error=None。普通 bf16 全 GPU 在 generation 时仍缺约20 MiB并 OOM。修正版 bnb8 已完成跨进程持久化 put/get，消除了显存问题；get 正确返回 4471-AZ。不过量化使 get 先 list 再 get，80 text + 3 chip，decode 74.231s（1.078 tok/s），wall 86.491s；与基线直接 get 的54 text+2 chip不同，因此任务延迟更慢。head mismatch=0、memory_loaded=true、task_sanity=true，峰值11.492 GiB。

保留 chip 权重与特殊列不代表 Qwen 文本逐 token 等同 bf16；文本层量化仍会改变语言输出与工具调用选择。put 检查 reference StoreSession lock；get 加载 npz 后沿用上游行为关闭 reference lock，因此 get 的 lock_error=None 不能解释为完整 reference 对拍通过。需结合 head 检查、工具响应和最终返回的 4471-AZ 判定。

# Quant E2E ledger

Goal: complete fast all-GPU quantized RvChat and fresh-process put/get, Chinese report, lead handoff.
Constraints: writes only gpu/quant, gpu/out, named report; no torchao; single GPU; cutoff 2026-10-30 00:25 CST.
Verified: naive bnb8 fits but gibberish due to mute column ~-15.06 contaminating row scales; preserve nonzero residual-input chip suffix columns in bf16 side projections. Exact chip compact only.
Active: serial matrix job pwsh-9e3aa3a0e5864819.
Verified: corrected bnb8 Hi sane 9text/3.79s peak11.49GiB; put69text+3chip/62.61s correct; get fresh npz80text+3chip/74.23s correct (list then get), head0 all. HF initial prompt return normalization failed before generate, fixed.
Next: complete NF4 and retry bnb8 HF, finalize Chinese report.

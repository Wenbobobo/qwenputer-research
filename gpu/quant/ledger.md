# Q1 ledger
Goal: Short empirical RDNA3 precision map, Chinese recommendation, GPU release flag.
Core: All Python via WSL run_py.sh; single GPU task; only allowed output paths; no checkpoint access.
Verified: Torch bf16/fp16 ~78 TFLOPS, FP32 ~19, INT8 ~63–70 TOPS; ordinary float8 mm unavailable.
Verified: Triton INT8 44.22 TOPS; FP8 lowers to FP16 WMMA. Generic INT4 0.23–0.62x, specialized decode GEMV 1.40–1.51x BF16 Linear. bnb 8bit/NF4 pass; torchao0.18 import incompatible with torch2.9.
Complete: Chinese cited report written; GPU processes exited; Q1_DONE written and Lead notified. No GPU work remains.

"""Fresh signext gate with no FLA and circuit-safe GEMM settings; optional activation diagnostics."""
import sys, os, runpy
sys.modules['fla']=None
import torch
from chipformer.q35.fast import exact_matmuls
exact_matmuls()
print('bf16_reduction',torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,flush=True)
sys.argv=['chipformer.rv.gpucheck','--decode','--progs','signext']
runpy.run_module('chipformer.rv.gpucheck',run_name='__main__')

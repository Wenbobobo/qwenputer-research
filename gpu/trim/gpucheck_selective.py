"""Run chipformer.rv.gpucheck with selective FLA (chip GDN layers -> reference torch kernels).
Usage (WSL): bash gpu/trim/gpucheck_selfla.sh   (PYTHONPATH = repro/qwenputer/experimental/riscv only)
SELECTIVE_FLA_OFF=1 runs gpucheck unpatched (full FLA) for comparison."""
import atexit, os, runpy, sys
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # gpu.trim (appended: must not shadow chipformer)
ON = os.environ.get('SELECTIVE_FLA_OFF') != '1'
if ON:
    from gpu.trim import selective_fla
    selective_fla.install()
    atexit.register(lambda: print('[selective_fla] calls', selective_fla.calls(), flush=True))
sys.argv = ['chipformer.rv.gpucheck', *sys.argv[1:]]
runpy.run_module('chipformer.rv.gpucheck', run_name='__main__', alter_sys=True)

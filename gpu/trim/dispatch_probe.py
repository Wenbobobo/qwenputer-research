import sys, inspect
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from gpu.trim import selective_fla as S
S.install()
from chipformer.rv import core,hf,ref
from chipformer.rv.programs import program
import transformers.models.qwen3_5.modeling_qwen3_5 as M
c=core.build(); p=program('signext'); core.snapshot(c,ref.Machine.from_program(p)); toks=ref.Machine.from_program(p).run(10**6)
m,lay,ids=hf.build_model(c,toks)
print('nlayers',c.n_layers(), 'model',type(m),flush=True)
for x in m.modules():
 if hasattr(x,'layer_idx') and hasattr(x,'in_proj_qkv'):
  print('GDN',type(x),type(x) is M.Qwen3_5GatedDeltaNet,x.layer_idx,x.forward,flush=True)
m=m.to('cuda')
try: print('RESULT',hf.check(c,m,lay,ids,toks,'prefill'),flush=True)
finally: print('COUNTS',S.calls(),flush=True)

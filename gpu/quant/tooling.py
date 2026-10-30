import sys,subprocess,json,traceback,gc
from pathlib import Path
if '--install' in sys.argv:
 r=subprocess.run([sys.executable,'-m','pip','install','bitsandbytes','torchao'],capture_output=True,text=True,timeout=240)
 Path('gpu/quant/out/install.log').write_text(r.stdout+'\n'+r.stderr); print(r.stdout[-5000:]+r.stderr[-2000:]); sys.exit(r.returncode)
import torch
rows=[]
def test(name,fn):
 try: rows.append(dict(name=name,status='ok',result=fn()))
 except Exception as e: rows.append(dict(name=name,status='error',error=repr(e),trace=traceback.format_exc()[-6000:]))
 print(json.dumps(rows[-1]),flush=True); Path('gpu/quant/out/tooling.json').write_text(json.dumps(rows,indent=2));gc.collect();torch.cuda.empty_cache()
def bnb(bits):
 import bitsandbytes as b
 cls=b.nn.Linear8bitLt if bits==8 else b.nn.Linear4bit
 kw=dict(has_fp16_weights=False) if bits==8 else dict(quant_type='nf4',compute_dtype=torch.bfloat16)
 layer=cls(1024,1024,bias=False,**kw).to('cuda'); x=torch.randn(4,1024,device='cuda',dtype=torch.bfloat16)
 with torch.no_grad(): y=layer(x)
 torch.cuda.synchronize(); return dict(version=b.__version__,shape=list(y.shape),dtype=str(y.dtype),finite=y.isfinite().all().item())
def ao(bits):
 import torchao
 from torchao.quantization import quantize_
 import torchao.quantization as q
 cfg=getattr(q,'Int8WeightOnlyConfig' if bits==8 else 'Int4WeightOnlyConfig')
 layer=torch.nn.Sequential(torch.nn.Linear(1024,1024,bias=False)).to('cuda',dtype=torch.bfloat16)
 quantize_(layer,cfg()); x=torch.randn(4,1024,device='cuda',dtype=torch.bfloat16)
 with torch.no_grad(): y=layer(x)
 torch.cuda.synchronize(); return dict(version=torchao.__version__,shape=list(y.shape),weight=str(type(layer[0].weight)),finite=y.isfinite().all().item())
for bits in [8,4]:test('bitsandbytes_'+str(bits),lambda bits=bits:bnb(bits))
for bits in [8,4]:test('torchao_'+str(bits),lambda bits=bits:ao(bits))
print('GPU_RELEASED')

import torch,triton,triton.language as tl,json,gc,traceback
from pathlib import Path
@triton.jit
def dot(A,B,C,N:tl.constexpr):
 m=tl.arange(0,32); n=tl.arange(0,32); k=tl.arange(0,64)
 a=tl.load(A+m[:,None]*N+k[None,:]); b=tl.load(B+k[:,None]*N+n[None,:])
 c=tl.dot(a,b);tl.store(C+m[:,None]*32+n[None,:],c)
r=[]
for dtype in [torch.float8_e4m3fn,torch.float8_e5m2]:
 try:
  a=torch.randn(64,64,device='cuda').to(dtype);b=a.clone();c=torch.empty(32,32,device='cuda');dot[(1,)](a,b,c,64);torch.cuda.synchronize();r.append(dict(dtype=str(dtype),status='ok',error=(c-a[:32].float()@b[:,:32].float()).abs().max().item()))
 except Exception as e:r.append(dict(dtype=str(dtype),status='error',error=repr(e),trace=traceback.format_exc()[-2500:]))
 print(json.dumps(r[-1]),flush=True);gc.collect();torch.cuda.empty_cache()
Path('gpu/quant/out/fp8_direct.json').write_text(json.dumps(r,indent=2))

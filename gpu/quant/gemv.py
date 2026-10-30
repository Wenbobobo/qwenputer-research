import torch,triton,triton.language as tl,json,gc
from pathlib import Path
@triton.jit
def gemv(A,W,C,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
 n=tl.program_id(0)*4+tl.arange(0,4); k=tl.arange(0,BK)
 p=tl.load(W+n[:,None]*(K//2)+k[None,:]//2,(n[:,None]<N)&(k[None,:]<K),0)
 w=(((p>>((k[None,:]%2)*4))&15).to(tl.float32)-8)*0.125
 a=tl.load(A+k,k<K,0).to(tl.float32)
 y=tl.sum(w*a[None,:],axis=1);tl.store(C+n,y,n<N)
def bench(f):
 for _ in range(3):f()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(50):f()
 b.record();b.synchronize();return a.elapsed_time(b)/50
rows=[]
for N in [9216,4608]:
 K=4608;p=torch.randint(0,256,(N,K//2),device='cuda',dtype=torch.uint8);w=torch.stack(((p&15).float()-8,(p>>4).float()-8),-1).reshape(N,K).to(torch.bfloat16)*.125
 a=torch.randn(K,device='cuda',dtype=torch.bfloat16);c=torch.empty(N,device='cuda',dtype=torch.bfloat16)
 f=lambda:gemv[(triton.cdiv(N,4),)](a,p,c,N,K,triton.next_power_of_2(K))
 ms=bench(f); base=bench(lambda:torch.mm(a[None,:],w.t()));ref=a[None,:]@w.t()
 rows.append(dict(N=N,K=K,int4_ms=ms,bf16_ms=base,speedup=base/ms,max_abs_error=(c-ref[0]).abs().max().item(),weight_gbs=p.numel()/ms/1e6));print(rows[-1],flush=True)
 del p,w,a,c,ref;gc.collect();torch.cuda.empty_cache()
Path('gpu/quant/out/gemv.json').write_text(json.dumps(rows,indent=2))

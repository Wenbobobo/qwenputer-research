import os,sys,json,time,gc,traceback
from pathlib import Path
import torch
import triton
import triton.language as tl
OUT=Path('gpu/quant/out'); OUT.mkdir(parents=True,exist_ok=True)
rows=[]
def save(): (OUT/(sys.argv[1]+'.json')).write_text(json.dumps(rows,indent=2))
def clean():
 gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
def run(name,f):
 try:
  r=f(); rows.append(dict(name=name,status='ok',**(r or {})))
 except Exception as e: rows.append(dict(name=name,status='error',error=repr(e),trace=traceback.format_exc()[-4000:]))
 print(json.dumps(rows[-1]),flush=True); save(); clean()
def bench(f,warm=3,rep=10):
 for _ in range(warm): f()
 torch.cuda.synchronize(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(rep): f()
 b.record(); b.synchronize(); return a.elapsed_time(b)/rep
@triton.jit
def mm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,MODE:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=64):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); k=tl.arange(0,BK)
 if MODE==1: acc=tl.full((BM,BN),0,tl.int32)
 else: acc=tl.full((BM,BN),0,tl.float32)
 for s in range(tl.cdiv(K,BK)):
  kk=s*BK+k
  if MODE==0: av=tl.load(A+m[:,None]*K+kk[None,:])
  else: av=tl.load(A+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),0.0)
  if MODE==2:
   packed=tl.load(B+n[None,:]*(K//2)+kk[:,None]//2,(n[None,:]<N)&(kk[:,None]<K),0)
   bv=(((packed >> ((kk[:,None]%2)*4)) & 15).to(tl.float32)-8)*0.125
   bv=bv.to(tl.bfloat16)
  else:
   bv=tl.load(B+kk[:,None]*N+n[None,:])
  acc+=tl.dot(av,bv)
 tl.store(C+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
@triton.jit
def cvt(A,B,S:tl.constexpr,TYPE:tl.constexpr,BLOCK:tl.constexpr=1024):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 x=tl.load(A+i,i<S,0)
 if TYPE==0: y=x.to(tl.float8e4nv).to(tl.bfloat16)
 else: y=x.to(tl.float8e5).to(tl.bfloat16)
 tl.store(B+i,y,i<S)
def torch_mm(n,dtype):
 a=torch.randn(n,n,device='cuda',dtype=torch.float32).to(dtype); b=torch.randn_like(a); c=torch.empty_like(a)
 ms=bench(lambda:torch.mm(a,b,out=c)); return dict(shape=[n,n,n],ms=ms,tflops=2*n**3/ms/1e9)
def fp8(dtype):
 a=torch.empty((512,512),device='cuda',dtype=dtype)
 x=torch.randn(512,512,device='cuda'); a=x.to(dtype); back=a.to(torch.float16)
 creation_cast=True
 try:
  b=a.t().contiguous(); ms=bench(lambda:torch.mm(a,b)); return dict(creation_cast=creation_cast,matmul_ms=ms)
 except Exception as e: return dict(creation_cast=creation_cast,matmul_error=repr(e))
def intmm(n):
 a=torch.randint(-8,8,(n,n),device='cuda',dtype=torch.int8); b=torch.randint(-8,8,(n,n),device='cuda',dtype=torch.int8)
 ms=bench(lambda:torch._int_mm(a,b)); return dict(shape=[n,n,n],ms=ms,tops=2*n**3/ms/1e9)
def tritonmm(n,dtype):
 a=torch.randint(-8,8,(n,n),device='cuda',dtype=torch.int8) if dtype==torch.int8 else torch.randn(n,n,device='cuda').to(dtype)
 b=a.clone(); c=torch.empty((n,n),device='cuda',dtype=torch.int32 if dtype==torch.int8 else torch.float32)
 f=lambda:mm[(triton.cdiv(n,32),triton.cdiv(n,64))](a,b,c,n,n,n,1 if dtype==torch.int8 else 0)
 ms=bench(f); ref=a.float()@b.float(); err=(c.float()-ref).abs().max().item()
 return dict(shape=[n,n,n],ms=ms,tops=2*n**3/ms/1e9,max_abs_error=err)
def tritoncvt(t):
 a=torch.randn(1048576,device='cuda',dtype=torch.bfloat16); b=torch.empty_like(a)
 ms=bench(lambda:cvt[(1024,)](a,b,a.numel(),t)); return dict(ms=ms,max_abs_error=(a-b).abs().max().item())
def int4(M,N,K):
 p=torch.randint(0,256,(N,K//2),device='cuda',dtype=torch.uint8)
 w=torch.stack(((p&15).float()-8,(p>>4).float()-8),dim=-1).reshape(N,K).to(torch.bfloat16)*0.125
 a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16); c=torch.empty(M,N,device='cuda',dtype=torch.bfloat16)
 f=lambda:mm[(triton.cdiv(M,32),triton.cdiv(N,64))](a,p,c,M,N,K,2)
 ms=bench(f,3,20); ref=a@w.t(); error=(c-ref).abs().max().item(); base=bench(lambda:torch.mm(a,w.t()),3,20)
 return dict(shape=[M,N,K],fused_int4_ms=ms,bf16_ms=base,speedup=base/ms,tflops=2*M*N*K/ms/1e9,max_abs_error=error,weight_mb=p.numel()/1e6)
print(json.dumps(dict(torch=torch.__version__,triton=triton.__version__,device=torch.cuda.get_device_name(),props=str(torch.cuda.get_device_properties(0)))),flush=True)
mode=sys.argv[1]
if mode=='torch':
 for n in [4096,8192]:
  for d in [torch.bfloat16,torch.float16,torch.float32]: run(str(d)+'_'+str(n),lambda n=n,d=d:torch_mm(n,d))
 for d in [torch.float8_e4m3fn,torch.float8_e5m2]: run(str(d),lambda d=d:fp8(d))
 for n in [4096,8192]: run('torch_int_mm_'+str(n),lambda n=n:intmm(n))
elif mode=='triton':
 run('triton_int8_4096',lambda:tritonmm(4096,torch.int8))
 for t in [0,1]:run('triton_fp8_conversion_'+str(t),lambda t=t:tritoncvt(t))
 for d in [torch.float8_e4m3fn,torch.float8_e5m2]:run('triton_dot_'+str(d),lambda d=d:tritonmm(512,d))
elif mode=='int4':
 for N in [9216,4608]:
  for M in [1,32,256]:run('int4_'+str((M,N,4608)),lambda M=M,N=N:int4(M,N,4608))
clean(); print('GPU_RELEASED',flush=True)

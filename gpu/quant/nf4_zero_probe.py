"""Check bnb NF4 quantization of zero-rich merged-like weights."""
import json,torch,bitsandbytes as bnb
r=[]
for compress in [True,False]:
 for kind in ['random','zerorows','zerocols','allzero']:
  torch.manual_seed(3)
  w=torch.randn(512,512,dtype=torch.bfloat16)*0.02
  if kind=='zerorows':w[256:]=0
  if kind=='zerocols':w[:,256:]=0
  if kind=='allzero':w.zero_()
  q=bnb.nn.Linear4bit(512,512,bias=False,compute_dtype=torch.bfloat16,quant_type='nf4',compress_statistics=compress)
  q.weight=bnb.nn.Params4bit(w,requires_grad=False,quant_type='nf4',compress_statistics=compress)
  q=q.cuda()
  x=torch.randn(1,512,device='cuda',dtype=torch.bfloat16)
  y=q(x)
  dq=bnb.functional.dequantize_4bit(q.weight.data,q.weight.quant_state)
  r.append(dict(compress=compress,kind=kind,finite=bool(y.isfinite().all()),weight_finite=bool(dq.isfinite().all()),maxerr=float((y-torch.nn.functional.linear(x,w.cuda())).abs().max())))
print(json.dumps(r,indent=2))

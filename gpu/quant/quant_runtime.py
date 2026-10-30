"""Quantized Qwenputer loading prototype; chip layers 0-11 remain bf16."""
import argparse,gc,json,time,traceback
from pathlib import Path
import torch
from torch import nn
MODEL='/mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem'
class PerChannelInt8Linear(nn.Module):
 def __init__(self,src):
  super().__init__(); w=src.weight.detach().float(); s=w.abs().amax(1).clamp_min(1e-8)/127
  self.register_buffer('qweight',torch.round(w/s[:,None]).clamp(-127,127).to(torch.int8)); self.register_buffer('scale',s.to(torch.bfloat16))
  if src.bias is not None:self.register_buffer('bias',src.bias.detach().to(torch.bfloat16))
  else:self.bias=None
 def forward(self,x): return torch.nn.functional.linear(x,self.qweight.to(torch.bfloat16)*self.scale[:,None],self.bias)
def patches():
 from transformers import Qwen3_5ForCausalLM
 orig=Qwen3_5ForCausalLM.from_pretrained
 def load(cls,*a,**kw):
  kw.setdefault('dtype',torch.bfloat16);kw.setdefault('device_map','auto');kw.setdefault('max_memory',{0:'19GiB','cpu':'30GiB'})
  t=time.time();m=orig.__func__(cls,*a,**kw);print('[quant] load_s=%.1f map=%s'%(time.time()-t,getattr(m,'hf_device_map',None)),flush=True);m.to=lambda *x,**y:m;return m
 Qwen3_5ForCausalLM.from_pretrained=classmethod(load)
 origto=nn.Module.to
 def guarded(self,*a,**kw): return self if hasattr(self,'hf_device_map') else origto(self,*a,**kw)
 nn.Module.to=guarded
def idx(name):
 p=name.split('.')
 for i,x in enumerate(p[:-1]):
  if x=='layers' and p[i+1].isdigit():return int(p[i+1])
def replace(model,scheme):
 changed=[]
 for name,mod in list(model.named_modules()):
  if not isinstance(mod,nn.Linear) or idx(name) is None or idx(name)<12:continue
  parent,attr=name.rsplit('.',1); par=model.get_submodule(parent)
  if scheme=='manual8': par._modules[attr]=PerChannelInt8Linear(mod)
  elif scheme=='torchao8':
   from torchao.quantization import quantize_,Int8WeightOnlyConfig
   quantize_(mod,Int8WeightOnlyConfig())
  else:
   import bitsandbytes as bnb
   if scheme=='bnb8': q=bnb.nn.Linear8bitLt(mod.in_features,mod.out_features,bias=mod.bias is not None,has_fp16_weights=False)
   else:q=bnb.nn.Linear4bit(mod.in_features,mod.out_features,bias=mod.bias is not None,quant_type='nf4',compute_dtype=torch.bfloat16)
   q.load_state_dict(mod.state_dict());par._modules[attr]=q
  changed.append(name)
 return changed
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scheme',default='auto',choices=['auto','bnb8','bnb4','torchao8','manual8']);ap.add_argument('--embed',default='gpu',choices=['gpu','cpu']);ap.add_argument('--output',default='gpu/out/quant_runtime.json');args=ap.parse_args()
 patches();import sys;sys.path[:0]=['/mnt/c/Projects/Qcompute/repro/qwenputer','/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv']
 from transformers import Qwen3_5ForCausalLM
 r={'request':vars(args),'chip_layers_preserved':True,'changed_layers':[],'attempts':[]};t=time.time()
 try:
  m=Qwen3_5ForCausalLM.from_pretrained(MODEL);r['load_s']=time.time()-t
  for s in (['bnb8','bnb4','torchao8','manual8'] if args.scheme=='auto' else [args.scheme]):
   try:
    ch=replace(m,s)
    if ch:r['scheme']=s;r['changed_layers']=ch;break
   except Exception as e:r['attempts'].append({'scheme':s,'error':repr(e),'trace':traceback.format_exc()[-3000:]})
  if not r.get('scheme'):raise RuntimeError('No quantization scheme succeeded')
  r['device_map']=getattr(m,'hf_device_map',None)
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats();r['gpu_max_memory_allocated_before_generation']=torch.cuda.max_memory_allocated()
  r['status']='replacements_complete; run generation through RvChat harness for end-to-end check'
 except Exception as e:r.update(status='error',error=repr(e),trace=traceback.format_exc()[-5000:])
 Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()

"""Quantized RvChat: CPU load -> exact chip compact -> Qwen-only quant -> GPU.
Never changes checkpoint or repro sources. Each invocation is a fresh process.
"""
import argparse, gc, json, sys, time, traceback
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'repro/qwenputer/experimental/riscv'), str(ROOT/'repro/qwenputer')]
from quant_runtime import PerChannelInt8Linear

class PreserveColumns(nn.Module):
    """Keep merged circuit-only input columns exact, outside quantization scales."""
    def __init__(self, quant, cols, weight, in_features, out_features):
        super().__init__()
        self.quant=quant
        self.register_buffer('cols',cols)
        self.register_buffer('extra_weight',weight)
        self.in_features,self.out_features=in_features,out_features
    def forward(self,x):
        return self.quant(x)+torch.nn.functional.linear(x.index_select(-1,self.cols),self.extra_weight)

def install(scheme, report):
    from transformers import Qwen3_5ForCausalLM
    from chipformer.q35 import fast as F
    original = Qwen3_5ForCausalLM.from_pretrained
    original_to = nn.Module.to
    def load(cls, *a, **kw):
        t = time.perf_counter()
        kw.pop('device_map', None)
        kw.update(dtype=torch.bfloat16, device_map=None)
        m = original.__func__(cls, *a, **kw).eval()
        report['cpu_load_s'] = time.perf_counter()-t
        print('[quant] CPU loaded', report['cpu_load_s'], flush=True)
        meta=json.loads((Path(a[0])/'chipformer.json').read_text())
        assert meta['chip_layers']==12 and len(m.model.layers)==44 and meta['chip_dims_offset']==2560
        report['compact'] = F.compact(m, qwen=False, n_chip=meta['chip_layers'])
        changed=[]
        preserved=[]
        import bitsandbytes as bnb
        for i in range(12, 44):
            layer=m.model.layers[i]
            for name, src in list(layer.named_modules()):
                if not isinstance(src, nn.Linear): continue
                parent, _, attr=name.rpartition('.')
                par=layer.get_submodule(parent) if parent else layer
                # Only residual-input maps have merged chip columns beyond d0.
                cols=None
                if src.in_features==m.config.hidden_size and ('in_proj' in name or name.endswith(('q_proj','k_proj','v_proj','gate_proj','up_proj'))):
                    cols=(src.weight[:,2560:]!=0).any(0).nonzero().flatten()+2560
                    if cols.numel():
                        extra=src.weight.detach().index_select(1,cols).clone()
                        src.weight.data[:,cols]=0
                        preserved.append({'module':f'model.layers.{i}.{name}','columns':cols.tolist(),'bytes':extra.numel()*extra.element_size()})
                if scheme=='manual8': q=PerChannelInt8Linear(src)
                elif scheme=='bnb8':
                    q=bnb.nn.Linear8bitLt(src.in_features,src.out_features,bias=src.bias is not None,has_fp16_weights=False)
                    q.weight=bnb.nn.Int8Params(src.weight.detach(),requires_grad=False,has_fp16_weights=False)
                    if src.bias is not None:q.bias=nn.Parameter(src.bias.detach(),requires_grad=False)
                else:
                    q=bnb.nn.Linear4bit(src.in_features,src.out_features,bias=src.bias is not None,quant_type='nf4',compute_dtype=torch.bfloat16)
                    q.weight=bnb.nn.Params4bit(src.weight.detach(),requires_grad=False,quant_type='nf4')
                    if src.bias is not None:q.bias=nn.Parameter(src.bias.detach(),requires_grad=False)
                if cols is not None and cols.numel():q=PreserveColumns(q,cols,extra,src.in_features,src.out_features)
                setattr(par,attr,q)
                changed.append(f'model.layers.{i}.{name}')
            print('[quant] replaced layer',i,flush=True)
        assert len(changed)==248, len(changed)
        del layer, src
        gc.collect()
        original_to(m,'cuda')
        torch.cuda.synchronize()
        report.update(changed_modules=changed, preserved_columns=preserved, bnb8_threshold=0.0 if scheme=='bnb8' else None, load_s=time.perf_counter()-t,
                      loaded_memory_bytes=torch.cuda.memory_allocated(),
                      all_parameters_cuda=all(p.device.type=='cuda' for p in m.parameters()))
        print('[quant] ready', {k:v for k,v in report.items() if k not in ('changed_modules','compact','preserved_columns')},flush=True)
        m._quant_ready=True
        m.to=lambda *a,**k:m
        return m
    Qwen3_5ForCausalLM.from_pretrained=classmethod(load)
    def guarded(self,*a,**kw):
        return self if getattr(self,'_quant_ready',False) else original_to(self,*a,**kw)
    nn.Module.to=guarded

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scheme',choices=['bnb8','bnb4','manual8'],default='bnb8')
    ap.add_argument('--ask',default='Hi')
    ap.add_argument('--max-new',type=int,default=12)
    ap.add_argument('--memory')
    ap.add_argument('--output',required=True)
    ap.add_argument('--hf',action='store_true')
    args=ap.parse_args()
    r={'config':vars(args),'chip_quantized':False,'embedding_head_dtype':'bf16','fast':True,'preserve_chip_input_columns':True}
    try:
        torch.cuda.reset_peak_memory_stats()
        install(args.scheme,r)
        from chipformer.rv import chat, ref, elf, bridge as BR
        t=time.perf_counter()
        model_path=ROOT/'.data/Qwen3.5-4B-Qwenputer-Filesystem'
        if args.hf:
            from transformers import Qwen3_5ForCausalLM, AutoTokenizer
            from types import SimpleNamespace
            model=Qwen3_5ForCausalLM.from_pretrained(model_path)
            tok=AutoTokenizer.from_pretrained(model_path)
            c=SimpleNamespace(model=model,tok=tok)
            c.prompt_ids=lambda msgs,tools:tok.apply_chat_template(msgs,tools=list(tools),add_generation_prompt=True,enable_thinking=False,tokenize=True)
            c.text_of=lambda ids:tok.decode(ids,skip_special_tokens=False)
        else:
            c=chat.RvChat(model_path,device='cuda',fast=True)
        r['init_s']=time.perf_counter()-t
        r['post_init_memory_bytes']=torch.cuda.memory_allocated()
        r['all_buffers_cuda']=all(b.device.type=='cuda' for b in c.model.buffers())
        r['quant_weight_dtypes']={str(mod.weight.dtype):sum(1 for z in c.model.modules() if type(z)==type(mod)) for mod in c.model.modules() if type(mod).__name__ in ('Linear8bitLt','Linear4bit')}
        from chipformer.rv import store as ST
        msgs=[{'role':'system','content':ST.MEM_SYSTEM},{'role':'user','content':args.ask}]
        ids=c.prompt_ids(msgs,(chat.CALC_TOOL,*ST.TOOLS))
        if not isinstance(ids,list):ids=ids['input_ids']
        ids=[int(i) for i in ids]
        r['nonfinite_logits_calls']=0
        def finite_hook(mod,inp,out):
            if not bool(out.isfinite().all()):r['nonfinite_logits_calls']+=1
        c.model.lm_head.register_forward_hook(finite_hook)
        firmware=ROOT/'repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf'
        if args.hf:
            x=torch.tensor([ids],device='cuda')
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.no_grad():out=c.model.generate(input_ids=x,max_new_tokens=args.max_new,do_sample=False)
            torch.cuda.synchronize()
            tokens=out[0,len(ids):].tolist();sec=time.perf_counter()-t
            r.update(text=c.text_of(tokens),text_tokens=len(tokens),wall_s=sec,tokens_per_s=len(tokens)/sec)
        else:
            remembered=bool(args.memory) and Path(args.memory).exists()
            if remembered:c.store_snap=chat.load_store_file(args.memory)
            def machine():
                m=ref.Machine.from_program(elf.program(firmware));BR.park(m);return m
            lock=None
            if not remembered:
                lock=ST.StoreSession(c.T,machine(),c.store_cfg);lock.history(ids)
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.no_grad():
                out=c.generate(ids,machine(),args.max_new,check_head=True,verbose=True,lock=lock)
            torch.cuda.synchronize();sec=time.perf_counter()-t
            if args.memory:chat.save_store_file(args.memory,c.save_store())
            r.update(out,text=c.text_of(out['tokens']),wall_s=sec,
                     text_tokens_per_s=out['text_tokens']/out['seconds'],
                     wall_text_tokens_per_s=out['text_tokens']/sec,
                     memory_loaded=remembered)
            r['checks_pass']=out['head_mismatch']==0 and out['lock_error'] is None
            r['ended_eos']=bool(out['tokens']) and out['tokens'][-1] in c.eos
            if 'locker' in args.ask:
                r['locker_answer_present']='4471-AZ' in r['text']
                r['tool_call_present']='<tool_call>' in r['text']
                r['task_sanity']=r['checks_pass'] and r['locker_answer_present'] and r['tool_call_present'] and r['ended_eos']
        r['status']='ok'
    except Exception as e:
        r.update(status='error',error=repr(e),trace=traceback.format_exc())
        traceback.print_exc()
    finally:
        r['max_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
        r['max_memory_reserved_bytes']=torch.cuda.max_memory_reserved()
        Path(args.output).write_text(json.dumps(r,indent=2,default=str))
        print(json.dumps(r,indent=2,default=str),flush=True)
    if r['status']!='ok':sys.exit(1)
if __name__=='__main__':main()

"""CPU plain-Hi generation: same trimmed model, BF16 head vs INT8 head."""
import gc
import json
import time
from pathlib import Path
import torch
from gpu.trim.common import CKPT, trim_model
from gpu.trim.int8_head import install_int8_head
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

torch.set_num_threads(10)
torch.set_grad_enabled(False)
r = {}
t = time.perf_counter()
m = Qwen3_5ForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).eval()
r['trim'] = trim_model(m)
gc.collect()
r['load_trim_s'] = time.perf_counter()-t
tok = AutoTokenizer.from_pretrained(CKPT)
ids = tok.apply_chat_template([{'role':'user','content':'Hi'}], add_generation_prompt=True,
                             enable_thinking=False, tokenize=True)
if not isinstance(ids,list): ids=ids['input_ids']

def run():
    x = torch.tensor([ids])
    cache = None
    tokens = []
    t = time.perf_counter()
    for _ in range(6):
        out = m(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = int(out.logits[0,-1,:248320].argmax())
        tokens.append(nxt)
        cache = out.past_key_values
        x = torch.tensor([[nxt]])
    return dict(tokens=tokens, text=tok.decode(tokens), seconds=time.perf_counter()-t)

r['baseline'] = run()
print('baseline',r['baseline'],flush=True)
r['head'] = install_int8_head(m)
gc.collect()
r['int8'] = run()
r['same_tokens'] = r['baseline']['tokens'] == r['int8']['tokens']
Path('gpu/out/int8_cpu_generate.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2),flush=True)

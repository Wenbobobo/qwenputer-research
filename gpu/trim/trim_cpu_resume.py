"""Resume the CPU equivalence check from saved original logits (trim_cpu_check.json.<k>.orig.pt):
load, trim, forced run on the trimmed model only, compare. Lower memory than the two-phase run."""
import argparse, gc, json, time
import torch
from gpu.trim.common import CKPT, trim_model
from gpu.trim.trim_cpu_check import run

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--key', default='store_hi')
    ap.add_argument('--ref', default='gpu/out/trim_cpu_check.json'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    print('start', a, flush=True)
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from chipformer.rv import store as ST, chat
    tok = AutoTokenizer.from_pretrained(CKPT)
    msgs = [{'role': 'system', 'content': ST.MEM_SYSTEM}, {'role': 'user', 'content': 'Hi'}]
    o = tok.apply_chat_template(msgs, tools=[chat.CALC_TOOL, *ST.TOOLS], add_generation_prompt=True,
                                enable_thinking=False, tokenize=True)
    ids = [int(v) for v in (o if isinstance(o, list) else o['input_ids'])]
    ref = torch.load(f'{a.ref}.{a.key}.orig.pt'); lg0, toks0 = ref['logits'], ref['tokens']
    print('ref loaded', len(ids), toks0, flush=True)
    m = Qwen3_5ForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        stats = trim_model(m); gc.collect(); print('trimmed', stats, flush=True)
        t = time.perf_counter(); lg1, _ = run(m, ids, len(toks0), forced=toks0)
    n = len(toks0); d = (lg0 - lg1).abs()
    am0, am1 = lg0[:, :248320].argmax(-1), lg1[:, :248320].argmax(-1)
    toks1 = [int(v) for v in lg1[-n:, :248320].argmax(-1)]
    res = {'prompt_len': len(ids), 'positions_compared': lg0.shape[0], 'orig_tokens': toks0,
           'trim_tokens': toks1, 'orig_text': tok.decode(toks0), 'trim_text': tok.decode(toks1),
           'max_abs_logit_diff': float(d.max()), 'mean_abs_logit_diff': float(d.mean()),
           'max_abs_text_vocab': float(d[:, :248320].max()), 'max_abs_gen_positions': float(d[-n:].max()),
           'logit_scale_max': float(lg0.abs().max()),
           'text_argmax_agree': f'{int((am0 == am1).sum())}/{len(am0)}',
           'full_argmax_agree': f'{int((lg0.argmax(-1) == lg1.argmax(-1)).sum())}/{len(am0)}',
           'bit_identical_positions': int((d.amax(-1) == 0).sum()), 'seconds': time.perf_counter() - t,
           'trim_stats': stats, 'resumed': True}
    print(a.key, res, flush=True)
    r = json.loads(open(a.ref).read()); r[a.key] = res
    open(a.ref, 'w').write(json.dumps(r, indent=2, default=str))

if __name__ == '__main__': main()

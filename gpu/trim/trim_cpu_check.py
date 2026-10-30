"""CPU equivalence: original vs trimmed Qwen layers on the full checkpoint (plain HF forward).
One load: original logits first, then trim in place, then the same token stream again."""
import argparse, gc, json, time
import torch
from gpu.trim.common import CKPT, check_dropped_zero, trim_model, ROOT  # noqa (sets sys.path)

def run(model, ids, n_new, forced=None):
    x = torch.tensor([ids])
    out = model(input_ids=x, use_cache=True)
    logits = [out.logits[0].float()[-64:]]  # last 64 prompt positions
    cache, toks = out.past_key_values, []
    for k in range(n_new):
        nxt = int(torch.argmax(logits[-1][-1, :248320])) if forced is None else forced[k]
        toks.append(nxt)
        if k == n_new - 1: break
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits.append(out.logits[0].float())
    return torch.cat(logits), toks

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-new', type=int, default=3)
    ap.add_argument('--output', required=True); ap.add_argument('--threads', type=int, default=14)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from chipformer.rv import store as ST, chat
    print('imports ok', flush=True)
    r = {'n_new': a.n_new}
    t = time.perf_counter()
    m = Qwen3_5ForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).eval()
    r['cpu_load_s'] = time.perf_counter() - t; print(r, flush=True)
    tok = AutoTokenizer.from_pretrained(CKPT)
    def ids_of(msgs, tools):
        o = tok.apply_chat_template(msgs, tools=list(tools) or None, add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True)
        return [int(v) for v in (o if isinstance(o, list) else o['input_ids'])]
    prompts = {'plain_hi': ids_of([{'role': 'user', 'content': 'Hi'}], ()),
               'store_hi': ids_of([{'role': 'system', 'content': ST.MEM_SYSTEM},
                                   {'role': 'user', 'content': 'Hi'}], (chat.CALC_TOOL, *ST.TOOLS))}
    ref = {}
    with torch.no_grad():
        for k, ids in prompts.items():
            t = time.perf_counter(); lg, toks = run(m, ids, a.n_new)
            ref[k] = (lg, toks); print(k, len(ids), 'orig', toks, time.perf_counter() - t, flush=True)
            torch.save({'logits': lg, 'tokens': toks}, f'{a.output}.{k}.orig.pt')
        t = time.perf_counter()
        r['zero_check'] = check_dropped_zero(m)
        r['trim_stats'] = trim_model(m); gc.collect()
        r['trim_s'] = time.perf_counter() - t; print(r['zero_check'], r['trim_stats'], flush=True)
        for k, ids in prompts.items():
            lg0, toks0 = ref[k]
            t = time.perf_counter(); lg1, _ = run(m, ids, a.n_new, forced=toks0)
            # greedy trimmed tokens: argmax at the same positions (identical prefix when they agree)
            toks1 = [int(v) for v in lg1[-a.n_new:, :248320].argmax(-1)]
            d = (lg0 - lg1).abs()
            am0, am1 = lg0[:, :248320].argmax(-1), lg1[:, :248320].argmax(-1)
            fa0, fa1 = lg0.argmax(-1), lg1.argmax(-1)
            gen = slice(lg0.shape[0] - a.n_new, None)
            r[k] = {'prompt_len': len(ids), 'positions_compared': lg0.shape[0],
                    'orig_tokens': toks0, 'trim_tokens': toks1,
                    'orig_text': tok.decode(toks0), 'trim_text': tok.decode(toks1),
                    'max_abs_logit_diff': float(d.max()), 'mean_abs_logit_diff': float(d.mean()),
                    'max_abs_text_vocab': float(d[:, :248320].max()),
                    'max_abs_gen_positions': float(d[gen].max()),
                    'logit_scale_max': float(lg0.abs().max()),
                    'text_argmax_agree': f'{int((am0 == am1).sum())}/{len(am0)}',
                    'full_argmax_agree': f'{int((fa0 == fa1).sum())}/{len(fa0)}',
                    'bit_identical_positions': int((d.amax(-1) == 0).sum()),
                    'seconds': time.perf_counter() - t}
            print(k, r[k], flush=True)
            open(a.output, 'w').write(json.dumps(r, indent=2, default=str))
    open(a.output, 'w').write(json.dumps(r, indent=2, default=str))

if __name__ == '__main__': main()

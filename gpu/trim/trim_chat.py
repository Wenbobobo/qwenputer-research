"""Trimmed RvChat on GPU: CPU load -> lossless Qwen trim (layers 12-43) -> exact chip compact
(upstream fast.compact, qwen=False, n_chip=12) -> whole model on CUDA, no offload.
Monkeypatches from_pretrained and Module.to only inside this process (pattern of
gpu/quant/quant_chat.py). Each invocation is a fresh process; run via gpu/run_py.sh."""
import argparse, gc, json, sys, time, traceback
from pathlib import Path
import torch
from torch import nn
from gpu.trim.common import CKPT, ROOT, check_dropped_zero, trim_model, SlimHead


def install(report, head_trim=False):
    import os
    if os.environ.get('TRIM_SELECTIVE_FLA') == '1':
        from gpu.trim.selective_fla import install as install_selective
        report['selective_fla'] = install_selective()
    from transformers import Qwen3_5ForCausalLM
    from chipformer.q35 import fast as F
    original = Qwen3_5ForCausalLM.from_pretrained
    original_to = nn.Module.to

    def load(cls, *a, **kw):
        t = time.perf_counter()
        kw.pop('device_map', None)
        kw.update(dtype=torch.bfloat16, device_map=None)
        m = original.__func__(cls, *a, **kw).eval()
        report['cpu_load_s'] = time.perf_counter() - t
        print('[trim] CPU loaded', report['cpu_load_s'], flush=True)
        with torch.no_grad():
            report['zero_check'] = check_dropped_zero(m)
            assert not report['zero_check']['nonzero_dropped'], report['zero_check']
            report['trim'] = trim_model(m)
            report['compact'] = F.compact(m, qwen=False, n_chip=12)
            if os.environ.get('TRIM_FP32_CHIP') == '1':
                from gpu.trim.fp32_chip import install as install_fp32_chip
                report['fp32_chip'] = install_fp32_chip(m, n_chip=12)
                report['chip'] = 'fp32 compact chip; bf16 boundary/Qwen'
            if os.environ.get('TRIM_INT8_HEAD') == '1':
                from gpu.trim.int8_head import install_int8_head
                report['int8_head'] = install_int8_head(
                    m, text_only=head_trim,
                    chunk_rows=int(os.environ.get('TRIM_INT8_CHUNK_ROWS', '16384')))
            elif head_trim:
                v0 = 248320
                head = SlimHead(m.lm_head.weight.detach(), v0)
                m.lm_head = head
                report['head_trim'] = {'cols': [head.c0, head.c1], 'rows': v0,
                                       'bytes': head.weight.numel() * 2}
        gc.collect()
        report['param_bytes_cpu'] = sum(p.numel() * p.element_size() for p in m.parameters())
        report['buffer_bytes_cpu'] = sum(b.numel() * b.element_size() for b in m.buffers())
        t2 = time.perf_counter()
        original_to(m, 'cuda')
        torch.cuda.synchronize()
        report.update(to_cuda_s=time.perf_counter() - t2, load_s=time.perf_counter() - t,
                      loaded_memory_bytes=torch.cuda.memory_allocated(),
                      all_parameters_cuda=all(p.device.type == 'cuda' for p in m.parameters()))
        print('[trim] ready', {k: v for k, v in report.items() if k not in ('compact',)}, flush=True)
        m._trim_ready = True
        m.to = lambda *a, **k: m
        return m
    Qwen3_5ForCausalLM.from_pretrained = classmethod(load)

    def guarded(self, *a, **kw):
        return self if getattr(self, '_trim_ready', False) else original_to(self, *a, **kw)
    nn.Module.to = guarded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ask', default='Hi')
    ap.add_argument('--max-new', type=int, default=12)
    ap.add_argument('--memory')
    ap.add_argument('--output', required=True)
    ap.add_argument('--head-trim', action='store_true')
    ap.add_argument('--hf', action='store_true')
    args = ap.parse_args()
    r = {'config': vars(args), 'chip': 'bf16 exact (fast.compact qwen=False)',
         'qwen': 'lossless trim 2561 cols / 2560 rows', 'fast': True}
    try:
        torch.cuda.reset_peak_memory_stats()
        install(r, args.head_trim)
        from chipformer.rv import chat, ref, elf, bridge as BR, store as ST
        t = time.perf_counter()
        c = chat.RvChat(CKPT, device='cuda', fast=True)
        r['init_s'] = time.perf_counter() - t
        r['after_init_memory_bytes'] = torch.cuda.memory_allocated()
        msgs = [{'role': 'system', 'content': ST.MEM_SYSTEM}, {'role': 'user', 'content': args.ask}]
        ids = c.prompt_ids(msgs, (chat.CALC_TOOL, *ST.TOOLS))
        r['prompt_len'] = len(ids)
        firmware = ROOT/'repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf'
        if args.hf:
            x = torch.tensor([ids], device='cuda')
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                out = c.model.generate(input_ids=x, max_new_tokens=args.max_new, do_sample=False)
            torch.cuda.synchronize()
            tokens = out[0, len(ids):].tolist(); sec = time.perf_counter() - t
            r.update(text=c.text_of(tokens), text_tokens=len(tokens), wall_s=sec,
                     tokens_per_s=len(tokens) / sec)
        else:
            remembered = bool(args.memory) and Path(args.memory).exists()
            if remembered:
                c.store_snap = chat.load_store_file(args.memory)
            def machine():
                m = ref.Machine.from_program(elf.program(firmware)); BR.park(m); return m
            lock = None
            if not remembered:
                lock = ST.StoreSession(c.T, machine(), c.store_cfg); lock.history(ids)
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                out = c.generate(ids, machine(), args.max_new, check_head=not args.head_trim,
                                 verbose=True, lock=lock)
            torch.cuda.synchronize(); sec = time.perf_counter() - t
            if args.memory:
                chat.save_store_file(args.memory, c.save_store())
            r.update(out, text=c.text_of(out['tokens']), wall_s=sec,
                     text_tokens_per_s=out['text_tokens'] / out['seconds'],
                     wall_text_tokens_per_s=out['text_tokens'] / sec, memory_loaded=remembered)
            # steady-state decode rate: a second identical generate (CUDA graph captured, warm)
            if not args.memory:
                torch.cuda.synchronize(); t = time.perf_counter()
                with torch.no_grad():
                    out2 = c.generate(ids, machine(), args.max_new, check_head=False)
                torch.cuda.synchronize()
                r['warm'] = {'text': c.text_of(out2['tokens']), 'seconds': out2['seconds'],
                             'wall_s': time.perf_counter() - t,
                             'text_tokens_per_s': out2['text_tokens'] / out2['seconds'],
                             'same_tokens': out2['tokens'] == out['tokens']}
        print(r.get('text'), flush=True)
        r['status'] = 'ok'
    except Exception as e:
        r.update(status='error', error=repr(e), trace=traceback.format_exc())
        traceback.print_exc()
    finally:
        import os
        if os.environ.get('TRIM_SELECTIVE_FLA') == '1':
            from gpu.trim.selective_fla import calls
            r['selective_fla_calls'] = calls()
        r['max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
        r['max_memory_reserved_bytes'] = torch.cuda.max_memory_reserved()
        Path(args.output).write_text(json.dumps(r, indent=2, default=str))
        print(json.dumps({k: v for k, v in r.items() if k not in ('compact', 'tokens')}, indent=2, default=str), flush=True)
    if r['status'] != 'ok':
        sys.exit(1)


if __name__ == '__main__':
    main()

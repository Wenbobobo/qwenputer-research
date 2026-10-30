"""Summarize serial measurement JSON and compare token transcripts."""
import json
from pathlib import Path
root = Path('gpu/out')
keys = ['trim_base_hi','trim_base_put','trim_base_get','int8_hi','int8_put','int8_get',
        'hip_hi','hip_int8_hi','hip_put','hip_get','tune_hi','tuned_hi','tuned_put','tuned_get']
rows = {}
for key in keys:
    p = root / (key + '.json')
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    rows[key] = {k:r.get(k) for k in ('status','error','text','wall_s','text_tokens',
                  'text_tokens_per_s','wall_text_tokens_per_s','head_mismatch','lock_error',
                  'max_memory_allocated_bytes','after_init_memory_bytes','int8_head','warm')}
comparisons = {}
for task in ('hi','put','get'):
    base = root / ('trim_base_' + task + '.json')
    if not base.exists():
        base = root / ('trim_' + task + '.json')
    if not base.exists():
        continue
    b = json.loads(base.read_text())
    for prefix in ('int8','hip','tuned'):
        p = root / (prefix + '_' + task + '.json')
        if p.exists():
            r = json.loads(p.read_text())
            bt, rt = b.get('tokens', []), r.get('tokens', [])
            comparisons[prefix+'_'+task] = dict(baseline=str(base),
                same_tokens=bool(bt) and bt==rt, same_text=b.get('text')==r.get('text'),
                first_token_diff=next((i for i,(x,y) in enumerate(zip(bt,rt)) if x!=y),
                                      min(len(bt),len(rt)) if len(bt)!=len(rt) else None),
                speed_delta_pct=(r['text_tokens_per_s']/b['text_tokens_per_s']-1)*100
                    if r.get('text_tokens_per_s') and b.get('text_tokens_per_s') else None)
result = dict(runs=rows, comparisons=comparisons)
(root / 'int8_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

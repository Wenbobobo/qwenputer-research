"""Summarize completed reports without loading torch or touching GPU."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
rows=[]
for path in sorted((root/'gpu/out').glob('quant_*.json')):
    d=json.loads(path.read_text())
    if not isinstance(d,dict):continue
    row={k:d.get(k) for k in ['status','load_s','init_s','seconds','wall_s','text_tokens','chip_tokens','text_tokens_per_s','tokens_per_s','head_mismatch','lock_error','memory_loaded','task_sanity','ended_eos','text','error']}
    row['file']=path.name
    for key in ['loaded_memory_bytes','post_init_memory_bytes','max_memory_allocated_bytes','max_memory_reserved_bytes']:
        if key in d:row[key.replace('_bytes','_GiB')]=round(d[key]/2**30,4)
    rows.append(row)
print(json.dumps(rows,ensure_ascii=False,indent=2))
(root/'gpu/out/quant_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

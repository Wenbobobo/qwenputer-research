import json
from pathlib import Path
H=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (H.parent/'h1/h1_results.jsonl').read_text().splitlines()]
out={}
for cond in ['C1','C2','C3','C4']:
 rs=[r for r in rows if r['cond']==cond and r.get('seed',0)==0 and int(r['key'].split('/')[-2].split('-')[-1])<10]
 out[cond]={kind:[sum(r['correct'] for r in rs if kind=='all' or r['kind']==kind),sum(1 for r in rs if kind=='all' or r['kind']==kind)] for kind in ['update','fact','all']}
(H/'matched_baselines.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))

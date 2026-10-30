import sys,json
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'h1'))
from h1_gen import own_text,score_fact,recall_cases
rs=[json.loads(x) for x in (H/'out/inw_questions.jsonl').read_text().splitlines()]
cs=[json.loads(x) for x in (H/'out/inw_sessions.jsonl').read_text().splitlines()]
assert len(rs)==30 and len({r['key'] for r in rs})==30 and len(cs)==10
cases=recall_cases(10,0)
for r in rs:
 i,j=map(int,[r['key'].split('/')[0].split('-')[1],r['key'].split('/')[1]])
 q=cases[i]['questions'][j]
 assert all(q[k]==r[k] for k in ('q','gold','old','kind'))
 assert score_fact(own_text(r['text']),q)==r['correct']
for c in cs:
 i=int(c['case'].split('-')[1]); assert [s['user'] for s in c['sessions']]==cases[i]['sessions']
 assert (H/'out/npz'/f"{c['case']}.npz").exists()
print('VALIDATED 10 generated cases, 30 unique questions, strict rescore, 10 saved npz')
print('question locks',sum(bool(r['info']['lock_error']) for r in rs),'head',sum(r['info']['head_mismatch'] for r in rs))

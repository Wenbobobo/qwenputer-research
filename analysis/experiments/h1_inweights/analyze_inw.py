"""Aggregate completed C5 questions; all score calculations reuse persisted booleans."""
import json, math, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (HERE/'out/inw_questions.jsonl').read_text().splitlines() if x]
sessions=[json.loads(x) for x in (HERE/'out/inw_sessions.jsonl').read_text().splitlines() if x]
def stat(rs, field='correct'):
    k=sum(bool(r[field]) for r in rs); n=len(rs)
    if not n: return dict(k=k,n=n,p=None)
    p=k/n; z=1.96; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(k=k,n=n,p=p,wilson95=[mid-half,mid+half])
summary={kind:stat([r for r in rows if kind=='all' or r['kind']==kind]) for kind in ('update','fact','all')}
summary['changed_new_first']=stat([r for r in rows if r['kind']=='update'],'new_first')
summary['per_case']={c['case']:{kind:stat([r for r in rows if r['key'].split('/')[0]==c['case'] and (kind=='all' or r['kind']==kind)]) for kind in ('update','fact','all')} for c in sessions}
infos=[r['info'] for r in rows]+[s for c in sessions for s in c['sessions']]
summary['runtime']={'turns':len(infos),'seconds':sum(i['seconds'] for i in infos),'head_mismatch':sum(i['head_mismatch'] for i in infos),'lock_errors':[i['lock_error'] for i in infos if i['lock_error']],'n_cases':len(sessions)}
# Historical store_after diagnostics failed to decode tuple keys. Derive a separate,
# explicitly transcript-based view of completed calls; do not treat it as GPU state proof.
transcript={}
for c in sessions:
    d={}; views=[]
    for s in c['sessions']:
        for call in re.findall(r'<tool_call>(.*?)</tool_call>',s['text'],re.S):
            fn=re.search(r'<function=(\w+)>',call)
            ps=dict(re.findall(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',call,re.S))
            if not fn: continue
            op=fn.group(1); name=ps.get('name','').strip(); text=ps.get('text','').strip()
            if op=='put': d[name]=text
            elif op=='append': d[name]=d.get(name,'')+text
            elif op=='delete': d.pop(name,None)
        views.append(dict(d))
    transcript[c['case']]=views
for r in rows:
    vs=transcript[r['key'].split('/')[0]]
    r['gold_in_store']=any(r['gold'].lower() in v.lower() for v in vs[-1].values())
    r['old_in_store']=any(r['old'].lower() in v.lower() for v in vs[-1].values()) if r['old'] else None
    r['old_ever_stored']=any(r['old'].lower() in v.lower() for s in vs for v in s.values()) if r['old'] else None
summary['store_diagnostic_authority']='completed tool-call transcripts, inferred; raw store_after tuple-key decoder failed'
summary['transcript_store_views']=transcript
summary['changed_diagnostics']=[{k:r[k] for k in ('key','gold','old','correct','new_first','gold_in_store','old_in_store','old_ever_stored')} for r in rows if r['kind']=='update']
badcases={c['case'] for c in sessions if any(s['lock_error'] or s['head_mismatch'] for s in c['sessions'])}
badcases.update(r['key'].split('/')[0] for r in rows if r['info']['lock_error'] or r['info']['head_mismatch'])
clean=[r for r in rows if r['key'].split('/')[0] not in badcases]
summary['runtime']['cases_with_lock_or_head_error']=sorted(badcases)
summary['lock_clean_subset']={kind:stat([r for r in clean if kind=='all' or r['kind']==kind]) for kind in ('update','fact','all')}
summary['five_case_checkpoint']={kind:stat([r for r in rows if int(r['key'].split('/')[0].split('-')[1])<5 and (kind=='all' or r['kind']==kind)]) for kind in ('update','fact','all')}
(HERE/'h1_inweights_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))

import json
def load(p):
    rows=[]
    for line in open(p, encoding='utf-8'):
        line=line.strip()
        if line: rows.append(json.loads(line))
    return rows
for name,p in [('fp32','gpu/out/c8_fp32/turns.jsonl'),('bf16','gpu/out/c8_bf16_pair/turns.jsonl')]:
    try: rows=load(p)
    except Exception as e:
        print(name,'ERR',e); continue
    print('==',name,'turns',len(rows))
    for r in rows:
        print('  %-16s text=%4s chip=%6s tok/s=%7.2f lock=%s head=%s' % (r.get('label'), r.get('text_tokens'), r.get('chip_tokens'), r.get('tok_s') or 0, r.get('lock_error'), r.get('head_mismatch')))
    print('  divergences:', [r.get('label') for r in rows if r.get('lock_error')])
import json
d=json.load(open('gpu/out/c8_summary.json',encoding='utf-8'))
for name in d:
    if name not in ('fp32','bf16'): continue
    blk=d[name]
    print('==',name,'turns',len(blk.get('turns',[])))
    for t in blk.get('turns',[]):
        print('  %-16s text=%4d chip=%5d tok/s=%7.2f lock=%s head=%s' % (t.get('label'), t.get('text_tokens',0), t.get('chip_tokens',0), t.get('tok_s',0), t.get('lock_error'), t.get('head_mismatch')))
    print('  divergences:', [t.get('label') for t in blk.get('turns',[]) if t.get('lock_error')])
    r=blk.get('runtime',{}); print('  load_s',r.get('load_s'),'init_s',r.get('init_s'),'loaded',r.get('loaded_memory_bytes'))
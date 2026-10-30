"""Serial fresh-process E2E matrix, with a Beijing 00:25 GPU cutoff."""
import datetime, json, os, subprocess, sys
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
zone=ZoneInfo('Asia/Shanghai')
now=datetime.datetime.now(zone)
cutoff=datetime.datetime(2026,10,30,0,25,tzinfo=zone)
for scheme in sys.argv[1:] or ['bnb8','bnb4']:
    for task,ask,limit in [('hi','Hi',12),('put','Remember that my locker code is 4471-AZ.',120),('get',"What's my locker code?",120),('hf_hi','Hi',12)]:
        stem=f'quant_{scheme}_{task}'
        cmd=[sys.executable,str(ROOT/'gpu/quant/quant_chat.py'),'--scheme',scheme,'--ask',ask,'--max-new',str(limit),'--output',str(ROOT/f'gpu/out/{stem}.json')]
        if task in ('put','get'):cmd+=['--memory',str(ROOT/f'gpu/out/quant_{scheme}_store.npz')]
        if task=='hf_hi':cmd+=['--hf']
        remaining=(cutoff-datetime.datetime.now(zone)).total_seconds()
        if remaining<=0:raise SystemExit('00:25 hard stop reached')
        print(datetime.datetime.now(zone).isoformat(),stem,flush=True)
        with (ROOT/f'gpu/out/{stem}.log').open('w') as log:
            try:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            except subprocess.TimeoutExpired:raise SystemExit('00:25 hard stop terminated GPU child')
        print(stem,'exit',result.returncode,flush=True)
        if result.returncode:
            print('Stop this scheme after failure; inspect its JSON/log.',flush=True)
            break
        data=json.loads((ROOT/f'gpu/out/{stem}.json').read_text())
        if task in ('put','get') and not data.get('task_sanity',False):
            print('WARNING task sanity failed:',stem,flush=True)

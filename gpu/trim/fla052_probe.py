import sys
from pathlib import Path
sys.path.insert(0,'/tmp/qcompute-a7-fla/052')
from chipformer.rv import core,hf,ref
from chipformer.rv.programs import program
c=core.build(); p=program('signext'); core.snapshot(c,ref.Machine.from_program(p)); toks=ref.Machine.from_program(p).run(10**6)
m,lay,ids=hf.build_model(c,toks); m=m.cuda()
print('TOKENS',hf.check(c,m,lay,ids,toks,'prefill',tokens=True),flush=True)

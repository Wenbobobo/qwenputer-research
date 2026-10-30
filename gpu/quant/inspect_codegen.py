import sys,json
sys.argv=['bench.py','inspect']
exec(compile(open('gpu/quant/bench.py').read(),'gpu/quant/bench.py','exec'))
for d in [torch.float8_e4m3fn,torch.float8_e5m2]:
 a=torch.randn(512,512,device='cuda').to(d);c=torch.empty(512,512,device='cuda'); k=mm[(16,8)](a,a,c,512,512,512,0)
 text=k.asm.get('amdgcn',''); name=str(d).split('.')[-1];Path('gpu/quant/out/'+name+'.amdgcn').write_text(text)
 print(name,json.dumps([s.strip() for s in text.splitlines() if any(x in s for x in ['wmma','mfma','cvt','dot'])][:35]))
 del a,c;clean()

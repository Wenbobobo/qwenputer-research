"""Plain HF Hi retry after the main matrix releases the GPU."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for scheme in ['bnb8','bnb4']:
    stem=f'quant_{scheme}_hf_hi'
    with (root/f'gpu/out/{stem}.log').open('w') as f:
        p=subprocess.run([sys.executable,str(root/'gpu/quant/quant_chat.py'),'--scheme',scheme,'--hf','--ask','Hi','--max-new','12','--output',str(root/f'gpu/out/{stem}.json')],stdout=f,stderr=subprocess.STDOUT)
    print(stem,p.returncode,flush=True)

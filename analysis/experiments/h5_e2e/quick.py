#!/usr/bin/env python3
"""Quick synthetic-only E2E validation."""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "analysis" / "memory_disk", ROOT / "repro" / "qwenputer" / "experimental" / "riscv"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

import memory_disk as MD
import safe_store as SS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from transformers import AutoTokenizer, AutoModel

def load_model(path):
    t0=time.perf_counter(); tok=AutoTokenizer.from_pretrained(str(path)); mod=AutoModel.from_pretrained(str(path)); mod.eval()
    return tok,mod,(time.perf_counter()-t0)*1000

def embed(texts,tok,mod,ml=512):
    x=tok(texts,padding=True,truncation=True,max_length=ml,return_tensors="pt")
    with torch.no_grad(): h=mod(**x).last_hidden_state[:,0]
    return torch.nn.functional.normalize(h,p=2,dim=1).cpu().numpy()

def run():
    tok,mod,load_ms=load_model(ROOT/".data"/"bge-small-en-v1.5")
    data=json.loads((ROOT/"analysis/experiments/h5/dataset.json").read_text(encoding="utf-8"))
    rows=[(r["name"],r["text"]) for r in data["facts"]]
    st=SS.SafeStore(seed=2035)
    put_t=[]; [put_t.append((time.perf_counter(),st.put(k,v),time.perf_counter())) for k,v in rows]
    puts=[{"ms":(t[2]-t[0])*1000,"ok":t[1].get("answer")=="ok\n","full":bool(t[1].get("full"))} for t in put_t]
    corpus=embed([k+" "+v for k,v in rows],tok,mod)
    queries=[{"q":q["query"],"gold_key":q["target"],"gold_ans":q["answer"]} for q in data["queries"][:50]]
    q_ms=[]; get_ms=[]; hits={1:0,3:0,5:0}; e2e={1:0,3:0,5:0}
    for q in queries:
        t0=time.perf_counter(); v=embed(["Represent this sentence for searching relevant passages: "+q["q"]],tok,mod,ml=128)[0]; q_ms.append((time.perf_counter()-t0)*1000)
        order=np.argsort(-(corpus@v)); gold=q["gold_key"]; candkeys=[rows[int(i)][0] for i in order[:5]]
        for k in (1,3,5):
            ck=candkeys[:k]; hits[k]+=int(gold in ck)
            gt=0.0; ok=False
            for c in ck: t=time.perf_counter(); val=st.get(c); gt+=(time.perf_counter()-t)*1000; ok=ok or (val and q["gold_ans"] in val)
            get_ms.append(gt); e2e[k]+=int(ok)
    n=len(queries)
    return {"n":n,"model_load_ms":load_ms,"router_recall":{f"at_{k}":hits[k]/n for k in (1,3,5)},"e2e_accuracy":{f"at_{k}":e2e[k]/n for k in (1,3,5)},"latency_ms":{"bge_query_p50":float(np.percentile(q_ms,50)),"bge_query_p95":float(np.percentile(q_ms,95)),"store_get_p50":float(np.percentile(get_ms,50))},"store":{"attempted":len(puts),"accepted":sum(p["ok"] for p in puts),"full":sum(p["full"] for p in puts)}}

if __name__=="__main__":
    r=run(); print(json.dumps(r,indent=2)); (ROOT/"analysis/experiments/h5_e2e/quick.json").write_text(json.dumps(r,indent=2),encoding="utf-8")

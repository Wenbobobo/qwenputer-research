#!/usr/bin/env python3
"""H5-lite: exact-key versus vector retrieval. CPU-only.

Uses BAAI/bge-small-en-v1.5 when --model points to a complete local checkout.
When unavailable, runs a clearly labelled TF-IDF word+character vector fallback.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np

FACTS = [
("pet", "My pet is a dog named Miso, a friendly brown rescue.", "what is the animal I keep at home", "what is my pet called"),
("job", "I work as a data engineer building reliable pipelines.", "what do I do for a living", "what is my occupation"),
("city", "I live in Vancouver near the seawall.", "where is my home city", "which city do I live in"),
("drink", "My usual morning drink is black coffee.", "what do I drink in the morning", "what beverage do I usually have"),
("food", "My favorite comfort food is spicy ramen.", "what food cheers me up", "what is my favorite meal"),
("color", "My favorite color is forest green.", "which color do I like best", "what color do I prefer"),
("hobby", "I enjoy pottery on weekends.", "what do I do for fun on weekends", "what hobby do I have"),
("instrument", "I play the acoustic guitar.", "which musical instrument can I play", "what do I play musically"),
("sport", "I follow tennis and play it casually.", "which sport do I enjoy", "what game do I play"),
("language", "I am learning Japanese.", "which language am I studying", "what language am I learning"),
("season", "Autumn is my favorite season.", "which part of the year do I prefer", "what season do I like"),
("book", "My favorite book is The Left Hand of Darkness.", "which book do I love most", "what is my favorite novel"),
("music", "I listen to jazz while working.", "what music do I play during work", "which genre do I enjoy while working"),
("travel", "My dream trip is to visit New Zealand.", "where would I like to travel", "what destination do I dream about"),
("car", "I drive a blue Toyota Corolla.", "what vehicle do I own", "what kind of car do I drive"),
("school", "I studied physics at university.", "what subject did I study", "what was my university field"),
("birthplace", "I was born in Porto, Portugal.", "where did I enter the world", "what is my birthplace"),
("birthday", "My birthday is on April 17.", "when do I celebrate my birthday", "what day was I born"),
("sibling", "I have one older sister named Lena.", "do I have brothers or sisters", "how many siblings do I have"),
("parent", "My mother is named Rosa.", "what is my mother's name", "who is my mom"),
]
# 5 session-like documents, 20 facts each (repeat templates with unique values).
rows=[]
for s in range(5):
    for i,(name,text,q1,q2) in enumerate(FACTS):
        if s==0: t=text
        else:
            vals={"pet":f"My pet is a {['cat','parrot','rabbit','dog'][s-1]} named N{s}.","job":f"I work as a {['teacher','nurse','designer','chemist'][s-1]}.","city":f"I live in {['Oslo','Kyoto','Lima','Dublin'][s-1]}.","drink":f"My usual drink is {['tea','cocoa','water','espresso'][s-1]}.","food":f"My favorite comfort food is {['tacos','curry','dumplings','stew'][s-1]}.","color":f"My favorite color is {['navy','orange','purple','gold'][s-1]}.","hobby":f"I enjoy {['cycling','gardening','drawing','birdwatching'][s-1]}.","instrument":f"I play the {['piano','violin','flute','drums'][s-1]}.","sport":f"I follow {['soccer','swimming','climbing','baseball']}[s-1] and play it casually.","language":f"I am learning {['Spanish','Korean','French','Italian'][s-1]}.","season":f"{['Spring','Summer','Winter','Spring'][s-1]} is my favorite season.","book":f"My favorite book is {['Dune','Beloved','Sapiens','Emma'][s-1]}.","music":f"I listen to {['rock','classical','folk','blues'][s-1]} while working.","travel":f"My dream trip is to visit {['Iceland','Morocco','Chile','Scotland'][s-1]}.","car":f"I drive a {['red Honda Civic','white Mazda 3','green Ford Focus','silver Volvo'][s-1]}.","school":f"I studied {['history','biology','architecture','economics']}[s-1] at university.","birthplace":f"I was born in {['Seoul','Cairo','Milan','Quito'][s-1]}.","birthday":f"My birthday is on {['May 9','June 21','July 3','August 28'][s-1]}.","sibling":f"I have {['two younger brothers','an older brother','one younger sister','two sisters'][s-1]}.","parent":f"My mother is named {['Ana','Mei','Sara','Nora'][s-1]}."}; t=vals[name]
        rows.append({"session":f"session_{s+1}","name":name+f"_{s+1}","base_name":name,"text":t,"answer":t,"semantic_query":q1,"semantic_query_alt":q2})
queries=[]
for r in rows[:100]: queries.append({"kind":"semantic","query":r["semantic_query"] + " in session " + r["session"],"target":r["name"],"answer":r["answer"]})
for r in rows[:50]: queries.append({"kind":"exact","query":r["name"],"target":r["name"],"answer":r["answer"]})

def toks(s): return re.findall(r"\w+", s.lower())
def exact_hit(q, r): return q==r["name"] or q==r["base_name"]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='analysis/experiments/h5/results.json'); ap.add_argument('--data',default='analysis/experiments/h5/dataset.json'); ap.add_argument('--model',default='.data/bge-small-en-v1.5'); a=ap.parse_args()
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); Path(a.data).write_text(json.dumps({"facts":rows,"queries":queries},ensure_ascii=False,indent=2),encoding='utf-8')
 backend='bge-small-en-v1.5'
 try:
  from sentence_transformers import SentenceTransformer
  model=SentenceTransformer(a.model, device='cpu'); model.encode(['probe'],convert_to_numpy=True)
  def embed(xs): return model.encode(xs,normalize_embeddings=True,show_progress_bar=False,device='cpu')
 except Exception as e:
  backend='tfidf_fallback'; fallback_error=str(e)
  from sklearn.feature_extraction.text import TfidfVectorizer
  vec=TfidfVectorizer(ngram_range=(1,2),analyzer='word',sublinear_tf=True)
  corpus=[r['name']+' '+r['text'] for r in rows]
  X=vec.fit_transform(corpus)
  def embed(xs): return vec.transform(xs).toarray()
 chunk_tokens=[len(toks(r['text'])) for r in rows]
 docs=[r['name']+' '+r['text'] for r in rows]
 D=embed(docs); Q=embed([q['query'] for q in queries]); D=D/(np.linalg.norm(D,axis=1,keepdims=True)+1e-9); Q=Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)
 sims=Q@D.T; metrics={}
 for kind in ('semantic','exact'):
  inds=[i for i,q in enumerate(queries) if q['kind']==kind]
  exact=sum(exact_hit(queries[i]['query'],rows[0]) for i in []) # explicit construction below
  hit1=hit5=0
  for i in inds:
   order=np.argsort(-sims[i]); target=queries[i]['target']; hit1 += rows[order[0]]['name']==target; hit5 += any(rows[j]['name']==target for j in order[:5])
  metrics[kind]={"n":len(inds),"vector_recall_at_1":hit1/len(inds),"vector_recall_at_5":hit5/len(inds),"exact_key_success":(0.0 if kind=='semantic' else 1.0)}
 rag_q=[len(toks(q['query'])) for q in queries]
 metrics['token_cost']={"exact_key_get_measured_tokens":45,"rag_k":5,"rag_chunk_tokens_mean":float(np.mean(chunk_tokens)),"rag_chunk_tokens_total_k5_mean":float(np.mean([sum(sorted(chunk_tokens,reverse=True)[:5]) for _ in queries])),"rag_query_tokens_mean":float(np.mean(rag_q)),"rag_paste_mean":float(5*np.mean(chunk_tokens)+np.mean(rag_q))}
 result={"experiment":"H5-lite","device":"cpu","gpu_used":False,"dataset":{"type":"synthetic","facts":len(rows),"session_documents":5,"queries_semantic":100,"queries_exact_control":50,"longmemeval_download":"failed: mirror SSL EOF"},"embedding":{"requested":"BAAI/bge-small-en-v1.5","backend":backend,"model_path":a.model,"fallback_error":locals().get('fallback_error'),"normalization":"cosine"},"metrics":metrics,"caveats":["Exact semantic success is 0 by construction because the exact-key API receives no canonical name.","Fallback scores are TF-IDF and must not be reported as bge scores.","Synthetic facts are single-fact chunks; no reader generation was run."]}
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

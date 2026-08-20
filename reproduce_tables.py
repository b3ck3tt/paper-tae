"""Reproduce the main tables of the paper from the released CSVs (no API, no OpenML access needed).

Reads results/*.csv and prints:
  Table 1  per-model nregret@1 / hit@1 / Spearman / coverage
  Table 2  mean-difference vs portfolio + Benjamini-Hochberg-corrected p (10/11 worse)
  Table 5  task-invariance / degeneracy of top-1 picks

Usage:  python reproduce_tables.py
"""
import numpy as np, pandas as pd
from collections import Counter
RNG = np.random.default_rng(0)

L  = pd.read_csv("results/llm_metrics_per_task.csv")            # model, tid, nregret@1, hit@1, spearman
B  = pd.read_csv("results/baselines_main200.csv")               # task_id, method, nregret@1, ...
RK = pd.read_csv("results/rankings_main200.csv")                # model, task_id, ranking
port = B[B.method == "global_default"].set_index("task_id")["nregret@1"]

LAB = {"claude-opus-4-8":"Opus 4.8","claude-sonnet-5":"Sonnet 5","claude-haiku-4-5-20251001":"Haiku 4.5",
 "claude-fable-5":"Fable 5","claude-sonnet-5-think":"Sonnet 5 +think","gpt-5":"GPT-5","gpt-5-mini":"GPT-5-mini",
 "deepinfra_deepseek-ai_DeepSeek-V3":"DeepSeek-V3","deepinfra_Qwen_Qwen2.5-72B-Instruct":"Qwen2.5-72B",
 "deepinfra_meta-llama_Llama-3.3-70B-Instruct":"Llama-3.3-70B",
 "deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct":"Llama-3.1-8B"}
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); adj=np.empty(n); prev=1
    for k,i in enumerate(o[::-1]):
        r=n-k; prev=min(prev,p[i]*n/r); adj[i]=prev
    return adj

print("\n=== Table 1: quality (baselines on the 200-task subset) ===")
for meth,lab in [("lambdamart","text-LTR"),("knn","kNN"),("global_default","portfolio")]:
    x=B[B.method==meth]["nregret@1"]
    print(f"  {lab:10s} nregret@1={x.mean():.3f}")
print(f"  {'-'*30}")
for m in sorted(L.model.unique(), key=lambda m:L[L.model==m]['nregret@1'].mean()):
    s=L[L.model==m]
    print(f"  {LAB.get(m,m):16s} nregret@1={s['nregret@1'].mean():.3f}  hit@1={s['hit@1'].mean():.3f}  Spearman={s['spearman'].mean():+.2f}")

print("\n=== Table 2: mean-difference vs portfolio, BH-corrected ===")
rows=[]
for m in L.model.unique():
    lm=L[L.model==m].set_index("tid")["nregret@1"]; c=sorted(set(lm.index)&set(port.index))
    d=(lm.reindex(c)-port.reindex(c)).values
    bs=np.array([RNG.choice(d,len(d),True).mean() for _ in range(10000)])
    p=max(2*min((bs<=0).mean(),(bs>=0).mean()),1e-4)
    rows.append([LAB.get(m,m),d.mean(),p])
adj=bh([r[2] for r in rows]); nworse=0
for (lab,dd,p),a in sorted(zip(rows,adj),key=lambda z:z[0][1]):
    w = a<0.05 and dd>0; nworse+=w
    print(f"  {lab:16s} Δ={dd:+.3f}  raw p={p:.3f}  BH p={a:.3f}  {'WORSE' if w else 'n.s.'}")
print(f"  => {nworse}/11 significantly worse than the portfolio after BH")

print("\n=== Table 5: task-invariance / degeneracy of top-1 picks ===")
sp=L.groupby("model")["spearman"].mean().to_dict()
degs=[]
for m,g in RK.groupby("model"):
    tops=[str(r).split(",")[0] for r in g["ranking"]]; c=Counter(tops); n=len(tops)
    H=-sum((k/n)*np.log2(k/n) for k in c.values())
    degs.append((LAB.get(m,m),H,len(c),max(c.values())/n,sp.get(m,np.nan)))
print(f"  {'model':16s} {'entropy':>7} {'uniq':>5} {'mode%':>6} {'Spearman':>9}")
for lab,H,u,ms,s in sorted(degs,key=lambda x:x[1]):
    print(f"  {lab:16s} {H:7.2f} {u:5d} {ms:6.2f} {s:9.2f}")
print(f"  (max entropy log2(200)={np.log2(200):.2f}; low entropy + ~0 Spearman = task-invariant collapse)")

"""CC18 task packs for the P4 harness (recommender/llm_paper4.py), built exactly like the long-tail pack:
candidates = every flow with an evaluation on the task (all toolkits, agg = max of predictive_accuracy), catalog =
union of candidates over the selected tasks, same INSTRUCTIONS, same anonymized meta-features (no name / id),
same suffix format. No API calls.

  A  all 72 OpenML-CC18 tasks, all toolkits (no MIN_CAND filter: the smallest task has 12 candidates)
     -> results/results_recommender_paper4_cc18all/pack.joblib
  B  the existing sklearn-only flow-level pack (69 tasks with >= 15 sklearn candidates), copied unchanged so the
     new run gets its own output directory   -> results/results_recommender_paper4_cc18sk/pack.joblib

Usage: python recommender/llm_pack_cc18_p4.py"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, shutil, joblib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_paper4 import INSTRUCTIONS, META, clean_flow, est_tokens

BUNDLE = "data/minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib"


def build_all():
    b = joblib.load(BUNDLE)
    ev = b["evals_df"][["task_id", "flow_id", "flow_name", "value"]].copy()
    ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce"); ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    ev = ev.dropna(subset=["flow_id", "value"]); ev["flow_id"] = ev["flow_id"].astype(int)
    agg = ev.groupby(["task_id", "flow_id"], as_index=False)["value"].max().rename(columns={"value": "acc"})
    fname = ev.drop_duplicates("flow_id").set_index("flow_id")["flow_name"].to_dict()
    tmeta = b["tasks_df"].set_index("tid")[META]
    tasks = sorted(int(t) for t in agg.task_id.unique() if t in tmeta.index)
    used = sorted(agg[agg.task_id.isin(tasks)]["flow_id"].unique())
    fid2cat = {f: f"F{i}" for i, f in enumerate(used)}
    prefix = INSTRUCTIONS + "\n".join(f"{fid2cat[f]}: {clean_flow(fname.get(f, ''))}" for f in used)
    packs, truth = [], {}
    for tid in tasks:
        g = agg[agg.task_id == tid]; mv = tmeta.loc[tid]
        meta_str = "; ".join(f"{c}={int(mv[c]) if pd.notna(mv[c]) else 'NA'}" for c in META)
        cids = [fid2cat[f] for f in g["flow_id"]]
        suffix = (f"\n\nDATASET (anonymized) meta-features: {meta_str}\n"
                  f"Candidate pipeline IDs ({len(cids)}): {','.join(cids)}\n"
                  f"Rank ALL of these candidate IDs best-to-worst, comma-separated:")
        packs.append({"tid": tid, "suffix": suffix, "cids": cids})
        truth[tid] = g[["flow_id", "acc"]].assign(cat=g["flow_id"].map(fid2cat)).to_dict("records")
    out = Path("results/results_recommender_paper4_cc18all"); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"prefix": prefix, "packs": packs, "truth": truth, "fid2cat": fid2cat}, out / "pack.joblib")
    nc = [len(p["cids"]) for p in packs]
    print(f"A: {len(packs)} tasks, catalog {len(used)} flows, candidates/task min {min(nc)} median {int(np.median(nc))} "
          f"mean {np.mean(nc):.0f} max {max(nc)}, prefix ~{est_tokens(prefix)} tok (chars/3.5)")


def copy_sklearn():
    out = Path("results/results_recommender_paper4_cc18sk"); out.mkdir(parents=True, exist_ok=True)
    shutil.copy("results/results_recommender_paper4_cc18flow/pack.joblib", out / "pack.joblib")
    d = joblib.load(out / "pack.joblib"); nc = [len(p["cids"]) for p in d["packs"]]
    print(f"B: {len(d['packs'])} tasks, catalog {len(d['fid2cat'])} flows, candidates/task min {min(nc)} median "
          f"{int(np.median(nc))} mean {np.mean(nc):.0f} max {max(nc)}, prefix ~{est_tokens(d['prefix'])} tok")


if __name__ == "__main__":
    build_all(); copy_sklearn()

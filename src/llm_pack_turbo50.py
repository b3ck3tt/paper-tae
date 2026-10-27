"""Control pack for the Llama-Turbo substitutes: 50 long-tail tasks drawn with a fixed seed (2026) from the P4
main 200-task pack (same prefix, suffixes and truth), so the Turbo rankings can be compared paired with the
original P4 Llama rankings. No API calls. Usage: python recommender/llm_pack_turbo50.py"""
import joblib, numpy as np
from pathlib import Path
d = joblib.load("results/results_recommender_paper4/pack.joblib")
rng = np.random.default_rng(2026)
idx = sorted(rng.choice(len(d["packs"]), size=50, replace=False))
packs = [d["packs"][i] for i in idx]
out = Path("results/results_recommender_paper4_turbo50"); out.mkdir(parents=True, exist_ok=True)
joblib.dump({"prefix": d["prefix"], "packs": packs, "truth": {p["tid"]: d["truth"][p["tid"]] for p in packs},
             "fid2cat": d["fid2cat"]}, out / "pack.joblib")
print("turbo50:", len(packs), "tasks, first tids", [p["tid"] for p in packs[:5]])

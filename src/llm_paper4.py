"""Paper 4 (full): LLM-as-recommender at flow level, with prompt caching + cost dry-run.

Design for cheap caching: every prompt = [ cached prefix: instructions + FULL flow catalog ] +
[ per-task suffix: anonymized metafeatures + candidate flow IDs + "rank them" ]. The catalog is
byte-identical across tasks -> written once, read at ~0.1x thereafter.

Modes:
  build            build catalog + per-task packs (+ hidden ground truth/baselines), save to disk
  dryrun [model]   estimate tokens + project cost per model (NO API, NO key needed)
  run <model> [N]  call the model with caching on N tasks; save rankings + real token usage
  eval             score LLM rankings vs trained baselines (text-LTR/kNN/portfolio/random)

Cross-vendor (Paper 4 robustness: the negative result should not be Claude-specific). The provider is
inferred from the model id, and each provider reads its own key (env var or ~/.<provider>_key, chmod 600):
  anthropic  claude-*                              ANTHROPIC_API_KEY  / ~/.anthropic_key
  openai     gpt-*, o1/o3/o4-*, or openai:<id>     OPENAI_API_KEY     / ~/.openai_key
  deepinfra  deepinfra:<org/model>                 DEEPINFRA_API_KEY  / ~/.deepinfra_key
OpenAI + DeepInfra share the OpenAI-compatible client (only base_url differs), so one code path covers
GPT and any open-weight family DeepInfra hosts. Examples:
  run gpt-5 ; run gpt-5-mini
  run "deepinfra:meta-llama/Llama-3.3-70B-Instruct" ; run "deepinfra:Qwen/Qwen2.5-72B-Instruct"
  run "deepinfra:deepseek-ai/DeepSeek-V3" ; run "deepinfra:meta-llama/Meta-Llama-3.1-8B-Instruct"
Keep the SAME pack (build once) so every vendor ranks the identical tasks/candidates -> comparable.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, re, time, joblib, numpy as np, pandas as pd
from pathlib import Path
OUT = Path(os.environ.get("P4_OUT", "results/results_recommender_paper4")); OUT.mkdir(exist_ok=True)
SCR = Path("scratch")
# Sub-experiments (full-1236 stability, repeated runs) override output dir / pack path so they don't
# collide with the main 200-task results. Default: pack lives inside OUT.
PACK = Path(os.environ.get("P4_PACK", str(OUT / "pack.joblib")))

RNG = np.random.default_rng(int(os.environ.get("P4_SEED", "7")))  # P4_SEED for a fresh subset (stability)
N_TASKS = int(os.environ.get("N_TASKS", "200"))
MIN_CAND = 15
MAX_ATTEMPTS = int(os.environ.get("P4_MAX_ATTEMPTS", "3"))   # retries for empty/truncated answers
META = ["NumberOfInstances","NumberOfFeatures","NumberOfClasses","NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures","MajorityClassSize","MinorityClassSize","NumberOfInstancesWithMissingValues"]

# approx pricing $/Mtok (input, output, cache-write, cache-read). ALL APPROXIMATE -- VERIFY before spend.
# For OpenAI-compatible providers there is no separate cache-write cost (caching is automatic): cw=inp;
# OpenAI discounts cached reads (~cr=0.5*inp), DeepInfra generally does not (cr=inp).
PRICING = {
    # --- Anthropic ---
    "claude-opus-4-8":            dict(inp=15.0, out=75.0, cw=18.75, cr=1.50),
    "claude-sonnet-5":            dict(inp=3.0,  out=15.0, cw=3.75,  cr=0.30),
    "claude-haiku-4-5-20251001":  dict(inp=0.80, out=4.0,  cw=1.00,  cr=0.08),
    "claude-fable-5":             dict(inp=10.0, out=50.0, cw=12.50, cr=1.00),  # premium-priced
    # --- OpenAI (VERIFY: 2026 prices) ---
    "gpt-5":                      dict(inp=1.25, out=10.0, cw=1.25,  cr=0.125),
    "gpt-5-mini":                 dict(inp=0.25, out=2.0,  cw=0.25,  cr=0.025),
    # --- DeepInfra open-weight (VERIFY; no cache discount) ---
    "deepinfra:meta-llama/Llama-3.3-70B-Instruct":     dict(inp=0.23, out=0.40, cw=0.23, cr=0.23),
    "deepinfra:Qwen/Qwen2.5-72B-Instruct":             dict(inp=0.23, out=0.40, cw=0.23, cr=0.23),
    "deepinfra:deepseek-ai/DeepSeek-V3":               dict(inp=0.40, out=0.89, cw=0.40, cr=0.40),
    "deepinfra:meta-llama/Meta-Llama-3.1-8B-Instruct": dict(inp=0.03, out=0.05, cw=0.03, cr=0.03),
    # Turbo (FP8) substitutes; the original Llama ids are no longer served (DeepInfra models list, 27 Oct 2026)
    "deepinfra:meta-llama/Llama-3.3-70B-Instruct-Turbo":     dict(inp=0.10, out=0.32, cw=0.10, cr=0.10),
    "deepinfra:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": dict(inp=0.02, out=0.04, cw=0.02, cr=0.02),
}
if os.environ.get("P4_PRICES_2026_10"):   # current list prices for the CC18 re-run (DeepInfra models list, 27 Oct 2026)
    PRICING["deepinfra:Qwen/Qwen2.5-72B-Instruct"] = dict(inp=0.36, out=0.40, cw=0.36, cr=0.36)
    PRICING["deepinfra:deepseek-ai/DeepSeek-V3"] = dict(inp=0.32, out=0.89, cw=0.32, cr=0.32)

# Spend guard across several concurrent runs: P4_BUDGET_DIRS = ':'-separated output dirs whose api_log.jsonl
# entries are summed (list prices above); a run stops before a new call once the total reaches P4_BUDGET_STOP.
def spent_so_far():
    tot = 0.0
    for d in filter(None, os.environ.get("P4_BUDGET_DIRS", "").split(":")):
        f = Path(d) / "api_log.jsonl"
        if not f.exists(): continue
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if "inp" not in r: continue
            pr = _price(r["model"][:-6] if r["model"].endswith("-think") else r["model"])
            tot += (r["inp"]*pr["inp"] + r["out"]*pr["out"] + r["cw"]*pr["cw"] + r["cr"]*pr["cr"]) / 1e6
    return tot
def _price(model): return PRICING.get(model) or dict(inp=1.0, out=3.0, cw=1.0, cr=1.0)  # fallback
def est_tokens(s): return int(len(s) / 3.5)  # rough; real usage comes from API response

def clean_flow(name):
    n = str(name or "")
    n = re.sub(r"\(.*?\)", "", n)                       # drop param blobs
    n = n.replace("sklearn.", "").replace("weka.classifiers.", "weka.").replace("mlr.", "mlr.")
    return n.strip()[:80]

def load_longtail():
    coll = joblib.load("data/minimal_cache_cc18/longtail_collected.joblib")
    tasks_all = joblib.load("data/minimal_cache_cc18/longtail_tasks.joblib")
    ev = pd.concat(coll["evals"].values(), ignore_index=True)
    ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce"); ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    ev = ev.dropna(subset=["flow_id","value"]); ev["flow_id"] = ev["flow_id"].astype(int)
    agg = ev.groupby(["task_id","flow_id"], as_index=False)["value"].max().rename(columns={"value":"acc"})
    fname = ev.drop_duplicates("flow_id").set_index("flow_id")["flow_name"].to_dict()
    return agg, fname, tasks_all

INSTRUCTIONS = (
 "You are an expert AutoML assistant. Given a dataset described ONLY by anonymized numeric "
 "meta-features (no name), and a list of candidate machine-learning pipelines (referenced by ID from "
 "the catalog below), rank the candidates from most to least likely to achieve the highest predictive "
 "accuracy on that dataset. Use your knowledge of how algorithm families behave on datasets of "
 "different size, dimensionality, and class balance. Output ONLY the candidate IDs in ranked order, "
 "best first, comma-separated (e.g. F12,F3,F45). No explanation.\n\nPIPELINE CATALOG:\n")

def build():
    agg, fname, tasks_all = load_longtail()
    tmeta = tasks_all.set_index("tid")[META]
    cand = agg.groupby("task_id").size(); tasks = cand[cand >= MIN_CAND].index.to_numpy()
    RNG.shuffle(tasks); tasks = sorted(tasks[:N_TASKS])
    # catalog = union of candidate flows across selected tasks
    used = sorted(agg[agg.task_id.isin(tasks)]["flow_id"].unique())
    fid2cat = {f: f"F{i}" for i, f in enumerate(used)}
    catalog = "\n".join(f"{fid2cat[f]}: {clean_flow(fname.get(f,''))}" for f in used)
    prefix = INSTRUCTIONS + catalog
    packs = []; truth = {}
    for tid in tasks:
        g = agg[agg.task_id == tid]
        if tid not in tmeta.index: continue
        mv = tmeta.loc[tid]
        meta_str = "; ".join(f"{c}={int(mv[c]) if pd.notna(mv[c]) else 'NA'}" for c in META)
        cids = [fid2cat[f] for f in g["flow_id"]]
        suffix = (f"\n\nDATASET (anonymized) meta-features: {meta_str}\n"
                  f"Candidate pipeline IDs ({len(cids)}): {','.join(cids)}\n"
                  f"Rank ALL of these candidate IDs best-to-worst, comma-separated:")
        packs.append({"tid": int(tid), "suffix": suffix, "cids": cids})
        truth[int(tid)] = g[["flow_id","acc"]].assign(cat=g["flow_id"].map(fid2cat)).to_dict("records")
    joblib.dump({"prefix": prefix, "packs": packs, "truth": truth, "fid2cat": fid2cat},
                PACK)
    ntok_prefix = est_tokens(prefix)
    ntok_suffix = int(np.mean([est_tokens(p["suffix"]) for p in packs]))
    ncand = int(np.mean([len(p["cids"]) for p in packs]))
    print(f"built: {len(packs)} tasks, catalog {len(used)} flows")
    print(f"prefix ~{ntok_prefix} tok (cached), mean suffix ~{ntok_suffix} tok, mean {ncand} candidates/task")
    return ntok_prefix, ntok_suffix, ncand, len(packs)

def dryrun():
    d = joblib.load(PACK)
    prefix = d["prefix"]; packs = d["packs"]
    ntok_prefix = est_tokens(prefix)
    tok_suffix = [est_tokens(p["suffix"]) for p in packs]
    est_out = [max(60, len(p["cids"]) * 4) for p in packs]   # ~4 tok per "F123," in output
    N = len(packs)
    print(f"\n{N} tasks | prefix(cached) ~{ntok_prefix} tok | suffix sum ~{sum(tok_suffix)} | output sum ~{sum(est_out)}\n")
    print(f"{'model':>28} {'naive $':>9} {'cached $':>9} {'cached+batch $':>15}")
    for m, pr in PRICING.items():
        # naive: full prefix every call
        naive = (N*ntok_prefix + sum(tok_suffix))/1e6*pr["inp"] + sum(est_out)/1e6*pr["out"]
        # cached: write prefix once (cw), read prefix N-1 times (cr), suffixes full, output full
        cached = (ntok_prefix/1e6*pr["cw"] + (N-1)*ntok_prefix/1e6*pr["cr"]
                  + sum(tok_suffix)/1e6*pr["inp"] + sum(est_out)/1e6*pr["out"])
        print(f"{m:>28} {naive:>9.2f} {cached:>9.2f} {cached*0.5:>15.2f}")
    print("\n(estimates: tokens ~chars/3.5, pricing approximate — verify; real usage logged on first live call)")

_KEYENV = {"anthropic": ("ANTHROPIC_API_KEY", ".anthropic_key"),
           "openai":    ("OPENAI_API_KEY",    ".openai_key"),
           "deepinfra": ("DEEPINFRA_API_KEY", ".deepinfra_key")}
_BASEURL = {"openai": None, "deepinfra": "https://api.deepinfra.com/v1/openai"}

def _load_key(provider):
    env, fname = _KEYENV[provider]
    k = os.environ.get(env)
    if k: return k
    p = Path.home() / fname
    if p.exists(): return p.read_text().strip()
    raise SystemExit(f"No API key for {provider}: set {env} or write ~/{fname}")

def _provider_and_model(model):
    """Infer (provider, api_model) from the model id. `model` has any '-think' suffix already stripped."""
    if model.startswith("deepinfra:"): return "deepinfra", model.split(":", 1)[1]
    if model.startswith("openai:"):    return "openai", model.split(":", 1)[1]
    if model.startswith(("gpt", "o1", "o3", "o4")): return "openai", model
    return "anthropic", model

def _safe(model):  # model id -> filesystem-safe token (deepinfra ids contain '/' and ':')
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)

def _parse_ranking(text, valid):
    ids = re.findall(r"F\d+", text or "")
    seen = set(); order = []
    for i in ids:
        if i in valid and i not in seen:
            seen.add(i); order.append(i)
    return order

def _call_anthropic(client, api_model, prefix, suffix, ncand, think):
    kw = dict(model=api_model,
              system=[{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}],
              messages=[{"role": "user", "content": suffix}])
    if think:                     # Claude 5 extended thinking: adaptive, medium effort, GENEROUS ceiling
        kw["thinking"] = {"type": "adaptive"}
        kw["output_config"] = {"effort": "medium"}
        kw["max_tokens"] = min(20000, ncand * 8 + 10000)
    elif "fable" in api_model:    # Fable forces adaptive thinking -> leave room for thinking + ranking
        kw["max_tokens"] = min(16000, ncand * 8 + 6000)
    else:                         # others: disable thinking, tight budget for the ranking
        kw["max_tokens"] = min(8000, ncand * 8 + 500)
        kw["thinking"] = {"type": "disabled"}
    msg = client.messages.create(**kw)
    txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    u = msg.usage
    return txt, dict(inp=u.input_tokens, out=u.output_tokens,
                     cw=getattr(u, "cache_creation_input_tokens", 0) or 0,
                     cr=getattr(u, "cache_read_input_tokens", 0) or 0,
                     request_id=getattr(msg, "_request_id", None), response_id=msg.id, resp_model=msg.model,
                     stop=msg.stop_reason, truncated=msg.stop_reason == "max_tokens")

def _call_openai(client, api_model, prefix, suffix, ncand):
    # prefix as system message (auto prefix-caching on OpenAI); suffix as user turn.
    kw = dict(model=api_model,
              messages=[{"role": "system", "content": prefix}, {"role": "user", "content": suffix}])
    reasoning = api_model.startswith(("gpt-5", "o1", "o3", "o4"))  # reasoning models: different token param, no temp
    if reasoning:
        # Bound the reasoning: at default (high) effort gpt-5 can reason without terminating on hard ranking
        # tasks (call hangs / never emits the ranking). "medium" caps it so calls finish; the budget then
        # only needs headroom for bounded reasoning + the ranking (too tight -> EMPTY/truncated ranking).
        kw["reasoning_effort"] = "low"
        kw["max_completion_tokens"] = min(24000, ncand * 8 + 16000)
    else:
        kw["max_tokens"] = min(8000, ncand * 8 + 500); kw["temperature"] = 0
    resp = client.chat.completions.create(**kw)
    txt = resp.choices[0].message.content or ""
    u = resp.usage
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    ptd = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(ptd, "cached_tokens", 0) or 0) if ptd is not None else 0
    fin = resp.choices[0].finish_reason
    return txt, dict(inp=pt - cached, out=ct, cw=0, cr=cached,  # cached reads billed at cr rate
                     request_id=getattr(resp, "_request_id", None), response_id=resp.id, resp_model=resp.model,
                     stop=fin, truncated=fin == "length")

def run(model, N=None):
    d = joblib.load(PACK)
    prefix, packs = d["prefix"], d["packs"]
    if N: packs = packs[:N]
    think = model.endswith("-think")                 # e.g. "claude-sonnet-5-think" -> thinking ablation
    base = model[:-6] if think else model
    provider, api_model = _provider_and_model(base)
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=_load_key(provider), timeout=600.0, max_retries=2)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=_load_key(provider), base_url=_BASEURL[provider],
                        timeout=900.0, max_retries=2)   # 900s: bounded-reasoning calls can still run a few min
    print(f"provider={provider} api_model={api_model}" + (" (+thinking)" if think else ""))
    resfile = OUT / f"llm_{_safe(model)}.joblib"
    done = joblib.load(resfile) if resfile.exists() else {}
    usage_tot = dict(inp=0, out=0, cw=0, cr=0)
    for i, p in enumerate(packs):
        tid = p["tid"]
        prev = done.get(tid) or done.get(str(tid))
        if prev and ("ranking" in prev or prev.get("final")):   # skip successes and final failures; retry errors
            continue
        stop = float(os.environ.get("P4_BUDGET_STOP", "0") or 0)
        if stop and spent_so_far() >= stop:
            joblib.dump(done, resfile)
            print(f"BUDGET STOP: spend across {os.environ.get('P4_BUDGET_DIRS')} >= ${stop:.0f}", flush=True)
            raise SystemExit(3)
        valid = set(p["cids"]); ncand = len(p["cids"])
        attempts = []
        for attempt in range(MAX_ATTEMPTS):            # empty or truncated answers are re-asked, never dropped
            try:
                if provider == "anthropic":
                    txt, u = _call_anthropic(client, api_model, prefix, p["suffix"], ncand, think)
                else:
                    txt, u = _call_openai(client, api_model, prefix, p["suffix"], ncand)
                ranking = _parse_ranking(txt, valid)
                with open(OUT / "responses.jsonl", "a") as fh:      # raw answer text, for parser/empty audits
                    fh.write(json.dumps({"tid": int(tid), "model": model, "attempt": attempt,
                                         "response_id": u.get("response_id"), "text": txt}) + "\n")
                for k in usage_tot: usage_tot[k] += u[k]
                rec = {"attempt": attempt, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model,
                       **{k: u[k] for k in ("request_id", "response_id", "resp_model", "stop", "truncated",
                                            "inp", "out", "cw", "cr")}, "n_parsed": len(ranking)}
                with open(OUT / "api_log.jsonl", "a") as fh:
                    fh.write(json.dumps({"tid": int(tid), **rec}) + "\n")
                rec["ranking"] = ranking; rec["usage"] = u      # every attempt's ranking is kept (protocol rule, 27 Oct)
                attempts.append(rec)
                if ranking and not u["truncated"]:
                    break
            except Exception as e:
                attempts.append({"attempt": attempt, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 "error": str(e)[:200]})
                with open(OUT / "api_log.jsonl", "a") as fh:
                    fh.write(json.dumps({"tid": int(tid), "model": model, **attempts[-1]}) + "\n")
                ranking, u = None, None
        # protocol rule (27 Oct 2026): a normally finished attempt with a ranking wins; otherwise the attempt with the
        # LONGEST valid ranking (selection by length only, never by accuracy); none with a valid id -> missing.
        good = [x for x in attempts if x.get("ranking") and not x.get("truncated")]
        part = [x for x in attempts if x.get("ranking")]
        pick = good[-1] if good else (max(part, key=lambda x: len(x["ranking"])) if part else None)
        ranking, u = (pick["ranking"], pick["usage"]) if pick else (None, None)
        for x in attempts:                                   # keep the result file compact: rankings of attempts
            x.pop("usage", None)
        if ranking:
            done[tid] = {"ranking": ranking, "n_parsed": len(ranking), "n_cand": len(valid),
                         "picked_attempt": pick["attempt"],
                         # per-task usage = a "reasoning effort / complexity" signal (esp. tok_out for thinking)
                         "tok_in": u["inp"], "tok_out": u["out"],
                         "tok_cache_read": u["cr"], "tok_cache_write": u["cw"],
                         "truncated": u["truncated"], "attempts": attempts}
        else:
            answered = attempts and all("error" not in x for x in attempts)   # model answered every time: final
            done[tid] = {"error": (attempts[-1].get("error") if attempts else "no attempt") or "empty ranking",
                         "attempts": attempts, "final": bool(answered)}
        if (i + 1) % 3 == 0 or i + 1 == len(packs):
            joblib.dump(done, resfile)
            print(f"  {i+1}/{len(packs)} done", flush=True)
    joblib.dump(done, resfile)
    pr = _price(base)
    cost = (usage_tot["inp"]/1e6*pr["inp"] + usage_tot["out"]/1e6*pr["out"]
            + usage_tot["cw"]/1e6*pr["cw"] + usage_tot["cr"]/1e6*pr["cr"])
    ok = sum(1 for v in done.values() if "ranking" in v)
    err = next((v["error"] for v in done.values() if "error" in v), None)
    print(f"\n{model}: {ok}/{len(done)} tasks ranked | REAL usage {usage_tot} | est cost ${cost:.2f}")
    if ok < len(done) and err: print(f"  first error: {err}")
    if usage_tot["cr"]:
        print(f"  (cache_read {usage_tot['cr']} vs input {usage_tot['inp']} -> caching active)")

def evaluate():
    from scipy.stats import spearmanr
    d = joblib.load(PACK); truth = d["truth"]
    models = [f.stem.replace("llm_", "") for f in OUT.glob("llm_*.joblib")]
    if not models: raise SystemExit("no llm_<model>.joblib results yet")
    rows = []
    for model in models:
        res = joblib.load(OUT / f"llm_{model}.joblib")
        for tid, r in res.items():
            tid = int(tid)
            if "ranking" not in r or not r["ranking"] or tid not in truth: continue
            recs = truth[tid]; acc = {x["cat"]: x["acc"] for x in recs}
            best = max(acc.values()); worst = min(acc.values()); rng = best - worst
            if rng <= 0: continue
            rank = r["ranking"]
            top = rank[0]
            nreg1 = (best - acc.get(top, worst)) / rng
            hit1 = int(acc.get(top, -1) == best)
            # spearman: predicted rank position (0=best) vs accuracy -> expect negative; report -rho
            pos = {c: i for i, c in enumerate(rank)}
            common = [c for c in acc if c in pos]
            rho = np.nan
            if len(common) > 2:
                rho = spearmanr([pos[c] for c in common], [acc[c] for c in common]).correlation
            rows.append({"model": model, "tid": tid, "nregret@1": nreg1, "hit@1": hit1, "spearman": -rho if rho==rho else np.nan})
    L = pd.DataFrame(rows)
    print("\n=== LLM-as-recommender (flow-level) vs trained baselines ===")
    print(L.groupby("model").agg(n=("tid","nunique"), nregret1=("nregret@1","mean"),
                                 hit1=("hit@1","mean"), spearman=("spearman","mean")).round(4).to_string())
    tasks = set(L["tid"])
    pt = pd.read_csv("results/results_recommender_longtail_full/ranking_metrics_per_task.csv")
    pt = pt[(pt.experiment=="tfidf_meta") & (pt.task_id.isin(tasks))]
    print(f"\nbaselines on the same {len(tasks)} tasks:")
    print(pt.groupby("method").agg(nregret1=("nregret@1","mean"), hit1=("hit@1","mean"),
                                   spearman=("spearman","mean")).round(4).to_string())
    L.to_csv(OUT / "llm_metrics_per_task.csv", index=False)
    print("\nsaved llm_metrics_per_task.csv")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build": build()
    elif mode == "dryrun":
        if not PACK.exists(): build()
        dryrun()
    elif mode == "run":
        model = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-5"
        N = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run(model, N)
    elif mode == "eval":
        evaluate()
    else:
        print("modes: build | dryrun | run <model> [N] | eval")

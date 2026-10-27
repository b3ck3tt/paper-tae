"""Scoring of the CC18 API re-run and the long-tail family cells. No API calls.

Flow level: nregret@1, hit@1, Spearman (as llm_paper4.evaluate), coverage = parsed / candidates.
Family level: flows -> estimator families with the P4 §4.3 map (paper-tae/family_level.py); a model's family order
  is the order of first appearance in its flow ranking; family accuracy = best flow accuracy in the family.
Portfolios, all leave-one-task-out on the same package (no information from the evaluated task):
  flow       flows ordered by mean accuracy on the other tasks (unseen flows last)
  family (a) families ordered by mean flow accuracy of the family on the other tasks (P4 definition)
  family (b) families ordered by the mean over the other tasks of the best accuracy in the family
Statistics per model and cell, on the common task set of the compared models: mean difference model - portfolio,
  paired bootstrap 95 % CI (10,000 resamples), Wilcoxon signed-rank p, Benjamini-Hochberg over the models of a cell.
Usage: python recommender/score_cc18_rerun.py B|A|LT|T  -> prints markdown, writes results/cc18_rerun_score_<pkg>.md"""
import sys, os, re, json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon

# Flow portfolio "as in P4": the best-on-average baseline of the recommender pipeline (task_group_kfold, trained
# on the other tasks only), per task, identical candidate sets (checked: n_candidates equal on every task).
P4_PORTFOLIO = {"B": "results/results_recommender_max/ranking_metrics_per_task.csv",
                "A": "results/results_recommender_tk_all/ranking_metrics_per_task.csv",
                "LT": "results/results_recommender_longtail_full/ranking_metrics_per_task.csv",
                "T": "results/results_recommender_longtail_full/ranking_metrics_per_task.csv"}


def p4_portfolio(pkg):
    pt = pd.read_csv(P4_PORTFOLIO[pkg]); pt = pt[(pt.experiment == "tfidf_meta") & (pt.method == "global_default")]
    return dict(zip(pt.task_id.astype(int), pt["nregret@1"]))


DIRS = {"B": "results/results_recommender_paper4_cc18sk", "A": "results/results_recommender_paper4_cc18all",
        "LT": "results/results_recommender_paper4", "T": "results/results_recommender_paper4_turbo50"}
FAM = [("random_forest", "random.?forest|randomforest"), ("extra_trees", "extra.?tree"),
       ("gradient_boosting", "gradientboost|gradient_boost|hist.?gradient|xgb|xgboost|lightgbm|gbm|adaboost|boosting"),
       ("svc", "svc|svm|support.?vector"), ("logistic", "logistic|logreg"), ("knn", "kneighbors|knn|neighbors"),
       ("naive_bayes", "naive.?bayes|gaussiannb|bernoullinb|multinomialnb"),
       ("decision_tree", "decisiontree|decision_tree|j48|reptree"), ("mlp", "mlp|perceptron|neural"),
       ("linear", "sgd|ridge|lasso|elasticnet|linearmodel|passiveaggr"), ("bagging", "bagging"),
       ("discriminant", "discriminant|lda|qda")]
LABEL = {"claude-haiku-4-5-20251001": "Haiku 4.5", "claude-sonnet-5": "Sonnet 5", "claude-sonnet-5-think": "Sonnet 5 + thinking",
         "claude-opus-4-8": "Opus 4.8", "claude-fable-5": "Fable 5", "gpt-5": "GPT-5", "gpt-5-mini": "GPT-5-mini",
         "deepinfra_deepseek-ai_DeepSeek-V3": "DeepSeek-V3", "deepinfra_Qwen_Qwen2.5-72B-Instruct": "Qwen2.5-72B",
         "deepinfra_meta-llama_Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B-Turbo (FP8)",
         "deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo": "Llama-3.1-8B-Turbo (FP8)",
         "deepinfra_meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B", "deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
RNG = np.random.default_rng(0)
P4P = {}


def family(name):
    n = str(name).lower()
    for f, pat in FAM:
        if re.search(pat, n):
            return f
    return "other"


def flow_names(pkg):
    if pkg == "LT" or pkg == "T":
        coll = joblib.load("data/minimal_cache_cc18/longtail_collected.joblib")
        ev = pd.concat(coll["evals"].values(), ignore_index=True)
    else:
        ev = joblib.load("data/minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")["evals_df"]
    ev = ev[["flow_id", "flow_name"]].copy(); ev["flow_id"] = pd.to_numeric(ev.flow_id, errors="coerce")
    return ev.dropna().drop_duplicates("flow_id").set_index("flow_id")["flow_name"].to_dict()


def load(pkg):
    d = Path(DIRS[pkg]); pack = joblib.load(d / "pack.joblib")
    names = flow_names(pkg)
    truth = {}
    for t, recs in pack["truth"].items():
        df = pd.DataFrame(recs); df["family"] = df.flow_id.map(lambda f: family(names.get(f, "")))
        truth[int(t)] = df
    models = {}
    excl = [x for x in os.environ.get("SCORE_EXCLUDE", "").split(",") if x]
    for f in sorted(d.glob("llm_*.joblib")):
        if ".context_limit" in f.name or f.name.endswith(".bak") or any(x in f.name for x in excl):
            continue
        models[f.stem[4:]] = joblib.load(f)
    return pack, truth, models


def nreg_flow(order, df):
    acc = dict(zip(df.cat, df.acc)); best, worst = max(acc.values()), min(acc.values())
    if best - worst <= 0:
        return None
    for c in order:
        if c in acc:
            return (best - acc[c]) / (best - worst)
    return None


def nreg_fam(order, df):
    facc = df.groupby("family").acc.max(); best, worst = facc.max(), facc.min()
    if best - worst <= 0:
        return None
    for f in order:
        if f in facc.index:
            return (best - facc[f]) / (best - worst)
    return None


def portfolios(truth):
    """Leave-one-task-out orders per task: flow, family (a), family (b)."""
    allr = pd.concat([df.assign(task=t) for t, df in truth.items()])
    best_in = {t: df.groupby("family").acc.max() for t, df in truth.items()}
    orders = {}
    for t, df in truth.items():
        oth = allr[allr.task != t]
        fmean = oth.groupby("cat").acc.mean()
        flow_order = sorted(df.cat, key=lambda c: -fmean.get(c, -1.0))
        fam_a = list(oth.groupby("family").acc.mean().sort_values(ascending=False).index)
        fam_b = list(pd.concat([best_in[u] for u in truth if u != t], axis=1).mean(axis=1)
                     .sort_values(ascending=False).index)
        orders[t] = (flow_order, fam_a, fam_b)
    return orders


def per_task(models, truth):
    """{model: {task: dict(flow, family, hit, rho, cov)}} for tasks with a non-empty ranking."""
    out = {}
    for m, res in models.items():
        rows = {}
        for t, v in res.items():
            t = int(t)
            if not v.get("ranking") or t not in truth:
                continue
            df = truth[t]; rk = v["ranking"]
            fr = nreg_flow(rk, df)
            if fr is None:
                continue
            fam_order = []
            cat2fam = dict(zip(df.cat, df.family))
            for c in rk:
                f = cat2fam.get(c)
                if f is not None and f not in fam_order:
                    fam_order.append(f)
            acc = dict(zip(df.cat, df.acc)); pos = {c: i for i, c in enumerate(rk)}
            common = [c for c in acc if c in pos]
            rho = spearmanr([pos[c] for c in common], [acc[c] for c in common]).correlation if len(common) > 2 else np.nan
            rows[t] = dict(flow=fr, family=nreg_fam(fam_order, df), hit=int(acc.get(rk[0], -1) == max(acc.values())),
                           rho=-rho if rho == rho else np.nan, cov=len(rk) / len(df))
        out[m] = rows
    return out


def boot_ci(d):
    d = np.asarray(d); bs = [RNG.choice(d, len(d)).mean() for _ in range(10000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); prev = 1.0
    for k, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - k)); adj[i] = prev
    return adj


def cell(pt, orders, truth, tasks, level, which):
    """Rows (model, mean, port_mean, diff, lo, hi, p) on `tasks`; level in flow|family; which in 0|1|2."""
    port = {}
    for t in tasks:
        if which == "p4":
            port[t] = P4P.get(t); continue
        o = orders[t][which]
        port[t] = nreg_flow(o, truth[t]) if level == "flow" else nreg_fam(o, truth[t])
    rows = []
    for m, r in pt.items():
        ts = [t for t in tasks if t in r and r[t][level] is not None and port[t] is not None]
        # round: equal regrets computed along different code paths differ by ~1e-16; Wilcoxon must see them as ties
        a = np.round(np.array([r[t][level] for t in ts]), 10); b = np.round(np.array([port[t] for t in ts]), 10); d = a - b
        lo, hi = boot_ci(d)
        p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
        rows.append([m, a.mean(), b.mean(), d.mean(), lo, hi, p, len(ts)])
    adj = bh([r[6] for r in rows])
    for r, q in zip(rows, adj):
        r.append(q)
    return rows


def completeness(pack, models):
    nc = {p["tid"]: len(p["cids"]) for p in pack["packs"]}
    lines = ["| model | complete ranking | partial ranking (median coverage) | missing |", "|---|---|---|---|"]
    for m, res in models.items():
        comp, part, cov = 0, 0, []
        for t, v in res.items():
            if not v.get("ranking"):
                continue
            full = len(v["ranking"]) == nc[int(t)] and not v.get("truncated")
            comp += full
            if not full:
                part += 1; cov.append(len(v["ranking"]) / nc[int(t)])
        miss = len(nc) - comp - part
        lines.append(f"| {LABEL.get(m, m)} | {comp} | {part}" + (f" ({np.median(cov):.2f})" if cov else "") + f" | {miss} |")
    return lines


def median_model(pt, orders, truth, tasks, level, which):
    """Per task: the median nregret over models; compared paired with the portfolio on the same tasks."""
    a, bb = [], []
    for t in tasks:
        if which == "p4":
            pv = P4P.get(t)
        else:
            o = orders[t][which]
            pv = nreg_flow(o, truth[t]) if level == "flow" else nreg_fam(o, truth[t])
        vals = [r[t][level] for r in pt.values() if t in r and r[t][level] is not None]
        if pv is None or not vals:
            continue
        a.append(np.median(vals)); bb.append(pv)
    a, bb = np.round(np.array(a), 10), np.round(np.array(bb), 10); d = a - bb; lo, hi = boot_ci(d)
    return a.mean(), bb.mean(), d.mean(), lo, hi, (wilcoxon(a, bb).pvalue if np.any(d != 0) else 1.0), len(a)


def missing_report(pkg, pack, models):
    nc = {p["tid"]: len(p["cids"]) for p in pack["packs"]}
    lines = ["| model | tasks with ranking | missing after 3 attempts | truncated attempts / all attempts | "
             "median #cand (missing) | median #cand (ranked) |", "|---|---|---|---|---|---|"]
    for m, res in models.items():
        ok = [int(t) for t, v in res.items() if v.get("ranking")]
        miss = [int(t) for t in nc if int(t) not in ok]
        att = [a for v in res.values() for a in v.get("attempts", []) if "stop" in a]
        tr = sum(a["truncated"] for a in att)
        lines.append(f"| {LABEL.get(m, m)} | {len(ok)}/{len(nc)} | {len(miss)}"
                     f"{' (' + ', '.join(map(str, miss[:6])) + ('…' if len(miss) > 6 else '') + ')' if miss else ''} | "
                     f"{tr}/{len(att)} | {np.median([nc[t] for t in miss]) if miss else '—'} | "
                     f"{np.median([nc[t] for t in ok]):.0f} |")
    return lines


def main(pkg):
    pack, truth, models = load(pkg)
    pt = per_task(models, truth); orders = portfolios(truth)
    common = sorted(set.intersection(*[set(r) for r in pt.values()])) if pt else []
    allr = pd.concat(list(truth.values()))
    md = [f"# CC18 re-run scoring — package {pkg} ({len(truth)} tasks, {len(pack['fid2cat'])} catalog flows)\n",
          f"Family map: P4 §4.3. 'other' = {100 * np.mean([family_is_other for family_is_other in allr.drop_duplicates('cat').family.eq('other')]):.1f} % "
          f"of distinct candidate flows, {100 * allr.family.eq('other').mean():.1f} % of candidate rows.\n",
          "## Complete / partial / missing\n"] + completeness(pack, models) + ["", "## Coverage and missing tasks\n"] + missing_report(pkg, pack, models)
    md += [f"\nCommon task set (every model has a ranking): **{len(common)}** tasks.\n"]
    global P4P
    P4P = p4_portfolio(pkg)
    for level, which, name in [("flow", "p4", "flow portfolio (P4: best-on-average, task_group_kfold)"),
                               ("flow", 0, "flow portfolio, sensitivity: leave-one-task-out within the package"),
                               ("family", 1, "family portfolio (a) mean family accuracy"),
                               ("family", 2, "family portfolio (b) mean best-in-family accuracy")]:
        rows = cell(pt, orders, truth, common, level, which)
        med = median_model(pt, orders, truth, common, level, which)
        md += [f"\n## {level} level vs {name} — common set\n",
               "| model | model nregret@1 | portfolio | Δ (model − portfolio) | 95 % CI | Wilcoxon p | BH q | n |",
               "|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: x[1]):
            md.append(f"| {LABEL.get(r[0], r[0])} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:+.3f} | [{r[4]:+.3f}, {r[5]:+.3f}] | "
                      f"{r[6]:.2g} | {r[8]:.2g} | {r[7]} |")
        md.append(f"| **median over models** (per task) | {med[0]:.3f} | {med[1]:.3f} | {med[2]:+.3f} | [{med[3]:+.3f}, {med[4]:+.3f}] | "
                  f"{med[5]:.2g} | — | {med[6]} |")
        n_beat = sum(1 for r in rows if r[3] < 0); n_sig = sum(1 for r in rows if r[3] < 0 and r[8] < 0.05)
        md.append(f"\nModels below the portfolio: {n_beat} of {len(rows)} (BH q < 0.05: {n_sig}); "
                  f"above: {sum(1 for r in rows if r[3] > 0)} (BH q < 0.05: {sum(1 for r in rows if r[3] > 0 and r[8] < 0.05)}).")
    md += ["\n## Flow level, hit@1 / Spearman / coverage — common set\n", "| model | hit@1 | Spearman | coverage |", "|---|---|---|---|"]
    for m, r in pt.items():
        md.append(f"| {LABEL.get(m, m)} | {np.mean([r[t]['hit'] for t in common]):.2f} | "
                  f"{np.nanmean([r[t]['rho'] for t in common]):.2f} | {np.mean([r[t]['cov'] for t in common]):.2f} |")
    md += ["\n## All available tasks per model (NOT comparable across models)\n",
           "| model | n | flow nregret@1 | family nregret@1 |", "|---|---|---|---|"]
    for m, r in pt.items():
        md.append(f"| {LABEL.get(m, m)} | {len(r)} | {np.mean([v['flow'] for v in r.values()]):.3f} | "
                  f"{np.mean([v['family'] for v in r.values() if v['family'] is not None]):.3f} |")
    out = "\n".join(md) + "\n"
    Path(f"results/cc18_rerun_score_{pkg}{os.environ.get('SCORE_TAG', '')}.md").write_text(out)
    print(out)
    return pt, orders, truth, common


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "B")

"""Reproduce Table 4 (granularity x benchmark x baseline strength) and the two CC18 re-run checks of the paper
from the released CSVs (no API, no OpenML access needed).

Cells (one protocol, common task set per benchmark = tasks on which every model returned a ranking):
  flow level     model nregret@1 vs. the flow portfolio of the main study (best-on-average, trained without the
                 evaluated task, task-grouped CV; per task in results/baselines_main200.csv and
                 results/cc18/portfolio_cc18.csv)
  family level   flows -> coarse estimator families (regular expressions on the flow name, unmatched -> "other");
                 a model's family order = order of first appearance in its flow ranking; a family is credited with
                 the best accuracy of its flows. Two family portfolios, both leave-one-task-out:
                   (a) families ordered by the mean accuracy of all member flows on the other tasks
                   (b) families ordered by the mean over the other tasks of the best accuracy within the family
                 Sensitivity: the same with the "other" bucket removed for model and portfolio alike.
Statistics: per model, Wilcoxon signed-rank on the paired per-task regrets (regrets rounded to 10 decimals so
that ties are exact), Benjamini-Hochberg over the 11 models of a cell; a model counts as significantly
better/worse if BH q < 0.05.

Also prints:
  - the Llama-Turbo (FP8) control: the -Turbo models used on CC18 vs. the original Llama ids of the main study on
    50 long-tail tasks (paired difference, bootstrap 95 % CI, Wilcoxon p, top-1 agreement);
  - serving determinism (Appendix A): DeepSeek-V3 on the 69 CC18 tasks, August 2026 run vs. the logged October
    2026 re-run with a byte-identical prompt at temperature 0.

Expected output (Table 4):
  CC18       flow  DeepSeek-V3 0.086 vs 0.090   better 0 / worse 10 of 11
             (a)   Opus 4.8    0.047 vs 0.254   10 / 0   (without "other": 5 / 0)
             (b)   Opus 4.8    0.047 vs 0.066    0 / 7   (without "other": 0 / 7)
  long tail  flow  DeepSeek-V3 0.137 vs 0.114    0 / 6
             (a)   Sonnet 5 + thinking 0.114 vs 0.228   9 / 0   (without "other": 2 / 0)
             (b)   Sonnet 5 + thinking 0.114 vs 0.058   0 / 11  (without "other": 0 / 9)
  Turbo control: 70B -0.006 [-0.048, +0.031], 8B +0.066 [-0.009, +0.140] (Wilcoxon p 0.006)
  DeepSeek-V3 Aug vs Oct: 0.141 -> 0.086, top-1 differs on 24 of 69, paired difference 0.055 [0.011, 0.106]

Usage:  python table4.py
"""
import re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

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


def family(name):
    n = str(name).lower()
    for f, pat in FAM:
        if re.search(pat, n):
            return f
    return "other"


def nreg_flow(order, df):
    acc = dict(zip(df.candidate_id, df.accuracy)); best, worst = max(acc.values()), min(acc.values())
    if best - worst <= 0:
        return None
    for c in order:
        if c in acc:
            return (best - acc[c]) / (best - worst)
    return None


def nreg_fam(order, df):
    facc = df.groupby("family").accuracy.max(); best, worst = facc.max(), facc.min()
    if best - worst <= 0:
        return None
    for f in order:
        if f in facc.index:
            return (best - facc[f]) / (best - worst)
    return None


def fam_order(ranking, df):
    c2f = dict(zip(df.candidate_id, df.family)); out = []
    for c in ranking:
        f = c2f.get(c)
        if f is not None and f not in out:
            out.append(f)
    return out


def family_portfolios(truth):
    """Leave-one-task-out family orders (a) and (b) per task."""
    allr = pd.concat([df.assign(task=t) for t, df in truth.items()])
    best_in = {t: df.groupby("family").accuracy.max() for t, df in truth.items()}
    out = {}
    for t in truth:
        oth = allr[allr.task != t]
        a = list(oth.groupby("family").accuracy.mean().sort_values(ascending=False).index)
        b = list(pd.concat([best_in[u] for u in truth if u != t], axis=1).mean(axis=1).sort_values(ascending=False).index)
        out[t] = (a, b)
    return out


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def compare(model_reg, port_reg):
    """{model: {task: regret}}, {task: regret} -> rows (model, mean, port, better?, worse?) with BH."""
    rows = []
    for m, r in model_reg.items():
        ts = [t for t in r if port_reg.get(t) is not None]
        a = np.round([r[t] for t in ts], 10); b = np.round([port_reg[t] for t in ts], 10)
        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        rows.append([m, a.mean(), b.mean(), a.mean() - b.mean(), p])
    for r, q in zip(rows, bh([r[4] for r in rows])):
        r.append(q)
    return sorted(rows, key=lambda r: r[1])


def cell_text(rows):
    best = rows[0]
    better = sum(r[3] < 0 and r[5] < 0.05 for r in rows); worse = sum(r[3] > 0 and r[5] < 0.05 for r in rows)
    return f"{LABEL.get(best[0], best[0])} {best[1]:.3f} vs {best[2]:.3f}; better {better} / worse {worse} of {len(rows)}"


def load_truth(gt_csv, names=None):
    gt = pd.read_csv(gt_csv)
    if "flow_name" not in gt:
        gt = gt.merge(names, on="candidate_id", how="left")
    gt["family"] = gt.flow_name.map(family)
    return {int(t): df.reset_index(drop=True) for t, df in gt.groupby("task_id")}


def load_rankings(csv):
    r = pd.read_csv(csv); r = r[r.ranking.fillna("") != ""]
    out = {}
    for m, g in r.groupby("model"):
        out[m] = {int(t): rk.split(",") for t, rk in zip(g.task_id, g.ranking)}
    return out


def benchmark(name, truth, rankings, flow_port):
    flow = {m: {t: nreg_flow(rk, truth[t]) for t, rk in r.items() if t in truth} for m, r in rankings.items()}
    flow = {m: {t: v for t, v in r.items() if v is not None} for m, r in flow.items()}
    common = sorted(set.intersection(*[set(r) for r in flow.values()]))
    print(f"\n=== {name}: {len(rankings)} models, common task set {len(common)} tasks ===")
    rows = compare({m: {t: r[t] for t in common} for m, r in flow.items()}, flow_port)
    print(f"  flow            {cell_text(rows)}")
    for drop_other in (False, True):
        tr = truth if not drop_other else {t: df[df.family != "other"].reset_index(drop=True) for t, df in truth.items()}
        tr = {t: df for t, df in tr.items() if df.family.nunique() >= 2}
        ports = family_portfolios(tr)
        fam = {m: {t: nreg_fam(fam_order(rk, tr[t]), tr[t]) for t, rk in r.items() if t in tr and t in common}
               for m, r in rankings.items()}
        fam = {m: {t: v for t, v in r.items() if v is not None} for m, r in fam.items()}
        fcommon = sorted(set.intersection(*[set(r) for r in fam.values()]))
        tag = "without 'other'" if drop_other else "with our family map"
        for k, lab in ((0, "(a) mean-member"), (1, "(b) best-member")):
            port = {t: nreg_fam(ports[t][k], tr[t]) for t in fcommon}
            rows = compare({m: {t: r[t] for t in fcommon} for m, r in fam.items()}, port)
            print(f"  family {lab:16s} [{tag}, {len(fcommon)} tasks]  {cell_text(rows)}")


def boot_ci(d, rng, B=10000):
    bs = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired(name, a_rk, b_rk, truth, rng):
    ts = [t for t in truth if t in a_rk and t in b_rk and nreg_flow(a_rk[t], truth[t]) is not None]
    a = np.round([nreg_flow(a_rk[t], truth[t]) for t in ts], 10); b = np.round([nreg_flow(b_rk[t], truth[t]) for t in ts], 10)
    d = b - a; lo, hi = boot_ci(d, rng); p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    agree = np.mean([a_rk[t][0] == b_rk[t][0] for t in ts])
    print(f"  {name}: n={len(ts)}  {a.mean():.3f} -> {b.mean():.3f}  difference {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"Wilcoxon p {p:.2g}  top-1 agreement {agree:.2f} ({sum(a_rk[t][0] != b_rk[t][0] for t in ts)} differ)")


if __name__ == "__main__":
    names = pd.read_csv("results/catalog.csv")[["candidate_id", "flow_name"]]
    # CC18: the 69 OpenML-CC18 tasks with >= 15 scikit-learn candidates (catalog 1,475 flows), October 2026 re-run
    cc_truth = load_truth("results/cc18/ground_truth_cc18.csv")
    cc_port = dict(pd.read_csv("results/cc18/portfolio_cc18.csv").set_index("task_id")["nregret@1"])
    benchmark("CC18", cc_truth, load_rankings("results/cc18/rankings_cc18.csv"), cc_port)
    # long tail: the 200-task main study
    lt_truth = load_truth("results/ground_truth.csv", names)
    base = pd.read_csv("results/baselines_main200.csv")
    lt_port = dict(base[base.method == "global_default"].set_index("task_id")["nregret@1"])
    lt_rank = load_rankings("results/rankings_main200.csv")
    benchmark("long tail", lt_truth, lt_rank, lt_port)

    print("\n=== Llama-Turbo (FP8) control, 50 long-tail tasks: original -> Turbo ===")
    turbo = load_rankings("results/turbo50/rankings_turbo50.csv")
    rng = np.random.default_rng(0)                 # one generator for both models (70B first), as in the paper
    for orig, tb in (("deepinfra_meta-llama_Llama-3.3-70B-Instruct", "deepinfra_meta-llama_Llama-3.3-70B-Instruct-Turbo"),
                     ("deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct", "deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo")):
        paired(LABEL[orig], lt_rank[orig], turbo[tb], {t: lt_truth[t] for t in turbo[tb]}, rng)

    print("\n=== Serving determinism (Appendix A): DeepSeek-V3 on CC18, August -> October 2026 ===")
    aug = load_rankings("results/cc18/rankings_cc18_deepseek_aug2026.csv")["deepinfra_deepseek-ai_DeepSeek-V3"]
    octo = load_rankings("results/cc18/rankings_cc18.csv")["deepinfra_deepseek-ai_DeepSeek-V3"]
    paired("DeepSeek-V3 (positive = October better)", octo, aug, cc_truth, np.random.default_rng(0))

"""Reproduce the long-tail family-level cell of Table 4 (granularity x heterogeneity) from the released CSVs
(no API, no OpenML access needed).

Each OpenML flow is mapped to a coarse estimator family by regular expressions on its name (first match wins,
unmatched flows -> "other"). For each of the 200 long-tail tasks the family accuracy is the best accuracy of its
flows; family nregret@1 = (best family - chosen family) / (best - worst family), tasks with a constant range are
skipped.
  - LLM: the model's flow ranking is collapsed to the order of first appearance of each family.
  - Portfolio: families ordered by their mean flow accuracy over all 200 tasks (best-on-average), as at flow level.
Also prints the size of the "other" bucket (share of catalog flows and of candidate rows).

Expected output: portfolio family nregret@1 = 0.228; best LLM 0.143 (Sonnet 5 + thinking); 9 of 11 models
below the portfolio; "other" = 29.1 % of the 1,613 catalog flows, 48.0 % of candidate rows.

Usage:  python family_level.py
"""
import re
import numpy as np, pandas as pd

FAM = [("random_forest", "random.?forest|randomforest"), ("extra_trees", "extra.?tree"),
       ("gradient_boosting", "gradientboost|gradient_boost|hist.?gradient|xgb|xgboost|lightgbm|gbm|adaboost|boosting"),
       ("svc", "svc|svm|support.?vector"), ("logistic", "logistic|logreg"), ("knn", "kneighbors|knn|neighbors"),
       ("naive_bayes", "naive.?bayes|gaussiannb|bernoullinb|multinomialnb"),
       ("decision_tree", "decisiontree|decision_tree|j48|reptree"), ("mlp", "mlp|perceptron|neural"),
       ("linear", "sgd|ridge|lasso|elasticnet|linearmodel|passiveaggr"), ("bagging", "bagging"),
       ("discriminant", "discriminant|lda|qda")]


def family(name):
    n = str(name).lower()
    for f, pat in FAM:
        if re.search(pat, n):
            return f
    return "other"


cat = pd.read_csv("results/catalog.csv")                         # candidate_id, flow_id, flow_name
fam_of = {c: family(n) for c, n in zip(cat.candidate_id, cat.flow_name)}
GT = pd.read_csv("results/ground_truth.csv")                     # task_id, candidate_id, accuracy
GT["family"] = GT.candidate_id.map(fam_of)
RK = pd.read_csv("results/rankings_main200.csv")                 # model, task_id, ranking

fam_acc = {t: g.groupby("family").accuracy.max().to_dict() for t, g in GT.groupby("task_id")}
port_order = list(GT.groupby("family").accuracy.mean().sort_values(ascending=False).index)


def nregret(order, facc):
    best, worst = max(facc.values()), min(facc.values())
    if best - worst <= 0:
        return None
    for f in order:
        if f in facc:
            return (best - facc[f]) / (best - worst)
    return None


port = [r for r in (nregret(port_order, fa) for fa in fam_acc.values()) if r is not None]
print(f"portfolio (family level)  nregret@1 = {np.mean(port):.3f}  (n={len(port)})")
rows = []
for m, g in RK.groupby("model"):
    regs = []
    for t, rk in zip(g.task_id, g.ranking):
        if t not in fam_acc or not isinstance(rk, str) or not rk:
            continue
        order = []
        for c in rk.split(","):
            f = fam_of.get(c)
            if f is not None and f not in order:
                order.append(f)
        r = nregret(order, fam_acc[t])
        if r is not None:
            regs.append(r)
    rows.append((m, np.mean(regs), len(regs)))
for m, r, n in sorted(rows, key=lambda x: x[1]):
    print(f"  {m:45s} {r:.3f}  (n={n})  {'beats' if r < np.mean(port) else 'loses to'} the family portfolio")
other_flows = sum(f == "other" for f in fam_of.values())
print(f'"other" bucket: {other_flows}/{len(fam_of)} catalog flows ({100 * other_flows / len(fam_of):.1f} %), '
      f'{(GT.family == "other").sum()}/{len(GT)} candidate rows ({100 * (GT.family == "other").mean():.1f} %)')

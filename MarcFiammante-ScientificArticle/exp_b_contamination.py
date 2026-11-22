#!/usr/bin/env python3
"""Experiment B (revision): controlled outlier contamination on Dry Bean (k = 7).
Outliers are ADDED: a random clean row shifted by m x (feature std) with a random sign per feature,
for m in {5, 20, 100} and contamination p in {0, 0.5, 1, 2, 5}% of N. Every encoder, PCA and scaler
is fitted on the contaminated data (the encoder sees the outliers). ARI is scored on clean rows only.
3 repetitions per cell. Needs fast_feature_clustering.py, common.py, dry-beans.csv."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from common import *

K = 7
X0, y = load_beans(); N, C = X0.shape; sd = X0.std(0)


def methods(X):
    Z = StandardScaler().fit_transform(X); R = RobustScaler().fit_transform(X)
    out = {"k-means++, standardised": KMeans(K, n_init=10, random_state=0).fit_predict(Z),
           "k-means++, robust-scaled": KMeans(K, n_init=10, random_state=0).fit_predict(R),
           "trimmed k-means (1%), standardised": trimmed_kmeans(Z, K, 0.01, n_init=5)[0]}
    P = PCA(3, whiten=True, random_state=0).fit_transform(Z)
    for enc in ("minmax", "robust", "quantile"):
        f = patent_fit(P, K, enc, bits=2, reassign=True)
        out[f"patent, {enc}"] = f.labels_
        g = patent_fit(P, K, enc, bits=2, reassign=False)
        out[f"hybrid, {enc}"] = KMeans(K, init=seeds_from(g.labels_, Z, K), n_init=1).fit_predict(Z)
    return out


rows = []
for m in (5, 20, 100):
    for p in (0.0, 0.005, 0.01, 0.02, 0.05):
        for rep in range(3 if p > 0 else 1):
            rng = np.random.default_rng(100 * rep + int(1000 * p) + m)
            n_out = int(round(p * N))
            base = X0[rng.integers(0, N, n_out)]
            Xo = base + m * sd * rng.choice([-1, 1], size=(n_out, C))
            X = np.vstack([X0, Xo])
            for name, lab in methods(X).items():
                rows.append({"m": m, "p_%": 100 * p, "rep": rep, "method": name, "ARI_clean": ari(y, lab[:N])})
        print(f"m={m} done", flush=True) if p == 0.05 else None
df = pd.DataFrame(rows); df.to_csv("results/exp_b_contamination_runs.csv", index=False)
tab = df.groupby(["method", "m", "p_%"]).ARI_clean.mean().unstack(["m", "p_%"])
tab.to_csv("results/exp_b_contamination_summary.csv")
with pd.option_context("display.width", 300, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
    print(tab)

#!/usr/bin/env python3
"""Experiment A (revision): robust k-means baselines vs the patent method and the hybrid,
on California Housing (K = 5) and Dry Bean (K = 7).
Separates preprocessing from algorithm: every method is scored by silhouette in TWO COMMON spaces
(standardised Z and winsorised-standardised W), plus ARI vs the varieties (Dry Bean) or eta^2 of
house value (California Housing), cluster balance, bootstrap stability (B refits of the whole
pipeline, ARI vs the full-data fit on the rows each resample contains, 95% interval) and fit time (mean of 5 after a warm-up).
Usage: python exp_a_robust_baselines.py [--boot 50]"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from common import *


def eta2(v, lab):
    m = v.mean(); return sum((lab == c).sum() * (v[lab == c].mean() - m) ** 2 for c in np.unique(lab)) / ((v - m) ** 2).sum()


def pipelines(K, n_init, dataset):
    """Each entry: X_raw -> labels. Everything (scalers, PCA, encoders) is fitted inside."""
    def km(S): return KMeans(K, n_init=n_init, random_state=42).fit_predict(S)
    def qt(X): return QuantileTransformer(n_quantiles=1000, output_distribution="uniform").fit_transform(X)
    P = {
        "k-means++, StandardScaler": lambda X: km(StandardScaler().fit_transform(X)),
        "k-means++, RobustScaler": lambda X: km(RobustScaler().fit_transform(X)),
        "k-means++, winsorised 1-99% + standardised": lambda X: km(winsor_std(X)),
        "k-means++, quantile-transformed": lambda X: km(qt(X)),
        "trimmed k-means (1%), standardised": lambda X: trimmed_kmeans(StandardScaler().fit_transform(X), K, 0.01, n_init=min(n_init, 10))[0],
    }
    if dataset == "housing":
        P["patent alone (min-max, 1 bit)"] = lambda X: patent_fit(X, K, "minmax").labels_
        P["patent alone (robust, 1 bit, reassign)"] = lambda X: patent_fit(X, K, "robust", reassign=True).labels_
        def hyb(X):
            W = winsor_std(X); f = patent_fit(X, K, "robust")
            return KMeans(K, init=seeds_from(f.labels_, W, K), n_init=1).fit_predict(W)
        P["hybrid: patent seeds (robust, 1 bit) + 1 Lloyd run in W"] = hyb
    else:
        def pc(X): return PCA(3, whiten=True, random_state=0).fit_transform(StandardScaler().fit_transform(X))
        P["patent alone (3 PCs, robust, 2 bits, reassign)"] = lambda X: patent_fit(pc(X), K, "robust", 2, True).labels_
        def hyb(X):
            Z = StandardScaler().fit_transform(X); f = patent_fit(pc(X), K, "robust", 2)
            return KMeans(K, init=seeds_from(f.labels_, Z, K), n_init=1).fit_predict(Z)
        P["hybrid: patent seeds + 1 Lloyd run in Z"] = hyb
        def hybw(X):
            W = winsor_std(X); f = patent_fit(pc(X), K, "robust", 2)
            return KMeans(K, init=seeds_from(f.labels_, W, K), n_init=1).fit_predict(W)
        P["hybrid: patent seeds + 1 Lloyd run in W"] = hybw
    return P


def evaluate(name, X, K, n_init, truth=None, price=None, B=50):
    Z = StandardScaler().fit_transform(X); W = winsor_std(X)
    rows = []; rng = np.random.default_rng(7); boots = [rng.integers(0, len(X), len(X)) for _ in range(B)]
    for m, fn in pipelines(K, n_init, name).items():
        lab, t = timed(lambda: fn(X), 5)
        sizes = np.bincount(lab, minlength=K); p = sizes[sizes > 0] / len(X)
        # bootstrap: refit on a resample and compare labels on the rows the resample contains
        # (first occurrence of each row), with no label transfer: method-neutral (Hennig 2007)
        stab = []
        for idx in boots:
            lb = fn(X[idx]); rows_u, first = np.unique(idx, return_index=True)
            stab.append(ari(lab[rows_u], lb[first]))
        r = {"dataset": name, "method": m, "fit_ms": 1e3 * t, "smallest_%": 100 * sizes.min() / len(X), "effective_k": np.exp(-(p * np.log(p)).sum()),
             "sil_Z": silhouette_score(Z, lab, sample_size=5000, random_state=42), "sil_W": silhouette_score(W, lab, sample_size=5000, random_state=42),
             "stability_ARI_mean": np.mean(stab), "stability_ARI_ci_low": np.percentile(stab, 2.5), "stability_ARI_ci_high": np.percentile(stab, 97.5)}
        if truth is not None: r["ARI_truth"] = ari(truth, lab)
        if price is not None: r["eta2_price"] = eta2(price, lab)
        rows.append(r); print(f"{name:8s} {m:58s} {r['fit_ms']:8.1f} ms  stab {r['stability_ARI_mean']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=50); a = ap.parse_args()
    Xh, price, _ = load_housing(); Xb, yb = load_beans()
    rows = evaluate("housing", Xh, 5, 10, price=price, B=a.boot) + evaluate("beans", Xb, 7, 30, truth=yb, B=a.boot)
    df = pd.DataFrame(rows); df.to_csv("results/exp_a_robust_baselines.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))

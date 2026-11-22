#!/usr/bin/env python3
"""Experiment D (revision): statistical validation at higher resolution on California Housing.
B bootstrap refits (default 200) and P column-permutation null datasets (default 999) per method.
Reports bootstrap ARI with 95% percentile intervals, permutation p-values (b+1)/(P+1)
[Phipson & Smyth 2010] and Holm-adjusted p-values across the methods tested.
Configurations (K, encoding, depth) are FIXED in advance: the same six methods as Section 6.2.
Usage: python stats_validation.py [--csv housing.csv] [--boot 200] [--perm 999] [--jobs 2]"""
import argparse, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from housing_patent_vs_kmeans import load_data
from validate_clusters import METHODS

SEED = 42


def sil(S, lab):
    return np.nan if len(np.unique(lab)) < 2 else silhouette_score(S, lab, sample_size=5000, random_state=SEED)


def run_method(args):
    name, X, B, P = args
    fit = METHODS[name]
    rng = np.random.default_rng(SEED)            # same resamples for every method
    N = len(X)
    t0 = time.perf_counter()
    predict, own = fit(X); ref = predict(X); real = sil(own(X), ref)
    boot = []
    labs = []
    for b in range(B):
        idx = rng.integers(0, N, N)
        p, _ = fit(X[idx]); l = p(X); labs.append(l); boot.append(adjusted_rand_score(ref, l))
    pr = np.random.default_rng(SEED + 1)
    pairs = [tuple(pr.choice(B, 2, replace=False)) for _ in range(min(200, B * (B - 1) // 2))]
    pair_ari = [adjusted_rand_score(labs[i], labs[j]) for i, j in pairs]
    null = []
    prm = np.random.default_rng(SEED + 2)
    for _ in range(P):
        Xp = np.column_stack([prm.permutation(X[:, c]) for c in range(X.shape[1])])
        p, o = fit(Xp); null.append(sil(o(Xp), p(Xp)))
    null = np.array(null); boot = np.array(boot)
    return {"method": name, "boot_B": B, "ARI_boot_mean": boot.mean(), "ARI_boot_ci_low": np.percentile(boot, 2.5),
            "ARI_boot_ci_high": np.percentile(boot, 97.5), "ARI_boot_min": boot.min(), "ARI_pairwise_mean": np.mean(pair_ari),
            "perm_P": P, "sil_real": real, "sil_null_mean": np.nanmean(null), "sil_null_q99": np.nanpercentile(null, 99),
            "sil_null_max": np.nanmax(null), "margin": real - np.nanmean(null),
            "n_null_ge_real": int(np.sum(null >= real)), "p_perm": (1 + np.sum(null >= real)) / (1 + P),
            "seconds": time.perf_counter() - t0}


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--perm", type=int, default=999); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df, _, src = load_data(a.csv); X = df.values.astype(np.float64)
    print(f"Data: {src}; {a.boot} bootstraps, {a.perm} permutations, {len(METHODS)} methods", flush=True)
    with Pool(a.jobs) as pool:
        rows = []
        for r in pool.imap_unordered(run_method, [(m, X, a.boot, a.perm) for m in METHODS]):
            print(f"done: {r['method']} in {r['seconds']:.0f}s", flush=True); rows.append(r)
    res = pd.DataFrame(rows).set_index("method").loc[list(METHODS)].reset_index()
    res["p_holm"] = holm(res.p_perm)
    res.to_csv(os.path.join(a.out, "stats_validation.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(res.to_string(index=False))

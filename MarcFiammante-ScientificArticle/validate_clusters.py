#!/usr/bin/env python3
"""
Cluster validity tests for California Housing, k=5
==================================================
1. Stability (bootstrap): refit the whole pipeline (encoder + clustering) on B bootstrap
   samples, predict labels for all 20,640 rows, and compare with the full-data fit (ARI).
   Also reports mean pairwise ARI between bootstrap runs. ARI > 0.8 is usually read as
   stable, 0.6-0.8 moderately stable, < 0.6 unstable.
2. Null test (column permutation): shuffle every column independently. This keeps each
   feature's distribution but destroys all dependence between features, i.e. any real
   cluster structure. Re-run the identical pipeline P times. If the real silhouette is
   not clearly above the null distribution, the "clusters" are just cuts through a
   continuum.

Usage
    python validate_clusters.py [--csv housing.csv] [--boot 20] [--perm 20] [--out results]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from fast_feature_clustering import FastFeatureClustering, FeatureChannelEncoder
from housing_patent_vs_kmeans import load_data

K, SEED = 5, 42


# Each method: fit(X_fit) -> (predict_fn on any X, own-space transform for silhouette)
def kmeans_std(Xf):
    sc = StandardScaler().fit(Xf)
    km = KMeans(K, n_init=10, random_state=SEED).fit(sc.transform(Xf))
    return lambda X: km.predict(sc.transform(X)), sc.transform


def kmeans_uint8(Xf):
    enc = FeatureChannelEncoder("minmax").fit(Xf)
    km = KMeans(K, n_init=10, random_state=SEED).fit(enc.transform(Xf).astype(float))
    return lambda X: km.predict(enc.transform(X).astype(float)), lambda X: enc.transform(X).astype(float)


def patent(method, bits=None, reassign=False):
    def fit(Xf):
        enc = FeatureChannelEncoder(method).fit(Xf)
        f = FastFeatureClustering(K, bits=bits, reassign=reassign).fit(enc.transform(Xf))
        return lambda X: f.predict(enc.transform(X)), lambda X: enc.transform(X).astype(float)
    return fit


METHODS = {
    "KMeans k=5 (standardised)": kmeans_std,
    "KMeans k=5 (uint8 min-max)": kmeans_uint8,
    "Patent minmax, 1 bit": patent("minmax"),
    "Patent minmax, 2 bits + reassign": patent("minmax", 2, True),
    "Patent robust, 1 bit + reassign": patent("robust", None, True),
    "Patent quantile, 1 bit + reassign": patent("quantile", None, True),
}


def sil(Xs, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(Xs, labels, sample_size=5000, random_state=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, _, source = load_data(args.csv)
    X = df.values.astype(np.float64)
    N = len(X)
    Z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, N, N) for _ in range(args.boot)]
    perms = [np.column_stack([rng.permutation(X[:, c]) for c in range(X.shape[1])])
             for _ in range(args.perm)]
    print(f"Data: {source} {X.shape}; {args.boot} bootstraps, {args.perm} permutations\n")

    rows = []
    for name, fit in METHODS.items():
        t0 = time.perf_counter()
        predict, own = fit(X)
        ref = predict(X)
        real_own, real_std = sil(own(X), ref), sil(Z, ref)

        # 1. bootstrap stability
        boot_labels = []
        for idx in boots:
            p, _ = fit(X[idx])
            boot_labels.append(p(X))
        ari_ref = np.array([adjusted_rand_score(ref, b) for b in boot_labels])
        ari_pair = np.array([adjusted_rand_score(boot_labels[i], boot_labels[j])
                             for i in range(len(boot_labels)) for j in range(i + 1, len(boot_labels))])

        # 2. permutation null
        null_own, null_std = [], []
        for Xp in perms:
            p, o = fit(Xp)
            lp = p(Xp)
            null_own.append(sil(o(Xp), lp))
            null_std.append(sil(StandardScaler().fit_transform(Xp), lp))
        null_own, null_std = np.array(null_own), np.array(null_std)

        def z(real, null):
            return (real - np.nanmean(null)) / (np.nanstd(null) + 1e-12)

        rows.append({
            "method": name,
            "ARI_boot_vs_full_mean": ari_ref.mean(), "ARI_boot_vs_full_min": ari_ref.min(),
            "ARI_boot_pairwise_mean": ari_pair.mean(),
            "sil_own_real": real_own, "sil_own_null_mean": np.nanmean(null_own),
            "sil_own_null_max": np.nanmax(null_own), "sil_own_z": z(real_own, null_own),
            "sil_std_real": real_std, "sil_std_null_mean": np.nanmean(null_std),
            "sil_std_null_max": np.nanmax(null_std), "sil_std_z": z(real_std, null_std),
            "p_value_own": (1 + np.sum(null_own >= real_own)) / (1 + len(null_own)),
            "seconds": time.perf_counter() - t0,
        })
        r = rows[-1]
        print(f"{name:36s} stability ARI {r['ARI_boot_vs_full_mean']:.2f} (min {r['ARI_boot_vs_full_min']:.2f}) | "
              f"sil_own real {real_own:.3f} vs null {r['sil_own_null_mean']:.3f} (max {r['sil_own_null_max']:.3f}) "
              f"| {r['seconds']:.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "validation.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print("\n" + res.to_string(index=False))
    print(f"\nWritten {args.out}/validation.csv")


if __name__ == "__main__":
    main()

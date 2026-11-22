"""Shared helpers for the revision experiments."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from fast_feature_clustering import FastFeatureClustering, FeatureChannelEncoder


def load_housing(csv="housing_statlib.csv"):
    from housing_patent_vs_kmeans import load_data
    df, y, src = load_data(csv if os.path.exists(csv) else None)
    return df.values.astype(np.float64), np.asarray(y, float), src


def load_beans(csv="dry-beans.csv"):
    d = pd.read_csv(csv, encoding="utf-8-sig")
    return d.drop(columns="Class").values.astype(np.float64), pd.factorize(d["Class"])[0]


def winsor_std(X, lo=1, hi=99):
    a, b = np.percentile(X, [lo, hi], axis=0)
    return StandardScaler().fit_transform(np.clip(X, a, b))


def trimmed_kmeans(S, K, alpha=0.01, n_init=10, max_iter=100, seed=0):
    """Trimmed k-means (Cuesta-Albertos et al. 1997): the alpha share of farthest points is
    ignored when updating centroids. Returns labels for all points (nearest centroid)."""
    rng = np.random.default_rng(seed); best = None; N = len(S); keep = int(np.ceil((1 - alpha) * N))
    for r in range(n_init):
        C, _ = kmeans_plusplus(S, K, random_state=int(rng.integers(1 << 30)))
        for it in range(max_iter):
            d = ((S[:, None, :] - C[None]) ** 2).sum(-1); lab = d.argmin(1); dm = d[np.arange(N), lab]
            core = np.argsort(dm)[:keep]
            newC = np.stack([S[core][lab[core] == c].mean(0) if np.any(lab[core] == c) else C[c] for c in range(K)])
            if np.allclose(newC, C): break
            C = newC
        cost = np.sort(dm)[:keep].sum()
        if best is None or cost < best[0]: best = (cost, lab.copy(), C.copy())
    return best[1], best[2]


def patent_fit(P, K, encoding, bits=None, reassign=False):
    U = FeatureChannelEncoder(encoding).fit_transform(P)
    return FastFeatureClustering(K, bits=bits, reassign=reassign).fit(U)


def seeds_from(labels, S, K):
    return np.stack([S[labels == c].mean(0) for c in range(K)])


def timed(fn, repeats=5):
    fn(); t0 = time.perf_counter()
    for _ in range(repeats): out = fn()
    return out, (time.perf_counter() - t0) / repeats

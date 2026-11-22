#!/usr/bin/env python3
"""Experiment C (revision): does the hybrid need ~K-1 channels, or ~ the intrinsic dimension r of the
cluster-centre configuration? N = 50,000 points; K Gaussian centres placed in a random r-dimensional
subspace of an ambient D-dimensional space; isotropic unit noise in all D dimensions.
Grid: D in {20, 100}, K in {10, 20}, r in {2, 4, 8} (r <= K-1), centre scale chosen for moderate
separation. Hybrid: whitened PCA to d channels (d in {2, 3, r, K-1}), robust encoding, bits from the
patent rule capped by the occupancy constraint, then ONE Lloyd run in the ambient space.
Baselines: k-means++ single run (5 seeds: mean and worst) and n_init = 10."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from common import *

N = 50_000
rows = []
for D in (20, 100):
    for K in (10, 20):
        for r in (2, 4, 8):
            rng = np.random.default_rng(1000 * D + 10 * K + r)
            Q, _ = np.linalg.qr(rng.normal(size=(D, r)))                       # orthonormal D x r
            centres = rng.normal(size=(K, r)) * (3.0 * np.sqrt(K) ** (1 / r) * 2)  # spread grows mildly with K
            y = rng.integers(0, K, N)
            X = (centres[y] @ Q.T + rng.normal(size=(N, D))).astype(np.float32)
            single = [ari(y, KMeans(K, n_init=1, random_state=s).fit_predict(X)) for s in range(5)]
            best10 = ari(y, KMeans(K, n_init=10, random_state=0).fit_predict(X))
            base = {"D": D, "K": K, "r": r, "kmpp_single_mean": np.mean(single), "kmpp_single_worst": min(single), "kmpp_n10": best10}
            for d in sorted({2, 3, r, K - 1}):
                P = PCA(d, whiten=True, random_state=0, svd_solver="randomized").fit_transform(X)
                bmax = max(1, int(np.floor(np.log2(N) / d)))                    # occupancy constraint
                f = patent_fit(P, K, "robust", bits=None)
                if f.active_bits_ > bmax:
                    f = patent_fit(P, K, "robust", bits=bmax)
                km = KMeans(K, init=seeds_from(f.labels_, X, K), n_init=1).fit(X)
                rows.append({**base, "d": d, "bits": f.active_bits_, "cells": f.n_initial_cells_,
                             "hybrid_ARI": ari(y, km.labels_), "hybrid_iters": km.n_iter_, "patent_ARI": ari(y, f.labels_)})
            print(f"D={D} K={K} r={r}: k-means++ single {np.mean(single):.3f} (worst {min(single):.3f}), n_init=10 {best10:.3f} | " +
                  " ".join(f"d={x['d']}:{x['hybrid_ARI']:.3f}" for x in rows if (x['D'], x['K'], x['r']) == (D, K, r)), flush=True)
pd.DataFrame(rows).to_csv("results/exp_c_dimension.csv", index=False)

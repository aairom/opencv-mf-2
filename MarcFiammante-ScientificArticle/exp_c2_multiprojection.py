#!/usr/bin/env python3
"""Experiment C2 (revision): multi-projection seeding. For each configuration of Experiment C,
run the hybrid for every d in {2, 3, 4, 6, 8} that satisfies the occupancy constraint at 1 bit
(d <= log2 N), and keep the run with the LOWEST final inertia (no labels used).
Compared with k-means++ n_init = 5 and n_init = 10 (same number of Lloyd runs, or twice)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from common import *

N = 50_000; DS = (2, 3, 4, 6, 8)
rows = []
for D in (20, 100):
    for K in (10, 20):
        for r in (2, 4, 8):
            rng = np.random.default_rng(1000 * D + 10 * K + r)
            Q, _ = np.linalg.qr(rng.normal(size=(D, r)))
            centres = rng.normal(size=(K, r)) * (3.0 * np.sqrt(K) ** (1 / r) * 2)
            y = rng.integers(0, K, N)
            X = (centres[y] @ Q.T + rng.normal(size=(N, D))).astype(np.float32)
            Pall = PCA(max(DS), whiten=True, random_state=0, svd_solver="randomized").fit_transform(X)
            best = None
            for d in DS:
                bmax = max(1, int(np.floor(np.log2(N) / d)))
                f = patent_fit(Pall[:, :d], K, "robust", bits=None)
                if f.active_bits_ > bmax: f = patent_fit(Pall[:, :d], K, "robust", bits=bmax)
                km = KMeans(K, init=seeds_from(f.labels_, X, K), n_init=1).fit(X)
                if best is None or km.inertia_ < best[0]: best = (km.inertia_, d, km.labels_)
            k5 = KMeans(K, n_init=5, random_state=0).fit(X); k10 = KMeans(K, n_init=10, random_state=0).fit(X)
            rows.append({"D": D, "K": K, "r": r, "multi_proj_ARI": ari(y, best[2]), "chosen_d": best[1],
                         "multi_proj_inertia": best[0], "kmpp_n5_ARI": ari(y, k5.labels_), "kmpp_n10_ARI": ari(y, k10.labels_),
                         "kmpp_n10_inertia": k10.inertia_})
            x = rows[-1]
            print(f"D={D} K={K} r={r}: multi-projection ARI {x['multi_proj_ARI']:.3f} (d={x['chosen_d']}, inertia {best[0]/k10.inertia_-1:+.4%} vs n_init=10) | n_init=5 {x['kmpp_n5_ARI']:.3f} | n_init=10 {x['kmpp_n10_ARI']:.3f}", flush=True)
pd.DataFrame(rows).to_csv("results/exp_c2_multiprojection.csv", index=False)

#!/usr/bin/env python3
"""Revision analyses on California Housing (and Dry Bean for quantisation error):
 (1) Ward: patent merge vs brute-force weighted Ward vs nearest-neighbour-chain weighted Ward
     (Lance-Williams updates, O(L^2) time and memory) on the same occupied cells.
 (2) Occupancy distribution by encoding and depth: occupied share, singleton cells, median / p90 cell size.
 (3) Quantisation error vs depth: within-cell variance, centroid displacement, inertia, ARI vs a
     reference partition (k-means on the same continuous channels), and for Dry Bean ARI vs varieties."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from common import *
from fast_feature_clustering import cross_product_reduce, feature_index, index_to_values


def ward_brute(c0, n0, k):
    c = [x.astype(float) for x in c0]; n = list(map(float, n0)); grp = [[i] for i in range(len(c))]
    while len(c) > k:
        A = np.array(c); nn = np.array(n); best = (np.inf, 0, 0)
        for a in range(len(c) - 1):
            d = ((A[a + 1:] - A[a]) ** 2).sum(1) * nn[a] * nn[a + 1:] / (nn[a] + nn[a + 1:])
            j = int(d.argmin())
            if d[j] < best[0]: best = (d[j], a, a + 1 + j)
        _, a, b = best
        c[a] = (c[a] * n[a] + c[b] * n[b]) / (n[a] + n[b]); n[a] += n[b]; grp[a] += grp[b]
        del c[b], n[b], grp[b]
    lab = np.empty(len(c0), int)
    for g, m in enumerate(grp): lab[m] = g
    return lab


def ward_nn_chain(c0, n0, k):
    """Weighted Ward by nearest-neighbour chain with Lance-Williams updates (Murtagh 1983; Muellner 2011).
    Works on squared Ward merge costs; O(L^2) time and memory."""
    L = len(c0); n = n0.astype(float).copy()
    D2 = ((c0[:, None, :] - c0[None]) ** 2).sum(-1)
    W = (n[:, None] * n[None] / (n[:, None] + n[None])) * D2          # Ward merge cost
    np.fill_diagonal(W, np.inf)
    alive = np.ones(L, bool); members = {i: [i] for i in range(L)}; merges = []
    chain = []
    while alive.sum() > 1:
        if not chain: chain = [int(np.flatnonzero(alive)[0])]
        a = chain[-1]; row = np.where(alive, W[a], np.inf); row[a] = np.inf
        b = int(row.argmin())
        if len(chain) > 1 and W[a, chain[-2]] <= row[b]: b = chain[-2]
        if len(chain) > 1 and b == chain[-2]:
            chain.pop(); chain.pop()
            merges.append((W[a, b], a, b))
            na, nb = n[a], n[b]; idx = np.flatnonzero(alive); idx = idx[(idx != a) & (idx != b)]
            nk = n[idx]
            W[a, idx] = ((na + nk) * W[a, idx] + (nb + nk) * W[b, idx] - nk * W[a, b]) / (na + nb + nk)  # Lance-Williams (Ward)
            W[idx, a] = W[a, idx]; alive[b] = False; W[b, :] = np.inf; W[:, b] = np.inf
            n[a] = na + nb; members[a] += members.pop(b)
        else:
            chain.append(b)
    # cut: replay the merges in increasing cost order to stop at k clusters (reducible linkage)
    parent = list(range(L))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    groups = L
    for cost, a, b in sorted(merges):
        if groups == k: break
        parent[find(b)] = find(a); groups -= 1
    roots = {}; lab = np.array([roots.setdefault(find(i), len(roots)) for i in range(L)])
    return lab


def sse_cells(c0, n0, lab):
    s = 0
    for g in np.unique(lab):
        m = lab == g; mu = (c0[m] * n0[m, None]).sum(0) / n0[m].sum(); s += (n0[m] * ((c0[m] - mu) ** 2).sum(1)).sum()
    return s


X, price, src = load_housing()
print("Data:", src)
out = []
for enc in ("minmax", "quantile"):
    U = FeatureChannelEncoder(enc).fit_transform(X)
    g = FastFeatureClustering(5, post_process_refine=False).fit(U); t = g.tree_
    c0 = t.colors[:t.n_leaves]; n0 = t.counts[:t.n_leaves].astype(float); tot = sse_cells(c0, n0, np.zeros(len(c0), int))
    (_, tp) = timed(lambda: FastFeatureClustering(5, post_process_refine=False).fit(U).tree_, 5)
    lab_b, tb = timed(lambda: ward_brute(c0, n0, 5), 3)
    lab_n, tn = timed(lambda: ward_nn_chain(c0, n0, 5), 5)
    out.append({"encoding": enc, "cells": len(c0), "R2_patent": 1 - sse_cells(c0, n0, g.key_labels_) / tot,
                "R2_ward_brute": 1 - sse_cells(c0, n0, lab_b) / tot, "R2_ward_nnchain": 1 - sse_cells(c0, n0, lab_n) / tot,
                "ARI_brute_vs_nnchain": ari(lab_b, lab_n), "patent_fit_ms": 1e3 * tp, "merge_tree_ms": 1e3 * g.timings_["merge_tree_s"],
                "ward_brute_ms": 1e3 * tb, "ward_nnchain_ms": 1e3 * tn})
w = pd.DataFrame(out); w.to_csv("results/ward_timing.csv", index=False); print(w.to_string(index=False))

# (2) occupancy distribution
occ = []
for enc in ("minmax", "robust", "quantile"):
    U = FeatureChannelEncoder(enc).fit_transform(X)
    for n in (1, 2, 3):
        lv, _ = cross_product_reduce(U, n, "round"); k = feature_index(lv, n); _, cnt = np.unique(k, return_counts=True)
        occ.append({"encoding": enc, "bits": n, "possible": 2 ** (8 * n) if n * 8 < 63 else float(2 ** (8 * n)), "occupied": len(cnt),
                    "occupied_share_%": 100 * len(cnt) / 2 ** (8 * n), "singleton_cells_%": 100 * np.mean(cnt == 1),
                    "rows_in_singletons_%": 100 * (cnt == 1).sum() / len(X), "median_cell": np.median(cnt), "p90_cell": np.percentile(cnt, 90),
                    "max_cell": cnt.max()})
o = pd.DataFrame(occ); o.to_csv("results/occupancy_distribution.csv", index=False)
with pd.option_context("display.width", 250, "display.float_format", "{:.3g}".format): print(o.to_string(index=False))

# (3) quantisation error vs depth
def quant_study(name, S, K, y=None, enc="robust"):
    U = FeatureChannelEncoder(enc).fit_transform(S); Uf = U.astype(float)
    ref = KMeans(K, n_init=10, random_state=0).fit(Uf)
    rows = []
    for n in (1, 2, 3, 4, 5):
        lv, al = cross_product_reduce(U, n, "round"); key = feature_index(lv, n)
        _, inv = np.unique(key, return_inverse=True); inv = inv.ravel()
        cnt = np.bincount(inv); sums = np.zeros((len(cnt), U.shape[1])); np.add.at(sums, inv, Uf)
        wcv = ((Uf - sums[inv] / cnt[inv, None]) ** 2).sum() / ((Uf - Uf.mean(0)) ** 2).sum()
        if len(cnt) <= K: continue
        f = FastFeatureClustering(K, bits=n).fit(U)
        P = f.palette_  # refined centroids, 0..255 units
        disp = np.sqrt(((P[:, None] - ref.cluster_centers_[None]) ** 2).sum(-1)).min(1).mean()
        inertia = sum(((Uf[f.labels_ == c] - P[c]) ** 2).sum() for c in range(K))
        r = {"dataset": name, "bits": n, "cells": len(cnt), "within_cell_var_share_%": 100 * wcv,
             "centroid_displacement": disp, "inertia_vs_kmeans_%": 100 * (inertia / ref.inertia_ - 1),
             "ARI_vs_kmeans_partition": ari(ref.labels_, f.labels_)}
        if y is not None: r["ARI_vs_truth"] = ari(y, f.labels_)
        rows.append(r)
    return rows
Xb, yb = load_beans(); Zb = StandardScaler().fit_transform(Xb); Pb = PCA(3, whiten=True, random_state=0).fit_transform(Zb)
q = pd.DataFrame(quant_study("California Housing (8 ch, robust)", X, 5) + quant_study("Dry Bean (3 PCs, robust)", Pb, 7, yb))
q.to_csv("results/quantisation_error.csv", index=False)
with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format): print(q.to_string(index=False))

#!/usr/bin/env python3
"""
Fast Feature Clustering — N-channel generalisation of IBM Patent US 11,763,422 B2
================================================================================
"Fast color clustering for preprocessing an image" (M. Fiammante, J.-A. Broyelle)

Generalises the reference implementation in github.com/aairom/opencv-mf
(fast_color_clustering.py) from 3-channel colour images to *any* tabular data:

    tabular rows (N x C, arbitrary units)
        |  FeatureChannelEncoder  : each feature -> uint8 0..255 "fictitious channel"
        v
    multi-channel image (H x W x C, uint8)      e.g. California Housing: 129 x 160 x 8
        |  (1) cross-product reduction   q = round((2^n - 1) * v / 255)         [Claims 1, 7, 14]
        |  (2) left alignment            v' = q << (8 - n)                      [Claims 1, 2]
        |  (3) map generation            key = q0 q1 ... q(C-1)  (n*C bits)     [Claims 5, 12, 19]
        |        flat histogram if n*C is small, sorted hash map otherwise
        |  (4) binary merge tree         merge least-frequent "colour" into its
        |                                nearest neighbour, count-weighted      [Claims 3, 4, 10, 11]
        |  (5) post-processing           centroids = mean of source rows        [Claims 6, 13]
        v
    K clusters + palette (K x C) + key->label lookup table (O(1) per row at inference)

What changes when C > 3
-----------------------
* Bit budget: 2^(n*C) > K + margin. With C = 8 and K = 5 this gives n = 1 bit per feature,
  i.e. each feature is split at mid-range into a 256-cell hypercube histogram.
* Key width is n*C bits. Up to 64 bits the key is one uint64; above 64 bits (e.g. 100 channels at
  1 bit) the rows of quantised levels themselves are the keys ("multiword" map: np.unique over rows
  for fitting, a dictionary of row bytes for prediction).
* Prediction: a previously seen key is one lookup (O(1) dictionary / O(log L) sorted array); an unseen
  key falls back to the nearest palette entry, O(K C) per row.
* A dense LUT over 2^(n*C) entries is only practical for small n*C; above
  `flat_map_max_bits` the map is a sorted key array + searchsorted (the patent's
  "hash map" variant). Unseen keys at predict-time go to the nearest palette entry.
* CIELAB is meaningless for fictitious channels; distance is (optionally weighted)
  Euclidean in the left-aligned 0..255 space.
* The merge loop works on a compacted list of live nodes (O(live * C) per merge),
  with the same tie-breaking as the reference (lowest node id), so for C = 3 it
  reproduces the reference palette and labels exactly.

Optional, NOT part of the patent: `reassign=True` re-labels every row to its nearest
refined centroid (a single Lloyd step) after Claims 6/13 post-processing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FeatureChannelEncoder",
    "compute_significant_bits",
    "cross_product_reduce",
    "feature_index",
    "index_to_values",
    "MergeTree",
    "build_merge_tree",
    "FastFeatureClustering",
    "rows_to_image",
]


# =============================================================================
# 0. Tabular features -> fictitious uint8 channels
# =============================================================================

class FeatureChannelEncoder:
    """Map each column of a numeric table to an integer channel in 0..255.

    method
        "minmax"   : linear between column min and max (literal 0..255 scaling).
        "robust"   : linear between the `clip` percentiles, values outside clipped.
                     Prevents a few outliers (e.g. AveOccup up to 1243) from
                     squeezing the whole column into a handful of grey levels.
        "quantile" : empirical CDF (rank) -> uniform 0..255. Every channel uses the
                     full dynamic range, as an equalised image histogram would.
    """

    def __init__(self, method: str = "quantile", clip: Tuple[float, float] = (1.0, 99.0),
                 n_quantiles: int = 1024):
        if method not in ("minmax", "robust", "quantile"):
            raise ValueError("method must be 'minmax', 'robust' or 'quantile'")
        self.method = method
        self.clip = clip
        self.n_quantiles = n_quantiles

    def fit(self, X) -> "FeatureChannelEncoder":
        X = np.asarray(X, dtype=np.float64)
        if self.method == "minmax":
            self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        elif self.method == "robust":
            self.lo_ = np.percentile(X, self.clip[0], axis=0)
            self.hi_ = np.percentile(X, self.clip[1], axis=0)
        else:
            p = np.linspace(0.0, 1.0, self.n_quantiles)
            self.quantiles_ = np.quantile(X, p, axis=0)          # (Q, C)
            self.refs_ = p * 255.0
        self.n_channels_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.method == "quantile":
            U = np.empty_like(X)
            for c in range(X.shape[1]):
                q = self.quantiles_[:, c]
                # average forward/backward interpolation handles repeated quantiles (ties)
                fwd = np.interp(X[:, c], q, self.refs_)
                bwd = -np.interp(-X[:, c], -q[::-1], -self.refs_[::-1])
                U[:, c] = 0.5 * (fwd + bwd)
        else:
            span = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
            U = (np.clip(X, self.lo_, self.hi_) - self.lo_) / span * 255.0
        return np.clip(np.rint(U), 0, 255).astype(np.uint8)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, U) -> np.ndarray:
        """Approximate original-unit values for channel levels (e.g. a palette)."""
        U = np.asarray(U, dtype=np.float64)
        if self.method == "quantile":
            return np.stack([np.interp(U[:, c], self.refs_, self.quantiles_[:, c])
                             for c in range(U.shape[1])], axis=1)
        return self.lo_ + U / 255.0 * (self.hi_ - self.lo_)


def rows_to_image(U: np.ndarray) -> np.ndarray:
    """Lay N rows out as an H x W x C image, H*W == N, H the largest divisor <= sqrt(N)."""
    n = U.shape[0]
    h = max(d for d in range(1, int(np.sqrt(n)) + 1) if n % d == 0)
    return U.reshape(h, n // h, U.shape[1])


# =============================================================================
# 1-3. Cross-product reduction, left alignment, key concatenation (C channels)
# =============================================================================

def compute_significant_bits(k: int, margin: Optional[int] = None, channels: int = 3) -> int:
    """Smallest n with 2^(n*channels) > K + margin (Step 205, Claim 1). margin defaults to K//2."""
    if k < 1:
        raise ValueError("k must be at least 1")
    margin = max(1, k // 2) if margin is None else margin
    for bits in range(1, 9):
        if (1 << (bits * channels)) > k + margin:
            return bits
    return 8


def cross_product_reduce(img: np.ndarray, bits: int, rounding: str = "round"):
    """q = round((2^n - 1) * v / 255) via a 256-entry LUT, and v' = q << (8 - n).

    Works for any array shape (..., C) of uint8 — the LUT is applied element-wise.
    """
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in 1..8")
    if img.dtype != np.uint8:
        raise TypeError("expected uint8 input")
    shift = 8 - bits
    if rounding == "round":
        lut = np.round(np.arange(256, dtype=np.float32) * ((1 << bits) - 1) / 255.0).astype(np.uint8)
        levels = lut[img]
    elif rounding == "floor":
        levels = img >> shift
    else:
        raise ValueError("rounding must be 'round' or 'floor'")
    return levels, (levels << shift).astype(np.uint8)


def feature_index(levels: np.ndarray, bits: int) -> np.ndarray:
    """Concatenate the n significant bits of every channel: key = q0 q1 ... q(C-1)."""
    C = levels.shape[-1]
    if bits * C > 64:
        raise ValueError("bits * channels must be <= 64")
    key = np.zeros(levels.shape[:-1], dtype=np.uint64)
    for c in range(C):
        key = (key << np.uint64(bits)) | levels[..., c].astype(np.uint64)
    return key


def index_to_values(keys: np.ndarray, bits: int, channels: int, left_align: bool = True) -> np.ndarray:
    """Decode keys back to (M, C) channel values (left-aligned to 0..255 by default)."""
    keys = np.asarray(keys, dtype=np.uint64)
    mask = np.uint64((1 << bits) - 1)
    out = np.empty(keys.shape + (channels,), dtype=np.float64)
    for c in range(channels):
        out[..., c] = (keys >> np.uint64(bits * (channels - 1 - c))) & mask
    if left_align:
        out *= (1 << (8 - bits))
    return out


# =============================================================================
# 4. Binary merge tree (FIG. 3), C-dimensional
# =============================================================================

@dataclass
class MergeTree:
    colors: np.ndarray                     # (2L-1, C) node values
    counts: np.ndarray                     # (2L-1,)  node frequencies
    parent: np.ndarray                     # (2L-1,)  parent id, -1 for roots
    merges: List[Tuple[int, int, int]] = field(default_factory=list)  # (least_freq, closest, new)
    n_leaves: int = 0

    def get_roots(self) -> np.ndarray:
        return np.flatnonzero((self.parent < 0) & (self.counts > 0))

    def get_leaf_to_root_map(self) -> np.ndarray:
        p = self.parent
        r = np.where(p < 0, np.arange(len(p)), p)
        while True:
            nxt = np.where(p[r] < 0, r, p[r])
            if np.array_equal(nxt, r):
                return r[: self.n_leaves]
            r = nxt


def build_merge_tree(colors0: np.ndarray, counts0: np.ndarray, k: int,
                     weights: Optional[Sequence[float]] = None) -> MergeTree:
    """Least-frequent-first, nearest-neighbour, count-weighted merging until K nodes remain."""
    L, C = colors0.shape
    n_nodes = max(2 * L - 1, L)
    colors = np.zeros((n_nodes, C)); colors[:L] = colors0
    counts = np.zeros(n_nodes, dtype=np.int64); counts[:L] = counts0
    parent = np.full(n_nodes, -1, dtype=np.int64)
    tree = MergeTree(colors, counts, parent, [], L)
    if L <= k:
        return tree

    w = np.ones(C) if weights is None else np.sqrt(np.asarray(weights, dtype=np.float64))
    # compacted live set: slot -> node id
    feat = colors0.astype(np.float64) * w
    cnt = counts0.astype(np.float64).copy()
    ids = np.arange(L, dtype=np.int64)
    n, next_id = L, L

    while n > k:
        cs = cnt[:n]
        cand = np.flatnonzero(cs == cs.min())                 # Step 218: least frequent
        si = cand[np.argmin(ids[cand])] if cand.size > 1 else cand[0]
        diff = feat[:n] - feat[si]
        d = np.einsum("ij,ij->i", diff, diff)                 # Steps 220-222: distances
        d[si] = np.inf
        cand = np.flatnonzero(d == d.min())
        sj = cand[np.argmin(ids[cand])] if cand.size > 1 else cand[0]

        i, j = ids[si], ids[sj]
        ni, nj = counts[i], counts[j]
        cm = (colors[i] * ni + colors[j] * nj) / (ni + nj)    # Step 224: weighted merge
        new = next_id; next_id += 1
        colors[new], counts[new] = cm, ni + nj
        parent[i] = parent[j] = new                           # Step 226: tree update
        tree.merges.append((int(i), int(j), int(new)))

        feat[sj], cnt[sj], ids[sj] = cm * w, ni + nj, new     # merged node takes j's slot
        last = n - 1                                           # drop i's slot (swap with last)
        if si != last:
            feat[si], cnt[si], ids[si] = feat[last], cnt[last], ids[last]
        n -= 1
    return tree


# =============================================================================
# 5. Estimator
# =============================================================================

class FastFeatureClustering:
    """Patent US 11,763,422 B2 clustering for uint8 data with any number of channels.

    Input: uint8 array of shape (..., C) — an (H, W, C) multi-channel image or (N, C) rows.
    """

    def __init__(self, n_clusters: int = 5, bits: Optional[int] = None, margin: Optional[int] = None,
                 auto_increment_bits: bool = True, rounding: str = "round",
                 weights: Optional[Sequence[float]] = None, flat_map_max_bits: int = 24,
                 post_process_refine: bool = True, reassign: bool = False, force_multiword: bool = False):
        self.n_clusters = int(n_clusters)
        self.bits, self.margin = bits, margin
        self.auto_increment_bits = auto_increment_bits
        self.rounding = rounding
        self.weights = weights
        self.flat_map_max_bits = flat_map_max_bits
        self.post_process_refine = post_process_refine
        self.reassign = reassign
        self.force_multiword = force_multiword   # testing only: use the multiword path even when nC <= 64

    # -- map generation (Step 212): flat histogram or sorted hash map ---------------
    def _histogram(self, keys: np.ndarray, bits: int, C: int):
        if bits * C <= self.flat_map_max_bits:
            hist = np.bincount(keys.astype(np.int64), minlength=1 << (bits * C))
            present = np.flatnonzero(hist).astype(np.uint64)
            return present, hist[present.astype(np.int64)], "flat"
        present, cnts = np.unique(keys, return_counts=True)
        return present, cnts, "hash"

    def fit(self, img: np.ndarray, verbose: bool = False) -> "FastFeatureClustering":
        X = self._rows(img)
        N, C = X.shape
        t0 = time.perf_counter()
        bits = int(self.bits) if self.bits is not None else compute_significant_bits(
            self.n_clusters, self.margin, channels=C)

        while True:                                           # Steps 206-216 + Claim 2
            levels, _ = cross_product_reduce(X, bits, self.rounding)
            multiword = self.force_multiword or bits * C > 64
            if multiword:                                     # key wider than 64 bits: rows of levels are the keys
                rows_u, inv, cnts = np.unique(levels, axis=0, return_inverse=True, return_counts=True)
                inv = inv.ravel(); present = rows_u; map_kind = "multiword"
            else:
                keys = feature_index(levels, bits)
                present, cnts, map_kind = self._histogram(keys, bits, C)
            if (len(present) > self.n_clusters or not self.auto_increment_bits or bits >= 8):
                break
            if verbose:
                print(f"unique cells {len(present)} <= K, bits {bits} -> {bits + 1}")
            bits += 1

        self.active_bits_, self.n_channels_, self.map_kind_ = bits, C, map_kind
        self.multiword_ = multiword
        self.n_initial_cells_ = len(present)
        t1 = time.perf_counter()

        if multiword:
            colors0 = present.astype(np.float64) * (1 << (8 - bits))
        else:
            colors0 = index_to_values(present, bits, C, left_align=True)
        tree = build_merge_tree(colors0, cnts, self.n_clusters, self.weights)
        self.tree_ = tree
        t2 = time.perf_counter()

        roots = tree.get_roots()
        roots = roots[np.argsort(-tree.counts[roots], kind="stable")]
        node_to_label = np.full(len(tree.parent), -1, dtype=np.int64)
        node_to_label[roots] = np.arange(len(roots))
        self.tree_palette_ = tree.colors[roots].copy()          # float, 0..255 scale
        self.palette_counts_ = tree.counts[roots].copy()
        self.key_labels_ = node_to_label[tree.get_leaf_to_root_map()].astype(np.int32)
        self.palette_ = self.tree_palette_.copy()
        if multiword:                                            # dict: row bytes -> label
            self.keys_ = None
            self.row_map_ = {r.tobytes(): int(l) for r, l in zip(present, self.key_labels_)}
            labels = self.key_labels_[inv]
        else:
            self.keys_ = present                                 # sorted
            labels = self.key_labels_[np.searchsorted(self.keys_, keys)]
        if self.post_process_refine or self.reassign:            # Claims 6, 13
            Xf = X.astype(np.float64)
            sums = np.zeros((len(roots), C)); np.add.at(sums, labels, Xf)
            sizes = np.bincount(labels, minlength=len(roots))
            ok = sizes > 0
            self.palette_[ok] = sums[ok] / sizes[ok, None]
            if self.reassign:                                    # optional, non-patent
                labels = self._nearest(Xf, self.palette_)
        self.labels_ = labels
        self.timings_ = {"histogram_s": t1 - t0, "merge_tree_s": t2 - t1,
                         "total_s": time.perf_counter() - t0}
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        X = self._rows(img)
        shape = np.asarray(img).shape[:-1]
        if self.reassign:
            return self._nearest(X.astype(np.float64), self.palette_).reshape(shape)
        levels, _ = cross_product_reduce(X, self.active_bits_, self.rounding)
        if self.multiword_:
            got = np.array([self.row_map_.get(r.tobytes(), -1) for r in levels], dtype=np.int32)
            miss = got < 0
            if miss.any():                                       # unseen cells -> nearest palette, O(K C)
                got[miss] = self._nearest(levels[miss].astype(np.float64) * (1 << (8 - self.active_bits_)), self.tree_palette_)
            return got.reshape(shape)
        keys = feature_index(levels, self.active_bits_)
        pos = np.minimum(np.searchsorted(self.keys_, keys), len(self.keys_) - 1)
        found = self.keys_[pos] == keys
        labels = np.empty(len(keys), dtype=np.int32)
        labels[found] = self.key_labels_[pos[found]]
        if (~found).any():                                       # unseen cells -> nearest palette
            vals = index_to_values(keys[~found], self.active_bits_, self.n_channels_)
            labels[~found] = self._nearest(vals, self.tree_palette_)
        return labels.reshape(shape)

    def fit_predict(self, img: np.ndarray) -> np.ndarray:
        self.fit(img)
        return self.labels_.reshape(np.asarray(img).shape[:-1])

    def _nearest(self, X: np.ndarray, P: np.ndarray) -> np.ndarray:
        w = np.ones(P.shape[1]) if self.weights is None else np.asarray(self.weights, float)
        d = ((X[:, None, :] - P[None, :, :]) ** 2 * w).sum(-1)
        return np.argmin(d, axis=1).astype(np.int32)

    @staticmethod
    def _rows(img) -> np.ndarray:
        a = np.asarray(img)
        if a.dtype != np.uint8:
            raise TypeError("expected uint8 channels (use FeatureChannelEncoder first)")
        return np.ascontiguousarray(a.reshape(-1, a.shape[-1]))

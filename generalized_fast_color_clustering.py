#!/usr/bin/env python3
"""
Generalized Fast Color Clustering for Arbitrary Feature Spaces
===============================================================
Extends IBM Patent US 11,763,422 B2 (fast_color_clustering.py)
from 3-channel (RGB) images to C-channel tabular data.

Core algorithm (patent pipeline, generalized):
  1. Compute optimal bit-depth n: smallest n s.t. 2^(n*C) > K + K//2.
  2. Cross-product reduce: q_c = round((2^n - 1) * v / 255) for each channel c.
  3. Composite key: key = q_0 | (q_1 << n) | (q_2 << 2n) | ... (uint64).
  4. Histogram: count frequencies of each distinct composite key.
  5. Hierarchical merging: while n_active > K:
       - pick least-frequent node i
       - find nearest neighbour j (Euclidean on quantized centroid)
       - merge: new_color = (c_i*n_i + c_j*n_j) / (n_i+n_j), new_count = n_i+n_j
       - replace i, j with new node
  6. Final centroids: mean of original (un-quantized) rows in each cluster.
  7. LUT: map every composite key to a cluster label for O(1) prediction.

Bit-exactness with reference (C=3):
  When C=3, the quantization and merging logic produces the same cluster
  assignments as the reference _sources/fast_color_clustering.py because:
  - identical rounding formula: round((2^n-1)*v/255)
  - identical tie-breaking: np.argmin (first minimum wins)
  - identical weighted-average merge
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "compute_significant_bits",
    "cross_product_reduce_multichannel",
    "composite_key",
    "GeneralizedFastColorClustering",
]


# ---------------------------------------------------------------------------
# Step 1: bit-depth
# ---------------------------------------------------------------------------

def compute_significant_bits(k: int, channels: int, margin: Optional[int] = None) -> int:
    """Smallest n s.t. 2^(n*channels) > k + margin.

    Parameters
    ----------
    k : int
        Target number of clusters.
    channels : int
        Number of feature channels C.
    margin : int, optional
        Defaults to k // 2 (patent default).

    Returns
    -------
    int
        Bits per channel (1..8, clamped).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if margin is None:
        margin = max(1, k // 2)
    target = k + margin
    for n in range(1, 9):
        if (1 << (n * channels)) > target:
            return n
    return 8


# ---------------------------------------------------------------------------
# Step 2: cross-product quantization (vectorized, works for any C)
# ---------------------------------------------------------------------------

def cross_product_reduce_multichannel(data_uint8: np.ndarray, bits: int) -> np.ndarray:
    """Quantize uint8 C-channel data to `bits` significant bits per channel.

    Uses the same formula as the reference implementation:
      q = clip(round(v * 2^n / 256), 0, 2^n - 1)

    This matches  _sources/fast_color_clustering.py  cross_product_reduce()
    and ensures bit-exact agreement for C=3.

    The patent also describes q = round((2^n-1)*v/255) — this gives the same
    result for most values but differs near 127/128 for n=1; we use the
    reference code's formula for compatibility.

    Parameters
    ----------
    data_uint8 : ndarray of shape (N, C), dtype uint8
        Input rows where every value is in [0, 255].
    bits : int
        Significant bits per channel (1..8).

    Returns
    -------
    levels : ndarray of shape (N, C), dtype uint8
        Quantized levels q_c in [0, 2^bits - 1].
    """
    if data_uint8.dtype != np.uint8:
        raise TypeError("data_uint8 must have dtype uint8")
    if not (1 <= bits <= 8):
        raise ValueError("bits must be in 1..8")
    scale = np.float32(1 << bits)   # 2^n
    max_q = int(scale) - 1
    # Reference formula: q = clip(round(v * 2^n / 256), 0, 2^n-1)
    q = np.rint(data_uint8.astype(np.float32) * (scale / 256.0))
    q = np.minimum(q, max_q).astype(np.uint8)
    return q


# ---------------------------------------------------------------------------
# Step 3: composite uint64 key
# ---------------------------------------------------------------------------

def composite_key(levels: np.ndarray, bits: int) -> np.ndarray:
    """Pack C-channel quantized levels into a single uint64 key.

    Uses MSB-first ordering (channel 0 at highest bits) to match the reference
    implementation's rrrgggbbb convention:

      key = (q_0 << (C-1)*n) | (q_1 << (C-2)*n) | ... | q_{C-1}

    For C=3 this produces the same integer as the reference color_index():
      key = (q0 << 2n) | (q1 << n) | q2    ← identical to reference

    Parameters
    ----------
    levels : ndarray of shape (N, C), dtype uint8
    bits : int

    Returns
    -------
    keys : ndarray of shape (N,), dtype uint64
    """
    C = levels.shape[1]
    keys = np.zeros(levels.shape[0], dtype=np.uint64)
    for c in range(C):
        shift = (C - 1 - c) * bits
        keys |= levels[:, c].astype(np.uint64) << np.uint64(shift)
    return keys


# ---------------------------------------------------------------------------
# Step 4-5: histogram + hierarchical merging
# ---------------------------------------------------------------------------

def _build_histogram(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unique_keys, counts) sorted by key value."""
    unique_keys, counts = np.unique(keys, return_counts=True)
    return unique_keys, counts.astype(np.int64)


def _key_to_centroid(keys: np.ndarray, bits: int, channels: int,
                     left_align: bool = True) -> np.ndarray:
    """Decode uint64 composite keys back to float64 centroids.

    By default returns left-aligned values in [0, 255] (q << (8-bits)),
    matching the reference _index_to_aligned_color() behavior.
    Set left_align=False to get raw quantized levels in [0, 2^bits-1].

    Uses MSB-first ordering (channel 0 at highest bits), matching composite_key().
    """
    mask = np.uint64((1 << bits) - 1)
    centroids = np.zeros((len(keys), channels), dtype=np.float64)
    for c in range(channels):
        shift = (channels - 1 - c) * bits
        centroids[:, c] = ((keys >> np.uint64(shift)) & mask).astype(np.float64)
    if left_align:
        centroids *= float(1 << (8 - bits))   # q << (8-n)
    return centroids


def _merge_tree(
    centroids: np.ndarray,
    counts: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hierarchical least-frequent merging (patent Tree Module / FIG. 3).

    Parameters
    ----------
    centroids : (L, C) float64  — initial palette centroids
    counts    : (L,) int64      — pixel/row counts per initial color
    k         : int             — target number of clusters

    Returns
    -------
    labels_leaf : (L,) int   — cluster label for every initial leaf node
    final_centroids : (k, C) — merged centroids (in quantized space)
    """
    L = len(counts)
    # Preallocate node arrays large enough for all merges
    N = max(2 * L - 1, 1)
    node_c = np.zeros((N, centroids.shape[1]), dtype=np.float64)
    node_n = np.zeros(N, dtype=np.int64)
    parent = np.full(N, -1, dtype=np.int64)
    node_c[:L] = centroids
    node_n[:L] = counts

    alive = np.zeros(N, dtype=bool)
    alive[:L] = True
    # Use a float copy of counts as the "sort key" (inf for dead nodes)
    sort_key = np.where(alive, node_n.astype(np.float64), np.inf)

    nxt = L
    n_alive = L

    while n_alive > k:
        # a) least-frequent alive node
        i = int(np.argmin(sort_key))
        # b) distances from i to all other alive nodes (Euclidean on quantized centroid)
        diff = node_c - node_c[i]                          # (N, C)
        dist = np.einsum("ij,ij->i", diff, diff)            # (N,)
        dist[~alive] = np.inf
        dist[i] = np.inf
        # c) closest node
        j = int(np.argmin(dist))
        # d) weighted-average merge
        ni, nj = int(node_n[i]), int(node_n[j])
        new_c = (node_c[i] * ni + node_c[j] * nj) / (ni + nj)
        new_n = ni + nj
        new_idx = nxt
        nxt += 1
        node_c[new_idx] = new_c
        node_n[new_idx] = new_n
        # e) update tree
        parent[i] = new_idx
        parent[j] = new_idx
        alive[i] = alive[j] = False
        alive[new_idx] = True
        sort_key[i] = sort_key[j] = np.inf
        sort_key[new_idx] = float(new_n)
        n_alive -= 1

    # Collect final cluster roots
    roots = np.flatnonzero(alive[:nxt])

    # Map every leaf to its root via pointer jumping
    r = np.where(parent < 0, np.arange(N), parent).copy()
    # Clamp r to valid indices before indexing
    for _ in range(int(math.ceil(math.log2(N + 1))) + 2):
        nxt_r = np.where(parent[r] < 0, r, parent[r])
        if np.array_equal(nxt_r, r):
            break
        r = nxt_r
    leaf_roots = r[:L]

    # Assign label 0..k-1 to roots sorted by descending frequency
    order = np.argsort(-node_n[roots])
    roots_sorted = roots[order]
    root_to_label = np.full(N, -1, dtype=np.int64)
    root_to_label[roots_sorted] = np.arange(len(roots_sorted))
    labels_leaf = root_to_label[leaf_roots]

    final_centroids = node_c[roots_sorted]
    return labels_leaf, final_centroids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class GeneralizedFastColorClustering:
    """Generalized Fast Color Clustering for C-channel uint8 tabular data.

    Attributes set after fit():
        n_bits_        : int        — bits per channel used
        n_initial_     : int        — number of distinct composite keys
        palette_       : (K, C) float64 — final centroids (quantized space)
        centroids_orig_: (K, C) float64 — centroids in original (uint8) space
        lut_           : dict[uint64->int] — lookup table key->cluster label
        fit_time_      : float      — wall-clock seconds for fit
    """

    n_clusters: int = 5
    # Internally computed bits; set to None for auto
    _bits: int = field(default=0, init=False, repr=False)

    def fit(self, data_uint8: np.ndarray) -> "GeneralizedFastColorClustering":
        """Fit the model on N×C uint8 data.

        Parameters
        ----------
        data_uint8 : (N, C) ndarray, dtype uint8
        """
        import time
        t0 = time.perf_counter()

        N, C = data_uint8.shape
        K = self.n_clusters

        # Step 1: optimal bit depth
        n = compute_significant_bits(K, C)
        self._bits = n

        # Step 2: quantize
        levels = cross_product_reduce_multichannel(data_uint8, n)

        # Step 3: composite keys
        keys = composite_key(levels, n)

        # Step 4: histogram
        unique_keys, counts = _build_histogram(keys)
        self.n_initial_ = len(unique_keys)

        # Decode initial centroids from composite keys
        init_centroids = _key_to_centroid(unique_keys, n, C)

        # Step 5: merge until K clusters remain
        if self.n_initial_ <= K:
            # Edge case: already <= K distinct colors
            labels_leaf = np.arange(self.n_initial_, dtype=np.int64)
            self.palette_ = init_centroids
        else:
            labels_leaf, self.palette_ = _merge_tree(init_centroids, counts, K)

        # Build LUT: unique_key -> label
        self.lut_: Dict[int, int] = {
            int(unique_keys[i]): int(labels_leaf[i])
            for i in range(len(unique_keys))
        }

        # Assign labels to all rows
        row_labels = np.array([self.lut_[int(k)] for k in keys], dtype=np.int64)

        # Step 6: final centroids as mean of original uint8 values
        self.centroids_orig_ = np.zeros((K, C), dtype=np.float64)
        for lab in range(K):
            mask = row_labels == lab
            if mask.any():
                self.centroids_orig_[lab] = data_uint8[mask].astype(np.float64).mean(0)

        # Store for predict()
        self._unique_keys = unique_keys
        self._labels_leaf = labels_leaf
        self._n_bits = n
        self._C = C

        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, data_uint8: np.ndarray) -> np.ndarray:
        """Predict cluster labels for N×C uint8 data.

        Parameters
        ----------
        data_uint8 : (N, C) ndarray, dtype uint8

        Returns
        -------
        labels : (N,) int64
        """
        levels = cross_product_reduce_multichannel(data_uint8, self._n_bits)
        keys = composite_key(levels, self._n_bits)
        # Map via LUT; unseen keys -> nearest palette entry
        K, C = self.n_clusters, self._C
        labels = np.empty(len(keys), dtype=np.int64)
        seen_mask = np.array([int(k) in self.lut_ for k in keys])
        if seen_mask.any():
            labels[seen_mask] = np.array(
                [self.lut_[int(k)] for k in keys[seen_mask]], dtype=np.int64
            )
        if not seen_mask.all():
            # Unseen keys: nearest palette in quantized space
            unseen_centroids = _key_to_centroid(keys[~seen_mask], self._n_bits, C)
            diff = (unseen_centroids[:, None, :] - self.palette_[None, :, :]) ** 2
            labels[~seen_mask] = np.argmin(diff.sum(2), axis=1)
        return labels

    def fit_predict(self, data_uint8: np.ndarray) -> np.ndarray:
        self.fit(data_uint8)
        return self.predict(data_uint8)

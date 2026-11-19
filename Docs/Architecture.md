# Architecture

## Overview

This project generalises IBM Patent US 11,763,422 B2 (Fast Color Clustering) from 3-channel (RGB) images to C-channel tabular data. The key insight is treating each tabular row as a "pixel" and each scaled feature as a "colour channel".

## Module Map

```mermaid
flowchart LR
    subgraph generalized_fast_color_clustering.py
        A[compute_significant_bits\nn: smallest n s.t. 2^nC > K+K÷2]
        B[cross_product_reduce_multichannel\nq = clip·round·v·2^n÷256]
        C[composite_key uint64\nMSB-first: ch0 at highest bits]
        D[_build_histogram\nnp.unique → counts]
        E[_key_to_centroid\ndecode + left-align to 0-255]
        F[_merge_tree\nleast-freq hierarchical merge]
        G[GeneralizedFastColorClustering\nfit / predict / fit_predict]
        A --> B --> C --> D --> E --> F --> G
    end

    subgraph analysis.py
        H[Section 0 – bit-exactness C=3]
        I[Section 1 – load + scale uint8]
        J[Section 2 – 129×160×8 tensor]
        K[Section 3 – benchmark + plots]
        L[Section 4 – robustness]
    end

    G --> K
    G --> L
```

## Algorithm Detail

### Step 1 — Bit Depth (patent claim 1)

```
n = min{n ∈ 1..8 : 2^(n·C) > K + max(1, K÷2)}
```

For K=5, C=8: n=1 (2^8 = 256 > 7). For K=5, C=3: n=1 (2^3 = 8 > 7).

### Step 2 — Cross-Product Quantisation

```
q_c = clip(round(v_c · 2^n / 256), 0, 2^n − 1)
```

Matches the reference code formula exactly. This is equivalent to the patent's
`round((2^n−1)·v/255)` for most values and ensures bit-exact agreement.

### Step 3 — Composite Key (uint64, MSB-first)

```
key = (q_0 << (C-1)·n) | (q_1 << (C-2)·n) | … | q_{C-1}
```

For C=3 this reduces to `(q0 << 2n) | (q1 << n) | q2`, identical to the reference `color_index()`.

### Step 4 — Histogram

`np.unique(keys, return_counts=True)` gives the L distinct keys and their pixel counts. L ≤ 2^(n·C).

### Step 5 — Hierarchical Merge Tree (patent FIG. 3)

```
while n_active > K:
    i = argmin(counts[alive])                       # least-frequent
    j = argmin(‖centroid_i − centroid_j‖²)          # nearest neighbour
    merge(i, j): new_centroid = (c_i·n_i + c_j·n_j)/(n_i+n_j)
                 new_count    = n_i + n_j
```

Centroids are maintained in left-aligned [0,255] space for metric consistency.

### Step 6 — Final Centroids

After label assignment, final centroids are computed as the mean of original uint8 rows in each cluster (`model.centroids_orig_`).

### Step 7 — LUT

A Python dict maps every seen composite key to a cluster label. Unseen keys (at predict time) are assigned to the nearest palette centroid.

## Bit-Exactness Proof (C=3)

| Property | Reference (C=3) | Generalised (any C) |
|---|---|---|
| Quantisation formula | `round(v·2^n/256)` | same |
| Key encoding | `(q0<<2n)\|(q1<<n)\|q2` | `(q0<<(C-1)n)\|…` = same for C=3 |
| Centroid initialisation | left-aligned `q<<(8-n)` | same via `_key_to_centroid(left_align=True)` |
| Merge distance metric | Euclidean on left-aligned | same |
| Tie-breaking | `np.argmin` (first min) | same |
| **ARI on astronaut image** | **—** | **1.000000** ✓ |

## Data Flow — California Housing

```mermaid
flowchart TD
    Raw["Raw data\n20640 × 8\nfloat64"] 
    Raw -->|MinMaxScaler×255| MM["uint8 minmax\n20640 × 8"]
    Raw -->|RobustScaler norm| RB["uint8 robust\n20640 × 8"]
    Raw -->|QuantileTransformer| QT["uint8 quantile\n20640 × 8"]
    MM --> T1["Tensor 129×160×8"]
    RB --> T2["Tensor 129×160×8"]
    QT --> T3["Tensor 129×160×8"]
    MM --> FastCC["FastCC K=5\n~0.007 s\nη²=0.355"]
    MM --> KMu8["KMeans uint8 K=5\n~0.112 s\nη²=0.014"]
    Raw --> StdKM["KMeans+StandardScaler K=5\n~0.154 s\nη²=0.053"]
```




# Generalized Fast Color Clustering for Tabular Data

Extends IBM Patent **US 11,763,422 B2** (*Fast Color Clustering for Preprocessing an Image*) from 3-channel RGB images to arbitrary C-channel feature spaces, demonstrated on the California Housing dataset.

## Architecture

```mermaid
flowchart TD
    A[Raw tabular data\n20640 × 8] --> B[Scale to uint8\nminmax / robust / quantile]
    B --> C[129×160×8 tensor\nfictitious image]
    C --> D[compute_significant_bits\nn: 2^nC > K + K/2]
    D --> E[cross_product_reduce_multichannel\nq = clip·round·v·2^n÷256]
    E --> F[composite_key uint64\nMSB-first packing]
    F --> G[Histogram / hash map\ncount distinct keys]
    G --> H[_merge_tree\nhierarchical least-freq merge]
    H --> I[K cluster labels\nper row]
    I --> J[Benchmark vs KMeans\ntime · silhouette · η² · ARI]
    I --> K[Robustness\n20-fold bootstrap + null test]
```

## Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant A as analysis.py
    participant G as GeneralizedFastColorClustering
    participant R as Reference (C=3 only)

    U->>A: python analysis.py
    A->>R: Section 0 - bit-exactness test (astronaut image)
    R-->>A: ARI = 1.000000 (bit-exact)
    A->>G: Section 1-2 - load California Housing, scale, reshape tensor
    A->>G: Section 3 - fit/predict all three methods
    G-->>A: labels, centroids, fit_time
    A->>A: Compute silhouette, eta2, ARI
    A->>A: Plot cluster maps and feature heatmaps
    A->>G: Section 4 - 20-fold bootstrap and null hypothesis test
    A-->>U: output/ plots + report.txt
```

## Key Results (K=5, minmax scaling)

| Method | Time (s) | Silhouette | η² (price) | ARI vs Std |
|---|---|---|---|---|
| KMeans + StandardScaler | 0.154 | 0.295 | 0.053 | 1.000 |
| KMeans + uint8 | 0.112 | 0.323 | 0.014 | 0.423 |
| **Generalized FastCC** | **0.007** | 0.131 | **0.355** | 0.356 |

- **FastCC is ~22× faster** than KMeans and captures strong house-price structure (η² = 0.355).
- Bootstrap stability: ARI = 0.862 ± 0.090 across 20 folds.
- Null hypothesis: silhouette degrades 0.036 on shuffled data (p < 0.0001), confirming real signal.

## Project Structure

```
.
├── generalized_fast_color_clustering.py  # Generalized C-channel algorithm
├── analysis.py                           # Full analysis pipeline
├── requirements.txt
├── _sources/                             # Reference implementation (C=3)
│   ├── fast_color_clustering.py
│   └── US11763422.pdf
├── output/                               # Timestamped plots + report
└── Docs/
    ├── Quickstart.md
    └── Architecture.md
```

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python analysis.py
```

## License

MIT License — research adaptation of IBM Patent US 11,763,422 B2 for academic / educational use.

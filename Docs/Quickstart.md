# Quickstart

## Prerequisites

- Python 3.11+
- `uv` package manager

## Install

```bash
cd opencv-mf-2
uv venv .venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Run the full analysis

```bash
python analysis.py
```

All outputs land in `output/` with a `YYYYmmdd_HHMMSS_` timestamp prefix:

| File | Contents |
|---|---|
| `*_bitexact_verification.png` | Side-by-side C=3 reference vs generalised cluster maps |
| `*_tensor_<method>.png` | 129×160×8 false-colour channel images (3 scalings) |
| `*_cluster_maps.png` | Cluster assignment map (3 methods) next to house-price ground truth |
| `*_feature_heatmaps.png` | Per-cluster feature-profile heatmaps |
| `*_metrics_summary.png` | Bar-chart of time / silhouette / η² / ARI |
| `*_scaling_comparison.png` | FastCC cluster map under three scalings |
| `*_robustness.png` | Bootstrap ARI histogram + null silhouette distribution |
| `*_report.txt` | Full printed log of all metrics |

## Run only the algorithm

```python
from generalized_fast_color_clustering import GeneralizedFastColorClustering
import numpy as np

# Any (N, C) uint8 matrix
X_u8 = np.random.randint(0, 256, (20640, 8), dtype=np.uint8)

model = GeneralizedFastColorClustering(n_clusters=5)
labels = model.fit_predict(X_u8)

print(labels.shape)          # (20640,)
print(model._bits)           # bits/channel used
print(model.n_initial_)      # number of distinct composite keys
print(model.centroids_orig_) # (5, 8) centroids in uint8 space
```

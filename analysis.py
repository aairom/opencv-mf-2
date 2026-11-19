#!/usr/bin/env python3
"""
Generalized Fast Color Clustering — California Housing Analysis
===============================================================
IBM Patent US 11,763,422 B2 extended to C-channel tabular data.

Sections
--------
0.  Bit-exactness verification (C=3 vs reference implementation)
1.  Data loading + three uint8 scalings (minmax / robust / quantile)
2.  129×160×8 tensor representation + visualisation
3.  Benchmark: (a) KMeans+StandardScaler, (b) KMeans+uint8, (c) FastCC
    Metrics: time, cluster sizes, silhouette, η², ARI
    Plots  : cluster maps, feature-profile heatmaps
4.  Robustness: 20-fold bootstrap stability + null-hypothesis test

Outputs are written to output/<timestamp>_*.{png,txt}
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

# Reference implementation (C=3 only, requires opencv)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_sources"))
try:
    from fast_color_clustering import (
        FastColorClustering as _RefFCC,
        cross_product_reduce,
        color_index,
    )
    _HAS_REF = True
except ImportError:
    _HAS_REF = False
    warnings.warn("Reference fast_color_clustering not importable (opencv missing?).")

from generalized_fast_color_clustering import (
    GeneralizedFastColorClustering,
    compute_significant_bits,
    cross_product_reduce_multichannel,
    composite_key,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

def outpath(name: str) -> str:
    return os.path.join(OUT, f"{TS}_{name}")

REPORT_LINES: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    REPORT_LINES.append(msg)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
K = 5          # target clusters
N_ROWS = 20640  # California Housing size
IMG_H, IMG_W, N_FEAT = 129, 160, 8   # 129*160 = 20640 exactly
assert IMG_H * IMG_W == N_ROWS, "Tensor shape must match dataset size"


def scale_to_uint8(X: np.ndarray, method: str) -> np.ndarray:
    """Scale each column of X independently to [0,255] uint8."""
    if method == "minmax":
        sc = MinMaxScaler(feature_range=(0, 255))
        out = sc.fit_transform(X)
    elif method == "robust":
        sc = RobustScaler()
        scaled = sc.fit_transform(X)
        # Clip to [0,1] after min/max normalisation per column
        col_min = scaled.min(axis=0)
        col_max = scaled.max(axis=0)
        out = (scaled - col_min) / (col_max - col_min + 1e-9) * 255.0
    elif method == "quantile":
        sc = QuantileTransformer(n_quantiles=1000, output_distribution="uniform",
                                  random_state=42)
        out = sc.fit_transform(X) * 255.0
    else:
        raise ValueError(method)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def eta_squared(labels: np.ndarray, y: np.ndarray) -> float:
    """η² = SS_between / SS_total (one-way ANOVA effect size)."""
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    ss_between = sum(
        np.sum(labels == c) * (y[labels == c].mean() - grand_mean) ** 2
        for c in np.unique(labels)
        if (labels == c).any()
    )
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def cluster_sizes(labels: np.ndarray, k: int) -> list[int]:
    return [int((labels == c).sum()) for c in range(k)]


# ===========================================================================
# SECTION 0 — Bit-exactness verification (C=3)
# ===========================================================================
def verify_bit_exactness() -> None:
    log("=" * 70)
    log("SECTION 0 — Bit-exactness Verification (C=3 vs Reference)")
    log("=" * 70)

    if not _HAS_REF:
        log("  SKIP: reference fast_color_clustering not available (need opencv).")
        return

    try:
        from skimage import data as skdata
        test_img_rgb = skdata.astronaut()   # (512, 512, 3) uint8
    except Exception:
        rng = np.random.default_rng(0)
        test_img_rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        log("  Using synthetic 64×64 random image (skimage not available).")

    H, W, _ = test_img_rgb.shape
    log(f"  Test image: {H}×{W}×3 uint8")

    # Compute matching bits using our generalized function (same formula as patent)
    ref_bits = compute_significant_bits(K, channels=3)

    # --- Reference (C=3) ---
    ref_model = _RefFCC(n_colors=K, bits=ref_bits, rounding="round", distance="rgb",
                        channel_order="rgb")
    ref_model.fit(test_img_rgb)
    ref_labels_2d = ref_model.predict(test_img_rgb)       # (H, W)
    ref_labels = ref_labels_2d.ravel()

    # --- Generalized (C=3) ---
    flat = test_img_rgb.reshape(-1, 3)                    # (H*W, 3) uint8
    gen_model = GeneralizedFastColorClustering(n_clusters=K)
    gen_model.fit(flat)
    gen_labels = gen_model.predict(flat)

    # ARI should be 1.0 for identical assignments (up to label permutation)
    ari = adjusted_rand_score(ref_labels, gen_labels)
    log(f"  Reference bits:    {ref_model.bits}")
    log(f"  Generalized bits:  {gen_model._bits}")
    log(f"  Reference n_initial_colors: {ref_model.n_initial_colors_}")
    log(f"  Generalized n_initial:      {gen_model.n_initial_}")
    log(f"  ARI (ref vs generalized):   {ari:.6f}   (1.0 = bit-exact)")
    if ari > 0.999:
        log("  ✓ BIT-EXACT: generalized reproduces the reference for C=3.")
    else:
        log(f"  ⚠ CLOSE but not 1.0 (ARI={ari:.6f}) — tie-breaking may vary.")

    # Side-by-side visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img_rgb)
    axes[0].set_title("Original astronaut image")
    axes[0].axis("off")
    axes[1].imshow(ref_labels_2d.reshape(H, W), cmap="tab10", aspect="auto")
    axes[1].set_title(f"Reference FastCC (C=3, bits={ref_bits})")
    axes[1].axis("off")
    axes[2].imshow(gen_labels.reshape(H, W), cmap="tab10", aspect="auto")
    axes[2].set_title(f"Generalized FastCC (C=3, bits={gen_model._bits})")
    axes[2].axis("off")
    fig.suptitle(f"Bit-exactness Verification  ARI={ari:.6f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath("bitexact_verification.png"), dpi=100, bbox_inches="tight")
    plt.close()
    log("  Saved: bitexact_verification.png")
    log()


# ===========================================================================
# SECTION 1 — Data loading + three uint8 scalings
# ===========================================================================
def load_and_scale():
    log("=" * 70)
    log("SECTION 1 — California Housing Dataset + uint8 Scaling")
    log("=" * 70)

    housing = fetch_california_housing(as_frame=False)
    X_raw = housing.data              # (20640, 8)
    y = housing.target                # house price
    feat_names = list(housing.feature_names)

    log(f"  Shape: {X_raw.shape}   Features: {feat_names}")
    log(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    scalings = {}
    for method in ("minmax", "robust", "quantile"):
        X_u8 = scale_to_uint8(X_raw, method)
        scalings[method] = X_u8
        log(f"  {method:10s} -> uint8 shape {X_u8.shape}, "
            f"mean={X_u8.mean():.1f}, std={X_u8.std():.1f}")
    log()
    return X_raw, y, feat_names, scalings


# ===========================================================================
# SECTION 2 — 129×160×8 tensor representation
# ===========================================================================
def visualize_tensor(scalings: dict, feat_names: list) -> None:
    log("=" * 70)
    log("SECTION 2 — 129×160×8 Tensor Representation")
    log("=" * 70)

    for method, X_u8 in scalings.items():
        tensor = X_u8.reshape(IMG_H, IMG_W, N_FEAT)   # (129, 160, 8)
        log(f"  {method}: tensor shape {tensor.shape}")

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle(
            f"8-Channel Image Representation — {method} scaling\n"
            f"(129×160 pixels, each pixel = one California Housing row)",
            fontsize=11,
        )
        for c, ax in enumerate(axes.flat):
            im = ax.imshow(tensor[:, :, c], cmap="viridis", aspect="auto",
                           interpolation="nearest")
            ax.set_title(feat_names[c], fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(outpath(f"tensor_{method}.png"), dpi=120, bbox_inches="tight")
        plt.close()
        log(f"    Saved: {TS}_tensor_{method}.png")
    log()


# ===========================================================================
# SECTION 3 — Benchmark
# ===========================================================================
def run_benchmark(X_raw, y, scalings, feat_names):
    log("=" * 70)
    log("SECTION 3 — Benchmark: KMeans(Standard) / KMeans(uint8) / FastCC")
    log("=" * 70)

    # Use minmax-scaled uint8 as the "working" scaled data (most neutral)
    X_u8 = scalings["minmax"]

    results = {}

    # ------------------------------------------------------------------ (a)
    log("  (a) StandardScaler + KMeans")
    sc = StandardScaler()
    X_std = sc.fit_transform(X_raw)
    t0 = time.perf_counter()
    km_std = KMeans(n_clusters=K, n_init=10, random_state=42)
    lab_a = km_std.fit_predict(X_std)
    dt_a = time.perf_counter() - t0
    sil_a = silhouette_score(X_std, lab_a, sample_size=5000, random_state=42)
    eta_a = eta_squared(lab_a, y)
    results["KMeans+Standard"] = dict(
        labels=lab_a, time=dt_a,
        silhouette=sil_a, eta2=eta_a,
        sizes=cluster_sizes(lab_a, K),
    )
    log(f"    time={dt_a:.3f}s  silhouette={sil_a:.4f}  η²={eta_a:.4f}")
    log(f"    cluster sizes: {cluster_sizes(lab_a, K)}")

    # ------------------------------------------------------------------ (b)
    log("  (b) KMeans directly on uint8 data")
    t0 = time.perf_counter()
    km_u8 = KMeans(n_clusters=K, n_init=10, random_state=42)
    lab_b = km_u8.fit_predict(X_u8.astype(np.float32))
    dt_b = time.perf_counter() - t0
    sil_b = silhouette_score(X_u8.astype(np.float32), lab_b, sample_size=5000, random_state=42)
    eta_b = eta_squared(lab_b, y)
    results["KMeans+uint8"] = dict(
        labels=lab_b, time=dt_b,
        silhouette=sil_b, eta2=eta_b,
        sizes=cluster_sizes(lab_b, K),
    )
    log(f"    time={dt_b:.3f}s  silhouette={sil_b:.4f}  η²={eta_b:.4f}")
    log(f"    cluster sizes: {cluster_sizes(lab_b, K)}")

    # ------------------------------------------------------------------ (c)
    log("  (c) Generalized Fast Color Clustering")
    t0 = time.perf_counter()
    fcc = GeneralizedFastColorClustering(n_clusters=K)
    lab_c = fcc.fit_predict(X_u8)
    dt_c = time.perf_counter() - t0
    sil_c = silhouette_score(X_u8.astype(np.float32), lab_c, sample_size=5000, random_state=42)
    eta_c = eta_squared(lab_c, y)
    n_bits = fcc._bits
    results["FastCC"] = dict(
        labels=lab_c, time=dt_c,
        silhouette=sil_c, eta2=eta_c,
        sizes=cluster_sizes(lab_c, K),
        n_bits=n_bits,
        n_initial=fcc.n_initial_,
    )
    log(f"    time={dt_c:.3f}s  silhouette={sil_c:.4f}  η²={eta_c:.4f}")
    log(f"    bits/channel={n_bits}  n_initial_keys={fcc.n_initial_}")
    log(f"    cluster sizes: {cluster_sizes(lab_c, K)}")

    # ------------------------------------------------------------------ ARI
    ari_ab = adjusted_rand_score(lab_a, lab_b)
    ari_ac = adjusted_rand_score(lab_a, lab_c)
    ari_bc = adjusted_rand_score(lab_b, lab_c)
    log(f"\n  ARI cross-comparison:")
    log(f"    ARI(Standard vs uint8) = {ari_ab:.4f}")
    log(f"    ARI(Standard vs FastCC) = {ari_ac:.4f}")
    log(f"    ARI(uint8 vs FastCC)    = {ari_bc:.4f}")
    for name in results:
        results[name]["ari_vs_std"] = adjusted_rand_score(lab_a, results[name]["labels"])

    log()

    # ------------------------------------------------------------------ Plots
    _plot_cluster_maps(results, y, feat_names)
    _plot_heatmaps(results, X_u8, feat_names)
    _plot_metrics_summary(results)

    return results, fcc


def _plot_cluster_maps(results, y, feat_names):
    """Cluster assignment maps (reshaped to 129×160)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    methods = list(results.keys())
    cmaps = ["tab10", "Set1", "Set2"]

    for col, name in enumerate(methods):
        lab = results[name]["labels"]
        lab_img = lab.reshape(IMG_H, IMG_W)
        y_img = y.reshape(IMG_H, IMG_W)
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        im1 = ax_top.imshow(lab_img, cmap=cmaps[col % len(cmaps)],
                            aspect="auto", interpolation="nearest")
        ax_top.set_title(f"Cluster Map\n{name}", fontsize=10)
        ax_top.axis("off")
        plt.colorbar(im1, ax=ax_top, fraction=0.046)

        im2 = ax_bot.imshow(y_img, cmap="plasma", aspect="auto",
                            interpolation="nearest")
        ax_bot.set_title("House Price (ground truth)", fontsize=9)
        ax_bot.axis("off")
        plt.colorbar(im2, ax=ax_bot, fraction=0.046)

    fig.suptitle("Cluster Assignment Maps (129×160 spatial layout) vs House Price",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath("cluster_maps.png"), dpi=120, bbox_inches="tight")
    plt.close()
    log("    Saved: cluster_maps.png")


def _plot_heatmaps(results, X_u8, feat_names):
    """Cluster feature profile heatmaps — mean uint8 value per cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = list(results.keys())
    for ax, name in zip(axes, methods):
        lab = results[name]["labels"]
        profile = np.zeros((K, N_FEAT))
        for c in range(K):
            mask = lab == c
            if mask.any():
                profile[c] = X_u8[mask].mean(axis=0)
        im = ax.imshow(profile, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=255)
        ax.set_title(f"Feature Profile\n{name}", fontsize=10)
        ax.set_xlabel("Feature channel (uint8 mean)")
        ax.set_ylabel("Cluster")
        ax.set_xticks(range(N_FEAT))
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"C{i}" for i in range(K)])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Cluster Feature Profiles — Mean uint8 Value per Feature per Cluster",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath("feature_heatmaps.png"), dpi=120, bbox_inches="tight")
    plt.close()
    log("    Saved: feature_heatmaps.png")


def _plot_metrics_summary(results):
    """Bar chart summary of key metrics."""
    methods = list(results.keys())
    metrics = {
        "Time (s)": [results[m]["time"] for m in methods],
        "Silhouette": [results[m]["silhouette"] for m in methods],
        "η² (price)": [results[m]["eta2"] for m in methods],
        "ARI vs Std": [results[m]["ari_vs_std"] for m in methods],
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, (title, vals) in zip(axes, metrics.items()):
        bars = ax.bar(methods, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.4f}", ha="center", fontsize=7)
    fig.suptitle("Benchmark Metrics Summary", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath("metrics_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()
    log("    Saved: metrics_summary.png")


# ===========================================================================
# SECTION 4 — Robustness validation
# ===========================================================================
def run_robustness(X_u8, results, y):
    log("=" * 70)
    log("SECTION 4 — Robustness Validation")
    log("=" * 70)

    rng = np.random.default_rng(42)
    base_labels = results["FastCC"]["labels"]
    base_sil = results["FastCC"]["silhouette"]

    # ---- (a) 20-fold bootstrap stability ----
    log("  (a) 20-fold bootstrap stability (ARI between original and bootstrap)")
    N = len(X_u8)
    boot_aris = []
    for fold in range(20):
        idx = rng.choice(N, size=N, replace=True)
        X_boot = X_u8[idx]
        fcc_boot = GeneralizedFastColorClustering(n_clusters=K)
        fcc_boot.fit(X_boot)
        # Predict on FULL original dataset
        lab_boot_full = fcc_boot.predict(X_u8)
        ari = adjusted_rand_score(base_labels, lab_boot_full)
        boot_aris.append(ari)
        if (fold + 1) % 5 == 0:
            log(f"    fold {fold+1:2d}/20  ARI={ari:.4f}")

    boot_aris = np.array(boot_aris)
    log(f"\n  Bootstrap ARI — mean={boot_aris.mean():.4f} ± {boot_aris.std():.4f}")
    log(f"  Bootstrap ARI — min={boot_aris.min():.4f}  max={boot_aris.max():.4f}")

    # ---- (b) Null hypothesis: shuffled columns ----
    log("\n  (b) Null hypothesis test — shuffled-column data")
    null_sils = []
    for rep in range(20):
        X_shuf = X_u8.copy()
        for c in range(X_shuf.shape[1]):
            X_shuf[:, c] = rng.permutation(X_shuf[:, c])
        fcc_null = GeneralizedFastColorClustering(n_clusters=K)
        fcc_null.fit(X_shuf)
        lab_null = fcc_null.predict(X_shuf)
        try:
            null_sils.append(
                silhouette_score(X_shuf.astype(np.float32), lab_null,
                                  sample_size=3000, random_state=int(rep))
            )
        except Exception:
            null_sils.append(0.0)

    null_sils = np.array(null_sils)
    log(f"  Original silhouette:        {base_sil:.4f}")
    log(f"  Null silhouette mean ± std: {null_sils.mean():.4f} ± {null_sils.std():.4f}")
    degradation = base_sil - null_sils.mean()
    log(f"  Silhouette degradation:     {degradation:.4f}  "
        f"({'signal confirmed' if degradation > 0.01 else 'weak signal'})")

    # one-sample t-test: null silhouette < original?
    t_stat, p_val = stats.ttest_1samp(null_sils, base_sil, alternative="less")
    log(f"  One-sided t-test (null < real): t={t_stat:.3f}  p={p_val:.4f}")

    # ---- Plots ----
    _plot_robustness(boot_aris, null_sils, base_sil)

    log()
    return boot_aris, null_sils


def _plot_robustness(boot_aris, null_sils, base_sil):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bootstrap ARI distribution
    ax1.hist(boot_aris, bins=10, color="#4c72b0", edgecolor="white", alpha=0.8)
    ax1.axvline(boot_aris.mean(), color="red", linestyle="--",
                label=f"Mean={boot_aris.mean():.4f}")
    ax1.set_title("Bootstrap Stability (20 folds)\nARI vs Original Clustering")
    ax1.set_xlabel("ARI")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)

    # Null vs real silhouette
    ax2.hist(null_sils, bins=10, color="#dd8452", edgecolor="white", alpha=0.8,
             label=f"Null (mean={null_sils.mean():.4f})")
    ax2.axvline(base_sil, color="blue", linestyle="--", linewidth=2,
                label=f"Real data={base_sil:.4f}")
    ax2.set_title("Null Hypothesis Test\nSilhouette: Real vs Shuffled Data")
    ax2.set_xlabel("Silhouette Coefficient")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)

    fig.suptitle("Robustness Validation — FastCC on California Housing", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath("robustness.png"), dpi=120, bbox_inches="tight")
    plt.close()
    log("    Saved: robustness.png")


# ===========================================================================
# Scaling method comparison (supplementary)
# ===========================================================================
def compare_scaling_methods(scalings, y):
    log("=" * 70)
    log("SUPPLEMENTARY — Scaling Method Comparison (FastCC, K=5)")
    log("=" * 70)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (method, X_u8) in zip(axes, scalings.items()):
        fcc = GeneralizedFastColorClustering(n_clusters=K)
        lab = fcc.fit_predict(X_u8)
        sil = silhouette_score(X_u8.astype(np.float32), lab, sample_size=5000, random_state=42)
        eta = eta_squared(lab, y)
        lab_img = lab.reshape(IMG_H, IMG_W)
        ax.imshow(lab_img, cmap="tab10", aspect="auto", interpolation="nearest")
        ax.set_title(f"{method} scaling\nsil={sil:.4f}  η²={eta:.4f}", fontsize=10)
        ax.axis("off")
        log(f"  {method:10s}: silhouette={sil:.4f}  η²={eta:.4f}  "
            f"n_initial={fcc.n_initial_}  bits={fcc._bits}")
    fig.suptitle("FastCC Cluster Maps — Three Scaling Methods", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath("scaling_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    log("    Saved: scaling_comparison.png")
    log()


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log("=" * 70)
    log("Generalized Fast Color Clustering — California Housing")
    log(f"Run: {TS}")
    log("=" * 70)
    log()

    verify_bit_exactness()

    X_raw, y, feat_names, scalings = load_and_scale()
    visualize_tensor(scalings, feat_names)
    compare_scaling_methods(scalings, y)

    results, fcc = run_benchmark(X_raw, y, scalings, feat_names)
    boot_aris, null_sils = run_robustness(scalings["minmax"], results, y)

    # Final summary table
    log("=" * 70)
    log("FINAL SUMMARY TABLE")
    log("=" * 70)
    hdr = f"{'Method':<22} {'Time(s)':>8} {'Silhouette':>11} {'η²':>8} {'ARI_vs_Std':>11}"
    log(hdr)
    log("-" * len(hdr))
    for name, res in results.items():
        log(f"{name:<22} {res['time']:>8.3f} {res['silhouette']:>11.4f} "
            f"{res['eta2']:>8.4f} {res['ari_vs_std']:>11.4f}")
    log()
    log(f"Bootstrap stability (FastCC): ARI = {boot_aris.mean():.4f} ± {boot_aris.std():.4f}")
    log(f"Null silhouette degradation:  {results['FastCC']['silhouette'] - null_sils.mean():.4f}")
    log()

    # Save text report
    report_path = outpath("report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()

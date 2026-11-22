#!/usr/bin/env python3
"""
California Housing: US 11,763,422 B2 multi-channel clustering vs classical KMeans (k=5)
=====================================================================================
Each of the 8 features is normalised to an integer 0..255 and becomes a fictitious
channel; the 20,640 block groups are laid out as a 129 x 160 x 8 image and clustered
with FastFeatureClustering (fast_feature_clustering.py). The baseline is the KMeans
pipeline from housingkmeans.py (StandardScaler + KMeans(k=5, n_init=10, rs=42)).

Usage
    python housing_patent_vs_kmeans.py                 # uses sklearn.fetch_california_housing
    python housing_patent_vs_kmeans.py --csv housing.csv   # fallback: StatLib/Géron CSV layout
Outputs (in --out, default ./results): PNG figures + metrics CSV.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_feature_clustering import FastFeatureClustering, FeatureChannelEncoder, rows_to_image

K = 5
SEED = 42
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
            "Latitude", "Longitude"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical slots 1-5
CMAP = ListedColormap(PALETTE)
INK, MUTED = "#1f1f1e", "#6b6a64"


# -----------------------------------------------------------------------------
def load_data(csv: str | None):
    if csv is None:
        try:
            from sklearn.datasets import fetch_california_housing
            h = fetch_california_housing(as_frame=True)
            return h.data[FEATURES].copy(), h.target.values, "sklearn.fetch_california_housing"
        except Exception as e:  # offline / blocked
            raise SystemExit(f"fetch_california_housing failed ({e}); pass --csv housing.csv")
    r = pd.read_csv(csv)
    bed = r.total_bedrooms
    if bed.isna().any():  # Géron's copy blanks 207 values; impute with median bedroom/room ratio
        bed = bed.fillna(r.total_rooms * (bed / r.total_rooms).median())
    df = pd.DataFrame({
        "MedInc": r.median_income, "HouseAge": r.housing_median_age,
        "AveRooms": r.total_rooms / r.households, "AveBedrms": bed / r.households,
        "Population": r.population, "AveOccup": r.population / r.households,
        "Latitude": r.latitude, "Longitude": r.longitude})
    return df, r.median_house_value.values / 1e5, f"CSV {os.path.basename(csv)}"


def eta2(y, labels):
    """Correlation ratio: share of the variance of y explained by the partition."""
    m = y.mean()
    between = sum((labels == c).sum() * (y[labels == c].mean() - m) ** 2 for c in np.unique(labels))
    return between / ((y - m) ** 2).sum()


def align_to(ref, labels, k=K):
    """Relabel `labels` so that clusters best match `ref` ids (Hungarian on contingency)."""
    ct = np.zeros((k, k), dtype=int)
    np.add.at(ct, (ref, labels), 1)
    r, c = linear_sum_assignment(-ct)
    mapping = np.empty(k, dtype=int); mapping[c] = r
    return mapping[labels]


def evaluate(name, labels, Z, U_space, y, ref, t):
    sizes = np.bincount(labels, minlength=K)
    p = sizes[sizes > 0] / sizes.sum()
    return {
        "method": name,
        "time_ms": 1e3 * t,
        "sizes": " / ".join(map(str, sorted(sizes, reverse=True))),
        "smallest_%": 100 * sizes.min() / sizes.sum(),
        "effective_k": float(np.exp(-(p * np.log(p)).sum())),
        "silhouette_std": silhouette_score(Z, labels, sample_size=5000, random_state=SEED),
        "silhouette_own": silhouette_score(U_space, labels, sample_size=5000, random_state=SEED),
        "davies_bouldin_std": davies_bouldin_score(Z, labels),
        "eta2_MedHouseVal": eta2(y, labels),
        "ARI_vs_KMeans": adjusted_rand_score(ref, labels) if ref is not None else 1.0,
        "NMI_vs_KMeans": normalized_mutual_info_score(ref, labels) if ref is not None else 1.0,
    }


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, y, source = load_data(args.csv)
    X = df.values.astype(np.float64)
    print(f"Data: {source}  {X.shape}")

    # ---- A. classical baseline (exactly the attached housingkmeans.py) -------------
    Z = StandardScaler().fit_transform(X)
    t0 = time.perf_counter()
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
    t_km = time.perf_counter() - t0
    ref = km.labels_
    rows, labels_by = [], {"KMeans k=5 (standardised)": ref}
    rows.append(evaluate("KMeans k=5 (standardised)", ref, Z, Z, y, None, t_km))

    # ---- encodings: 8 features -> 8 fictitious uint8 channels ---------------------
    enc = {m: FeatureChannelEncoder(m).fit(X) for m in ("minmax", "robust", "quantile")}
    U = {m: e.transform(X) for m, e in enc.items()}
    images = {m: rows_to_image(u) for m, u in U.items()}          # (129, 160, 8)
    print("Fictitious image shape:", images["minmax"].shape)

    # ---- B. KMeans on the very same 0..255 channels (isolates algorithm vs encoding)
    t0 = time.perf_counter()
    kmu = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(U["minmax"].astype(np.float64))
    labels_by["KMeans k=5 (uint8 min-max)"] = kmu.labels_
    rows.append(evaluate("KMeans k=5 (uint8 min-max)", kmu.labels_, Z, U["minmax"].astype(float),
                         y, ref, time.perf_counter() - t0))

    # ---- C..: patent pipeline, grid over encoding x bits x reassign --------------
    models = {}
    for m in ("minmax", "robust", "quantile"):
        for bits in (None, 2, 3):
            for reassign in (False, True):
                f = FastFeatureClustering(K, bits=bits, reassign=reassign)
                t0 = time.perf_counter()
                lab = f.fit_predict(images[m]).ravel()               # fit on the 8-channel image
                t = time.perf_counter() - t0
                b = f.active_bits_
                name = f"Patent {m}, {b} bit{'s' if b > 1 else ''}{' (auto)' if bits is None else ''}" \
                       f"{' + reassign' if reassign else ''}"
                models[name] = f
                labels_by[name] = lab
                r = evaluate(name, lab, Z, U[m].astype(float), y, ref, t)
                r["cells"] = f.n_initial_cells_
                r["map"] = f.map_kind_
                rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(res.drop(columns=["sizes"]).to_string(index=False))
        print(res[["method", "sizes"]].to_string(index=False))

    # ---- inference speed of the fitted LUT --------------------------------------
    f = models["Patent minmax, 1 bit (auto)"]
    t0 = time.perf_counter()
    for _ in range(20):
        f.predict(images["minmax"])
    t_pred = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        km.predict(Z)
    t_kmp = (time.perf_counter() - t0) / 20
    print(f"\nPredict 20,640 rows: patent LUT {t_pred*1e3:.2f} ms | KMeans.predict {t_kmp*1e3:.2f} ms")

    # =============================================================================
    # Figures
    # =============================================================================
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    lon, lat = df["Longitude"].values, df["Latitude"].values

    def geo(ax, lab, title):
        lab = align_to(ref, lab)
        ax.scatter(lon, lat, c=lab, cmap=CMAP, vmin=-0.5, vmax=K - 0.5, s=3, alpha=0.75, linewidths=0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(1.15)
        for s in ax.spines.values(): s.set_visible(False)
        sizes = np.bincount(lab, minlength=K)
        txt = "  ".join(f"C{c}:{sizes[c]}" for c in range(K))
        ax.text(0.0, -0.04, txt, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")

    show = ["KMeans k=5 (standardised)", "KMeans k=5 (uint8 min-max)",
            "Patent minmax, 1 bit (auto)", "Patent minmax, 1 bit (auto) + reassign",
            "Patent robust, 1 bit (auto)", "Patent quantile, 1 bit (auto)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, name in zip(axes.ravel(), show):
        rr = res.set_index("method").loc[name]
        geo(ax, labels_by[name], f"{name}\nsil(std)={rr.silhouette_std:.3f}   "
                                 f"η²(price)={rr.eta2_MedHouseVal:.3f}   ARI={rr.ARI_vs_KMeans:.2f}")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=PALETTE[c], label=f"Cluster {c}")
               for c in range(K)]
    fig.legend(handles=handles, loc="upper center", ncol=K, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("California Housing, k=5 — cluster ids aligned to classical KMeans (Hungarian matching)",
                 y=0.965, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out, "maps_comparison.png"), dpi=130)
    plt.close(fig)

    # --- the fictitious 8-channel image and its label images
    img = images["minmax"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2))
    for c in range(8):
        ax = axes.ravel()[c]
        ax.imshow(img[..., c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"ch{c}: {FEATURES[c]}", fontsize=9, loc="left"); ax.axis("off")
    for ax, name, t in ((axes.ravel()[8], "Patent minmax, 1 bit (auto)", "Patent labels"),
                        (axes.ravel()[9], "KMeans k=5 (standardised)", "KMeans labels")):
        ax.imshow(align_to(ref, labels_by[name]).reshape(img.shape[:2]), cmap=CMAP, vmin=-0.5,
                  vmax=K - 0.5, interpolation="nearest")
        ax.set_title(t, fontsize=9, loc="left"); ax.axis("off")
    fig.suptitle(f"Features as a {img.shape[0]}×{img.shape[1]}×{img.shape[2]} uint8 image (min-max, "
                 "rows in dataset order)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fictitious_image.png"), dpi=130)
    plt.close(fig)

    # --- contingency + cluster profiles
    pat = align_to(ref, labels_by["Patent minmax, 1 bit (auto)"])
    ct = pd.crosstab(pd.Series(ref, name="KMeans"), pd.Series(pat, name="Patent")).reindex(
        index=range(K), columns=range(K), fill_value=0)
    zdf = pd.DataFrame(Z, columns=FEATURES)
    zdf["MedHouseVal"] = (y - y.mean()) / y.std()
    prof_km = zdf.groupby(ref).mean().reindex(range(K))
    prof_pt = zdf.groupby(pat).mean().reindex(range(K))
    vmax = 2.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1.7, 1.7]})
    ax = axes[0]
    ax.imshow(np.log1p(ct.values), cmap="Blues")
    for i in range(K):
        for j in range(K):
            v = ct.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=8,
                    color="white" if np.log1p(v) > 0.6 * np.log1p(ct.values.max()) else INK)
    ax.set_xlabel("Patent cluster (min-max, 1 bit)"); ax.set_ylabel("KMeans cluster")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_title("Contingency table", loc="left", fontsize=10)
    for ax, prof, t in ((axes[1], prof_km, "KMeans cluster profiles"),
                        (axes[2], prof_pt, "Patent cluster profiles")):
        im = ax.imshow(np.clip(prof.values, -vmax, vmax), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(prof.shape[1])); ax.set_xticklabels(prof.columns, rotation=40, ha="right")
        ax.set_yticks(range(K)); ax.set_yticklabels([f"C{c}" for c in range(K)])
        for i in range(K):
            for j in range(prof.shape[1]):
                v = prof.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}", ha="center", va="center",
                            fontsize=7.5, color="white" if abs(v) > 1.3 else INK)
        ax.set_title(t + " (mean z-score; MedHouseVal not used for clustering)", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.03, label="z (clipped ±2)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "profiles_contingency.png"), dpi=130)
    plt.close(fig)
    print(f"\nFigures and metrics written to {args.out}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Rerun Experiments A (robust baselines) and D (200 bootstraps, 999 permutations)
# on the OFFICIAL scikit-learn copy of California Housing, under WSL2.
# Usage: unzip revision_release.zip into a folder, cd into it, then:  bash run_AD_laptop.sh
set -euo pipefail
cd "$(dirname "$0")"

# 0. Make sure the official data are used: with no housing_statlib.csv in the folder,
#    the scripts fall back to sklearn.datasets.fetch_california_housing().
if [ -f housing_statlib.csv ]; then
  mv housing_statlib.csv housing_statlib.csv.unused
  echo "Moved housing_statlib.csv aside so the official sklearn copy is used."
fi
pip install --quiet ucimlrepo matplotlib   # numpy, scipy, scikit-learn, pandas already installed

# 1. Dry Bean (needed by Experiment A), written in the layout the scripts expect
if [ ! -f dry-beans.csv ]; then
  python - <<'EOF'
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=602)
df = ds.data.features.copy()
df["Class"] = ds.data.targets.iloc[:, 0].values
df.to_csv("dry-beans.csv", index=False)
print("dry-beans.csv:", df.shape)
EOF
fi

# 2. Record the environment next to the results
mkdir -p results_laptop
python - > results_laptop/environment.txt <<'EOF'
import os, sys, platform, numpy, scipy, sklearn, pandas
print(platform.platform()); print("python", sys.version.split()[0])
print("numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__, "pandas", pandas.__version__)
print("cpu_count", os.cpu_count())
EOF
cat results_laptop/environment.txt

# 3. Experiment A: timings with 6 threads (one per physical core), nothing else running
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
mkdir -p results
time python exp_a_robust_baselines.py --boot 50 | tee results_laptop/exp_a_log.txt
cp results/exp_a_robust_baselines.csv results_laptop/exp_a_robust_baselines.csv

# 4. Experiment D: 6 methods in parallel, one thread each (about 15-25 min on 6 cores)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time python stats_validation.py --boot 200 --perm 999 --jobs 6 --out results_laptop | tee results_laptop/exp_d_log.txt

# 5. Compare with the server results shipped in the zip (rebuilt StatLib copy)
python - <<'EOF'
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
a_srv = pd.read_csv("exp_a_robust_baselines.csv"); a_lap = pd.read_csv("results_laptop/exp_a_robust_baselines.csv")
cols = ["fit_ms", "effective_k", "sil_Z", "stability_ARI_mean", "eta2_price", "ARI_truth"]
m = a_srv.merge(a_lap, on=["dataset", "method"], suffixes=("_server", "_laptop"))
print("\nExperiment A: server vs laptop")
print(m[["dataset", "method"] + [f"{c}_{s}" for c in cols for s in ("server", "laptop")]].round(3).to_string(index=False))
d_srv = pd.read_csv("stats_validation.csv"); d_lap = pd.read_csv("results_laptop/stats_validation.csv")
cols = ["ARI_boot_mean", "ARI_boot_ci_low", "ARI_boot_min", "sil_real", "sil_null_mean", "margin", "p_perm", "p_holm"]
m = d_srv.merge(d_lap, on="method", suffixes=("_server", "_laptop"))
print("\nExperiment D: server vs laptop")
print(m[["method"] + [f"{c}_{s}" for c in cols for s in ("server", "laptop")]].round(4).to_string(index=False))
EOF
echo "Done. Send me the files in results_laptop/ to update Tables 1b and 2."

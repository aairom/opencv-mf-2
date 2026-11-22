# Revision experiments (response to peer review, 22 November 2026)

Put these files in one folder with `housing_statlib.csv` (optional: without it, California Housing is
loaded with sklearn's fetch_california_housing) and `dry-beans.csv` (UCI Dry Bean, id 602).

| Script | Manuscript | Runtime on 2 cores |
| --- | --- | --- |
| stats_validation.py --boot 200 --perm 999 | Section 6.2, Table 2 | about 45 min |
| exp_a_robust_baselines.py --boot 50 | Section 6.1, Table 1b | about 10 min |
| exp_b_contamination.py | Section 4.5, Table 4.5 | about 5 min |
| exp_c_dimension.py, exp_c2_multiprojection.py | Section 7.5, Table 9 | about 10 min each |
| exp_ward_occupancy_quant.py | Tables 4.3, 4.4, 4.6 | about 2 min |
| verify_c3.py <path to opencv-mf clone> | bit-exact check (needs opencv-python, scikit-image) | seconds |

The CSV files are the results reported in the revised manuscript (server: Intel Xeon 2.1 GHz, 2 cores,
Python 3.13, NumPy 2.5, scikit-learn 1.9; rebuilt StatLib copy of California Housing).
fast_feature_clustering.py now supports keys wider than 64 bits (multiword path).

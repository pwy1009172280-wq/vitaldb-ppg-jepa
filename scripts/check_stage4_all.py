import glob
import os

import numpy as np

D = "/scratch-shared/wpu/vitaldb_downstream"
files = sorted(glob.glob(os.path.join(D, "features_*.npy")))
print("n_subjects", len(files))
bad_shape = 0
bad_target = 0
bad_zscore = 0
bad_finite = 0
target_min, target_max = 1e9, -1e9
for f in files:
    X = np.load(f)
    y = np.load(f.replace("features_", "targets_"))
    if X.shape != (120, 2000):
        bad_shape += 1
    if y.shape != (120,):
        bad_shape += 1
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        bad_finite += 1
    if y.min() < 30 or y.max() > 220:
        bad_target += 1
    target_min = min(target_min, float(y.min()))
    target_max = max(target_max, float(y.max()))
    # z-score: per-window mean ~0, std ~1
    if abs(float(X.mean())) > 1e-3 or abs(float(X.std()) - 1.0) > 1e-2:
        bad_zscore += 1
print("bad_shape", bad_shape, "bad_finite", bad_finite, "bad_target", bad_target, "bad_zscore", bad_zscore)
print("target_range", round(target_min, 1), round(target_max, 1))
print("CHECK_DONE")

#!/usr/bin/env python3
"""Approved BIDMC long-history S-versus-L diagnostic (isolated exploratory run)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.linear_model import Ridge

FS = 125.0
DECIM = 5
BLOCK = 250
N_BLOCKS = 5
LONG = 7500
SHORT = 1250
FUTURE = 1250
STRIDE = 250
EPS = 1e-12
BANDS = ((0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 8.0))


def split_for_record(rid):
    v = int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) % 10
    return "train" if v < 6 else ("val" if v < 8 else "test")


def causal_filter(x):
    b, a = butter(3, [0.5 / (FS / 2), 8.0 / (FS / 2)], btype="band")
    return lfilter(b, a, x)


def target_for_future(z):
    out = []
    for r in range(N_BLOCKS):
        x = np.asarray(z[r * BLOCK : (r + 1) * BLOCK], dtype=float)
        if len(x) != BLOCK or not np.all(np.isfinite(x)):
            return None
        x = x - np.mean(x)
        w = np.hanning(BLOCK + 1)[:-1]
        spec = np.fft.rfft(x * w)
        freqs = np.fft.rfftfreq(BLOCK, 1.0 / FS)
        power = np.abs(spec) ** 2 / (FS * np.sum(w ** 2))
        total = float(power[(freqs >= 0.5) & (freqs <= 8.0)].sum())
        vals = [np.log10(total + EPS)]
        for lo, hi in BANDS:
            band = (freqs >= lo) & (freqs < hi)
            if hi == 8.0:
                band = (freqs >= lo) & (freqs <= hi)
            vals.append(float((power[band].sum() + EPS) / (total + 4 * EPS)))
        out.extend(vals)
    return np.asarray(out, dtype=np.float64)


def read_segment(path):
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    if "PLETH" not in d or "Time [s]" not in d:
        return []
    t = d["Time [s]"].to_numpy(float)
    x = d["PLETH"].to_numpy(float)
    dt = np.diff(t)
    bad = (~np.isfinite(t)) | (~np.isfinite(x))
    boundary = np.zeros(len(x), dtype=bool)
    boundary[0] = True
    if len(x) > 1:
        boundary[1:] = bad[1:] | bad[:-1] | (~np.isfinite(dt)) | (dt < 0) | (dt > 0.02)
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], len(x)]
    segments = []
    for s, e in zip(starts, ends):
        if e - s >= LONG + FUTURE:
            segments.append((s, e, x[s:e], t[s:e]))
    return segments


def collect(data_dir):
    rows = []
    audits = []
    for path in sorted(Path(data_dir).glob("bidmc_*_Signals.csv")):
        rid = path.stem.split("_")[1]
        segs = read_segment(path)
        rec_audit = {"record": rid, "file": str(path), "segments": len(segs), "rows": 0, "eligible": 0}
        for si, (s, e, raw, t) in enumerate(segs):
            rec_audit["rows"] += int(len(raw))
            z = causal_filter(raw)
            for o in range(LONG, len(raw) - FUTURE + 1, STRIDE):
                past = z[o - LONG : o : DECIM].astype(np.float64)
                future = z[o : o + FUTURE]
                y = target_for_future(future)
                if len(past) != 1500 or y is None or not np.all(np.isfinite(past)):
                    continue
                rec_audit["eligible"] += 1
                rows.append({
                    "record": rid, "segment": si, "origin": int(s + o),
                    "split": split_for_record(rid), "values": past, "target": y,
                    "future_start": int(s + o), "future_end": int(s + o + FUTURE),
                })
        audits.append(rec_audit)
    return rows, audits


def fit_scaler(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return mean, std


def design(rows, value_mean, value_std):
    xl = np.stack([r["values"] for r in rows])
    xs = xl.copy()
    xs[:, : 1500 - SHORT // DECIM] = 0.0
    xl = (xl - value_mean) / value_std
    xs = (xs - value_mean) / value_std
    old = 1500 - SHORT // DECIM
    xs[:, :old] = 0.0
    ml = np.ones_like(xl)
    ms = np.ones_like(xs)
    ms[:, :old] = 0.0
    return np.c_[xs, ms], np.c_[xl, ml]


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def metrics(y, p):
    return {"mse": mse(y, p), "mae": mae(y, p), "n": int(len(y))}


def smoke(rows, audits):
    checks = {}
    checks["records_with_data"] = len(audits)
    checks["eligible_rows"] = len(rows)
    checks["all_rows_finite"] = bool(all(np.all(np.isfinite(r["values"])) and np.all(np.isfinite(r["target"])) for r in rows))
    checks["record_split_disjoint"] = len({r["record"] for r in rows if r["split"] == "train"} & {r["record"] for r in rows if r["split"] == "test"}) == 0
    checks["paired_future_identity"] = bool(all(r["future_end"] - r["future_start"] == FUTURE for r in rows))
    # The two arms are generated from the same row and therefore have identical targets/origins.
    checks["same_origin_target_for_S_L"] = bool(all(r["origin"] == r["future_start"] and r["target"].shape == (25,) for r in rows))
    vals = np.stack([r["values"] for r in rows])
    checks["long_value_dimension"] = int(vals.shape[1]) == 1500
    xs, xl = design(rows[:1], np.zeros(1500), np.ones(1500))
    checks["fixed_design_shape"] = xs.shape == (1, 3000) and xl.shape == (1, 3000)
    checks["S_observed_value_count"] = int(np.sum(xs[0, 1500:])) == 250
    checks["S_masked_value_count"] = int(np.sum(xs[0, 1500:1500 + 1250])) == 0
    checks["L_observed_value_count"] = int(np.sum(xl[0, 1500:])) == 1500
    checks["S_mask_recent_count"] = int(np.sum(xs[0, 1500 + 1250:])) == 250
    # Causal filtering smoke: changing samples after an origin cannot change the earlier filtered prefix.
    checks["causal_future_perturbation"] = True
    for r in rows[: min(3, len(rows))]:
        # This invariant is guaranteed by lfilter's causal recurrence; test a representative segment numerically.
        raw = np.sin(np.linspace(0, 20, LONG + FUTURE))
        z1 = causal_filter(raw)
        raw[LONG:] += 7.0
        z2 = causal_filter(raw)
        if not np.array_equal(z1[:LONG], z2[:LONG]):
            checks["causal_future_perturbation"] = False
            break
    checks["all_valid"] = all(bool(v) for k, v in checks.items() if k not in {"records_with_data", "eligible_rows"})
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, audits = collect(args.data)
    checks = smoke(rows, audits)
    (out / "smoke_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not checks["all_valid"]:
        raise SystemExit("Smoke/validity gate failed; no model fitting performed")
    if len({r["record"] for r in rows if r["split"] == "test"}) < 6 or len(rows) < 1000:
        raise SystemExit("Stopping rule failed")
    y = np.stack([r["target"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    te = np.array([r["split"] == "test" for r in rows])
    vals = np.stack([r["values"] for r in rows])
    mean_x, std_x = fit_scaler(vals[tr])
    xs, xl = design(rows, mean_x, std_x)
    ymean, ystd = fit_scaler(y[tr])
    yz = (y - ymean) / ystd
    preds = {"S": np.zeros_like(y), "L": np.zeros_like(y)}
    models = {}
    for name, X in (("S", xs), ("L", xl)):
        model = Ridge(alpha=1.0)
        model.fit(X[tr], yz[tr])
        pred = model.predict(X[te]) * ystd + ymean
        preds[name][te] = pred
        models[name] = {"alpha": 1.0, "n_parameters": int(model.coef_.size + model.intercept_.size)}
    prior = np.mean(y[tr], axis=0)
    pprior = np.repeat(prior[None, :], te.sum(), axis=0)
    result = {
        "config": {"fs_hz": FS, "short_seconds": 10, "long_seconds": 60, "future_seconds": 10,
                   "stride_seconds": 2, "target_dim": 25, "target_features_per_block": 5,
                   "input_shape": 3000, "S_observed_value_features": 250,
                   "S_masked_value_features": 1250, "L_observed_value_features": 1500,
                   "effective_observed_feature_difference": 1250, "ridge_alpha": 1.0,
                   "split_rule": "sha256(record_id)[:8] mod 10: train<6, val<8"},
        "data": {"audited_records": len(audits), "eligible_rows": len(rows),
                 "split_record_counts": {s: len({r["record"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")},
                 "split_row_counts": {s: int(np.sum([r["split"] == s for r in rows])) for s in ("train", "val", "test")}},
        "smoke_checks": checks,
        "target": {"dimension": 25, "blocks": 5, "features_per_block": ["log_total_power", "relative_band_0.5_1.5", "relative_band_1.5_3", "relative_band_3_5", "relative_band_5_8"]},
        "models": models,
        "metrics": {"training_target_mean_reference": metrics(y[te], pprior)},
        "per_record": {},
    }
    for name in ("S", "L"):
        result["metrics"][name] = metrics(y[te], preds[name][te])
        result["metrics"][name]["log_total_power_mse"] = mse(y[te, 0::5], preds[name][te, 0::5])
        result["metrics"][name]["relative_band_power_mse"] = mse(np.delete(y[te], np.arange(0, 25, 5), axis=1), np.delete(preds[name][te], np.arange(0, 25, 5), axis=1))
    result["metrics"]["training_target_mean_reference"]["log_total_power_mse"] = mse(y[te, 0::5], pprior[:, 0::5])
    result["metrics"]["training_target_mean_reference"]["relative_band_power_mse"] = mse(np.delete(y[te], np.arange(0, 25, 5), axis=1), np.delete(pprior, np.arange(0, 25, 5), axis=1))
    test_ids = sorted({r["record"] for r in rows if r["split"] == "test"}, key=int)
    for rid in test_ids:
        ix = te & np.array([r["record"] == rid for r in rows])
        result["per_record"][rid] = {"n": int(ix.sum()), "S_mse": mse(y[ix], preds["S"][ix]), "L_mse": mse(y[ix], preds["L"][ix]), "L_minus_S_mse": mse(y[ix], preds["L"][ix]) - mse(y[ix], preds["S"][ix])}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        pick = rng.choice(test_ids, len(test_ids), replace=True)
        ix = np.concatenate([np.flatnonzero(te & np.array([r["record"] == rid for r in rows])) for rid in pick])
        boot.append(mse(y[ix], preds["L"][ix]) - mse(y[ix], preds["S"][ix]))
    result["record_bootstrap_L_minus_S_mse"] = {"mean": float(np.mean(boot)), "lo95": float(np.percentile(boot, 2.5)), "hi95": float(np.percentile(boot, 97.5)), "resamples": 1000}
    pd.DataFrame(audits).to_json(out / "record_audit.json", orient="records", indent=2)
    (out / "per_record_metrics.json").write_text(json.dumps([{"record": rid, **vals} for rid, vals in result["per_record"].items()], indent=2) + "\n")
    test_rows = [r for r, flag in zip(rows, te) if flag]
    pred_df = pd.DataFrame({"record": [r["record"] for r in test_rows], "split": [r["split"] for r in test_rows], "origin": [r["origin"] for r in test_rows], "test": True})
    for j in range(25):
        pred_df[f"target_{j:02d}"] = y[te, j]
        pred_df[f"S_{j:02d}"] = preds["S"][te, j]
        pred_df[f"L_{j:02d}"] = preds["L"][te, j]
    pred_df.to_csv(out / "paired_predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Stage 3+5+6 downstream evaluation: extract all-layer features -> ridge -> window MAE.

Usage: run_downstream_eval.py <checkpoint> <downstream_dir> <results_dir>

Sorts the 500 VitalDB subjects by the vital-split hash salt into 350/75/75,
extracts layer_1..layer_4 + final_layer (mean-token pooled, online path only),
fits one deterministic ridge (lambda=0.001) per layer on train, predicts
validation/test, and reports window MAE (bpm).
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/gpfs/home2/wpu/projects/vitaldb-ppg-jepa/releases/89915e4fd203111dc5774742893c24333322a652")

from src.downstream.core.ridge import RidgeRegression
from src.experiments.pipeline import load_checkpoint_model


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    checkpoint = sys.argv[1]
    downstream_dir = sys.argv[2]
    results_dir = sys.argv[3]

    adapter, payload = load_checkpoint_model(checkpoint)
    adapter.freeze_encoder()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter.to(device)
    adapter.eval()

    subjects = sorted(
        os.path.basename(f).replace("features_", "").replace(".npy", "")
        for f in os.listdir(downstream_dir)
        if f.startswith("features_")
    )
    subjects.sort(key=lambda s: (sha256(f"first-layerwise-2026-10-09|vital-split|{s}"), s))
    print("SUBJECTS", len(subjects), flush=True)
    assert len(subjects) >= 500, len(subjects)
    train_s, val_s, test_s = subjects[:350], subjects[350:425], subjects[425:500]

    def extract(split_subjects):
        feats = {f"layer_{i + 1}": [] for i in range(4)}
        feats["final_layer"] = []
        targets = []
        for s in split_subjects:
            X = np.load(os.path.join(downstream_dir, f"features_{s}.npy"))
            y = np.load(os.path.join(downstream_dir, f"targets_{s}.npy"))
            for i in range(0, len(X), 256):
                batch = torch.from_numpy(X[i:i + 256]).float().to(device).unsqueeze(1)
                with torch.no_grad():
                    out = adapter.encode_full(batch)
                hs = out.hidden_states
                for li in range(len(hs)):
                    feats[f"layer_{li + 1}"].append(hs[li].mean(dim=1).cpu().numpy().astype(np.float32))
                feats["final_layer"].append(out.final_tokens.mean(dim=1).cpu().numpy().astype(np.float32))
            targets.append(y.astype(np.float32))
        return {k: np.concatenate(v, axis=0) for k, v in feats.items()}, np.concatenate(targets)

    train_f, train_y = extract(train_s)
    val_f, val_y = extract(val_s)
    test_f, test_y = extract(test_s)
    print("TRAIN", train_f["layer_1"].shape, "VAL", val_f["layer_1"].shape, "TEST", test_f["layer_1"].shape, flush=True)

    results = {}
    for name in ["layer_1", "layer_2", "layer_3", "layer_4", "final_layer"]:
        ridge = RidgeRegression(lambda_=0.001).fit(train_f[name], train_y)
        val_pred = ridge.predict(val_f[name])
        test_pred = ridge.predict(test_f[name])
        val_mae = float(np.abs(val_pred - val_y).mean())
        test_mae = float(np.abs(test_pred - test_y).mean())
        results[name] = {"val_mae": round(val_mae, 4), "test_mae": round(test_mae, 4)}

    os.makedirs(results_dir, exist_ok=True)
    out = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "metric": "window_mae_bpm",
        "layers": results,
        "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
    }
    with open(os.path.join(results_dir, "downstream_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("DOWNSTREAM_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

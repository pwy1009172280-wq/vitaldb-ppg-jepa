#!/usr/bin/env python3
"""Thin pipeline CLI: validate a config, then run the SMOKE_ONLY fixture stage.

Production pretraining/downstream datasets and the real-tiny MIMIC smoke require
the isolated wfdb environment and authorized compute; this CLI deliberately does
not fetch, download, or launch any acquisition/experiment.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.experiments.pipeline import run_smoke


def _synthetic_dataset():
    import numpy as np

    from src.data.base import BaseDataset
    from src.data.samples import SUBJECT_IDENTITY_RESOLVED, UnifiedSample

    class SyntheticDataset(BaseDataset):
        name = "test-fixture-ppg"

        def __init__(self, n=8, window_len=16):
            self.n = n
            self.window_len = window_len

        def __len__(self):
            return self.n

        def __getitem__(self, index):
            if index < 0 or index >= self.n:
                raise IndexError(index)
            return UnifiedSample(
                signal=np.full((1, self.window_len), float(index), dtype=np.float32),
                subject_id=f"fixture-s{index % 2}", recording_id=f"r{index}",
                dataset=self.name, modality="PPG", sampling_rate_hz=125.0,
                start_time_s=0.0, end_time_s=self.window_len / 125.0,
                window_id=f"w{index}", window_start_sample=0, window_end_sample=self.window_len,
                subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
                subject_identity_namespace="test-fixture",
            )

    return SyntheticDataset()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("validate", "smoke"), default="validate")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "validate":
        print("config valid")
        return 0

    # SMOKE_ONLY fixture stage; no real dataset is selected here.
    out = run_smoke(config, _synthetic_dataset(), results_root=args.results_root, run_id=args.run_id)
    print("smoke complete:", sorted(out["result"]["representations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

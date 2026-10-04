"""Open a few real source-native records and record a small smoke result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ptb-xl", "ppg-dalia"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.dataset == "ptb-xl":
        from src.datasets.ptb_xl import PTBXLReader

        reader = PTBXLReader(args.root)
        indices = (0, next(i for i, row in enumerate(reader.rows) if row.subject_id != reader.rows[0].subject_id))
    else:
        from src.datasets.ppg_dalia import PPGDaLiAReader

        reader = PPGDaLiAReader(args.root)
        indices = (0,)
    samples = [reader[index] for index in indices]
    if not samples or any(sample.signal.ndim != 2 for sample in samples):
        raise RuntimeError("reader did not return channel x time signals")
    if len({sample.subject_id for sample in samples}) != len(samples) and args.dataset == "ptb-xl":
        raise RuntimeError("real-record smoke subjects collapsed")
    payload = {
        "status": "PASS",
        "dataset": args.dataset,
        "reader": type(reader).__name__,
        "reader_version": "1.0",
        "sample_count": len(samples),
        "subjects": [sample.subject_id for sample in samples],
        "recordings": [sample.recording_id for sample in samples],
        "shapes": [list(sample.signal.shape) for sample in samples],
        "sampling_rates_hz": [sample.sampling_rate_hz for sample in samples],
        "preprocessing_applied": False,
    }
    output = args.root / "metadata" / "reader_smoke.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

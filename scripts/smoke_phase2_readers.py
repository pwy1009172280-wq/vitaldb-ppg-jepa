"""Read one real source-native record per approved Phase 2 dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.index import read_jsonl, sha256_file
from src.datasets.phase2 import (
    CPSC2018Reader, ChapmanShaoxingReader, GeorgiaReader, LUDBReader,
    MITBIHReader, PPG_BPReader, WESADReader,
)


ROOT = Path("/projects/prjs2287/biosignal_bank/datasets")
READERS = {
    "ludb": LUDBReader,
    "mit-bih": MITBIHReader,
    "chapman-shaoxing": ChapmanShaoxingReader,
    "georgia": GeorgiaReader,
    "cpsc2018": CPSC2018Reader,
    "ppg-bp": PPG_BPReader,
    "wesad": WESADReader,
}


def main() -> None:
    for dataset, reader_cls in READERS.items():
        root = ROOT / dataset
        index = root / "metadata" / "records.jsonl"
        t0 = time.perf_counter()
        rows = tuple(read_jsonl(index))
        load_s = time.perf_counter() - t0
        reader = reader_cls(root)
        first = reader[0]
        repeated = reader[0]
        if (first.subject_id, first.recording_id) != (repeated.subject_id, repeated.recording_id):
            raise RuntimeError(f"unstable identity: {dataset}")
        payload = {
            "status": "PASS",
            "dataset": dataset,
            "reader": reader_cls.__name__,
            "reader_version": "1.0",
            "sample_count": 1,
            "subjects": [first.subject_id],
            "recordings": [first.recording_id],
            "shapes": [list(first.signal.shape)],
            "sampling_rates_hz": [first.sampling_rate_hz],
            "preprocessing_applied": False,
            "index_rows": len(rows),
            "index_bytes": index.stat().st_size,
            "index_sha256": sha256_file(index),
            "index_load_seconds": load_s,
        }
        (root / "metadata" / "reader_smoke.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

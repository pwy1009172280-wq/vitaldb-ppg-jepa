"""Build metadata-only Phase 2 indexes; source data is never rewritten."""

from pathlib import Path

from src.data.index import sha256_file
from src.datasets.phase2 import build_ecg_index, build_ppg_bp_index, build_wesad_index


ROOT = Path("/projects/prjs2287/biosignal_bank/datasets")


def main() -> None:
    specs = [
        ("ludb", "1.0.1", "incoming/ludb-1.0.1/data"),
        ("mit-bih", "1.0.0", "incoming/mitdb-1.0.0"),
        ("chapman-shaoxing", "1.0.0", "incoming/ecg-arrhythmia-1.0.0/WFDBRecords"),
        ("georgia", "1.0.2", "incoming/challenge-2020-georgia-1.0.2/physionet.org/files/challenge-2020/1.0.2/training/georgia"),
        ("cpsc2018", "1.0.2", "incoming/challenge-2020-cpsc2018-1.0.2/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018"),
    ]
    for dataset, version, subdir in specs:
        path = build_ecg_index(dataset, ROOT / dataset, version, subdir)
        print(dataset, sum(1 for _ in path.open()), sha256_file(path))
    for dataset, builder in (("ppg-bp", build_ppg_bp_index), ("wesad", build_wesad_index)):
        path = builder(ROOT / dataset)
        print(dataset, sum(1 for _ in path.open()), sha256_file(path))


if __name__ == "__main__":
    main()

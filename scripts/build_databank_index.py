"""Build source-native canonical record indexes without touching raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.ppg_dalia import build_index as build_ppg_dalia_index
from src.datasets.ptb_xl import build_index as build_ptb_xl_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ptb-xl", "ppg-dalia"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.dataset == "ptb-xl":
        index = build_ptb_xl_index(args.root)
        from src.datasets.ptb_xl import PTBXLReader
        reader = PTBXLReader(args.root, index)
    else:
        index = build_ppg_dalia_index(args.root)
        from src.datasets.ppg_dalia import PPGDaLiAReader
        reader = PPGDaLiAReader(args.root, index)
    print(f"dataset={args.dataset}")
    print(f"index={index}")
    print(f"rows={len(reader)}")
    print(f"subjects={len(reader.subject_ids())}")


if __name__ == "__main__":
    main()

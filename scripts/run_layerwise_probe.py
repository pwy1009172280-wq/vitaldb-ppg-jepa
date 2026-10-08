#!/usr/bin/env python3
"""LEGACY layerwise probe path; not for new experiments."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.downstream.layerwise import (
    current_git_commit, join_labels, load_feature_bundle, read_group_split,
    read_labels, run_layerwise_ridge, save_layerwise_results,
)


def main():
    parser = argparse.ArgumentParser(description='Run fixed-alpha layer-wise Ridge diagnostics.')
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--missing', choices=('drop', 'error'), default='drop')
    args = parser.parse_args()
    bundle = load_feature_bundle(args.features)
    labels = read_labels(args.labels, args.target)
    joined = join_labels(bundle, labels, args.target, missing=args.missing)
    split = read_group_split(args.split)
    result = run_layerwise_ridge(joined, split, alpha=args.alpha)
    csv_path, json_path = save_layerwise_results(
        result, args.output, checkpoint=bundle.metadata.get('checkpoint'), git_commit=current_git_commit())
    print(csv_path); print(json_path)


if __name__ == '__main__':
    main()

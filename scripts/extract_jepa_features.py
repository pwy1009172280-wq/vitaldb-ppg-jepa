#!/usr/bin/env python3
"""LEGACY feature extraction path; use scripts/run_downstream.py for new experiments."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.processed_dataset import ProcessedPPGDataset
from src.downstream.layerwise import extract_features, load_frozen_jepa_checkpoint, save_feature_bundle


def main():
    parser = argparse.ArgumentParser(description='Extract frozen full-sequence JEPA layer features.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--processed-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    model, _, payload = load_frozen_jepa_checkpoint(args.checkpoint, args.device)
    dataset = ProcessedPPGDataset(args.manifest, args.processed_root)
    bundle = extract_features(model, dataset, batch_size=args.batch_size, device=args.device,
                              num_workers=args.num_workers, checkpoint=str(Path(args.checkpoint).resolve()))
    bundle.metadata['resolved_config'] = payload['resolved_config']
    print(save_feature_bundle(bundle, args.output))


if __name__ == '__main__':
    main()

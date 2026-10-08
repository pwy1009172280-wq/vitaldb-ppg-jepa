"""LEGACY/COMPATIBILITY PPG/JEPA layer-wise diagnostics; do not extend.

This module intentionally separates frozen feature extraction from labels and
from the regression probe. It remains import-compatible while new downstream
work belongs in ``src.downstream.core``, ``readers``, and ``runners``.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..data.processed_dataset import ProcessedPPGDataset
from ..models.factory import build_model
from ..pretrain.config import (
    Data2VecConfig, DataConfig, JEPAConfig, MAEConfig, ModelConfig,
    SSLConfig, TrainConfig,
)


@dataclass
class FeatureBundle:
    layer_names: tuple[str, ...]
    features: dict[str, np.ndarray]
    caseid: np.ndarray
    tid: np.ndarray
    window_index: np.ndarray
    metadata: dict


@dataclass
class JoinedLabels:
    bundle: FeatureBundle
    target: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class GroupSplit:
    case_to_split: dict[str, str]
    identity: str

    def indices(self, caseids: np.ndarray) -> dict[str, np.ndarray]:
        labels = np.asarray([self.case_to_split.get(str(c), '') for c in caseids])
        missing = sorted({str(c) for c, s in zip(caseids, labels) if not s})
        if missing:
            raise ValueError(f'caseids missing from explicit split: {missing}')
        return {name: np.flatnonzero(labels == name) for name in ('train', 'val', 'test')}


def mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool only across tokens: [B, N, D] -> [B, D]."""
    if tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError(f'expected non-empty [B, N_tokens, D], got {tuple(tokens.shape)}')
    return tokens.mean(dim=1)


def _config_from_checkpoint(resolved: dict) -> SSLConfig:
    if not isinstance(resolved, dict):
        raise ValueError('checkpoint resolved_config must be a mapping')
    if resolved.get('method') != 'jepa':
        raise ValueError(f"checkpoint method must be 'jepa', got {resolved.get('method')!r}")
    try:
        return SSLConfig(
            method='jepa',
            model=ModelConfig(**resolved['model']),
            data=DataConfig(**resolved['data']),
            train=TrainConfig(**resolved['train']),
            mae=MAEConfig(**resolved.get('mae', {})),
            data2vec=Data2VecConfig(**resolved.get('data2vec', {})),
            jepa=JEPAConfig(**resolved['jepa']),
        ).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid or incomplete JEPA resolved_config: {exc}') from exc


def load_frozen_jepa_checkpoint(checkpoint: str | Path, device: str | torch.device = 'cpu'):
    """Reconstruct, strictly load, eval, and freeze a JEPA checkpoint."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError('checkpoint must contain a mapping')
    if payload.get('format_version') != 1:
        raise ValueError(f"unsupported checkpoint format_version: {payload.get('format_version')!r}")
    if 'model' not in payload or 'resolved_config' not in payload:
        raise ValueError('checkpoint must contain model and resolved_config')
    cfg = _config_from_checkpoint(payload['resolved_config'])
    model = build_model(cfg)
    try:
        model.load_state_dict(payload['model'], strict=True)
    except RuntimeError as exc:
        raise ValueError(f'checkpoint model state_dict does not match reconstructed JEPA: {exc}') from exc
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, payload


def _metadata_values(batch, key: str) -> list[str]:
    values = batch[key]
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    return [str(value) for value in values]


@torch.no_grad()
def extract_features(model, dataset, *, batch_size: int = 32, device: str | torch.device = 'cpu', num_workers: int = 0, checkpoint: str | None = None) -> FeatureBundle:
    """Extract pooled block and final representations in batches."""
    if batch_size <= 0:
        raise ValueError('batch_size must be > 0')
    model.eval()
    device = torch.device(device)
    layer_chunks: dict[str, list[np.ndarray]] = {}
    caseids: list[str] = []
    tids: list[str] = []
    windows: list[int] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in loader:
        waveform = batch['waveform'].to(device)
        representation = model.encode_full(waveform)
        tensors = {f'block_{i + 1}': state for i, state in enumerate(representation.hidden_states)}
        tensors['final'] = representation.final_tokens
        for name, tensor in tensors.items():
            pooled = mean_pool_tokens(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
            layer_chunks.setdefault(name, []).append(pooled)
        caseids.extend(_metadata_values(batch, 'caseid'))
        tids.extend(_metadata_values(batch, 'tid'))
        raw_windows = batch['window_index']
        windows.extend(int(value) for value in (raw_windows.tolist() if isinstance(raw_windows, torch.Tensor) else raw_windows))
    layer_names = tuple(layer_chunks)
    features = {name: np.concatenate(chunks, axis=0) for name, chunks in layer_chunks.items()}
    n = len(caseids)
    if any(array.shape[0] != n for array in features.values()):
        raise RuntimeError('feature and metadata row counts differ')
    return FeatureBundle(
        layer_names, features, np.asarray(caseids, dtype=str), np.asarray(tids, dtype=str),
        np.asarray(windows, dtype=np.int64),
        {'checkpoint': checkpoint, 'pooling': 'mean_tokens', 'n_samples': n,
         'feature_dtype': 'float32'},
    )


def save_feature_bundle(bundle: FeatureBundle, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'layer_names': np.asarray(bundle.layer_names, dtype=str),
        'caseid': bundle.caseid,
        'tid': bundle.tid,
        'window_index': bundle.window_index,
        'metadata_json': np.asarray(json.dumps(bundle.metadata, sort_keys=True)),
    }
    payload.update(bundle.features)
    np.savez_compressed(output, **payload)
    return output


def load_feature_bundle(path: str | Path) -> FeatureBundle:
    with np.load(path, allow_pickle=False) as data:
        layer_names = tuple(str(value) for value in data['layer_names'].tolist())
        features = {name: np.asarray(data[name]) for name in layer_names}
        bundle = FeatureBundle(
            layer_names, features, np.asarray(data['caseid'], dtype=str),
            np.asarray(data['tid'], dtype=str), np.asarray(data['window_index'], dtype=np.int64),
            json.loads(str(data['metadata_json'].item())),
        )
    n = len(bundle.caseid)
    if len(bundle.tid) != n or len(bundle.window_index) != n or any(v.shape[0] != n for v in bundle.features.values()):
        raise ValueError('feature bundle metadata and feature row counts do not match')
    return bundle


def read_labels(path: str | Path, target: str) -> dict[tuple[str, str, int], float]:
    required = {'caseid', 'tid', 'window_index', target}
    labels: dict[tuple[str, str, int], float] = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'label table missing columns: {sorted(missing)}')
        for row_number, row in enumerate(reader, 2):
            key = (str(row['caseid']), str(row['tid']), int(row['window_index']))
            if key in labels:
                raise ValueError(f'duplicate label key at row {row_number}: {key}')
            labels[key] = float(row[target])
    return labels


def join_labels(bundle: FeatureBundle, labels: dict[tuple[str, str, int], float], target: str, *, missing: str = 'drop') -> JoinedLabels:
    if missing not in ('drop', 'error'):
        raise ValueError("missing must be 'drop' or 'error'")
    keys = list(zip(bundle.caseid, bundle.tid, bundle.window_index.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError('feature bundle contains duplicate sample keys')
    keep = np.asarray([key in labels for key in keys], dtype=bool)
    missing_count = int((~keep).sum())
    if missing == 'error' and missing_count:
        raise ValueError(f'{missing_count} feature rows have no matching label')
    selected = np.flatnonzero(keep)
    joined = FeatureBundle(
        bundle.layer_names, {name: values[selected] for name, values in bundle.features.items()},
        bundle.caseid[selected], bundle.tid[selected], bundle.window_index[selected], dict(bundle.metadata),
    )
    values = np.asarray([labels[key] for key in (keys[index] for index in selected)], dtype=np.float64)
    metadata = {'target': target, 'n_features': len(keys), 'n_labels': len(labels),
                'n_joined': len(selected), 'n_missing_labels': missing_count, 'missing_policy': missing}
    return JoinedLabels(joined, values, metadata)


def read_group_split(path: str | Path) -> GroupSplit:
    mapping: dict[str, str] = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'caseid', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('split file must contain caseid and split columns')
        for row_number, row in enumerate(reader, 2):
            caseid, split = str(row['caseid']), str(row['split'])
            if split not in {'train', 'val', 'test'}:
                raise ValueError(f'invalid split {split!r} at row {row_number}')
            if caseid in mapping:
                raise ValueError(f'duplicate caseid in split file: {caseid}')
            mapping[caseid] = split
    if not mapping:
        raise ValueError('split file is empty')
    groups = {name: {case for case, split in mapping.items() if split == name} for name in ('train', 'val', 'test')}
    if groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test']:
        raise ValueError('caseid overlap across explicit splits')
    return GroupSplit(mapping, str(Path(path)))


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {'mae': None, 'rmse': None, 'r2': None}
    result: dict[str, float | None] = {
        'mae': float(mean_absolute_error(y_true, prediction)),
        'rmse': float(math.sqrt(mean_squared_error(y_true, prediction))),
        'r2': None,
    }
    if len(y_true) >= 2 and np.ptp(y_true) > 0:
        result['r2'] = float(r2_score(y_true, prediction))
    return result


def run_layerwise_ridge(joined: JoinedLabels, split: GroupSplit, *, alpha: float = 1.0) -> dict:
    if alpha <= 0:
        raise ValueError('Ridge alpha must be > 0')
    indices = split.indices(joined.bundle.caseid)
    if not len(indices['train']):
        raise ValueError('explicit split has no training samples')
    rows = []
    for layer in joined.bundle.layer_names:
        x = joined.bundle.features[layer]
        scaler = StandardScaler().fit(x[indices['train']])
        probe = Ridge(alpha=alpha).fit(scaler.transform(x[indices['train']]), joined.target[indices['train']])
        predictions = {name: probe.predict(scaler.transform(x[idx])) if len(idx) else np.empty(0) for name, idx in indices.items()}
        val_metrics = _metrics(joined.target[indices['val']], predictions['val'])
        test_metrics = _metrics(joined.target[indices['test']], predictions['test'])
        rows.append({'layer': layer, 'n_train': int(len(indices['train'])), 'n_val': int(len(indices['val'])),
                     'n_test': int(len(indices['test'])), 'mae': test_metrics['mae'],
                     'rmse': test_metrics['rmse'], 'r2': test_metrics['r2'],
                     'val_mae': val_metrics['mae'], 'val_rmse': val_metrics['rmse'], 'val_r2': val_metrics['r2']})
    return {'rows': rows, 'metadata': {'alpha': alpha, 'scaler': 'StandardScaler_fit_on_train_only',
            'target': joined.metadata['target'], 'split': split.identity, 'join': joined.metadata,
            'pooling': joined.bundle.metadata.get('pooling'),
            'feature_metadata': joined.bundle.metadata}}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_layerwise_results(result: dict, output_dir: str | Path, *, checkpoint: str | None = None, git_commit: str | None = None) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(result['metadata'])
    metadata.update({'checkpoint': checkpoint, 'git_commit': git_commit, 'timestamp': datetime.now(timezone.utc).isoformat()})
    rows = result['rows']
    fields = ['layer', 'mae', 'rmse', 'r2', 'n_train', 'n_val', 'n_test', 'val_mae', 'val_rmse', 'val_r2']
    csv_path = output_dir / 'layerwise_results.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    json_path = output_dir / 'layerwise_results.json'
    json_path.write_text(json.dumps(_json_safe({'metadata': metadata, 'rows': rows}), indent=2, sort_keys=True))
    return csv_path, json_path


def current_git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

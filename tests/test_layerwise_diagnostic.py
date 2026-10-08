import csv
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.downstream.layerwise import (
    FeatureBundle,
    extract_features,
    join_labels,
    load_feature_bundle,
    load_frozen_jepa_checkpoint,
    mean_pool_tokens,
    read_group_split,
    read_labels,
    run_layerwise_ridge,
    save_feature_bundle,
)
from src.models.common import BackboneConfig
from src.models.factory import build_model
from src.pretrain.config import (
    Data2VecConfig, DataConfig, JEPAConfig, MAEConfig, ModelConfig,
    SSLConfig, TrainConfig,
)


class TinyPPG(Dataset):
    def __init__(self):
        self.rows = []
        for case_index in range(6):
            for window_index in range(2):
                self.rows.append((f'case-{case_index}', f'tid-{case_index}', window_index,
                                  torch.randn(1, 8)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        caseid, tid, window_index, waveform = self.rows[index]
        return {'caseid': caseid, 'tid': tid, 'window_index': window_index, 'waveform': waveform}


def tiny_config():
    return SSLConfig(
        method='jepa', model=ModelConfig(patch_size=4, patch_stride=4, embed_dim=8, depth=2,
        num_heads=2, mlp_ratio=2., dropout=0.), data=DataConfig(input_length=8),
        train=TrainConfig(), mae=MAEConfig(), data2vec=Data2VecConfig(),
        jepa=JEPAConfig(predictor_dim=8, predictor_depth=1, predictor_num_heads=2),
    ).validate()


def checkpoint_payload(model, config):
    return {'format_version': 1, 'method': 'jepa', 'model': model.state_dict(),
            'resolved_config': {'method': 'jepa', 'model': asdict(config.model),
            'data': asdict(config.data), 'train': asdict(config.train), 'mae': asdict(config.mae),
            'data2vec': asdict(config.data2vec), 'jepa': asdict(config.jepa)}}


def write_labels(path, bundle, values, *, duplicate=False):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['caseid', 'tid', 'window_index', 'target'])
        for index, value in enumerate(values):
            writer.writerow([bundle.caseid[index], bundle.tid[index], bundle.window_index[index], value])
        if duplicate:
            writer.writerow([bundle.caseid[0], bundle.tid[0], bundle.window_index[0], values[0]])


def write_split(path):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['caseid', 'split'])
        for case_index in range(6):
            writer.writerow([f'case-{case_index}', 'train' if case_index < 2 else 'val' if case_index == 2 else 'test'])


def test_mean_pooling_and_frozen_checkpoint_extraction(tmp_path):
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(mean_pool_tokens(tokens), tokens.mean(dim=1))
    config = tiny_config(); source = build_model(config)
    checkpoint = tmp_path / 'checkpoint.pt'; torch.save(checkpoint_payload(source, config), checkpoint)
    model, loaded_config, _ = load_frozen_jepa_checkpoint(checkpoint)
    assert loaded_config.method == 'jepa' and not any(parameter.requires_grad for parameter in model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    bundle = extract_features(model, TinyPPG(), batch_size=3, checkpoint=str(checkpoint))
    assert bundle.layer_names == ('block_1', 'block_2', 'final')
    assert all(bundle.features[name].shape == (12, 8) for name in bundle.layer_names)
    assert bundle.caseid[0] == 'case-0' and bundle.tid[0] == 'tid-0' and bundle.window_index[0] == 0
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(not parameter.requires_grad for parameter in model.parameters())
    saved = save_feature_bundle(bundle, tmp_path / 'features.npz')
    loaded = load_feature_bundle(saved)
    assert loaded.layer_names == bundle.layer_names and np.array_equal(loaded.caseid, bundle.caseid)
    assert all(np.array_equal(loaded.features[name], bundle.features[name]) for name in bundle.layer_names)


def test_label_join_split_and_ridge_end_to_end(tmp_path):
    config = tiny_config(); source = build_model(config)
    checkpoint = tmp_path / 'checkpoint.pt'; torch.save(checkpoint_payload(source, config), checkpoint)
    model, _, _ = load_frozen_jepa_checkpoint(checkpoint)
    bundle = extract_features(model, TinyPPG(), batch_size=4)
    labels_path = tmp_path / 'labels.csv'; values = np.arange(len(bundle.caseid), dtype=float)
    write_labels(labels_path, bundle, values)
    split_path = tmp_path / 'split.csv'; write_split(split_path)
    joined = join_labels(bundle, read_labels(labels_path, 'target'), 'target')
    split = read_group_split(split_path); result = run_layerwise_ridge(joined, split, alpha=1.0)
    assert [row['layer'] for row in result['rows']] == ['block_1', 'block_2', 'final']
    assert all(row['n_train'] == 4 and row['n_val'] == 2 and row['n_test'] == 6 for row in result['rows'])
    assert all(np.isfinite(row['mae']) and np.isfinite(row['rmse']) and np.isfinite(row['r2']) for row in result['rows'])


def test_label_alignment_missing_and_duplicate_are_explicit(tmp_path):
    bundle = FeatureBundle(('block_1',), {'block_1': np.ones((2, 3), dtype=np.float32)},
                           np.array(['a', 'b']), np.array(['t', 't']), np.array([0, 0]), {})
    labels = tmp_path / 'labels.csv'; write_labels(labels, bundle, [1., 2.], duplicate=True)
    try:
        read_labels(labels, 'target')
        assert False, 'duplicate label key should fail'
    except ValueError as exc:
        assert 'duplicate' in str(exc)
    labels.write_text('caseid,tid,window_index,target\na,t,0,1\n')
    joined = join_labels(bundle, read_labels(labels, 'target'), 'target')
    assert joined.metadata['n_missing_labels'] == 1 and joined.metadata['n_joined'] == 1


def test_group_split_overlap_and_informative_layer_detection(tmp_path):
    overlap = tmp_path / 'overlap.csv'; overlap.write_text('caseid,split\na,train\na,test\n')
    try:
        read_group_split(overlap)
        assert False, 'overlapping caseid should fail'
    except ValueError as exc:
        assert 'duplicate caseid' in str(exc)
    rng = np.random.default_rng(7); n = 30; target = rng.normal(size=n)
    caseids = np.asarray([f'case-{i // 10}' for i in range(n)])
    bundle = FeatureBundle(('layer_A', 'layer_B'),
        {'layer_A': (target[:, None] + .01 * rng.normal(size=(n, 4))).astype(np.float32),
         'layer_B': rng.normal(size=(n, 4)).astype(np.float32)},
        caseids, np.asarray(['tid'] * n), np.arange(n), {})
    labels = {(str(caseids[i]), 'tid', int(i)): float(target[i]) for i in range(n)}
    split = tmp_path / 'split.csv'; split.write_text('caseid,split\ncase-0,train\ncase-1,val\ncase-2,test\n')
    result = run_layerwise_ridge(join_labels(bundle, labels, 'target'), read_group_split(split), alpha=1.0)
    rows = {row['layer']: row for row in result['rows']}
    assert rows['layer_A']['r2'] > rows['layer_B']['r2']

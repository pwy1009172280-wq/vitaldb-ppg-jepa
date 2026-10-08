"""Feature cache atomic write (A12): no temp residue, reload intact."""

import numpy as np

from src.downstream.core import (
    ExtractedFeature,
    FeatureCacheKey,
    load_feature_cache,
    save_feature_cache,
)


def test_cache_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    feature = ExtractedFeature(
        "rep",
        np.zeros((2, 3), dtype=np.float32),
        ({"subject_id": "s1"}, {"subject_id": "s2"}),
        "train",
        "reader", "1", "mean", "ckpt",
    )
    key = FeatureCacheKey("d", "s", "p", "c", "reader", "1", "rep", "mean", "1")
    path = tmp_path / "features.npz"
    save_feature_cache(feature, path, key)
    assert path.exists()
    assert list(tmp_path.glob(".feature.*.tmp")) == []
    loaded = load_feature_cache(path, key)
    assert np.array_equal(loaded.values, feature.values)

"""Downstream entry points; legacy diagnostic exports remain compatible.

Importing dataset-agnostic layer_probe must not eagerly import the legacy
PPG adapter, SSL model factory, or scikit-learn diagnostic implementation.
"""

from importlib import import_module

__all__ = [
    'FeatureBundle', 'JoinedLabels', 'GroupSplit', 'extract_features',
    'join_labels', 'load_feature_bundle', 'load_frozen_jepa_checkpoint',
    'mean_pool_tokens', 'read_group_split', 'read_labels',
    'run_layerwise_ridge', 'save_feature_bundle', 'save_layerwise_results',
]


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".layerwise", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

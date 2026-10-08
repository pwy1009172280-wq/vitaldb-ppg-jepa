"""Compatibility facade for the original phase-one layer-probe prototype.

New downstream work belongs in ``src.downstream.core``, ``readers``, and
``runners``. Existing exports remain available for old tests and scripts.
"""

from .config import (
    DownstreamConfig, EncoderConfig, ExperimentConfig, LayerProbeConfig, ProbeConfig,
    load_layer_probe_config,
)
from .experiment import run_layer_probe
from .features import (
    ExtractedFeatures, FeatureBatch, collate_feature_samples, extract_layer_features,
    load_extracted_features, load_frozen_encoder, read_encoder_depths, save_extracted_features,
)
from .probes import LinearProbe, evaluate_probe, feature_batches

__all__ = [
    "DownstreamConfig", "EncoderConfig", "ExperimentConfig", "LayerProbeConfig", "ProbeConfig",
    "load_layer_probe_config", "run_layer_probe", "ExtractedFeatures", "FeatureBatch",
    "collate_feature_samples", "extract_layer_features", "load_extracted_features",
    "load_frozen_encoder", "read_encoder_depths", "save_extracted_features", "LinearProbe",
    "evaluate_probe", "feature_batches",
]
"""LEGACY/COMPATIBILITY layer-probe package; use downstream.core for new runs."""

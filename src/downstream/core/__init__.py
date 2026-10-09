"""Model-agnostic downstream representation and probe contracts."""

from .aggregate import aggregate_predictions
from .cache import (
    CACHE_SCHEMA_VERSION,
    FeatureCacheKey,
    checkpoint_content_hash,
    compute_sample_set_hash,
    load_feature_cache,
    save_feature_cache,
)
from .encoder import Representation, freeze_encoder
from .extraction import FeatureBatch, ExtractedFeature, collate_feature_samples, extract_features
from .metrics import evaluate_predictions
from .probe import (
    ClassVocabulary,
    FeatureNormalizer,
    LinearHead,
    PredictionBatch,
    ProbeModel,
    make_prediction_batch,
)
from .ridge import RidgeRegression
from .protocol import DownstreamPolicy
from .results import load_results, save_results
from .readiness import (
    DATA_BLOCKED,
    DATA_READY_FOR_EVALUATION,
    PROTOCOL_BLOCKED,
    READY_FOR_EVALUATION,
    EvaluationReadiness,
    assess_evaluation_readiness,
)
from .tasks import LabelTaskAdapter, TaskAdapter, TaskTarget, adapt_targets

__all__ = [
    "CACHE_SCHEMA_VERSION", "FeatureCacheKey", "checkpoint_content_hash",
    "compute_sample_set_hash", "load_feature_cache", "save_feature_cache",
    "Representation", "freeze_encoder", "FeatureBatch", "ExtractedFeature",
    "collate_feature_samples", "extract_features", "evaluate_predictions",
    "ClassVocabulary", "FeatureNormalizer", "LinearHead", "PredictionBatch",
    "ProbeModel", "make_prediction_batch", "RidgeRegression", "DownstreamPolicy",
    "load_results",
    "save_results", "aggregate_predictions", "TaskAdapter", "TaskTarget",
    "LabelTaskAdapter", "adapt_targets", "EvaluationReadiness",
    "DATA_BLOCKED", "DATA_READY_FOR_EVALUATION", "PROTOCOL_BLOCKED", "READY_FOR_EVALUATION",
    "assess_evaluation_readiness",
]

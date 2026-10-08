"""Small, strict configuration schema shared by pipeline entry points."""

from dataclasses import dataclass, field
from typing import Any

PI_DECISION_REQUIRED = "PI_DECISION_REQUIRED"

# Frozen PPG-JEPA project boundary: the only permitted pretraining dataset is
# MIMIC-III Waveform Matched, modality PPG. VitalDB is downstream-only; ECG
# datasets live in the general bank but are not current PPG pretraining inputs.
PPG_PRETRAIN_DATASET = "mimic3wdb-matched"
PPG_PRETRAIN_MODALITY = "PPG"
FORBIDDEN_PRETRAIN_DATASETS = frozenset({
    "vitaldb",
    "ptb-xl", "cpsc2018", "georgia", "chapman-shaoxing", "mit-bih", "ludb",
    "icentia11k", "mimic-iv-ecg", "mimic4", "code-15",
})
FIXTURE_NAMESPACE_PREFIXES = ("test-fixture", "fixture", "synthetic")


@dataclass(frozen=True)
class PipelineConfig:
    dataset: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] = field(default_factory=dict)

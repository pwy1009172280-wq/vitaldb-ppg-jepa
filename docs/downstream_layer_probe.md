# Downstream phase 1: frozen encoder depth probes

This additive module lives in `src/downstream/layer_probe`. It does not change
UnifiedSample, PipelineConfig, Trainer, or ExperimentRunManifest. The existing
PPG-specific `downstream.layerwise` diagnostic API remains available; its package
exports are now lazy so importing the new API does not load concrete adapters
or model factories.

## Boundaries and interfaces

```text
trusted pretrained checkpoint + caller-owned encoder factory
    -> strict state loading -> eval + requires_grad(False)
split-aware, preprocessed UnifiedSample loader
    -> FeatureBatch(signal, original samples)
    -> no-grad encode_full / injected representation reader
    -> layer_1 ... layer_N, final_layer -> fixed mean-token pooling
    -> provenance-preserving feature matrices / optional NPZ cache
    -> training-only standardization -> linear head -> existing Trainer
    -> sample-level classification/regression metrics
    -> existing run manifest + per-layer probe checkpoints + metrics.json
```

`load_frozen_encoder(checkpoint, encoder_factory)` receives a factory taking the
checkpoint payload and returning the matching existing module. It strictly loads
`payload['model']`, the state key shared by the existing pretraining and generic
training checkpoints. The caller owns reconstruction from `resolved_config`;
there is no method-specific factory or JEPA branch in this module. Only load
trusted checkpoints: legacy checkpoint metadata requires `weights_only=False`.

`extract_layer_features()` takes the frozen module, loader, requested layers,
required SplitContext, and checkpoint reference. `collate_feature_samples()` can
be passed directly as the existing PyTorch DataLoader's collate_fn. Every
sample's subject must belong to its declared role. Windows must already have
equal shapes: extraction does not pad, augment, or preprocess raw waveforms.

The default reader consumes the existing JEPA `encode_full()` representation:
`layer_N` means block N's output, **before** the final normalization;
`final_layer` means `final_tokens`, **after** final normalization. It evaluates
the full unmasked input through the existing online/context encoder. This is
not an EMA-target comparison. A caller-owned RepresentationReader can expose
the same layer-name mapping without any model changes. Default pooling averages
the token dimension; it is a deliberately simple readout, not a claim that all
temporal information is preserved. Invalid/padded samples are rejected by the
default reader rather than silently pooled. A mask-aware reader can explicitly
return an already pooled `[batch, dim]` representation.

Feature caches contain float matrices and JSON sample references (subject,
dataset, recording, window ID/bounds, timing, channels/units, provenance,
metadata, and labels). They omit waveforms and validity-mask arrays. Duplicate
sample identities, missing layers, misaligned rows, empty splits, and non-finite
features fail. NPZ loading uses `allow_pickle=False`. Cache labels and identifiers
are sensitive metadata even though the cache contains no raw waveform.
Per-layer readout metadata distinguishes framework mean-token pooling from a
reader-provided, already pooled representation; this is also recorded in results.

## Configuration and execution

Use `load_layer_probe_config('configs/layer_probe.yaml')`; the original pipeline
loader intentionally still rejects these new top-level sections. Resolved
configs reuse `save_config()` and the original run manifest serialization.
`downstream.task` is the task family (`classification`/`regression`), while
`downstream.target` identifies the sample label used by the experiment.

```python
from src.downstream.layer_probe import load_layer_probe_config, run_layer_probe
from src.experiments import EvaluationProtocol
from src.training import OptimizerFactory
import torch

config = load_layer_probe_config('configs/layer_probe.yaml')
optimizers = OptimizerFactory()
optimizers.register('sgd', torch.optim.SGD)
protocol = EvaluationProtocol(
    task=config.downstream.task,
    protocol_type='linear_probe',
    metrics=(config.downstream.metric,),
    aggregation_level='sample',
    backbone_frozen=True,
)
result = run_layer_probe(
    config, loaders,  # train/validation/test; yield FeatureBatch
    encoder_factory=encoder_factory,  # caller's existing architecture builder
    optimizer_factory=optimizers,
    split=subject_split,
    dataset_manifest_ref=dataset_manifest_ref,
    subject_split_ref=subject_split_ref,
    evaluation_protocol=protocol,
    pretraining_subject_ids=pretraining_subject_ids,
)
```

`run_layer_probe()` can be registered directly with the existing
ExperimentRegistry under `layer_probe`; there is no new global registry.
The entry point checks declared and observed subject disjointness, verifies
each sample's split membership, and checks pretraining subjects against
validation/test subjects before fitting any probe. Explicitly supplied
pretraining subject IDs may overlap downstream **training** subjects, but never
validation/test subjects. Supply the full pretraining exposure set, including
subjects used for pretraining checkpoint selection. Subject identifiers must
be globally meaningful/namespaced; this module cannot detect aliases across
unlinked source datasets. Manifest references remain opaque references under
the existing experiment contract; the caller must bind them to the supplied
split/loaders. An empty pretraining exposure set is an explicit caller assertion,
not proof of disjointness.

The linear head uses cross-entropy or MSE; only its weight/bias are optimized.
Mean/std and class vocabulary are fitted on training features only and saved
in the probe state. Evaluation labels absent from the training vocabulary fail.
Supported metrics are accuracy/macro_f1 and MAE/RMSE. Macro F1 uses the fixed
training vocabulary with zero for undefined class F1. Aggregation is explicitly
sample-level; no subject-level metric or statistical claim is implied.

Each layer uses the same head initialization seed, optimizer configuration,
epochs, and fixed-order batches. There is no layer selection using test results,
no hidden hyperparameter search, and no checkpoint selection by test score.
Phase 1 explicitly selects `last`; monitored task-metric selection is not
implemented here. Encoder/checkpoint/feature references are recorded separately
from probe checkpoints.

## Output compatibility

```text
results/<experiment_name>/
  config.yaml
  manifest.json                     # existing ExperimentRunManifest
  evaluation_protocol.json          # same object as in manifest
  metrics.json                      # layer/task/split metric records
  features/{train,validation,test}.npz   # optional
  checkpoints/<layer>/last.pt        # existing CheckpointManager format
  logs/<layer>.json                  # existing Trainer loss records
```

Metrics record sample/subject counts, encoder and probe checkpoint references,
and the evaluation protocol reference plus a SHA256 of its canonical JSON
content. Existing Trainer loss logs retain their reference-string hash semantics;
their hash is not the content hash used in final evaluation results. They share
the identical protocol reference and are separate files. Completed run
directories are never overwritten by this entry point.

## Deferred extensions

Fine-tuning, random encoders, other SSL comparisons, and complex statistics are
not implemented or accepted by this phase's config. A future fine-tuning runner
can reuse the existing TrainingModel/Trainer contract without changing frozen
feature extraction. A future random-baseline experiment can reuse the head,
split/protocol checks, feature cache, and result structure with separately
recorded initialization provenance. Supporting those protocols will require
explicit new entry/config handling; they cannot silently turn this frozen
experiment into a different experiment.

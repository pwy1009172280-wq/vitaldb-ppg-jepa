# Downstream Engineering Plan v1.0 implementation

The common downstream layer is now split into three boundaries:

```text
model-specific reader
        ↓
src/downstream/core/extraction.py + cache.py
        ↓
src/downstream/core/probe.py + metrics.py + aggregate.py
        ↓
src/downstream/runners/layer_probe.py
        ↓
existing ExperimentRunManifest / EvaluationProtocol / CheckpointManager
```

`core` never imports JEPA or assumes names such as `layer_1`. A reader exposes
`name`, `version`, and a `read(model, batch)` method returning opaque names mapped
to `Representation(kind="token"|"vector")`. The JEPA reader in
`readers/jepa.py` is the only place that maps `hidden_states` and `final_tokens`
to `layer_N` and `final_layer`.

Extraction requires a `SplitContext`, preserves `UnifiedSample` provenance, keeps
loader order, runs eval/no-grad with frozen parameters, and writes float32
features. Cache metadata strictly binds dataset manifest hash, sample-set hash,
preprocessing hash, checkpoint content hash, reader identity/version,
representation, pooling identity/version, dtype, and schema version. Git commit
is provenance only and is not a cache invalidation key.

Probe code separates train-only `FeatureNormalizer` and `ClassVocabulary` from a
pure `LinearHead`. `PredictionBatch` makes metric inputs explicit. AUROC/AUPRC
use continuous probabilities and return `None` when mathematically undefined;
regression metrics remain in target units. Subject aggregation is named and
rejects inconsistent classification labels within a subject.

`runners/layer_probe.py` uses the existing Trainer and CheckpointManager. Its
selection policy is explicitly `last` or `best_validation`; `best_test` is not
accepted. It enriches the existing run manifest with downstream policy metadata
and schema version without defining a second manifest or protocol object.

The old `src/downstream/layerwise.py` and the original `src/downstream/layer_probe`
exports remain import-compatible legacy facades. New downstream work belongs in
`core`, `readers`, and `runners`.

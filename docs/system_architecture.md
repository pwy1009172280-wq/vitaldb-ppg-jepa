# Biosignal research pipeline v1.1 architecture

This repository now has a dataset-agnostic skeleton for a long-lived
biosignal research pipeline. v1.1 defines contracts, metadata-only manifests,
subject-level splitting, and extension points; it does not implement JEPA, a
trainer, data downloads, or changes to stored data.

## Design goals

- MIMIC, VitalDB, ECG, and IMU adapters expose the same sample contract.
- Self-supervised pretraining and downstream evaluation consume the same
  dataset/preprocessing boundaries, while experiment configuration selects the
  task.
- Dataset I/O, preprocessing, model implementations, and experiment runners
  remain separate packages.
- Subject identity is preserved as first-class provenance and split checks fail
  before samples can be mixed across train/validation/test.

## Repository skeleton

```text
configs/
  pipeline_v1.yaml       # contract-oriented example config
docs/
  system_architecture.md
src/
  data/                  # UnifiedSample, BaseDataset, adapters
  datasets/              # future MIMIC/VitalDB/ECG/IMU adapters
  preprocessing/         # future modality-specific transforms
  models/                # existing model code; no new model in v1
  experiments/           # future pretraining/evaluation orchestration
  config/                # YAML loader and schema
tests/
  test_pipeline_contracts.py
```

## Core boundaries

`UnifiedSample` is the only sample object that downstream components should
consume. Its signal has canonical shape `(channels, time)` and carries
`dataset`, `modality`, `recording_id`, `subject_id`, timing, sampling rate,
labels, and metadata. Dataset-specific identifiers can remain in metadata, but
`subject_id` must be stable within a dataset and globally namespaced when
combining datasets (for example `mimic:subject-123`).

`BaseDataset` is a minimal read interface: `name`, `__len__`, `__getitem__`,
and `subject_ids()`. An adapter owns source-file parsing and returns
`UnifiedSample`; it does not own model or experiment logic.

`src/config` loads a YAML mapping into four intentionally loose sections:
`dataset`, `preprocessing`, `model`, and `experiment`. The loader rejects
unknown top-level sections so configuration evolution is explicit. Concrete
schemas can be tightened as each component is implemented.

## Subject leakage policy

Splits are subject-grouped, never window-randomized. A split planner must first
produce disjoint subject sets, then materialize windows. The v1 contract
provides `assert_subject_disjoint()` for samples and
`assert_subject_sets_disjoint()` for manifests. Both raise
`SubjectLeakageError` on overlap. This check should run in every future
experiment entry point and be recorded in its manifest.

For multi-dataset studies, use a globally namespaced subject key. If a source
does not provide a trustworthy subject identity, it is not safe for evaluation
until an adapter supplies one; do not silently derive it from a window or file
name.

## Intended data flow

```text
source records
    -> dataset adapter (BaseDataset)
    -> subject-grouped split planner + leakage check
    -> preprocessing transform(s)
    -> UnifiedSample batches
    -> self-supervised pretraining OR downstream evaluation
```

The v1 code stops before batching and training. Future preprocessing should be
pure/configurable where practical, and should retain provenance. Future model
code should not import MIMIC/VitalDB readers. Future experiment runners should
receive interfaces and resolved config, not inspect raw files.

## v1.1 contracts

`DatasetManifest` contains dataset identity, modality, subject IDs, and
optional recording metadata. `ExperimentManifest` records the task, datasets,
seed, resolved config, and a `SubjectSplit`. These objects are metadata-only;
constructing or serializing them never opens a raw dataset.

`plan_subject_split()` shuffles unique subject IDs with a local seeded RNG and
uses largest-remainder allocation for the requested train/validation/test
ratios. `find_subject_overlap()` and the two `assert_*_splits_disjoint()`
utilities provide explicit checks for manifests and materialized samples.

`PreprocessingRegistry` is a name-to-callable registry with composition and
return-type validation. It intentionally has no built-in transforms in v1.1;
future transforms should be registered by modality-specific preprocessing
modules.

## Experiment runs

The experiment layer is metadata-only. `create_run()` creates the canonical
layout:

```text
results/<experiment_name>/
  config.yaml
  manifest.json
  checkpoints/
  metrics.json
  logs/
```

`manifest.json` records the experiment name, resolved configuration,
dataset-manifest reference, subject-split reference, seed, UTC timestamp, and
the current git commit when available. `seed_everything()` seeds Python and
optionally available NumPy/PyTorch backends; it does not start a trainer.
`ExperimentRegistry` only registers and resolves experiment definitions for
future entry points.

## v1.2 review hardening

`UnifiedSample` now carries channel units, optional boolean validity masks,
generic provenance, and explicit window sample bounds while retaining
extensible metadata for clinical fields. `SplitAwareDataset` requires a
`SplitContext` and rejects samples outside its declared role.

Preprocessing is separated into `StatelessTransform` and `FittedTransform`.
Fitted transforms require a training `SplitContext`; their version and fitted
status are exposed through pipeline metadata. Pretraining subject sets can be
checked against downstream subjects before evaluation.

`EvaluationProtocol` records task, protocol type, metrics, aggregation level,
and whether the backbone is frozen. These are metadata contracts only; no
evaluation algorithms are included.

## Next extension points

1. Add manifest and split-planner interfaces with persisted subject-level
   split manifests.
2. Add one adapter at a time (MIMIC, VitalDB, ECG, IMU) and contract tests
   using synthetic fixtures only.
3. Define preprocessing registry and transform provenance.
4. Add model/pretraining/evaluation registries without coupling them to data
   adapters.
5. Add reproducible experiment manifests, metrics, and checkpoint metadata.

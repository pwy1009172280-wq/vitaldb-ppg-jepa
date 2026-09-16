# Project context

## Project goal
Build a self-supervised representation learning pipeline on VitalDB, starting
with single-modal PPG → PPG. The main method is JEPA with latent-to-latent
prediction. Planned comparisons are MAE (reconstruction), data2vec (EMA latent
prediction), vanilla JEPA, and a project-specific modified JEPA. The project may
develop into a conference submission.

## Current research strategy
First understand JEPA and PPG signal characteristics, then how VitalDB represents
PPG. Establish a clean vanilla JEPA implementation before introducing changes.
Motivate modifications through PPG properties rather than arbitrary architecture
changes. The downstream task, primary evaluation protocol (linear probing, full
fine-tuning, or both), final baseline list, and modification scope remain open.

## Data strategy
- Begin with PPG only, not the full multimodal dataset. Plan to download and
  preserve VitalDB metadata, track metadata, and all available PPG/PLETH waveforms
  needed for the project; this document does not initiate downloads.
- Keep raw data immutable and generate processed data separately. Maintain a
  dataset manifest with case/track information and data status.
- Before training, check missing segments, constant/flat signals, abnormal
  amplitudes, sampling-rate consistency, noise/artifacts, and resampling needs.
- Define case/patient splits before windowing to avoid leakage. Group cases by
  patient where identifiers permit; document limitations if they do not.

## PPG research topics
Study physiological meaning, pulse morphology, temporal scales, systolic peaks,
dicrotic notches, motion artifacts, baseline drift, low perfusion, saturation and
clipping, sampling rate, normalization, window length, patch size, temporal
masking, and possible downstream tasks. Connect these properties to JEPA design.

## Model comparison principles
Share the VitalDB split, preprocessing, windowing, sampling rate, and downstream
evaluation protocol. Use the same patching where scientifically appropriate,
comparable encoder capacity, and comparable pretraining budgets. Document any
necessary differences. The main intended difference is the SSL objective:

| Method | Intended objective/target |
| --- | --- |
| MAE | Raw signal/patch reconstruction |
| data2vec | EMA teacher latent target |
| Vanilla JEPA | Joint-embedding latent prediction |
| Modified JEPA | Project-specific extension, still to be defined |

These descriptions do not imply that EMA is exclusive to data2vec; exact
reference formulations and adaptations remain to be selected and documented.

## Environment strategy
Reproduce original methods in separate environments if necessary; record Python,
PyTorch, CUDA, and dependency versions, then freeze working environments. Later
migrate required components into a unified project environment for fair
comparison. Never automatically upgrade PyTorch or major dependencies to fix an
error; report conflicts before changing environments.

## External code provenance
For every external repository, record URL, branch, commit hash, license, and
purpose in this project. Document all adaptations; never silently change an
external baseline while presenting it as the original implementation.

## Experiment tracking
Record every meaningful run, including failures: run ID, date, model, Git commit,
config, seed, dataset manifest/split version, preprocessing version, pretraining
budget, downstream task, checkpoint, result, training time, GPU memory/compute
notes, observations, and failures.

## Reproducibility
Track random seeds, Git commit, configs, environment versions, data split,
preprocessing version, checkpoint selection rule, and dataset version/manifest.
Version evaluation protocols instead of silently changing them.

## Training robustness
Eventually support checkpoint save/load, interruption recovery, small-data and
tiny-overfit sanity tests, representation-collapse monitoring, multiple seeds,
and mean/std reporting where appropriate. For JEPA/data2vec, assess relevant
representation statistics as well as loss; loss alone does not establish health.

## Storage and Git
Keep code, configuration, and documentation in Git. Do not commit datasets,
checkpoints, large experiment outputs, secrets, or API keys. Store large artifacts
separately and back them up.

## Near-term execution order
1. Prepare environments and external repositories.
2. Study JEPA in depth.
3. Study PPG signal characteristics.
4. Study VitalDB PPG tracks and quality.
5. Build/download the PPG dataset and manifest.
6. Implement and validate vanilla JEPA.
7. Run a small sanity/tiny-overfit experiment.
8. Pretrain and fine-tune once end-to-end as an integration check; this does not
   settle the primary evaluation protocol.
9. Add data2vec and MAE baselines.
10. Design and implement a PPG-motivated JEPA modification.
11. Run controlled comparisons.
12. Perform ablations, failure analysis, and paper-ready evaluation.

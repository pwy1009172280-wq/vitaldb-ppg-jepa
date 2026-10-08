# Gate 2A layer-wise diagnostic implementation report

## Scope

Implemented only diagnostic infrastructure: frozen JEPA feature extraction,
mean pooling, explicit label alignment, case-aware splits, fixed-alpha Ridge,
metrics, CLIs, synthetic fixtures, and tests. No real data, SURF access,
formal pretraining, hierarchy, contrastive objective, nonlinear probe, or
encoder/training change was made.

## Repository audit

`JEPA1D.encode_full()` already uses the online patch embed and full unmasked
context encoder and returns all block hidden states plus final tokens. The
training checkpoint stores `format_version=1`, `method`, `model`, and
`resolved_config`, which are sufficient to reconstruct the exact model.
`ProcessedPPGDataset` already exposes waveform, caseid, tid, and window_index.
The project environment contains scikit-learn.

## Files changed

- `src/downstream/layerwise.py`: extraction, NPZ bundle, checkpoint loading,
  label join, group split, Ridge, metrics, and result serialization.
- `src/downstream/__init__.py`: public diagnostic helpers.
- `scripts/extract_jepa_features.py`: feature extraction CLI.
- `scripts/run_layerwise_probe.py`: separate probe CLI.
- `tests/test_layerwise_diagnostic.py`: synthetic unit and end-to-end tests.
- `docs/layerwise_diagnostic.md`: scope and usage documentation.
- this report file.

## Architecture / dataflow

```text
JEPA checkpoint
→ exact config reconstruction + strict state_dict load + eval/freeze
→ encode_full(full waveform)
→ mean pool block_1 ... block_L and final
→ compressed NPZ with sample keys
→ explicit label join
→ explicit caseid split
→ train-only StandardScaler + fixed-alpha Ridge
→ CSV/JSON layer table
```

## Tests

The new diagnostic tests passed. They cover pooling, frozen checkpoint
loading, all blocks and final features, metadata preservation, feature save and
load, label duplicates/missing labels, explicit split validation, train-only
scaling by implementation path, finite metrics, layer order, and end-to-end
Ridge execution.

## Synthetic smoke

The synthetic end-to-end test loaded a temporary JEPA checkpoint, extracted two
blocks plus final features, preserved caseid/tid/window_index, saved and loaded
NPZ features, joined labels, applied case-aware train/val/test splitting, and
ran Ridge with MAE/RMSE/R². A separate synthetic discriminating fixture made
`layer_A` informative and `layer_B` noisy; the pipeline detected the
informative layer. This is an infrastructure test, not a scientific result.

## Known limitations

No real downstream target or formal split is selected. CUDA is unavailable on
the current machine. Feature extraction currently accumulates the small pilot
bundle in memory before writing one compressed NPZ; larger runs may need
chunked storage. The existing project training package allowlist does not yet
include these downstream diagnostics because no cluster packaging was
requested in this gate.

## Scientific decisions intentionally left unresolved

- target variable and VitalDB label semantics
- formal case/patient split
- alpha value as a scientific choice
- target normalization policy
- interpretation of layer preferences
- whether hierarchy is needed

## Git diff summary

Only Gate 2A diagnostic files and this documentation/report were added. Existing
unrelated dirty-worktree changes were preserved.

## Final verdict

Gate 2A diagnostic infrastructure is implemented and locally validated on
synthetic data. No formal scientific experiment was run.

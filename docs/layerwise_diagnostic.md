# Gate 2A layer-wise diagnostic pilot

This is diagnostic infrastructure for comparing frozen standard-JEPA
representations. It does not establish that hierarchy is necessary.

## Scope

The pipeline loads a completed JEPA checkpoint, encodes complete unmasked
waveforms with `JEPA1D.encode_full(...)`, mean-pools every Transformer block
and the final representation, and optionally runs a fixed-alpha linear Ridge
regression probe. It does not change JEPA training, masking, EMA, loss,
checkpoint format, or encoder parameters.

Representations are:

- `block_1` through `block_L`: the backbone's existing per-block
  pre-final-LayerNorm outputs.
- `final`: the backbone's post-final-LayerNorm output.

Every layer uses the same mean pooling across the token dimension. No CLS
token, learned pooling, layer fusion, concatenation, nonlinear probe, or
classifier is implemented.

## Data and alignment

Feature extraction preserves `caseid`, `tid`, and `window_index` and stores
one array per layer in a compressed NPZ bundle. Labels are supplied separately
as a CSV containing `caseid`, `tid`, `window_index`, and a caller-selected
target column. Duplicate keys fail explicitly; missing labels are reported and
can either be dropped explicitly or treated as an error.

The split is also explicit CSV with `caseid,split`, where split is one of
`train`, `val`, or `test`. Case IDs cannot overlap and every joined feature
case must be listed. Windows are never randomly split across boundaries.

## Ridge probe and outputs

The probe uses `StandardScaler` fitted only on training features followed by
`sklearn.linear_model.Ridge` with a fixed positive alpha (default `1.0`). No
hyperparameter selection is performed. Test metrics are MAE, RMSE, and R²;
validation metrics are also recorded when available. Results are written as
`layerwise_results.csv` and `layerwise_results.json`, including checkpoint,
config, split, target, pooling, alpha, join statistics, timestamp, and Git
identity.

Example commands:

```bash
python scripts/extract_jepa_features.py \
  --checkpoint <CHECKPOINT> --manifest <MANIFEST> \
  --processed-root <ROOT> --output features.npz

python scripts/run_layerwise_probe.py \
  --features features.npz --labels labels.csv --split split.csv \
  --target <TARGET_NAME> --output layerwise-results
```

## Limitations

No VitalDB label is selected here. No formal train/validation/test split,
scientific alpha, target, or physiological interpretation is frozen. Synthetic
tests only verify that the pipeline can detect an intentionally informative
layer; they are not evidence about JEPA representations. This diagnostic does
not establish that hierarchy is necessary.

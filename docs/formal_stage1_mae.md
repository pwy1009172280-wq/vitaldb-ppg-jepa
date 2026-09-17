# Stage 1 formal MAE and processed cache

Preparation writes one float32 waveform `.npy` per source with shape `[accepted_windows,5000]`, one integer accepted-window-index `.npy`, and one case-level CSV manifest. Dataset initialization builds prefix sums and memory maps arrays; `__getitem__` never parses raw CSV. Preparation is source-by-source, atomic, resumable, and calls preprocessing v0 unchanged.

The formal MAE consumes preprocessed `[B,1,5000]` windows. It randomly masks an exact configurable number of individual patches, encodes visible tokens with original absolute positions, restores mask tokens, adds decoder sinusoidal positions once, and predicts preprocessed waveform patches. MSE is normalized over masked patch values only. It never reconstructs original VitalDB CSV waveforms. Config examples are initial values, not frozen science. Benchmark prototype files remain timing-only and are not imported.
# Stage 1 freeze checklist

- CPU tests pass.
- Data integration tests pass.
- Resume and preprocessing-version tests pass.
- Formal MAE CUDA forward/backward has executed successfully on the local RTX 3060; unified-runner GPU validation remains pending.

The raw downloader contract is a flat cache: each source-manifest `local_path` identifies a CSV filename directly under the configured `raw_root`; preparation uses the filename component and never writes raw files. The output manifest is authoritative for the current source manifest, ordered deterministically, and retains matching prior rows while unvisited sources are processed.

The Dataset exposes total manifest rows, complete/failed counts, and failed row metadata while indexing only COMPLETE rows. Future formal `train_ssl.py` should require `failed_count == 0` by default, with an explicit reviewed option for an incomplete cohort.

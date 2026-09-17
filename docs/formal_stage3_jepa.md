# Stage 3 formal vanilla JEPA baseline

This is a documented 1-D PPG JEPA adaptation, not an exact I-JEPA reproduction. Target positions are configurable contiguous, non-overlapping temporal blocks; context is their complement, so the context encoder sees context tokens only. The EMA target patch embed and encoder receive the full unmasked sequence, contextualize every token, and only then select target positions.

The predictor restores a full sequence from projected context latents and learned target-query tokens at the original absolute target positions. It never inserts raw target patch embeddings. Predictor positional encoding is applied once. Smooth L1 latent loss is computed only at target positions. The initial baseline requires `patch_stride == patch_size` to avoid support overlap. EMA is explicit after a future successful optimizer step. Formal masking uses an explicit CPU `torch.Generator`; masks are placed on the model device and its state will be checkpointed by the future runner.

All scientific values are configurable in `configs/smoke/jepa.yaml`. The module is intended for later `train_ssl.py` and Slurm integration and contains no machine-specific paths or device assumptions. Benchmark timing prototypes are unrelated to the formal implementation.

## Stage 3 freeze checklist

- CPU tests pass.
- JEPA semantic tests pass.
- JEPA CUDA forward/backward has executed successfully on the local RTX 3060; unified-runner GPU validation remains pending.

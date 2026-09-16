# Unified PPG self-supervised learning environment

The `ppg-ssl` environment is the shared runtime for the project's own one-dimensional physiological time-series implementations: MAE, data2vec-style latent prediction, vanilla JEPA, and modified JEPA. It is intentionally independent of the original reference repositories and does not depend on Fairseq.

## Pinned stack

| Component | Version |
|---|---:|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| h5py | 3.12.1 |
| matplotlib | 3.9.2 |
| PyYAML | 6.0.2 |
| einops | 0.8.0 |
| tqdm | 4.66.5 |
| tensorboard | 2.18.0 |
| protobuf | 4.25.5 |
| pytest | 8.3.3 |

PyTorch 2.6.0 with the CUDA 12.4 wheel is an official release combination and is suitable for the host NVIDIA driver. The wheel supplies the required CUDA runtime libraries; a local CUDA toolkit or `nvcc` is not required for standard model execution. WSL2 must expose the NVIDIA driver to the Linux process. The validated GPU is an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM.

Ordinary Python packages should be installed from the Tsinghua PyPI mirror when practical. Use the official PyTorch CUDA index for the PyTorch wheel when the mirror does not provide a reliable matching build. Set `TMPDIR=~/pip-tmp` for large downloads because the default temporary filesystem may be small.

The following are intentionally deferred: `torchvision`, `torchaudio`, Fairseq, and custom CUDA extensions. Add them only when a concrete implementation requires them. No compiled extension is part of the baseline environment.

## Recreation

```bash
conda env create -f envs/ppg-ssl.yml
conda activate ppg-ssl
TMPDIR=~/pip-tmp python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
```

If ordinary packages are installed separately, use `https://pypi.tuna.tsinghua.edu.cn/simple` as the command-local index URL. Do not change global pip configuration.

## Validation

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} python - <<'PY'
import torch, numpy, scipy, pandas, sklearn, h5py, matplotlib, yaml
import einops, tqdm, tensorboard, google.protobuf, pytest
assert torch.cuda.is_available()
x = torch.tensor([1., 2., 3.], device="cuda")
assert float((x * x).sum()) == 14.0
print(torch.__version__, torch.cuda.get_device_name(0))
PY
```

The environment specification is maintained in [`envs/ppg-ssl.yml`](../envs/ppg-ssl.yml).

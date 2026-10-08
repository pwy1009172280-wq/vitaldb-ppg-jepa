# Pipeline Quickstart

5 分钟内恢复使用 PPG-JEPA 纯 pipeline integration（SMOKE_ONLY fixture 路径）。

## 1. 环境（WSL 本地）

```bash
# 复用 frozen env（已含 torch/scipy/sklearn/pytest；不含 wfdb）
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ppg-ssl
cd ~/projects/vitaldb-ppg-jepa-integration   # 或任意 exact-SHA release checkout
python -c "import torch; print(torch.__version__)"   # 2.6.0+cu124
```

## 2. 校验 config（role guard + PI 占位符 + 嵌套键）

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.run_pipeline --config <your-config.yaml> --stage validate
# 输出 "config valid" 或明确的拒绝原因
```

配置四 section：`dataset / preprocessing / model / experiment`。pretrain 只允许 `mimic3wdb-matched`（PPG）
或显式 `test-fixture*` 且 `experiment.smoke_only=true`；VitalDB/ECG 作 pretrain 会被 role guard 拒绝。

## 3. 跑 fixture smoke（2 updates，CPU，~5s）

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_pipeline_smoke.py
# 或：python -m scripts.run_pipeline --config <smoke-config.yaml> --stage smoke --results-root results --run-id smoke-1
```

fixture smoke 走完整链：pretrain → checkpoint → frozen ≥2 层表示 → feature cache save/reload →
确定性 fixture 预测 → metric → result save/reload。结果在 `results/smoke-1/metrics.json`。

## 4. 全量回归

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/
# 219 passed
```

## 5. Snellius exact-SHA release（已部署）

```bash
ssh wpu@snellius.surf.nl
cd /gpfs/home2/wpu/projects/vitaldb-ppg-jepa/releases/90079556ebd329a556ca50bff017d3f39f2941c4
git rev-parse HEAD   # 9007955...（与本地一致）
/scratch-shared/wpu/micromamba-root/envs/ppg-ssl/bin/python smoke_check.py   # IMPORT_OK
```

## 6. 常见报错

- `dataset ... is forbidden for PPG-JEPA pretraining` → role guard 拒绝（pretrain 仅 MIMIC PPG）。
- `unresolved PI decision at config.xxx` → config 含 `PI_DECISION_REQUIRED` 占位，需 PI 决定。
- `MIMIC native reader requires ... wfdb` → 隔离 env 未装 wfdb（ENVIRONMENT blocker，非 bug）。
- `duplicate run_id / run directory already contains files` → 换 run_id 或显式 resume（禁止 overwrite）。

注意：正式 pretraining / downstream 科研实验需先走 Research Gate；本 quickstart 只覆盖 engineering smoke。

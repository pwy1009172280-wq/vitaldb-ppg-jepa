# Pipeline Developer Guide

面向未来 DeepSeek/Codex。记录 authoritative modules、contracts、invariants、extension points、tests、
forbidden shortcuts。

## 1. Authoritative modules

| 关注点 | 文件 | 角色 |
|---|---|---|
| Canonical serialization/hash | `src/provenance.py` | Serialization v1：`canonical_json` / `content_hash` / `file_sha256`；唯一 hash 权威 |
| Sample contract | `src/data/samples.py` | `UnifiedSample`：C×T + nullable subject_id + status + valid_mask + provenance |
| Subject/sample identity | `src/data/identity.py` | `sample_identity_hash` / `ordered_sample_set_hash`（含 valid_mask） |
| Split/leakage | `src/data/{access,leakage,base}.py` | RESOLVED-only subject 运算；guard 不过滤 |
| Config | `src/config/{schema,loader}.py` | 4 section 嵌套校验 + role guard + PI 占位符拒绝 |
| Preprocessing | `src/preprocessing/registry.py` | staged（record/window）+ sequential train-only fit + fitted state |
| Source access | `src/data/{source_access,mimic_metadata}.py` | archive 安全（路径/链接/限额）+ MIMIC 逻辑路径 |
| Model 构造 | `src/experiments/pipeline.py` | `build_model_from_config` / `reconstruct_model` / `resolve_device` |
| Trainer | `src/training/trainer.py` | fail-fast + 真预算 + device owner + signal-safe resume |
| Checkpoint | `src/training/checkpoint.py` | generic_v1 + epoch_complete + next_batch_idx + 内嵌 config |
| Downstream | `src/downstream/core/*` | extraction/cache/metrics(undefined)/readiness(fail-closed) |
| Runner | `src/downstream/runners/layer_probe.py` | protocol.aggregation_level 单权威 |

## 2. Contracts / invariants

- **`UnifiedSample.subject_id`**：RESOLVED ↔ 非空 str；UNRESOLVED ↔ None。改动载体字段需同步 reader/guard/test。
- **`BaseDataset`（继承 `Sequence`）**：`__getitem__` 越界**必须**抛 `IndexError`（`list(dataset)` 依赖它；不抛会死循环）。
- **fit 顺序**：`record_transforms[] → windows → window_transforms[]`；第 k 个 fitted 只看前 k−1 输出。
- **train-only**：fit/normalizer/vocabulary 只在 train split；validation/test 参与即泄漏。
- **device ownership**：AMP/autocast 按模型实际 device，不按机器 CUDA 存在性。
- **checkpoint 权威**：reconstruction 用内嵌 `resolved_config.model`；外部 config 仅 content-match。
- **aggregation 权威**：`EvaluationProtocol.aggregation_level`，runner 不接收冲突参数。
- **required metric undefined** → `MetricUndefinedError`（fail-closed）；optional 才允许 None。

## 3. Extension points

- 新 dataset reader：实现 `BaseDataset`（name/__len__/__getitem__ + 越界 IndexError），写 `records.jsonl`
  （`RecordIndexRow`），subject 状态显式 RESOLVED/UNRESOLVED。
- 新 preprocessing transform：继承 `StatelessTransform`/`FittedTransform`，声明 `stage`（record/window）、
  `version`、`state()`；经 `PreprocessingRegistry.register`。
- 新 checkpoint 字段：additive 到 payload，load 对旧 checkpoint 用 `.get(key, default)` 兼容。
- 新 config 字段：加入 `loader._ALLOWED_NESTED` 对应 section。

## 4. Tests

- 全量：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/`（219 passed）。
- 关键负向测试必须覆盖：nonfinite/empty/incomplete、sequential fit、valid_mask 身份、role guard、
  metric undefined、signal-safe resume、checkpoint 外部 mismatch、archive 路径/链接/限额。

## 5. Forbidden shortcuts

- 不 `git add .` / reset --hard / clean / global stash / force push / 自动 merge main。
- 不 `list(loader)` 全量物化；不用 batch-slice 冒充 optimizer-step 预算。
- 不静默过滤 incomplete 行；不把 unresolved subject 交给 set 求交当通过。
- 不手写 WFDB format-80 解码；MIMIC 只用官方 wfdb（缺失即 ENVIRONMENT blocker）。
- 不 catch `ValueError` 一律返回 None 后报 metric 完成。
- 不在 login node 跑正式训练；不 submit Slurm（除非获授权）。
- 不改 JEPA 机制（mask/loss/predictor/EMA）、不做科学默认值、不实现 teacher extraction（需另批）。

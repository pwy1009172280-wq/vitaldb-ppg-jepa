# Research Platform Release — Round 2 最终汇报

## 结论

Round 2 已把以后新实验推荐的主路径收成一条：

```text
real dataset
→ UnifiedSample
→ model-specific TrainingAdapter
→ generic Trainer
→ generic CheckpointManager
→ model-specific RepresentationReader
→ common downstream
→ standardized results
```

本轮是工程收口，没有修改 JEPA 的 masking、predictor、objective 或其他 scientific mechanism，也没有修改冻结的 Dataset Pipeline v1.2 和 Training Framework v1.1.1 语义。

## 1. Round 2 改了什么

- 增加了 `ProcessedPPGUnifiedAdapter`，把现有 `ProcessedPPGDataset` 薄薄地接到 `UnifiedSample`。
- 增加新训练入口：`scripts/train_unified.py`。
- 新训练入口使用 `JEPATrainingAdapter`、generic `Trainer` 和 generic `CheckpointManager`。
- 增加新 downstream 入口：`scripts/run_downstream.py`。
- downstream 入口复用 common extraction、reader、feature cache、probe runner、metrics 和 aggregation。
- downstream 现在持久化 predictions，而不是只保留内存中的 `PredictionBatch`。
- generic checkpoint 增加明确的 `generic_v1` 标记。
- `scripts/package_for_cluster.py` 增加新主路径所需文件。
- legacy 代码保留，并标记为 legacy / compatibility / 不用于新实验。

## 2. 新实验从哪里开始

训练从这里开始：

```bash
python scripts/train_unified.py --config <config.yaml>
```

下游评估从这里开始：

```bash
python scripts/run_downstream.py --config <downstream.yaml>
```

CLI 只负责 orchestration，不重新实现训练循环、特征提取、probe、metrics 或 split 逻辑。

## 3. 真实 PPG 如何进入 UnifiedSample

真实 PPG 仍由冻结的 `ProcessedPPGDataset` 读取。新增的 adapter 位于 dataset adapter 层：

```text
ProcessedPPGDataset
→ ProcessedPPGUnifiedAdapter
→ UnifiedSample
```

adapter 保留 waveform，不复制数据，并映射：

- `caseid` → `subject_id`
- `tid` → `recording_id`
- `window_index` → `window_id` 及 window sample bounds
- waveform → canonical `(channels, time)` signal
- preprocessing version、manifest、旧字段 → provenance
- sampling rate、channel name、PPG modality
- manifest 中实际存在的 `label_*` 字段 → labels

旧数据没有的字段不会凭空生成。对于当前 VitalDB PPG preprocessing，窗口位置和 500 Hz 采样率来自已有 preprocessing contract；adapter 不改变 preprocessing。

下游再通过 `SplitContext` 和 `FeatureBatch` 进入 common extraction，downstream core 不再直接认识 `caseid`、`tid` 或 `window_index`。

## 4. JEPA 正式训练主路径

当前正式路径是：

```text
JEPA1D
→ JEPATrainingAdapter
→ src/training.Trainer
→ generic CheckpointManager
```

正式入口为 `scripts/train_unified.py`。

旧的 `scripts/train_ssl.py` 仍然保留，用于历史 formal experiment reproduction；它不再是新实验推荐入口，也不继续扩展新功能。

## 5. Checkpoint：generic 与 legacy 的区别

### Generic checkpoint

新训练路径生成 generic checkpoint，标记为 `generic_v1`，由 `CheckpointManager` 保存，包含模型状态、optimizer、scheduler、AMP scaler、RNG、训练进度、resolved config、manifest reference 和 protocol reference。

新 downstream 路径优先并且实际要求 generic checkpoint。

### Legacy formal checkpoint

旧 `scripts/train_ssl.py` 继续读取和产生旧 formal checkpoint，用于历史实验复现。不会强制重写历史 checkpoint，也不会让旧格式污染 common downstream core。

## 6. Downstream 怎么运行

downstream 配置指定：

- encoder checkpoint
- encoder config
- dataset manifest 和 processed root
- subject split
- RepresentationReader
- representation 名称
- pooling
- probe task、target 和 metrics
- aggregation

执行：

```bash
python scripts/run_downstream.py --config <downstream.yaml>
```

JEPA 当前使用 `JEPARepresentationReader`，通过 `encode_full` 提供 layer representations 和 `final_layer`。

## 7. Downstream 保存哪些文件

每个标准 result directory 至少包含：

```text
config.yaml
manifest.json
evaluation_protocol.json
encoder_checkpoint.json
feature_cache_manifest.json
feature_cache/
checkpoints/
predictions/
metrics.json
logs/
```

其中：

- `config.yaml`：resolved downstream config
- `manifest.json`：run / experiment manifest、数据和 subject split reference、代码与环境信息、downstream policy
- `evaluation_protocol.json`：EvaluationProtocol
- `encoder_checkpoint.json`：checkpoint 路径和 SHA-256
- `feature_cache_manifest.json`：feature cache 文件、representation、split 和 cache key
- `feature_cache/*.npz`：机器可读、带 provenance 的 representations
- `checkpoints/`：probe checkpoint（适用时）
- `predictions/*.npz`：逐 sample 或 subject aggregation 后的 targets、predictions、scores、probabilities 和 subject IDs，带 schema version
- `metrics.json`：标准化 metrics 和结果行
- `logs/`：训练日志

## 8. Metrics 在哪里

最终 metrics 在：

```text
results/<experiment-name>/metrics.json
```

这里保存 task、representation、split、aggregation、metric、sample/subject 数量、encoder checkpoint reference 和 probe checkpoint reference。

## 9. Predictions 在哪里

逐 representation、逐 split 的 predictions 在：

```text
results/<experiment-name>/predictions/*.npz
```

后续 bootstrap、统计检验和 error analysis 可以直接读取这些文件，不需要重新跑 encoder 或 probe。

## 10. 最终 result directory 结构

```text
results/<experiment-name>/
├── config.yaml
├── manifest.json
├── evaluation_protocol.json
├── encoder_checkpoint.json
├── feature_cache_manifest.json
├── feature_cache/
├── checkpoints/
├── predictions/
├── metrics.json
└── logs/
```

## 11. 保留的 legacy 路径

以下代码没有删除：

- `scripts/train_ssl.py`
- `src/downstream/layerwise.py`
- `src/downstream/layer_probe/`
- `scripts/extract_jepa_features.py`
- `scripts/run_layerwise_probe.py`
- `src/models/ssl_benchmarks.py`

原因是它们仍然服务于旧实验复现、兼容性或既有诊断结果。它们已经用轻量标记说明：legacy / compatibility / not for new experiments。

## 12. Snellius packaging 状态

`scripts/package_for_cluster.py` 已保留原有 formal packaging 支持，并加入新主路径所需文件，包括：

- `src/training`
- JEPA training adapter
- `src/data/ppg_adapter.py`
- `src/downstream/core`
- `src/downstream/readers`
- `src/downstream/runners`
- `scripts/train_unified.py`
- `scripts/run_downstream.py`
- 必需的 experiments metadata 和 protocol 文件

没有删除旧 packaging 支持。

## 13. 测试结果

完整测试：

```text
127 passed, 5 skipped
```

Round 2 主路径及相关核心测试复核：

```text
12 passed
```

已验证：

- dataset bridge identity、signal、provenance 和 labels
- generic JEPA training path
- generic checkpoint save / reload
- frozen encoder
- JEPARepresentationReader
- common feature extraction
- feature cache
- linear probe
- metrics / aggregation
- standard result directory
- metrics 从磁盘重新读取
- predictions 从磁盘重新读取
- training CLI `--help`
- downstream CLI `--help`
- cluster packaging manifest

没有进行真实大规模训练。

## 14. Blocking issue

没有发现 blocking issue。

当前新 downstream 路径要求 generic `generic_v1` checkpoint；legacy formal checkpoint 仍需使用 legacy compatibility path。这是有意的 checkpoint 边界，不是未完成的隐式依赖。

## 15. 最终已经验证的完整路径

```text
dataset
→ ProcessedPPGDataset
→ ProcessedPPGUnifiedAdapter
→ UnifiedSample
→ JEPATrainingAdapter
→ generic Trainer
→ generic CheckpointManager
→ generic checkpoint
→ JEPARepresentationReader
→ common downstream extraction
→ feature cache
→ linear probe
→ metrics / aggregation
→ predictions
→ standardized results
```

这条完整 smoke test 没有中途依赖旧 formal / legacy training 或 downstream 路径。

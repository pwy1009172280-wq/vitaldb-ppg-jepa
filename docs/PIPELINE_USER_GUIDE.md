# Pipeline 用户指南（面向 PI）

本指南解释 PPG-JEPA 纯 pipeline integration 的完整链路与常见操作/报错，不涉及任何科学结论。

## 一、这条 pipeline 做什么

只做「工程正确性 plumbing」，不选择任何科研设置。主链：

```text
Data Bank（source/version/provenance，只读）
 → authoritative SourceAccess（archive/loose 成员访问，有界、防路径逃逸）
 → native reader + canonical identity/provenance（真实 subject / 未解析 subject）
 → UnifiedSample（统一样本契约，C×T + valid_mask + provenance）
 → record 级变换 → lazy window → window 级变换（staged，train-only sequential fit）
 → collate → JEPA adapter → generic Trainer → CheckpointManager
 → RepresentationReader（任意命名层）→ FeatureCache
 → TaskTarget → EvaluationProtocol → evaluator → metrics → standardized result
```

## 二、关键不变式（为什么可信）

1. **Subject identity 严格二分**：`subject_id` 只有在 `subject_identity_status=RESOLVED` 时才非空；
   UNRESOLVED（无患者证据）时 `subject_id=None`，且被 subject-aware 的 split/leakage 运算拒绝。
2. **Leakage 控制**：formal 模式必须提供可解析、可验内容的 pretrain subject artifact；
   下游 train/validation/test 两两不交，且 pretrain ∩ 下游三 split 为空；缺失即失败（不默认空）。
3. **无静默覆盖**：run_id 重复即失败；cache/checkpoint 原子写；resume 需显式。
4. **确定性**：seed-before-init、专用 mask RNG、optimizer groups、真 optimizer-step 预算、signal-safe 恢复。
5. **Checkpoint 自描述**：模型配置内嵌于 checkpoint；外部 YAML 只作内容一致性断言，mismatch 拒绝。

## 三、pretrain 边界（冻结）

- 本项目 pretrain **只允许** MIMIC-III Waveform Matched，且 **仅 PPG**（PLETH 由真实 header 证实）。
- VitalDB 是 **downstream-only**；ECG 数据留在通用 bank，**不进入** PPG pretrain。
- config 层 role guard 会直接拒绝 VitalDB / 任何 ECG dataset 作 pretrain。

## 四、常见操作

| 操作 | 命令 | 说明 |
|---|---|---|
| 校验 config | `python -m scripts.run_pipeline --config X.yaml --stage validate` | 拒绝未知键/PI 占位/越界角色 |
| fixture smoke | `pytest tests/test_pipeline_smoke.py` | 全链 correctness，SMOKE_ONLY |
| 全量回归 | `pytest tests/` | 219 passed |
| 查看 Snellius release | `ssh wpu@snellius.surf.nl` → `releases/<full_sha>` | exact-SHA，clean |

## 五、常见报错与含义

- `dataset X is forbidden for PPG-JEPA pretraining` → role guard，非 bug。
- `PI_DECISION_REQUIRED` → 某科学字段未定，pipeline 拒绝替 PI 填值。
- `SUBJECT_IDENTITY_UNRESOLVED ... subject-aware ...` → 该数据无患者身份证据，不能做 subject split。
- `MIMIC native reader requires wfdb` → 隔离环境缺 wfdb（ENVIRONMENT blocker）。
- `required metric X is undefined` → 该 metric 在当前预测上无法计算（如单类别），流程 fail-closed。

## 六、现状（2026-10-09）

- **PIPELINE_IMPLEMENTATION_STATUS: READY_FOR_PI_PIPELINE_ACCEPTANCE**（工程侧）。
- **REAL_TINY_SMOKE: BLOCKED** — MIMIC-III 下载未完成 + wfdb 未装；完成数据与环境后即可跑。
- 首个科研实验（dataset/task/split/metric/budget 等）**全部待 PI Research Gate 决策**，pipeline 未做任何默认。

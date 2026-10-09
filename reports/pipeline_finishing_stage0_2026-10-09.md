# Pipeline Finishing + Stage 0 Readiness — 2026-10-09

Scope: Pure Pipeline Integration 收尾（任务 1–6）+ first layerwise JEPA experiment Stage 0 就绪判定。
不重做 full audit、不重设计 pipeline、不改实验科学协议。

## 1. 收尾任务结果（任务 1–6）

| 任务 | 结果 | 证据 |
|---|---|---|
| 1. 隔离 wfdb env | DONE | Snellius `ppg-ssl-integration`（clone ppg-ssl + wfdb 4.1.2 + torch 2.6.0+cu124，frozen pins numpy 1.26.4/scipy 1.13.1/pandas 2.2.3）；frozen `ppg-ssl` 未污染 |
| 2. MIMIC bounded snapshot | DONE | `p00_0000.tar` sha256 校验 PASS；29 subjects / 2262 PLETH rows → `/scratch-shared/wpu/mimic_snapshot/records.jsonl`（canonical subject/record/segment + PLETH/WFDB metadata + continuity） |
| 3. 真实 MIMIC A1 + REAL_TINY_SMOKE | DONE | WFDB format-16 解码 `p000052/3533390_0016`（RESP/PLETH/III/V/II，125 Hz，32250 样本，全 finite）→ PLETH → UnifiedSample → tiny JEPA → checkpoint → layer_1/layer_2/final_layer → cache/result（accuracy 1.0） |
| 4. GPU A7 | DONE | 修复 Trainer 未移 batch 到 device 的缺陷（commit `89915e4`，+1 单测）；job 27794327 `gpu_a100` A100-SXM4-40GB：device=cuda + AMP + model/batch/extraction 全 PASS |
| 5. 新 exact-SHA release | DONE | `releases/89915e4fd203111dc5774742893c24333322a652`（clean，HEAD 校验一致）；旧 `9007955` 未覆盖 |
| 6. 重跑 acceptance | DONE | 本地 220 passed（+1 device 单测）；A1/A7/A14 fixture+real、Phase 3/5/6 全部 PASS |

## 2. Stage 0 就绪矩阵

| 项 | 状态 | 事实 |
|---|---|---|
| MIMIC 归档完整性 | PASS | 10 组（p00–p09）tar+sha256，0 `.partial`；p00_0000.tar sha256 匹配 |
| MIMIC PLETH 可用 | PASS | p00_0000 内 29 subjects / 2262 PLETH 段；format 16 解码验证 |
| 真实 WFDB 解码 | PASS | `wfdb.rdrecord` 成功（format 16，PLETH 通道，NU 单位） |
| VitalDB 数据 | **FAIL** | `/projects/prjs2287/biosignal_bank/datasets/vitaldb/` **0 个文件**（仅空目录 data/incoming/metadata/manifest）；无 `SNUADC/PLETH`、无 `Solar8000/HR`、无 subject mapping |
| GPU compute | PASS | `gpu_a100` 分区可用，1×A100 40GB job 成功 |
| ridge solver（§12） | PASS | `src/downstream/core/ridge.py` `RidgeRegression`：float64 Cholesky、intercept 不正则、std≤1e-8 维度记录并置 0、α=n×0.001；4 单测通过 |
| resource estimate | PASS | experiment 模型实例化：total 2,023,296（~2.02M）；online encoder 796,672（~0.80M）；predictor 396,800；与 plan §20 "0.8M/2M" 一致 |

## 3. 结论

```
STAGE0_VERDICT: FAIL — VitalDB downstream source 不可用（0 文件）
PIPELINE_FINISHING: COMPLETE（任务 1–6 全通过）
STAGE1_6: BLOCKED
  (a) VitalDB 下游数据缺失（500 患者 / SNUADC-PLETH / Solar8000-HR / subject mapping 均不存在）
  (b) first_layerwise_jepa_experiment_plan_2026-10-09.md 头部状态为
      "FROZEN CANDIDATE / NOT APPROVED FOR EXECUTION / READY_FOR_EXECUTION: NO"
```

按 plan §17/§20：VitalDB target/subject mapping 本地可用性须由 Stage 0 给出 PASS/FAIL；结果为 FAIL。
按 plan §18.1：未获 PI 批准或 source 就绪前不启动 Stage 1–6。

## 4. 需要的 PI 决定

1. VitalDB 数据缺失：是否授权/等待 VitalDB acquisition（`vitaldb_acquire.sbatch` 存在但未运行，data/ 为空）？
   - 本实施不自行启动下载（不修改 acquisition）。
2. first layerwise JEPA experiment 是否已由 PI 批准执行（plan 头部仍为 NOT APPROVED FOR EXECUTION）？
   - 若批准，需明确 §20 四项（900h/4-block128/2seed×10000、VitalDB 500/350-75-75/120窗、ridge λ=0.001、
     ≤24 GPU-h ≤200 GiB）。

（ridge solver 已实现并单测，无需 PI 决定。）

# Pipeline Integration Implementation Report — 2026-10-09

性质：Pure Pipeline Integration 实施记录（Phase 1–6）。未启动任何科研实验；未修改 JEPA 机制、acquisition、raw 数据；未做科学决策。

## 交付状态总览

```
PRESERVATION_COMMIT: f2ea3179d69eff42aa9c62e14f396fd05cfcf7f1
INTEGRATION_BRANCH:  codex/pipeline-integration-2026-10-08
LOCAL_HEAD:          90079556ebd329a556ca50bff017d3f39f2941c4
REMOTE_HEAD:         90079556ebd329a556ca50bff017d3f39f2941c4  (non-force pushed)
SNELLIUS_HEAD:       90079556ebd329a556ca50bff017d3f39f2941c4  (clean)
SNELLIUS_RELEASE_PATH: /gpfs/home2/wpu/projects/vitaldb-ppg-jepa/releases/90079556ebd329a556ca50bff017d3f39f2941c4
```

- 原 dirty checkout 保留在 `codex/pipeline-preservation-2026-10-08` 分支；`codex/databank-phase1`@`44dfe7a` 未动；`main` 未 merge、未 force push。
- 排除项（`.codex_tmp_*`、`acquisition/`、`data/`、`research/`、`reports/`、`registry_sync/`、根目录研究草稿）仍原样留在原 checkout，未删除。

## 环境与部署

```
ENVIRONMENT_LOCK:
  local WSL:   ppg-ssl  Python 3.10.21  torch 2.6.0+cu124  numpy 1.26.4  (no wfdb)
  Snellius:    /scratch-shared/wpu/micromamba-root/envs/ppg-ssl  Python 3.10.22  torch 2.6.0+cu124  (no wfdb)
  wfdb:        NOT installed in either frozen env; MIMIC native reader fails closed with ENVIRONMENT blocker
```

- 部署方式：本地 `git bundle` → scp → Snellius `git clone --branch`；`git rev-parse HEAD` 与 `git status --porcelain` 验证 clean exact-SHA。
- 远程 GitHub `origin` 已 non-force push integration branch（`9007955`）。
- Snellius 旧 code package（`/gpfs/home2/wpu/projects/vitaldb-ppg-jepa`，无 `.git`）未覆盖、未 pull。

## 关键能力状态

```
LEGACY_TO_GENERIC_PARITY:
  PASS — nonfinite fail-fast、empty loader、incomplete manifest（legacy 有 generic 现补）、
         seed-before-init、专用 mask RNG、adamw parameter groups、真 optimizer-step 预算、
         accumulation tail、AMP skip 不占预算、EMA/scheduler 时机、device ownership 均经测试。
SIGNAL_SAFE_INTRA_EPOCH_RESUME:
  PASS — stop event + safe-boundary checkpoint(epoch_complete=False, next_batch_idx) + cursor 恢复；
         中断+恢复总更新数 == 连续 trace（测试覆盖）。
CHECKPOINT_RECONSTRUCTION:
  PASS — checkpoint 内嵌 resolved model config 为权威；reconstruction 从内嵌 config 重建；
         外部 config 仅作 content-match assertion，mismatch 拒绝；epoch_complete 记录。
CANONICAL_HASH_AND_CACHE:
  PASS — src/provenance.py（Serialization v1，canonical float 编码 MINOR-4，NaN/Inf fail-closed）；
         sample/subject identity（含 valid_mask）由 src/data/identity.py 统一；cache 原子写。
```

## 验收

```
FIXTURE_SMOKE:  PASS  (run_smoke: pretrain -> checkpoint -> frozen ≥2 named representations
                       -> feature cache save/reload -> deterministic fixture predictions -> result save/reload)
REAL_TINY_SMOKE: BLOCKED — MIMIC-III 下载未完成 + wfdb 未装；不自行重启/扩大下载。

A1  Source/parser/identity   PARTIAL — mimic_metadata 逻辑路径/PLETH 选择/测试通过；
                             real WFDB 解码与 header conformance 因无 wfdb/真实源 BLOCKED
A2  Source access            PASS — 路径逃逸/symlink/.partial/字节上限/重复成员拒绝（合成 tar 测试）
A3  Sample/mask              PASS — native 坐标/有限值规则/valid_mask 身份敏感/重复 ID 失败
A4  Staged preprocessing     PASS — sequential fit（第 k 个见前 k−1 输出）+ stage 声明 + fitted state
A5  Config/model             PASS — 嵌套校验/role guard（拒 VitalDB/ECG pretrain）/PI 占位符/几何兼容
A6  JEPA/Trainer parity      PASS — nonfinite/empty/incomplete/seed/mask RNG/groups/预算/accum/AMP/EMA
A7  Device                   PARTIAL — CPU 路径（含 CUDA 机器 CPU 配置）；真实 GPU 未验（无 GPU 授权）
A8  Checkpoint/durability    PASS — 保存重载/纯 bundle 重建/外部 mismatch 拒绝/RNG/信号安全恢复/格式边界
A9  Representation           PASS — layer_k 与 final_layer=LN(layer_N) 语义、online 标识、freeze、非零 dropout
A10 Task/protocol/leakage    PASS — TEST_FIXTURE ID 对齐、missing/duplicate 失败、pretrain∩三 split、身份缺失败
A11 Metrics/aggregation      PASS — protocol.aggregation_level 单权威；required undefined 失败；不按 test metric 选层
A12 Cache/run/result         PASS — canonical vectors、mask/state 变 cache miss、原子写、run 重名拒绝、reload
A13 Readiness/provenance     PASS — fatal 不 ready、stale 不升级、dirty 捕获；ECG unresolved 不阻 PPG
A14 End-to-end               PASS — fixture real-tiny 链全通（synthetic fixture）；real MIMIC 分支 BLOCKED
```

## 测试

```
TESTS_PASSED:   219（local 全量；基线 140 → +79）
TESTS_FAILED:   0
TESTS_SKIPPED_AND_REASON: 无 skip；GPU 相关验收 A7 为 PENDING（无 GPU compute 授权），
                          real WFDB A1 为 BLOCKED（wfdb 缺失），非 pytest skip 而是环境 blocker。
Snellius 子集:   39 passed（非训练：provenance/subject_identity/config/mimic_metadata/source_access）
```

## 文档

```
DOCUMENTATION:
  QUICKSTART:      docs/PIPELINE_QUICKSTART.md
  USER_GUIDE:      docs/PIPELINE_USER_GUIDE.md
  DEVELOPER_GUIDE: docs/PIPELINE_DEVELOPER_GUIDE.md
```

## Blockers / 决策隔离

```
CURRENT_PPG_PIPELINE_BLOCKERS:
  1. MIMIC-III Waveform Matched acquisition 未完成（Snellius 进行中）→ REAL_TINY_SMOKE BLOCKED
  2. wfdb 未安装 → MIMIC native reader 运行时 ENVIRONMENT_BLOCKED（代码已就绪，fail-closed）
  3. （其余 7 组设计前 blockers 已在本实施中修复）

GENERIC_PLATFORM_BLOCKERS:
  Phase-2 ECG record-as-subject → UNRESOLVED（subject-aware 能力阻塞，不阻 MIMIC-only）
  VitalDB caseid 非患者 → UNRESOLVED（需 patient mapping 才可 RESOLVED）
  BIDMC lineage/overlap audit 未完成 → OVERLAP_AUDIT_BLOCKED
  其他 downstream candidates 的 annotation/target 未定 → 逐 dataset 能力阻塞
  ZIP/大 record 访问成本、decoder 随 env 漂移、全库/分布式能力 → 隔离或 fail-closed

DEFERRED_RESEARCH_DECISIONS (PI_DECISION_REQUIRED，未在本实施中决定):
  final downstream suite、首个 dataset/task/target 与 annotation 语义
  formal subject/site/time split 与跨库 overlap 证据
  formal preprocessing（采样率/window/filter/normalization/invalid handling）
  scientific optimizer/parameter-group/scheduler policy、shuffle/sampler
  online vs teacher 表示选择、layer set、probe family、primary metric/aggregation
  validation/model/checkpoint/layer selection、seeds/repeats/budgets/stopping
```

## 最终状态

```
PIPELINE_IMPLEMENTATION_STATUS: READY_FOR_PI_PIPELINE_ACCEPTANCE
```

说明：`READY_FOR_PI_PIPELINE_ACCEPTANCE` 仅指 pipeline 工程实现（fixture smoke 全通、exact-SHA release 已部署）。
REAL_TINY_SMOKE 因 MIMIC 数据与 wfdb 未就绪而 BLOCKED，属数据/环境前置条件，非代码缺口。未写 PILOT_READY /
RESEARCH_GATE_APPROVED / EXPERIMENT_APPROVED。

## 实施提交（integration branch）

代码提交（19 个，代码头 `9007955`）：

f2ea317（preservation，共享基底）→ dc8323e（provenance）→ 22cf386（subject identity carrier）→
fb538d1（identity.py）→ b6ffc9c（config+role guard）→ 5cc48cd（readiness）→ 34c0624（Trainer fail-fast/预算）→
767874f（param groups+mask RNG）→ b67948a（sequential fit）→ 31ee12f（mimic_metadata+source_access）→
d559697（模型构造）→ 461f175（checkpoint 自描述）→ c78c7a9（metric undefined）→ 8f06ee3（cache 原子写）→
5e9e6ea（signal-safe resume）→ d2fd20b（run_pretrain）→ e98c717（aggregation 单权威）→
3e323d2（fixture smoke）→ 9007955（mimic3 reader + run_pipeline CLI）。

（docs 三件套与本报告作为文档提交追加于代码头之后；其 `src/` 与 `9007955` 字节一致。）

---

## 附：本实施中发现并修复的关键问题（供 PI 知悉）

1. **`BaseDataset`（`Sequence`）契约**：`list(dataset)` 依赖 `__getitem__` 在越界时抛 `IndexError`；任何不抛的 reader
   会无限循环。已在测试 fixture 中修复并写入 developer guide 的 forbidden shortcuts。
2. **checkpoint 保存 adapter state（`jepa.*` 前缀）**：重建需包回 `JEPATrainingAdapter` 再 load，已统一。
3. **`install_signal_handlers` 会覆盖进程 SIGTERM 处理**：`timeout`/Slurm 预清场信号会被 Trainer 的 flag handler
   截获（设计如此，但需在 runbook 中提示外部 kill -9 作为硬终止）。
4. **write 工具对 `\\wsl$` 的 temp+rename 不可用**：实施全程采用「写 workspace → cp 进 worktree」模式（工具链细节，
   不影响代码）。

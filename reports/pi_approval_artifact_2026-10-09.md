# PI Approval Artifact — first layerwise JEPA experiment (2026-10-09)

Authoritative PI decision, superseding the frozen-candidate header of
`reports/first_layerwise_jepa_experiment_plan_2026-10-09.md`.

## 1. Research Gate

```
RESEARCH_GATE = GO
EXECUTION_STAGE_0_TO_6 = APPROVED
```

- Approved plan: `reports/first_layerwise_jepa_experiment_plan_2026-10-09.md`
  (frozen amendments from the MiMo review are accepted).
- Execution is subject to fail-closed readiness gates (Stage 0/1 PASS/FAIL facts);
  any change to a frozen scientific setting ⇒ `PI_DECISION_REQUIRED`.

## 2. VitalDB acquisition

```
VITALDB_ACQUISITION = APPROVED
```

Scope (authoritative Data Bank):
- official metadata / case information
- official caseid → subjectid mapping
- SNUADC/PLETH (input waveform)
- Solar8000/HR (target numeric track)
- source/index/metadata required to construct the final 500-subject cohort

Order: metadata / subject mapping / case inventory FIRST, then candidate cohort,
then required waveform + numeric tracks.

Destination: `/projects/prjs2287/biosignal_bank/datasets/vitaldb/`
(Data Bank authoritative; preprocessing/staging/training artifacts → scratch).

Prohibited:
- modify the approved target
- substitute `PLETH_HR` for `Solar8000/HR`
- use `caseid` as `subjectid`
- silently reduce below 500 subjects due to data shortfall
- change cohort/split protocol for download convenience
- copy the whole Data Bank to scratch
- delete or overwrite existing bank data

If actual acquisition volume / license / access mechanism / resource requirements
deviate materially ⇒ STOP → `PI_DECISION_REQUIRED`.

## 3. Ridge solver

```
RIDGE_SOLVER = APPROVED AS ENGINEERING IMPLEMENTATION
```

Deterministic float64 ridge, intercept unregularized, train-only feature
standardization, λ = 0.001, mean-SSE objective, α = n·λ conversion, relative
normal-equation residual ≤ 1e-8. Implemented in `src/downstream/core/ridge.py`
(`RidgeRegression`), 4 unit tests.

## 4. Recording

- Decision date: 2026-10-09
- Scientific owner / Research Gate: PI (Bo)
- Implementation/execution owner: DeepSeek

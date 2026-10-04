# Data Bank v1 Phase 1 implementation map

The frozen v1.2 contracts are reused without modification.

| Existing component | Decision |
| --- | --- |
| `src.data.samples.UnifiedSample` | Reused as-is; source-native channel, subject, recording, timing, units, and provenance are populated. |
| `src.data.base.BaseDataset` | Reused as-is by both readers. |
| `src.data.manifests.DatasetManifest` | Reused for future metadata-only dataset manifests; no second sample/split contract added. |
| `src.data.splits` / leakage checks | Reused as-is; this phase does not create experiment splits. |
| `src.preprocessing` | Not called; ingestion performs no model-specific preprocessing. |
| `src.data.index.RecordIndexRow` | New thin JSONL canonical index contract because the repository has no existing Parquet/index dependency. |
| `src.datasets.ptb_xl.PTBXLReader` | New source-native WFDB reader and index builder. Keeps PTB-XL 100 Hz and 500 Hz variants separate. |
| `src.datasets.ppg_dalia.PPGDaLiAReader` | New source-native ZIP/pickle reader and metadata index builder. Keeps device, modality, and native rates separate. |
| `scripts/check_databank_readiness.py` | New generic evidence-producing readiness checker driven by the existing readiness policy fields. |

No Trainer, JEPA, downstream, split policy, role policy, overlap policy, or
preprocessing semantics are changed by Phase 1.

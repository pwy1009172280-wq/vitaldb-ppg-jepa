# MIMIC-III WDB Matched PPG pipeline

This pipeline is separate from VitalDB. It treats every WFDB segment as a
physical continuity unit: segment rows are never concatenated. Headers and
manifests are small and can be inspected before any waveform data is fetched.

## Proposed sequence

```text
downloaded master/layout headers
        -> inventory (PLETH rows + failures)
        -> plan (exact segment .hea/.dat files)
        -> human review of counts and decisions
        -> resumable download of the plan
        -> one-segment-at-a-time processing
        -> validation + provenance manifest
        -> optional safe deletion of validated raw files
```

Inventory is read-only. Existing files are reused. Downloaded files are first
written to `.part` files and atomically renamed; an interrupted `.part` can be
resumed when the server supports HTTP Range requests. Processing skips an
already valid output and records failures by the caller/job log. The optional
`--delete-raw` flag requires a validated output whose SHA-256 identifies the
requested raw file and whose path is inside `--raw-root`.

Example commands (replace paths and URL with the approved MIMIC source):

```bash
python -m scripts.data.mimic3wdb_pipeline inventory \
  --headers-root /scratch-shared/wpu/datasets/pretrain/mimic3wdb-matched \
  --output-root /scratch-shared/wpu/datasets/pretrain/mimic3wdb-matched/metadata \
  --source-base-url https://physionet.org/files/mimic3wdb-matched/1.0/

python -m scripts.data.mimic3wdb_pipeline plan \
  --manifest .../metadata/ppg_manifest.csv \
  --output .../metadata/required_files.csv
```

Do not run `download` until the inventory and plan have been reviewed. The
current implementation intentionally leaves `resample_hz`, filtering,
normalization, and window length unset (`null`/`none`). Those are scientific
decisions, not silently chosen preprocessing.

## Required decisions before processing

* the approved MIMIC-III WDB Matched release/version and URL/authentication;
* the authoritative subject-ID mapping/regex in the downloaded headers;
* whether PLETH aliases beyond exact `PLETH` are included;
* target sampling rate, filtering, normalization, quality exclusions, and
  windowing policy;
* subject-level split policy and whether segment boundaries may be crossed by
  later training windows (the pipeline itself never crosses them).

The manifest retains source paths, signal metadata, segment sample counts,
continuity status, and source hashes needed to rebuild the compact outputs.

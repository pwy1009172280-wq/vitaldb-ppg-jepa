"""Freeze the approved 900/100 roles for the existing MIMIC candidate set.

The candidate and segment manifests remain immutable inputs. Existing freeze
artifacts are accepted only when their bytes match the deterministic result.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_key(subject_id: str) -> tuple[str, str]:
    value = f"first-layerwise-2026-10-09|mimic-role|{subject_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest(), subject_id


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_create(path: str, content: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as stream:
            if stream.read() == content:
                return
        raise RuntimeError(f"refusing to replace existing frozen artifact: {path}")
    fd, tmp = tempfile.mkstemp(prefix=".freeze-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        try:
            os.link(tmp, path)
        except FileExistsError:
            with open(path, "rb") as stream:
                if stream.read() != content:
                    raise RuntimeError(f"concurrent conflicting freeze artifact: {path}")
        finally:
            os.unlink(tmp)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-root", required=True)
    args = parser.parse_args()

    candidates = read_rows(args.candidate)
    segments = read_rows(args.segments)
    if not candidates or "subject_id" not in candidates[0]:
        raise SystemExit("candidate CSV must contain subject_id rows")
    ids = [row["subject_id"].strip() for row in candidates]
    if len(ids) != 1000 or len(set(ids)) != 1000 or any(not value for value in ids):
        raise SystemExit(f"expected exactly 1000 unique candidate subjects; got {len(ids)} rows / {len(set(ids))} unique")
    segment_counts: dict[str, int] = {}
    for row in segments:
        subject_id = row.get("subject_id", "").strip()
        if subject_id:
            segment_counts[subject_id] = segment_counts.get(subject_id, 0) + 1
    if set(segment_counts) != set(ids):
        missing = len(set(ids) - set(segment_counts))
        extra = len(set(segment_counts) - set(ids))
        raise SystemExit(f"candidate/segment manifest subject mismatch: missing={missing} extra={extra}")

    ordered = sorted(ids, key=subject_key)
    role_by_id = {subject_id: ("train" if i < 900 else "diagnostic", i + 1) for i, subject_id in enumerate(ordered)}
    candidate_rank = {subject_id: int(row.get("rank", i + 1)) for i, (subject_id, row) in enumerate(zip(ids, candidates))}
    import io

    buffer = io.StringIO(newline="")
    fields = ["subject_id", "role", "role_rank", "role_sha256", "candidate_rank", "segment_rows"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for subject_id in ordered:
        role, rank = role_by_id[subject_id]
        writer.writerow({
            "subject_id": subject_id,
            "role": role,
            "role_rank": rank,
            "role_sha256": subject_key(subject_id)[0],
            "candidate_rank": candidate_rank[subject_id],
            "segment_rows": segment_counts[subject_id],
        })
    cohort_bytes = buffer.getvalue().encode("utf-8")
    archive_rows = []
    for shard in sorted({row["shard"] for row in segments}):
        if os.path.basename(shard) != shard or not shard.endswith(".tar"):
            raise SystemExit(f"unsafe or invalid shard name in segment manifest: {shard!r}")
        archive = os.path.join(args.archive_root, shard.split("_")[0], shard)
        sidecar = archive + ".sha256"
        if not os.path.isfile(archive) or not os.path.isfile(sidecar) or os.path.exists(archive + ".partial"):
            raise SystemExit(f"archive/sidecar missing or partial: {archive}")
        with open(sidecar, encoding="ascii") as stream:
            sidecar_text = stream.read().strip()
        expected_hash = sidecar_text.split()[0] if sidecar_text else ""
        if len(expected_hash) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected_hash):
            raise SystemExit(f"invalid SHA256 sidecar: {sidecar}")
        archive_rows.append({
            "shard": shard,
            "archive_path": os.path.abspath(archive),
            "archive_bytes": os.path.getsize(archive),
            "expected_sha256": expected_hash.lower(),
            "sha256_sidecar_path": os.path.abspath(sidecar),
            "sha256_sidecar_sha256": file_sha256(sidecar),
        })
    archive_buffer = io.StringIO(newline="")
    archive_fields = ["shard", "archive_path", "archive_bytes", "expected_sha256", "sha256_sidecar_path", "sha256_sidecar_sha256"]
    archive_writer = csv.DictWriter(archive_buffer, fieldnames=archive_fields, lineterminator="\n")
    archive_writer.writeheader()
    archive_writer.writerows(archive_rows)
    archive_bytes = archive_buffer.getvalue().encode("utf-8")
    metadata = {
        "schema_version": 1,
        "protocol_id": "first-layerwise-2026-10-09",
        "selection": "existing 1000 candidate subjects; SHA256 mimic-role ordering; first 900 train, final 100 diagnostic",
        "candidate_path": os.path.abspath(args.candidate),
        "candidate_sha256": file_sha256(args.candidate),
        "segment_manifest_path": os.path.abspath(args.segments),
        "segment_manifest_sha256": file_sha256(args.segments),
        "archive_root": os.path.abspath(args.archive_root),
        "archive_snapshot_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_shards": len(archive_rows),
        "archive_bytes": sum(int(row["archive_bytes"]) for row in archive_rows),
        "archive_integrity_note": "Expected hashes and sidecar hashes are frozen; payload bytes are not rehashed here to preserve the approved source-read budget.",
        "candidate_subjects": 1000,
        "train_subjects": 900,
        "diagnostic_subjects": 100,
        "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
    }
    metadata_bytes = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode("utf-8")
    out = os.path.abspath(args.output_dir)
    cohort_path = os.path.join(out, "mimic_frozen_cohort.csv")
    metadata_path = os.path.join(out, "mimic_frozen_cohort.json")
    atomic_create(cohort_path, cohort_bytes)
    atomic_create(metadata_path, metadata_bytes)
    archive_snapshot_path = os.path.join(out, "mimic_archive_snapshot.csv")
    atomic_create(archive_snapshot_path, archive_bytes)
    atomic_create(archive_snapshot_path + ".sha256", f"{hashlib.sha256(archive_bytes).hexdigest()}  {os.path.basename(archive_snapshot_path)}\n".encode())
    atomic_create(cohort_path + ".sha256", f"{hashlib.sha256(cohort_bytes).hexdigest()}  {os.path.basename(cohort_path)}\n".encode())
    atomic_create(metadata_path + ".sha256", f"{hashlib.sha256(metadata_bytes).hexdigest()}  {os.path.basename(metadata_path)}\n".encode())
    print("MIMIC_COHORT_FROZEN", cohort_path)
    print("TRAIN_SUBJECTS 900")
    print("DIAGNOSTIC_SUBJECTS 100")
    print("COHORT_SHA256", hashlib.sha256(cohort_bytes).hexdigest())
    print("CANDIDATE_SHA256", metadata["candidate_sha256"])
    print("SEGMENT_MANIFEST_SHA256", metadata["segment_manifest_sha256"])
    print("ARCHIVE_SHARDS", metadata["archive_shards"])
    print("ARCHIVE_BYTES", metadata["archive_bytes"])


if __name__ == "__main__":
    main()

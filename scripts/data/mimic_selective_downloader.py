#!/usr/bin/env python3
"""Resumable, bounded-concurrency MIMIC waveform manifest downloader."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import http.client
import json
import os
import threading
import time
from urllib.parse import urlsplit
from pathlib import Path


class _HTTPResponse:
    def __init__(self, response):
        self._response = response
        self.status_code = response.status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        while True:
            block = self._response.read(chunk_size)
            if not block:
                return
            yield block

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._response.close()


class _PersistentHTTPClient:
    """One reusable HTTP(S) connection, owned by one downloader worker."""
    def __init__(self):
        self.connection = None
        self.endpoint = None

    def get(self, url, headers, timeout):
        parsed = urlsplit(url)
        endpoint = (parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        if self.connection is None or endpoint != self.endpoint:
            if self.connection is not None:
                self.connection.close()
            connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            self.connection = connection_type(endpoint[1], endpoint[2], timeout=timeout[1])
            self.endpoint = endpoint
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            self.connection.request("GET", path, headers=headers)
            return _HTTPResponse(self.connection.getresponse())
        except Exception:
            self.connection.close()
            self.connection = None
            self.endpoint = None
            raise


def load_manifest(path: Path, limit: int | None = None):
    rows = []
    with path.open() as stream:
        header = stream.readline().rstrip("\n").split("\t")
        indexes = {name: i for i, name in enumerate(header)}
        required = {"logical_dat_relpath", "physical_source_recommendation", "source_kind"}
        if not required.issubset(indexes):
            raise ValueError(f"manifest missing columns: {sorted(required - set(indexes))}")
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append({key: parts[index] for key, index in indexes.items()})
    if limit is not None:
        normal = [r for r in rows if r["source_kind"] == "matched_sha256_verified"]
        fallback = [r for r in rows if r["source_kind"] != "matched_sha256_verified"]
        rows = normal[: limit // 2] + fallback[: limit - limit // 2]
    return rows


def _checksum_keys(value: str):
    value = value.strip().lstrip("./")
    yield value
    parsed = urlsplit(value)
    if parsed.scheme and parsed.path:
        path = parsed.path.lstrip("/")
        yield path
        if "/files/" in path:
            yield path.split("/files/", 1)[1]
        if "/1.0/" in path:
            yield path.split("/1.0/", 1)[1]
    if "/files/" in value:
        yield value.split("/files/", 1)[1]
    if "/1.0/" in value:
        yield value.split("/1.0/", 1)[1]


def load_checksums(path: Path, rows):
    wanted = {}
    for row in rows:
        for key in _checksum_keys(row["physical_source_recommendation"]):
            wanted[key] = row
        for key in _checksum_keys(row["logical_dat_relpath"]):
            wanted.setdefault(key, row)
    checksums = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts
            for key in _checksum_keys(name):
                if key in wanted:
                    checksums[wanted[key]["logical_dat_relpath"]] = digest
    return checksums


def sha256(path: Path, block=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block_data in iter(lambda: stream.read(block), b""):
            digest.update(block_data)
    return digest.hexdigest()


class Downloader:
    def __init__(self, args, rows, checksums):
        self.args, self.rows, self.checksums = args, rows, checksums
        self.state_path = Path(args.state)
        self.failure_path = Path(args.failures)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.failure_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.state = {"updated": time.time(), "total": len(rows), "completed": 0,
                      "verified": 0, "failed": 0, "downloaded_bytes": 0,
                      "expected_bytes": 0, "failures": [], "items": {}}
        if self.state_path.exists() and not getattr(args, "reset_state", False):
            try:
                saved = json.loads(self.state_path.read_text())
                if isinstance(saved, dict):
                    self.state.update(saved)
            except (OSError, ValueError):
                pass
        self.state["total"] = len(rows)

    def session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = _PersistentHTTPClient()
            self._local.session = session
        return session

    def save(self):
        self.state["updated"] = time.time()
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".part")
        temporary.write_text(json.dumps(self.state, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, self.state_path)
        self.failure_path.write_text("\n".join(self.state["failures"]) + ("\n" if self.state["failures"] else ""))

    def _verified_existing(self, destination, expected):
        return destination.exists() and destination.is_file() and sha256(destination) == expected

    def one(self, row):
        logical = row["logical_dat_relpath"]
        source = row["physical_source_recommendation"]
        source_kind = row["source_kind"]
        destination = Path(self.args.destination) / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = self.checksums.get(logical)
        if not expected:
            return logical, "failed", 0, f"missing_checksum:{source}", source_kind, source
        if self._verified_existing(destination, expected):
            return logical, "verified", destination.stat().st_size, None, source_kind, source

        partial = Path(str(destination) + ".part")
        error = None
        for attempt in range(max(1, self.args.retries)):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                response = self.session().get(
                    source,
                    headers={"User-Agent": "vitaldb-ppg-jepa-mimic-downloader/1.0", **headers},
                    timeout=(30, 180),
                )
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                if not resumed:
                    offset = 0
                with response, partial.open(mode) as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if sha256(partial) != expected:
                    raise RuntimeError("sha256_mismatch")
                os.replace(partial, destination)
                status = "resumed" if resumed else "downloaded"
                return logical, status, destination.stat().st_size, None, source_kind, source
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                if attempt + 1 < max(1, self.args.retries):
                    time.sleep(min(120, self.args.backoff * (2 ** attempt)))
        return logical, "failed", 0, error, source_kind, source

    def record(self, result):
        logical, status, size, error, source_kind, source = result
        item = {"status": status, "size": size, "source_kind": source_kind,
                "source": source, "updated": time.time()}
        if error:
            item["error"] = error
        self.state["items"][logical] = item
        self.state["completed"] = len(self.state["items"])
        self.state["verified"] = sum(v["status"] in {"verified", "downloaded", "resumed"}
                                      for v in self.state["items"].values())
        self.state["failed"] = sum(v["status"] == "failed" for v in self.state["items"].values())
        self.state["downloaded_bytes"] = sum(v.get("size", 0) for v in self.state["items"].values())
        self.state["failures"] = [
            f"{key}\t{value['source_kind']}\t{value['source']}\t{value.get('error', '')}"
            for key, value in sorted(self.state["items"].items()) if value["status"] == "failed"
        ]
        self.save()

    def run(self):
        pending = []
        for row in self.rows:
            item = self.state["items"].get(row["logical_dat_relpath"], {})
            destination = Path(self.args.destination) / row["logical_dat_relpath"]
            expected = self.checksums.get(row["logical_dat_relpath"])
            if item.get("status") in {"verified", "downloaded", "resumed"} and expected:
                if self._verified_existing(destination, expected):
                    continue
            pending.append(row)
        with futures.ThreadPoolExecutor(max_workers=max(1, self.args.workers)) as executor:
            for result in futures.as_completed([executor.submit(self.one, row) for row in pending]):
                self.record(result.result())
        self.save()
        return 1 if self.state["failed"] else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checksums", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()
    rows = load_manifest(Path(args.manifest), args.limit)
    checksums = load_checksums(Path(args.checksums), rows)
    if len(rows) != len(checksums):
        raise SystemExit(f"checksum metadata missing for {len(rows) - len(checksums)} manifest targets")
    return Downloader(args, rows, checksums).run()


if __name__ == "__main__":
    raise SystemExit(main())

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import scripts.data.mimic_selective_downloader as downloader
from scripts.data.mimic_selective_downloader import Downloader, load_checksums, load_manifest


def test_matched_fallback_routing_resume_and_skip(tmp_path):
    payloads = {
        "/matched/a.dat": b"matched-source-payload",
        "/original/b.dat": b"official-original-fallback-payload",
    }
    seen = []

    class Response:
        def __init__(self, body, status):
            self.body, self.status_code = body, status

        def raise_for_status(self):
            assert self.status_code < 400

        def iter_content(self, _chunk_size):
            yield self.body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class Client:
        def get(self, url, headers, timeout):
            del timeout
            path = url.split("http://test", 1)[1]
            seen.append((path, headers.get("Range")))
            payload = payloads[path]
            start = int(headers["Range"].split("=", 1)[1].rstrip("-")) if headers.get("Range") else 0
            return Response(payload[start:], 206 if start else 200)

    original_client = downloader._PersistentHTTPClient
    downloader._PersistentHTTPClient = Client
    try:
        base = "http://test"
        manifest = tmp_path / "manifest.tsv"
        manifest.write_text(
            "logical_dat_relpath\tphysical_source_recommendation\tsource_kind\n"
            f"matched/a.dat\t{base}/matched/a.dat\tmatched_sha256_verified\n"
            f"fallback/b.dat\t{base}/original/b.dat\tofficial_original_fallback\n"
        )
        checksums = tmp_path / "SHA256SUMS"
        checksums.write_text(
            f"{hashlib.sha256(payloads['/matched/a.dat']).hexdigest()}  matched/a.dat\n"
            f"{hashlib.sha256(payloads['/original/b.dat']).hexdigest()}  original/b.dat\n"
        )
        rows = load_manifest(manifest)
        sums = load_checksums(checksums, rows)
        assert len(sums) == 2
        destination = tmp_path / "download"
        partial = destination / "matched/a.dat.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(payloads["/matched/a.dat"][:7])
        args = SimpleNamespace(destination=destination, state=tmp_path / "state.json",
                               failures=tmp_path / "failures.tsv", workers=2, retries=2,
                               backoff=0)
        assert Downloader(args, rows, sums).run() == 0
        assert (destination / "matched/a.dat").read_bytes() == payloads["/matched/a.dat"]
        assert (destination / "fallback/b.dat").read_bytes() == payloads["/original/b.dat"]
        assert not (destination / "matched/a.dat.part").exists()
        assert any(path == "/matched/a.dat" and value == "bytes=7-" for path, value in seen)
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["items"]["fallback/b.dat"]["source_kind"] == "official_original_fallback"
        before = len(seen)
        assert Downloader(args, rows, sums).run() == 0
        assert len(seen) == before
    finally:
        downloader._PersistentHTTPClient = original_client

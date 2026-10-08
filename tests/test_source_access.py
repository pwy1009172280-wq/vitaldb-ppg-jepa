"""Archive access safety (A2): partial rejection, path/symlink safety, byte cap."""

import io
import tarfile

import pytest

from src.data.source_access import (
    MaterializedMember,
    is_completed_source,
    iter_regular_members,
    materialize_members,
    normalize_member_name,
)


def _make_tar(path, entries):
    """entries: list of (name, bytes, kind) where kind in {'file','symlink','dir'}."""
    with tarfile.open(path, "w") as archive:
        for name, data, kind in entries:
            if kind == "file":
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif kind == "symlink":
                info = tarfile.TarInfo(name)
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                archive.addfile(info)


def test_normalize_member_name_rejects_escape():
    with pytest.raises(ValueError, match="absolute"):
        normalize_member_name("/etc/passwd")
    with pytest.raises(ValueError, match="escape"):
        normalize_member_name("../escape")
    with pytest.raises(ValueError, match="escape"):
        normalize_member_name("a/../../b")
    assert normalize_member_name("p00/p000020/seg.hea") == "p00/p000020/seg.hea"


def test_is_completed_source_rejects_partial(tmp_path):
    p = tmp_path / "x.tar"
    p.write_bytes(b"x")
    assert is_completed_source(p)
    partial = tmp_path / "x.tar.part"
    partial.write_bytes(b"x")
    assert not is_completed_source(partial)


def test_iter_regular_members_rejects_symlink(tmp_path):
    tar = tmp_path / "s.tar"
    _make_tar(tar, [("ok.hea", b"header", "file"), ("link", b"", "symlink")])
    with pytest.raises(ValueError, match="symlink"):
        list(iter_regular_members(tar))


def test_iter_regular_members_yields_files_only(tmp_path):
    tar = tmp_path / "s.tar"
    _make_tar(tar, [("ok.hea", b"header", "file"), ("subdir", b"", "dir")])
    assert list(iter_regular_members(tar)) == ["ok.hea"]


def test_materialize_bounded_and_safe(tmp_path):
    tar = tmp_path / "s.tar"
    _make_tar(tar, [("p00/p000020/seg.hea", b"header-bytes", "file")])
    out = tmp_path / "scratch"
    members = materialize_members(tar, ["p00/p000020/seg.hea"], out, max_bytes=1024)
    assert members == [MaterializedMember("p00/p000020/seg.hea", 12)]
    assert (out / "p00" / "p000020" / "seg.hea").read_bytes() == b"header-bytes"


def test_materialize_rejects_over_cap(tmp_path):
    tar = tmp_path / "s.tar"
    _make_tar(tar, [("big.hea", b"x" * 100, "file")])
    with pytest.raises(ValueError, match="cap"):
        materialize_members(tar, ["big.hea"], tmp_path / "scratch", max_bytes=10)


def test_materialize_rejects_symlink_member(tmp_path):
    tar = tmp_path / "s.tar"
    _make_tar(tar, [("link", b"", "symlink")])
    with pytest.raises(ValueError, match="symlink"):
        materialize_members(tar, ["link"], tmp_path / "scratch", max_bytes=1024)


def test_materialize_rejects_duplicate(tmp_path):
    tar = tmp_path / "s.tar"
    _make_tar(tar, [("dup.hea", b"a", "file")])
    with pytest.raises(ValueError, match="duplicate"):
        materialize_members(tar, ["dup.hea", "dup.hea"], tmp_path / "scratch", max_bytes=1024)

"""Deterministic test vectors for canonical serialization/hashing (Serialization v1)."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.provenance import canonical_json, content_hash, file_sha256


def test_float_int_and_negative_zero_are_distinct():
    assert canonical_json(1) == "1"
    assert canonical_json(1.0) == "1.0"
    assert canonical_json(-0.0) == "-0.0"
    assert canonical_json(0.1) == "0.1"
    assert canonical_json([1, 1.0]) == "[1,1.0]"


def test_nonfinite_floats_fail_closed():
    with pytest.raises(ValueError):
        canonical_json(float("nan"))
    with pytest.raises(ValueError):
        canonical_json(float("inf"))
    with pytest.raises(ValueError):
        canonical_json([1.0, float("-inf")])


def test_dict_sorted_and_str_keys_only():
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    with pytest.raises(TypeError):
        canonical_json({1: "x"})


def test_sets_and_paths_rejected():
    with pytest.raises(TypeError):
        canonical_json({1, 2})
    with pytest.raises(TypeError):
        canonical_json(Path("x"))


def test_dataclass_normalized_by_declared_fields():
    @dataclass
    class C:
        b: int
        a: int

    assert canonical_json(C(b=1, a=2)) == '{"a":2,"b":1}'


def test_content_hash_kind_scoped_and_deterministic():
    h1 = content_hash("cfg", {"a": 1})
    h2 = content_hash("cfg", {"a": 1})
    assert h1 == h2
    assert len(h1) == 64
    assert content_hash("cfg", {"a": 1}) != content_hash("other", {"a": 1})


def test_ndarray_descriptor_deterministic_and_shape_sensitive():
    a = np.array([True, False], dtype=bool)
    assert "__canonical_ndarray__" in canonical_json(a)
    assert content_hash("mask", a) == content_hash("mask", np.array([True, False]))
    assert content_hash("mask", a) != content_hash("mask", np.array([True, True]))
    # shape participates in the descriptor
    assert content_hash("mask", np.zeros((2, 2), dtype=bool)) != content_hash(
        "mask", np.zeros((4,), dtype=bool)
    )


def test_ndarray_unsupported_and_nonfinite_rejected():
    with pytest.raises(TypeError):
        canonical_json(np.array([1, "a"], dtype=object))
    with pytest.raises(TypeError):
        canonical_json(np.array(["a", "b"]))  # unicode dtype
    with pytest.raises(ValueError):
        canonical_json(np.array([1.0, float("nan")]))


def test_reserved_ndarray_tag_cannot_be_faked_by_user_mapping():
    with pytest.raises(TypeError):
        canonical_json({"__canonical_ndarray__": {"dtype": "int64", "shape": [], "bytes_sha256": "x"}})


def test_file_sha256_exact_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert file_sha256(p) == hashlib.sha256(b"abc").hexdigest()

"""Logical-path and channel helpers for the MIMIC native reader (A1 pre-requisite)."""

import pytest

from src.data.mimic_metadata import (
    MimicLogicalRef,
    parse_logical_path,
    parse_subject_component,
    select_channel_indices,
)


def test_parse_logical_path_extracts_group_subject_segment():
    ref = parse_logical_path("p00/p000020/3544749_0008.hea")
    assert ref == MimicLogicalRef("p00", "p000020", "3544749_0008", "p00/p000020/3544749_0008.hea")


def test_group_directory_is_not_a_subject():
    assert parse_subject_component("p00") is None
    assert parse_subject_component("p000020") == "p000020"
    assert parse_subject_component("root") is None
    assert parse_subject_component("queue") is None


def test_parse_rejects_group_as_subject():
    # "p00" cannot appear in the subject position
    with pytest.raises(ValueError, match="subject"):
        parse_logical_path("p00/3544749_0008.hea")


def test_parse_rejects_bad_segment_token():
    with pytest.raises(ValueError, match="segment"):
        parse_logical_path("p00/p000020/not-a-segment.hea")


def test_parse_rejects_non_hea_path():
    with pytest.raises(ValueError, match=".hea"):
        parse_logical_path("p00/p000020/3544749_0008.dat")


def test_select_channel_indices_matches_source_label():
    assert select_channel_indices(("II", "PLETH", "AVF"), "PLETH") == [1]
    assert select_channel_indices(("II", "PLETH", "AVF"), "pleth") == [1]
    assert select_channel_indices(("II", "AVF"), "PLETH") == []
    with pytest.raises(ValueError, match="source_label"):
        select_channel_indices(("II",), "")

from pathlib import Path

from scripts.data.mimic_required_list_parser import generate_required_list, parse_master


def test_p070491_comment_and_layout_regression(tmp_path: Path):
    headers = tmp_path / "headers_v2" / "p07" / "p070491"
    headers.mkdir(parents=True)
    master = headers / "p070491-master.hea"
    master.write_text(
        "p070491-master/5 3 125 2176989\n"
        "3053188_layout 0\n"
        "3053188_0001 1989\n"
        "3053188_0002 1044375\n"
        "3053188_0003 125\n"
        "3053188_0004 1130500\n"
        "# Location: micu\n"
        "~ 42\n"
    )

    refs, comments, layouts, gaps, malformed = parse_master(master, tmp_path / "headers_v2")

    assert refs == [
        "p07/p070491/3053188_0001.hea",
        "p07/p070491/3053188_0002.hea",
        "p07/p070491/3053188_0003.hea",
        "p07/p070491/3053188_0004.hea",
    ]
    assert (comments, layouts, gaps, malformed) == (1, 1, 1, [])


def test_source_derived_generation_deduplicates_and_rejects_malformed(tmp_path: Path):
    headers = tmp_path / "headers_v2" / "p00" / "p000001"
    headers.mkdir(parents=True)
    (tmp_path / "mimic_biosignal_master_headers.txt").write_text(
        "p00/p000001/a.hea\np00/p000001/b.hea\n"
    )
    content = (
        "record/2 2 125 100\n"
        "record_layout 0\n"
        "1000001_0001 50\n"
        "# metadata\n"
        "~ 5\n"
    )
    (headers / "a.hea").write_text(content)
    (headers / "b.hea").write_text(content)

    required, summary = generate_required_list(tmp_path)

    assert required == ["p00/p000001/1000001_0001.hea"]
    assert summary.duplicates == 1
    assert summary.comments == 2
    assert summary.layouts == 2
    assert summary.gaps == 2
    assert summary.malformed == 0

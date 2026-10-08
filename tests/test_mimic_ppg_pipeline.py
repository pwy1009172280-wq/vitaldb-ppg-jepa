import csv
from scripts.mimic_ppg_pipeline import inventory, manifest

def test_metadata_pipeline_keeps_unresolved_choices(tmp_path):
    (tmp_path/'RECORDS-waveforms').write_text('p00/p000001/p000001-2000-01-01-00-00\n')
    (tmp_path/'waveform_header_urls.txt').write_text('https://physionet.org/files/mimic3wdb-matched/1.0/p00/p000001/p000001-2000-01-01-00-00.hea\n')
    (tmp_path/'layout_header_urls.txt').write_text('https://physionet.org/files/mimic3wdb-matched/1.0/p00/p000001/1_layout.hea\n')
    a=type('A',(),dict(root=str(tmp_path),output='inventory.csv',report='report.json')); inventory(a)
    p=tmp_path/'headers_v2/p00/p000001/p000001-2000-01-01-00-00.hea'; p.parent.mkdir(parents=True)
    p.write_text('record 2 125 1000\nseg.dat 16 100/mV 0 0 0 0 0 PLETH\nseg.dat 16 100/mV 0 0 0 0 0 II\n')
    inventory(a)
    b=type('B',(),dict(root=str(tmp_path),inventory='inventory.csv',output='manifest.csv',pleth_regex=None)); manifest(b)
    row=next(csv.DictReader((tmp_path/'manifest.csv').open()))
    assert row['pleth_presence']=='unresolved' and row['continuity_information'].startswith('unresolved')

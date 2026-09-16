import csv
from pathlib import Path
from scripts.download_vitaldb_ppg import filter_tracks, filename

def test_filter_tracks_and_filename():
 rows=[{"caseid":"1","tname":"SNUADC/PLETH","tid":"10"},{"caseid":"2","tname":"ECG","tid":"11"},{"caseid":"3","tname":"SNUADC/PLETH","tid":"12"}]
 assert [r['tid'] for r in filter_tracks(rows)]==['10','12']; assert [r['tid'] for r in filter_tracks(rows,'3')]==['12']; assert filename(1,'10')=='case_0001__10.csv'

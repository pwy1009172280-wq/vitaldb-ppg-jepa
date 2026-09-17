#!/usr/bin/env python3
"""Build an auditable exact-file allowlisted formal-project archive."""
import argparse, io, tarfile
from pathlib import Path
FILES = tuple("""README.md
configs/smoke/mae.yaml
configs/smoke/data2vec.yaml
configs/smoke/jepa.yaml
configs/formal/README.md
envs/formal-ssl-requirements.txt
docs/formal_stage1_mae.md
docs/formal_stage2_data2vec.md
docs/formal_stage3_jepa.md
docs/cluster_runbook.md
scripts/__init__.py
scripts/prepare_vitaldb_ppg.py
scripts/train_ssl.py
scripts/package_for_cluster.py
slurm/train_mae.sbatch
slurm/train_data2vec.sbatch
slurm/train_jepa.sbatch
src/data/__init__.py
src/data/ppg_preprocessing.py
src/data/processed_dataset.py
src/models/common/__init__.py
src/models/common/config.py
src/models/common/masking.py
src/models/common/patch_embed.py
src/models/common/positional_encoding.py
src/models/common/transformer.py
src/models/common/types.py
src/models/mae/__init__.py
src/models/mae/model.py
src/models/data2vec/__init__.py
src/models/data2vec/masking.py
src/models/data2vec/model.py
src/models/jepa/__init__.py
src/models/jepa/masking.py
src/models/jepa/model.py
src/models/factory.py
src/pretrain/config.py
src/pretrain/optim.py
src/pretrain/runner_utils.py
tests/test_formal_stage1.py
tests/test_formal_stage2_data2vec.py
tests/test_formal_stage3_jepa.py
tests/test_formal_runner.py""".splitlines())
def manifest(root):
 root=Path(root).resolve(); return sorted(rel for rel in FILES if (root/rel).is_file())
def package(root, output):
 root=Path(root).resolve(); output=Path(output).resolve(); paths=manifest(root); output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(output,'w:gz') as tar:
  for rel in paths: tar.add(root/rel,arcname=rel)
  data=('\n'.join(paths)+'\n').encode(); info=tarfile.TarInfo('PACKAGE_MANIFEST.txt'); info.size=len(data); tar.addfile(info,io.BytesIO(data))
 return output,paths
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--output'); ap.add_argument('--list-only',action='store_true'); a=ap.parse_args(); paths=manifest(a.root)
 if a.list_only: print('\n'.join(paths))
 elif a.output: print(package(a.root,a.output)[0])
 else: ap.error('--output is required unless --list-only')

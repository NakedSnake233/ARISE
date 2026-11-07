"""Validate the post-run seal before any reuse of this frozen experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2];os.chdir(root)
subprocess.run([sys.executable,'scripts/care/seal_full_suite.py','--verify'],check=True)
state=json.loads(Path('artifacts/care/full-v3/status.json').read_text())
if state['state']=='complete':
    print('All configurations already complete; verified caches/models, no retraining or status overwrite.')
else:
    os.execv(sys.executable,[sys.executable,'-u','scripts/care/run_full_suite.py','--config','configs/care/full-v3.json'])

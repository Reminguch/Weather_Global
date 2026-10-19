"""Verify actual H200 hardware before running any model process."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hardware-dir',type=Path,required=True)
p.add_argument('worker',nargs=argparse.REMAINDER)
a=p.parse_args()
if a.worker and a.worker[0]=='--':a.worker=a.worker[1:]
if not a.worker or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm worker arguments required')
r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True,text=True,capture_output=True)
if not r.stdout.strip() or any('H200' not in row for row in r.stdout.splitlines()):raise RuntimeError('Expected H200: '+r.stdout)
a.hardware_dir.mkdir(parents=True,exist_ok=True)
report=dict(hardware=r.stdout.strip().splitlines(),job_id=os.environ['SLURM_JOB_ID'],restart=os.environ.get('SLURM_RESTART_COUNT','0'),node=os.environ.get('SLURM_JOB_NODELIST'),worker=a.worker)
path=a.hardware_dir/(report['job_id']+'-restart'+report['restart']+'.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
os.execv(sys.executable,[sys.executable,'-u']+a.worker)

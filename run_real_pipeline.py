"""One-shot download -> preprocess -> experiments -> reports, all local."""
import argparse
from pathlib import Path
import subprocess
import sys


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,10)))
    p.add_argument('--targets',type=int,nargs='+')
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--provider',choices=['upstream','nemar'],default='nemar')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--output',default='results/real/exploratory')
    a=p.parse_args()
    if len(set(a.subjects))<3 or not set(a.targets or a.subjects)<=set(a.subjects):
        p.error('Need at least 3 subjects; targets must be among subjects')
    root=Path(__file__).resolve().parent
    commands=[
        ['download_real_data.py','--subjects',*map(str,a.subjects),'--provider',a.provider,'--workers','4'],
        ['prepare_real_data.py','--subjects',*map(str,a.subjects)],
        ['run_real_batch.py','--subjects',*map(str,a.subjects),'--targets',*map(str,a.targets or a.subjects),
         '--seeds',*map(str,a.seeds),'--epochs',str(a.epochs),'--output',a.output],
    ]
    for command in commands:
        print('STAGE '+command[0],flush=True)
        subprocess.run([sys.executable,*command],cwd=root,check=True)

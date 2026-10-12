"""Sequential GPU jobs with failure logging and completed-run resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,10)))
    p.add_argument('--targets',type=int,nargs='+',default=list(range(1,10)))
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--input-scale',type=float,default=1.)
    p.add_argument('--output',default='results/real/exploratory')
    p.add_argument('--data-root', default='data/processed')
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--sampling-rate', type=float, default=125.)
    p.add_argument('--epoch-start', type=float, default=2.)
    p.add_argument('--epoch-end', type=float, default=6.)
    p.add_argument('--temporal-kernel', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--adapt-epochs', type=int, default=50)
    p.add_argument('--adapt-lr', type=float, default=0.001)
    p.add_argument('--budgets', type=int, nargs='+', default=[1,3,5,10])
    p.add_argument('--euclidean-alignment', action='store_true')
    a=p.parse_args()
    out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
    plan=vars(a)|{'created_unix':time.time(),'note':'Exploratory defaults, not a confirmatory protocol.'}
    plan_path=out/'batch_plan.json'
    if not plan_path.exists(): plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for target in a.targets:
        for seed in a.seeds:
            run=out/f'target_{target:02d}_seed_{seed}'
            if (run/'metrics.json').exists():
                previous=json.loads((run/'config.json').read_text(encoding='utf-8'))
                expected = {'subjects': a.subjects, 'epochs': a.epochs, 'input_scale_microvolt': a.input_scale,
                            'data_root': a.data_root, 'samples': a.samples, 'sampling_rate': a.sampling_rate,
                            'epoch_start': a.epoch_start, 'epoch_end': a.epoch_end,
                            'temporal_kernel': a.temporal_kernel, 'dropout': a.dropout,
                            'patience': a.patience, 'adapt_epochs': a.adapt_epochs,
                            'adapt_lr': a.adapt_lr, 'budgets': a.budgets}
                expected['euclidean_alignment'] = a.euclidean_alignment
                if any(previous.get(k) != v for k,v in expected.items()):
                    raise ValueError('Completed run has different configuration; choose new output')
                print(f'SKIP {run.name}',flush=True); continue
            command=[sys.executable,str(ROOT/'run_real_experiment.py'),'--subjects',*map(str,a.subjects),
                     '--target',str(target),'--seed',str(seed),'--epochs',str(a.epochs),
                     '--patience',str(a.patience),'--adapt-epochs',str(a.adapt_epochs),
                     '--adapt-lr',str(a.adapt_lr),'--input-scale',str(a.input_scale),
                     '--output',a.output,'--data-root',a.data_root,'--samples',str(a.samples),
                     '--sampling-rate',str(a.sampling_rate),'--epoch-start',str(a.epoch_start),
                     '--epoch-end',str(a.epoch_end),'--temporal-kernel',str(a.temporal_kernel),
                     '--dropout',str(a.dropout),'--budgets',*map(str,a.budgets)]
            if a.euclidean_alignment:
                command.append('--euclidean-alignment')
            print(f'START {run.name}',flush=True)
            with (out/f'{run.name}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Experiment failed: see {out}/{run.name}.log')
            print(f'COMPLETE {run.name}',flush=True)
    subprocess.run([sys.executable,str(ROOT/'summarize_real_results.py'),'--root',str(out)],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'audit_real_results.py'),'--root',str(out)],check=True,cwd=ROOT)


if __name__=='__main__': main()

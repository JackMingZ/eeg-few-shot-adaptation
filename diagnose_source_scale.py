"""Source-only scale diagnostic; never loads held-out target subject 1."""
import json
from pathlib import Path
import numpy as np
import torch
from src.real_eeg import load_data,EEGNet,seed_all
from run_real_experiment import train_source

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    X,labels,subjects,sessions,ids,hashes=load_data(ROOT/'data/processed',[2,3])
    out=ROOT/'results/real/source_only_diagnostic_repro';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_session=np.flatnonzero((subjects==3)&(sessions=='0train'))
    assert len(training_session)==288
    protocols={'within_source_runs':(training_session[:240],training_session[240:]),
               'cross_source':(np.flatnonzero(subjects==3),np.flatnonzero(subjects==2))}
    records=[]
    for mode,(train,val) in protocols.items():
        for scale in [1.,10.,100.]:
            seed_all(7);model=EEGNet().to(device)
            history=train_source(model,torch.from_numpy(X[:,None]/scale),torch.from_numpy(labels).long(),train,val,device,100,100)
            (out/f'{mode}_scale_{int(scale)}.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            records.append({'mode':mode,'scale':scale,'best':min(history,key=lambda r:r['validation_loss']),
                            'train_ids':ids[train].tolist(),'validation_ids':ids[val].tolist()})
    (out/'diagnostic.json').write_text(json.dumps({'purpose':'Source-only engineering diagnostic; not target test results',
          'seed':7,'cache_sha256':hashes,'results':records},indent=2),encoding='utf-8')

"""Evaluate a saved model on exactly its recorded final test IDs."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from src.real_eeg import EEGNet, load_data

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--method',choices=['source','head','full','affine_head','scale_head','csp_lda'],default='source')
    p.add_argument('--budget',type=int,default=0)
    a=p.parse_args();directory=Path(a.run_dir)
    cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    split=json.loads((directory/'splits.json').read_text(encoding='utf-8'))
    X,y,_,_,ids,hashes=load_data(ROOT/'data/processed',[cfg['target']])
    if hashes[str(cfg['target'])]!=cfg['cache_sha256'][str(cfg['target'])]:
        raise ValueError('Evaluation cache changed since training')
    by_id={key:i for i,key in enumerate(ids)}
    idx=np.asarray([by_id[key] for key in split['test']])
    if a.method in ['source','csp_lda'] and a.budget!=0: p.error('Source baselines require budget 0')
    stem=f'{a.method}_k{a.budget}'
    if a.method=='csp_lda':
        model=joblib.load(directory/'csp_lda.joblib')
        pred=model.predict(X[idx].astype(np.float64))
    else:
        torch.set_num_threads(4)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint=torch.load(directory/f'{stem}.pt',weights_only=True,map_location=device)
        model=EEGNet().to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
        output=[]
        with torch.no_grad():
            for batch in np.array_split(idx,max(1,int(np.ceil(len(idx)/128)))):
                x=torch.from_numpy(X[batch,None]/cfg['input_scale_microvolt']).to(device)
                output.append(model(x).argmax(1).cpu().numpy())
        pred=np.concatenate(output)
    with np.load(directory/f'{stem}_predictions.npz',allow_pickle=False) as original:
        if not np.array_equal(pred,original['y_pred']): raise AssertionError('Reloaded predictions changed')
    metrics={'method':a.method,'budget_per_class':a.budget,'n_test':len(idx),'predictions_identical':True,
             'accuracy':float(accuracy_score(y[idx],pred)),
             'macro_f1':float(f1_score(y[idx],pred,average='macro',zero_division=0)),
             'kappa':float(cohen_kappa_score(y[idx],pred)),
             'confusion_matrix':confusion_matrix(y[idx],pred,labels=range(4)).tolist()}
    (directory/f'{stem}_reevaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics))

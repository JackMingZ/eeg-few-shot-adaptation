"""Recompute metrics and verify saved trial partitions/labels/provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,confusion_matrix
from src.real_eeg import state_hash

ROOT=Path(__file__).resolve().parent


def audit(root):
    paths = sorted(Path(root).glob('target_*/metrics.json'))
    if not paths:
        raise ValueError('No completed runs: pass the parent of target_XX_seed_YY, not the run itself')
    csp_checked = False
    findings=[]
    cache_hashes={}
    data_roots={}
    for metrics_path in paths:
        directory=metrics_path.parent
        cfg=json.loads((directory/'config.json').read_text())
        split=json.loads((directory/'splits.json').read_text())
        for subject,expected in cfg['cache_sha256'].items():
            if subject not in cache_hashes:
                cache_root=ROOT/cfg.get('data_root','data/processed')
                cache=cache_root/f'subject_{int(subject):02d}.npz'
                cache_hashes[subject]=hashlib.sha256(cache.read_bytes()).hexdigest()
            assert cache_hashes[subject]==expected,'Input cache changed'
        source=torch.load(directory/'source.pt',map_location='cpu',weights_only=True)
        source_digest=state_hash(source['state_dict'])
        assert source_digest==source['source_sha256']
        partitions=[set(split[k]) for k in ['train','validation','calibration_pool','test']]
        for i,a in enumerate(partitions):
            for b in partitions[i+1:]: assert a.isdisjoint(b),'Partition overlap'
        if cfg.get('euclidean_alignment', False):
            ea=json.loads((directory/'euclidean_alignment.json').read_text(encoding='utf-8'))
            allowed={'source_train':set(split['train']),
                     'unlabeled_validation_domain':set(split['validation']),
                     'unlabeled_target_calibration_pool':set(split['calibration_pool'])}
            for record in ea['fit_trial_ids_by_role']:
                fitted=set(record['fit_trial_ids'])
                assert record['role'] in allowed and fitted <= allowed[record['role']]
                assert fitted.isdisjoint(partitions[3]), 'EA used final test trials'
        previous=set()
        for key in sorted(split['calibration'],key=int):
            selected=set(split['calibration'][key]); assert previous<=selected<=partitions[2]
            assert len(selected)==int(key)*4
            previous=selected
        for name,expected in cfg['code_sha256'].items():
            assert hashlib.sha256((directory/'code'/name).read_bytes()).hexdigest()==expected,'Code snapshot differs'
        cache_root=ROOT/cfg.get('data_root','data/processed')
        with np.load(cache_root/f'subject_{cfg["target"]:02d}.npz',allow_pickle=False) as d:
            label_by_id=dict(zip(d['trial_id'],d['y']))
            indices={key:i for i,key in enumerate(d['trial_id'])}
            test_x=d['X'][[indices[key] for key in split['test']]]
        for key,ids in split['calibration'].items():
            counts=np.bincount([label_by_id[i] for i in ids],minlength=4)
            assert np.all(counts==int(key))
        for row in json.loads(metrics_path.read_text()):
            stem=f'{row["method"]}_k{row["budget_per_class"]}'
            if row['method']!='csp_lda':
                assert row['source_sha256']==source_digest
                checkpoint=torch.load(directory/f'{stem}.pt',map_location='cpu',weights_only=True)
                assert checkpoint['source_sha256']==source_digest
            with np.load(directory/f'{stem}_predictions.npz',allow_pickle=False) as prediction:
                ids=prediction['trial_id']; truth=prediction['y_true']; pred=prediction['y_pred']
                assert list(ids)==split['test']
                assert np.array_equal(truth,[label_by_id[key] for key in ids])
                actual={'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,average='macro',zero_division=0),
                        'kappa':cohen_kappa_score(truth,pred)}
                for key,value in actual.items(): assert abs(row[key]-value)<1e-12
                assert np.array_equal(prediction['confusion_matrix'],confusion_matrix(truth,pred,labels=range(4)))
                if row['method']=='csp_lda':
                    csp_checked = True
                    reloaded=joblib.load(directory/'csp_lda.joblib')
                    assert np.array_equal(pred,reloaded.predict(test_x.astype(np.float64)))
                if row['method']=='source' and len(set(pred))==1:
                    findings.append({'run':directory.name,'warning':'Source checkpoint predicts a single class; not a useful transfer baseline.'})
    report={'completed_runs':len(paths),
            'checks_passed':['partition disjointness','nested class-balanced calibration','code snapshots','input cache hashes','shared source checkpoint hashes','prediction IDs and labels','recomputed metrics','confusion matrices'] + (['CSP checkpoint reload'] if csp_checked else []),
            'diagnostic_warnings':findings}
    (Path(root)/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    audit(p.parse_args().root)

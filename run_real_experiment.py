"""Real EEG experiment. Configuration fixed before inspecting test scores."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from torch import nn

from src.real_eeg import (EEGNet, seed_all, load_data, make_split, configure_adaptation, state_hash,
                          euclidean_alignment_transform, apply_euclidean_alignment)

ROOT = Path(__file__).resolve().parent


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def train_source(model, x, y, train, val, device, epochs, patience):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history, best, stale = [], float('inf'), 0
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in np.array_split(np.random.permutation(train), max(1, int(np.ceil(len(train) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch].to(device)), y[batch].to(device))
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite source loss')
            loss.backward()
            optimizer.step()
            model.constrain()
            losses.append((float(loss.detach()), len(batch)))
        model.eval()
        with torch.no_grad():
            logits = predict_logits(model, x, val, device)
            val_loss = float(loss_fn(logits, y[val]))
        history.append({'epoch': epoch, 'train_loss': sum(a*b for a,b in losses)/len(train),
                        'validation_loss': val_loss, 'validation_accuracy': float((logits.argmax(1) == y[val]).float().mean())})
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'epoch={epoch} val_loss={val_loss:.4f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_logits(model, x, indices, device):
    model.eval()
    logits=torch.cat([model(x[b].to(device)).cpu() for b in np.array_split(indices, max(1, int(np.ceil(len(indices)/128))))])
    if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite predictions')
    return logits


def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'kappa': float(cohen_kappa_score(y, pred))}


def run(args):
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, subjects, sessions, trial_ids, hashes = load_data(
        ROOT/args.data_root, args.subjects, args.samples, args.sampling_rate,
        (args.epoch_start, args.epoch_end))
    train, val, pool, test, ordered, val_subject = make_split(labels, subjects, sessions, args.target, args.seed, max(args.budgets))
    if args.euclidean_alignment:
        X_model = X.astype(np.float32).copy()
        fit_records = []
        for subject in sorted(set(subjects[train])):
            subject_indices = np.flatnonzero(subjects == subject)
            fit_indices = train[subjects[train] == subject]
            matrix = euclidean_alignment_transform(X[fit_indices])
            X_model[subject_indices] = apply_euclidean_alignment(X[subject_indices], matrix)
            fit_records.append({'subject': int(subject), 'fit_trial_ids': trial_ids[fit_indices].tolist(), 'role': 'source_train'})
        val_indices = np.flatnonzero(subjects == val_subject)
        val_matrix = euclidean_alignment_transform(X[val_indices])
        X_model[val_indices] = apply_euclidean_alignment(X[val_indices], val_matrix)
        fit_records.append({'subject': int(val_subject), 'fit_trial_ids': trial_ids[val].tolist(), 'role': 'unlabeled_validation_domain'})
        target_indices = np.flatnonzero(subjects == args.target)
        target_matrix = euclidean_alignment_transform(X[pool])
        X_model[target_indices] = apply_euclidean_alignment(X[target_indices], target_matrix)
        fit_records.append({'subject': int(args.target), 'fit_trial_ids': trial_ids[pool].tolist(), 'role': 'unlabeled_target_calibration_pool'})
    else:
        X_model = X
    # Fixed physical-unit scaling, no estimation using target data.
    x = torch.from_numpy((X_model / args.input_scale).astype(np.float32))[:, None]
    y = torch.from_numpy(labels).long()
    out = ROOT / args.output / f'target_{args.target:02d}_seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed experiment: {out}')
    config = vars(args) | {'device': str(device), 'torch': str(torch.__version__), 'cache_sha256': hashes,
                           'validation_subject': val_subject, 'input_scale_microvolt':args.input_scale,
                           'adaptation_batchnorm':'frozen source statistics for all methods',
                           'euclidean_alignment': bool(args.euclidean_alignment),
                           'status':'running'}
    config['code_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ['src/real_eeg.py','run_real_experiment.py','prepare_real_data.py']}
    for name in config['code_sha256']:
        snapshot=out/'code'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    split = {'train': trial_ids[train].tolist(), 'validation':trial_ids[val].tolist(),
             'calibration_pool':trial_ids[pool].tolist(), 'test':trial_ids[test].tolist(),
             'calibration':{str(k):trial_ids[np.concatenate([v[:k] for v in ordered])].tolist() for k in args.budgets}}
    (out/'splits.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    if args.euclidean_alignment:
        (out/'euclidean_alignment.json').write_text(json.dumps({'fit_trial_ids_by_role': fit_records,
            'uses_target_unlabeled_pool': True, 'shrinkage': 1e-6}, indent=2), encoding='utf-8')
    model = EEGNet(samples=args.samples, temporal_kernel=args.temporal_kernel, dropout=args.dropout).to(device)
    sync(device); start=time.perf_counter()
    history=train_source(model,x,y,train,val,device,args.epochs,args.patience)
    config['best_epoch']=min(history,key=lambda row:row['validation_loss'])['epoch']
    sync(device); source_seconds=time.perf_counter()-start
    (out/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    source_hash=state_hash(state)
    torch.save({'state_dict':state,'config':config,'source_sha256':source_hash},out/'source.pt')
    rows=[]

    def evaluate(model, method, budget, elapsed, trainable, peak):
        sync(device); before=time.perf_counter()
        logits=predict_logits(model,x,test,device)
        sync(device); inference=time.perf_counter()-before
        pred=logits.argmax(1).numpy()
        row={'target':args.target,'seed':args.seed,'method':method,'budget_per_class':budget,
             **metrics(labels[test],pred),'n_test':len(test),'parameters':sum(p.numel() for p in model.parameters()),
             'predicted_classes':int(len(np.unique(pred))),'n_train':len(train),'n_validation':len(val),
             'trainable_parameters':trainable,'adaptation_seconds':elapsed,
             'source_training_seconds':source_seconds,'inference_seconds':inference,
             'peak_cuda_bytes':peak,'source_sha256':source_hash}
        rows.append(row)
        stem=f'{method}_k{budget}'
        np.savez_compressed(out/f'{stem}_predictions.npz',trial_id=trial_ids[test],y_true=labels[test],
                            y_pred=pred,logits=logits.numpy(),confusion_matrix=confusion_matrix(labels[test],pred,labels=range(4)))
        checkpoint=out/f'{stem}.pt'
        torch.save({'state_dict':model.state_dict(),'config':config,'method':method,
                    'source_sha256':source_hash,'budget_per_class':budget},checkpoint)
        restored=EEGNet(samples=args.samples, temporal_kernel=args.temporal_kernel, dropout=args.dropout).to(device)
        restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state_dict'])
        reloaded=predict_logits(restored,x,test,device)
        if not torch.equal(logits.argmax(1),reloaded.argmax(1)) or not torch.allclose(logits,reloaded,atol=1e-6):
            raise AssertionError('Checkpoint reload differs')
        print(json.dumps(row),flush=True)

    evaluate(model,'source',0,0.,0,0)
    for budget in args.budgets:
        if budget == 0:
            continue
        calibration=np.concatenate([v[:budget] for v in ordered])
        for method in ('head','full','affine_head','scale_head'):
            seed_all(args.seed)
            adapted=EEGNet(samples=args.samples, temporal_kernel=args.temporal_kernel, dropout=args.dropout).to(device)
            adapted.load_state_dict(state)
            assert state_hash(adapted.state_dict()) == source_hash
            configure_adaptation(adapted,method)
            params=[p for p in adapted.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(params,lr=args.adapt_lr)
            xb,yb=x[calibration].to(device),y[calibration].to(device)
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            sync(device); before=time.perf_counter()
            adaptation_history=[]
            for _ in range(args.adapt_epochs):
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(adapted(xb),yb)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite adaptation loss')
                loss.backward(); optimizer.step(); adapted.constrain()
                adaptation_history.append(float(loss.detach()))
            sync(device); elapsed=time.perf_counter()-before
            peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0
            # Frozen parameters and BN buffers must remain bit-identical.
            for key,value in adapted.state_dict().items():
                parameter=dict(adapted.named_parameters()).get(key)
                if parameter is None or not parameter.requires_grad:
                    assert torch.equal(value.cpu(),state[key]), f'Frozen state changed: {key}'
            evaluate(adapted,method,budget,elapsed,sum(p.numel() for p in params),peak)
            (out/f'{method}_k{budget}_loss.json').write_text(json.dumps(adaptation_history),encoding='utf-8')

    if not args.skip_csp:
        import mne
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        mne.set_log_level('WARNING')
        csp=make_pipeline(CSP(n_components=8,reg='ledoit_wolf',log=True,norm_trace=False),
                          LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
        before=time.perf_counter()
        csp.fit(X_model[train].astype(np.float64),labels[train])
        seconds=time.perf_counter()-before
        pred=csp.predict(X_model[test].astype(np.float64))
        rows.append({'target':args.target,'seed':args.seed,'method':'csp_lda','budget_per_class':0,
                     **metrics(labels[test],pred),'n_test':len(test),'source_training_seconds':seconds})
        joblib.dump(csp,out/'csp_lda.joblib')
        reloaded_csp=joblib.load(out/'csp_lda.joblib')
        if not np.array_equal(pred,reloaded_csp.predict(X_model[test].astype(np.float64))):
            raise AssertionError('CSP checkpoint reload differs')
        np.savez_compressed(out/'csp_lda_k0_predictions.npz',trial_id=trial_ids[test],y_true=labels[test],y_pred=pred,
                            confusion_matrix=confusion_matrix(labels[test],pred,labels=range(4)))
        print(json.dumps(rows[-1]),flush=True)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    config['status']='complete'
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,10)))
    p.add_argument('--data-root', default='data/processed')
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--temporal-kernel', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--sampling-rate', type=float, default=125.)
    p.add_argument('--epoch-start', type=float, default=2.)
    p.add_argument('--epoch-end', type=float, default=6.)
    p.add_argument('--target',type=int,default=1)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--adapt-epochs',type=int,default=50)
    p.add_argument('--adapt-lr',type=float,default=0.001)
    p.add_argument('--input-scale',type=float,default=1.,help='Divide microvolt input by this constant; default preserves microvolts')
    p.add_argument('--budgets',type=int,nargs='+',default=[1,3,5,10])
    p.add_argument('--output',default='results/real/pilot')
    p.add_argument('--skip-csp',action='store_true')
    p.add_argument('--euclidean-alignment', action='store_true')
    args=p.parse_args()
    if args.target not in args.subjects or min(args.budgets)<0 or args.epochs<1 or args.adapt_epochs<1 or not np.isfinite(args.input_scale) or args.input_scale<=0:
        p.error('Invalid target, budgets or epoch count')
    run(args)

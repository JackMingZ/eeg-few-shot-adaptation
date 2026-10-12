"""Reload an EEGNet checkpoint; input is cached, preprocessed microvolt EEG."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.real_eeg import EEGNet


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True,help='NPZ with X shaped [trials,22,500], in microvolts')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=True)
    model=EEGNet().to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with np.load(a.data,allow_pickle=False) as d:
        x=d['X'].astype(np.float32)
        if x.ndim!=3 or x.shape[1:]!=(22,500) or not np.isfinite(x).all():
            raise ValueError('Input must be finite [N,22,500] preprocessed EEG')
        ids=d['trial_id'] if 'trial_id' in d else np.arange(len(x)).astype(str)
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            xb=torch.from_numpy(x[start:start+128,None]/checkpoint['config']['input_scale_microvolt']).to(device)
            outputs.append(model(xb).softmax(1).cpu().numpy())
    probabilities=np.concatenate(outputs)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,trial_id=ids,probabilities=probabilities,y_pred=probabilities.argmax(1))
    print(json.dumps({'samples':len(x),'output':str(out),'class_order':['left_hand','right_hand','feet','tongue']}))


if __name__=='__main__': main()

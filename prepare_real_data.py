"""Download BNCI2014-001 via MOABB and cache cue-aligned EEG trials."""
import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def prepare(subjects, output_dir='processed', sampling_rate=125., epoch_start=2., epoch_end=6.):
    data_root = ROOT / 'data'
    data_root.mkdir(exist_ok=True)
    os.environ['MNE_DATA'] = str(data_root)
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(data_root)
    os.environ['MOABB_DOWNLOAD_PROVIDER'] = 'upstream'
    warnings.filterwarnings('ignore', message="Montage name 'standard_1005' is deprecated", category=FutureWarning)
    import mne
    from moabb.datasets import BNCI2014_001

    mne.set_log_level('WARNING')
    dataset = BNCI2014_001(artifact_handling='ignore')
    destination = data_root / output_dir
    destination.mkdir(exist_ok=True)
    expected_samples = int(round((epoch_end - epoch_start) * sampling_rate))
    for subject in subjects:
        out = destination / f'subject_{subject:02d}.npz'
        if out.exists():
            with np.load(out, allow_pickle=False) as cached:
                if cached['X'].shape[1:] != (22, expected_samples) or not np.isfinite(cached['X']).all():
                    raise ValueError(f'Invalid cache {out}')
            print(f'CACHE {out}', flush=True)
            continue
        from download_real_data import fetch
        raw_records = [fetch(f'A{subject:02d}{session}.mat') for session in ('T', 'E')]
        raws = dataset.get_data(subjects=[subject])[subject]
        arrays, labels, sessions, ids = [], [], [], []
        channels = None
        for session, runs in sorted(raws.items()):
            for run_name, raw in sorted(runs.items()):
                events = mne.find_events(raw, shortest_event=1, verbose=False)
                events = events[np.isin(events[:, 2], list(dataset.event_id.values()))]
                if not len(events):
                    continue
                eeg = raw.copy().pick('eeg').load_data()
                # Fixed filter, applied independently to each recording/run.
                eeg.filter(4., 38., method='fir', phase='zero', verbose=False)
                epochs = mne.Epochs(eeg, events, event_id=dataset.event_id,
                                    tmin=epoch_start, tmax=epoch_end - 1. / raw.info['sfreq'],
                                    baseline=None, preload=True, reject_by_annotation=False,
                                    on_missing='ignore', verbose=False)
                epochs.resample(sampling_rate, verbose=False)
                x = epochs.get_data(copy=True).astype(np.float32) * 1e6
                if x.shape[1:] != (22, expected_samples) or not np.isfinite(x).all():
                    raise ValueError(f'Unexpected EEG shape or values: {x.shape}')
                channels = epochs.ch_names
                arrays.append(x)
                labels.extend((epochs.events[:, 2] - 1).tolist())
                sessions.extend([session] * len(x))
                ids.extend([f'A{subject:02d}/{session}/{run_name}/{int(e[0])}' for e in epochs.events])
        x = np.concatenate(arrays)
        y = np.asarray(labels, dtype=np.int64)
        if len(set(ids)) != len(ids) or set(y) != {0, 1, 2, 3}:
            raise ValueError('Invalid trial IDs or labels')
        meta = {
            'dataset': 'BNCI2014-001 / BCI Competition IV 2a', 'subject': subject,
            'source': 'https://lampx.tugraz.at/~bci/database/001-2014/',
            'classes': dataset.event_id, 'channels': channels, 'unit': 'microvolt',
            'sampling_rate': sampling_rate, 'filter_hz': [4, 38],
            'epoch_seconds_from_trial_start': [epoch_start, epoch_end], 'samples': expected_samples,
            'artifact_policy': 'Keep all trials, including provider-marked artifacts; no label-dependent rejection.',
            'sessions': {s: int(sum(v == s for v in sessions)) for s in sorted(set(sessions))},
            'mne': mne.__version__,
            'raw_files': raw_records,
        }
        np.savez_compressed(out, X=x, y=y, session=np.asarray(sessions),
                            trial_id=np.asarray(ids), metadata=json.dumps(meta))
        meta['cache_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
        out.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(f'PREPARED subject={subject} shape={x.shape} sessions={meta["sessions"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--output-dir', default='processed')
    parser.add_argument('--sampling-rate', type=float, default=125.)
    parser.add_argument('--epoch-start', type=float, default=2.)
    parser.add_argument('--epoch-end', type=float, default=6.)
    args = parser.parse_args()
    if not set(args.subjects) <= set(range(1, 10)):
        parser.error('subjects must be in 1..9')
    if args.sampling_rate <= 0 or args.epoch_end <= args.epoch_start:
        parser.error('invalid sampling rate or epoch interval')
    prepare(args.subjects, args.output_dir, args.sampling_rate, args.epoch_start, args.epoch_end)

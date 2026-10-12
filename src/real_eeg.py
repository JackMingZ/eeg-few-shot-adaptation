"""EEGNet-8,2 style PyTorch model and audited subject/session splits."""
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from torch import nn


def euclidean_alignment_transform(trials, shrinkage=1e-6):
    """Return an unsupervised covariance-whitening transform for (N,C,T) trials."""
    x = np.asarray(trials, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError('EA expects finite non-empty (trials, channels, samples) data')
    centered = x - x.mean(axis=2, keepdims=True)
    covariance = np.einsum('nct,ndt->cd', centered, centered, optimize=True)
    covariance /= float(x.shape[0] * x.shape[2])
    scale = max(float(np.trace(covariance) / covariance.shape[0]), np.finfo(np.float64).eps)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(covariance.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, scale * shrinkage)
    return ((vectors * (1.0 / np.sqrt(values))) @ vectors.T).astype(np.float32)


def apply_euclidean_alignment(trials, transform):
    x = np.asarray(trials, dtype=np.float32)
    if x.ndim != 3 or transform.shape != (x.shape[1], x.shape[1]):
        raise ValueError('EA transform does not match trial channel dimension')
    return np.einsum('cd,ndt->nct', transform, x, optimize=True).astype(np.float32)


class EEGNet(nn.Module):
    """EEGNet blocks with 125 Hz kernels (64 temporal, 16 separable).

    Paper/author-code inspired PyTorch port; not a reproduction of published
    results. SAME padding, BN eps/momentum, two pooling blocks, max norms.
    Identity channel affine parameters stay frozen during source training.
    """
    def __init__(self, channels=22, samples=500, classes=4, temporal_kernel=64, dropout=0.5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, channels, 1), requires_grad=False)
        self.offset = nn.Parameter(torch.zeros(1, 1, channels, 1), requires_grad=False)
        self.temporal = nn.Sequential(
            nn.ZeroPad2d(((temporal_kernel - 1) // 2, temporal_kernel // 2, 0, 0)), nn.Conv2d(1, 8, (1, temporal_kernel), bias=False),
            nn.BatchNorm2d(8, eps=1e-3, momentum=0.01))
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.block1 = nn.Sequential(nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
                                    nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)), nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.head = nn.Linear(16 * (samples // 4 // 8), classes)

    def forward(self, x):
        x = x * self.scale + self.offset
        return self.head(self.block2(self.block1(self.spatial(self.temporal(x)))).flatten(1))

    @torch.no_grad()
    def constrain(self):
        # Match the axes used by Keras depthwise spatial and Dense max_norm.
        for weight, limit in ((self.spatial.weight, 1.), (self.head.weight, 0.25)):
            if weight.requires_grad:
                norms = weight.flatten(1).norm(dim=1).clamp(min=1e-8)
                factors = (limit / norms).clamp(max=1)
                weight.mul_(factors.reshape((-1,) + (1,) * (weight.ndim - 1)))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(root, subjects, samples=500, sampling_rate=125., epoch_seconds=(2., 6.)):
    if len(subjects) != len(set(subjects)):
        raise ValueError('Subject list contains duplicates')
    parts = []
    hashes = {}
    for subject in subjects:
        path = Path(root) / f'subject_{subject:02d}.npz'
        with np.load(path, allow_pickle=False) as d:
            metadata=json.loads(str(d['metadata']))
            if (metadata.get('unit') != 'microvolt' or
                float(metadata.get('sampling_rate', -1)) != float(sampling_rate) or
                list(metadata.get('filter_hz', [])) != [4, 38] or
                list(metadata.get('epoch_seconds_from_trial_start', [])) != list(epoch_seconds)):
                raise ValueError(f'Unexpected preprocessing metadata: {path}')
            if d['X'].shape != (len(d['y']),22,samples) or not np.isfinite(d['X']).all():
                raise ValueError(f'Invalid EEG cache: {path}')
            if len(d['session']) != len(d['y']) or len(d['trial_id']) != len(d['y']):
                raise ValueError(f'Misaligned metadata: {path}')
            parts.append((d['X'], d['y'], np.full(len(d['y']), subject), d['session'], d['trial_id']))
        hashes[str(subject)] = hashlib.sha256(path.read_bytes()).hexdigest()
    merged=[np.concatenate([p[i] for p in parts]) for i in range(5)]
    if len(set(merged[4]))!=len(merged[4]):
        raise ValueError('Duplicate original trial IDs')
    return (*merged, hashes)


def make_split(y, subjects, sessions, target, seed, max_budget=10):
    available = sorted(set(subjects) - {target})
    if len(available) < 2:
        raise ValueError('Need target and at least two source subjects')
    # Cyclic next source subject is validation, independent of target scores.
    validation_subject = next((s for s in available if s > target), available[0])
    train = np.flatnonzero((subjects != target) & (subjects != validation_subject))
    val = np.flatnonzero(subjects == validation_subject)
    pool = np.flatnonzero((subjects == target) & (sessions == '0train'))
    test = np.flatnonzero((subjects == target) & (sessions == '1test'))
    groups = [train, val, pool, test]
    for i, a in enumerate(groups):
        if not len(a) or set(y[a]) != {0, 1, 2, 3}:
            raise ValueError('Empty partition or missing classes')
        for b in groups[i + 1:]:
            if np.intersect1d(a, b).size:
                raise ValueError('Partition leakage')
    rng = np.random.default_rng(seed)
    ordered = [rng.permutation(pool[y[pool] == k]) for k in range(4)]
    if any(len(v) < max_budget for v in ordered):
        raise ValueError('Insufficient calibration data')
    return train, val, pool, test, ordered, int(validation_subject)


def configure_adaptation(model, method):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            (method == 'full' and name not in ('scale', 'offset'))
            or (method in ('head', 'affine_head', 'scale_head') and name.startswith('head.'))
            or (method in ('affine_head', 'scale_head') and name == 'scale')
            or (method == 'affine_head' and name == 'offset'))
    # All adaptation uses fixed source BN statistics, including full fine-tune.
    # This avoids batch-size-dependent BN drift with 1-10 examples/class.
    model.eval()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_counts(model):
    """Return total, trainable, and frozen parameter counts for reporting.

    Keeping this calculation in the project avoids mixing the historical
    1,020-parameter synthetic demo with the current real-data model profile.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        'total': int(total),
        'trainable': int(trainable),
        'frozen': int(total - trainable),
    }

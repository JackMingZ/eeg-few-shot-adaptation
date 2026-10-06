from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def torch_status():
    try:
        import torch
        return {"available": True, "cuda": torch.cuda.is_available(), "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
    except Exception as exc:
        return {"available": False, "cuda": False, "device": "cpu", "error": str(exc)}


def run_gpu_eegnet(trials, labels, subjects, output_dir, seed=7, epochs=35, calibration_trials_per_class=5):
    """Train an EEGNet-style model, preferring CUDA and falling back to CPU."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = np.flatnonzero(~np.isin(subjects, [4, 5]))
    calibration_idx, test_idx = [], []
    rng = np.random.default_rng(seed)
    for subject in [4, 5]:
        for label in np.unique(labels):
            idx = np.flatnonzero((subjects == subject) & (labels == label))
            rng.shuffle(idx)
            calibration_idx.extend(idx[:calibration_trials_per_class])
            test_idx.extend(idx[calibration_trials_per_class:])
    calibration_idx, test_idx = np.array(calibration_idx), np.array(test_idx)

    class EEGNet(nn.Module):
        def __init__(self, channels, samples, classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, (1, 31), padding=(0, 15), bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
                nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
                nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
                nn.Conv2d(16, 16, (1, 1), bias=False), nn.BatchNorm2d(16), nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(16, classes)

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    def fit_eval(fit_idx, name):
        model = EEGNet(trials.shape[1], trials.shape[2], len(np.unique(labels))).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        x = torch.tensor(trials[fit_idx, None], dtype=torch.float32)
        y = torch.tensor(labels[fit_idx], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(trials[test_idx, None], dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
        return {"method": name, "accuracy": accuracy_score(labels[test_idx], pred), "macro_f1": f1_score(labels[test_idx], pred, average="macro"), "kappa": cohen_kappa_score(labels[test_idx], pred), "device": str(device), "parameters": sum(p.numel() for p in model.parameters())}

    results = [fit_eval(train_idx, "eegnet_source_only"), fit_eval(np.concatenate([train_idx, calibration_idx]), "eegnet_few_shot_calibration")]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "gpu_metrics.json").write_text(json.dumps({"status": torch_status(), "results": results}, indent=2), encoding="utf-8")
    return results

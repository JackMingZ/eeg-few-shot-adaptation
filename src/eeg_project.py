from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ExperimentConfig:
    seed: int = 7
    n_subjects: int = 6
    n_classes: int = 4
    trials_per_class: int = 36
    n_channels: int = 8
    n_samples: int = 250
    sampling_rate: float = 125.0
    calibration_trials_per_class: int = 5


def make_synthetic_eeg(cfg: ExperimentConfig):
    """Create controllable multi-subject EEG-like trials.

    The generator includes class-specific mu/beta rhythms and subject-specific
    gain, frequency drift, and noise. It is intended for pipeline validation,
    not for claiming physiological findings.
    """
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.n_samples) / cfg.sampling_rate
    class_freqs = np.array([[10.0, 18.0], [11.0, 20.0], [9.0, 25.0], [12.0, 15.0]])
    class_channel_weights = rng.uniform(0.6, 1.4, size=(cfg.n_classes, cfg.n_channels))
    rows = []
    for subject in range(cfg.n_subjects):
        gain = rng.uniform(0.7, 1.5, size=cfg.n_channels)
        drift = rng.normal(0.0, 0.65)
        noise_scale = rng.uniform(0.35, 0.8)
        for label in range(cfg.n_classes):
            for _ in range(cfg.trials_per_class):
                signal = rng.normal(0, noise_scale, size=(cfg.n_channels, cfg.n_samples))
                f1, f2 = class_freqs[label] + drift
                phase = rng.uniform(0, 2 * np.pi, size=cfg.n_channels)
                amp = class_channel_weights[label] * gain
                signal += amp[:, None] * (
                    0.65 * np.sin(2 * np.pi * f1 * t + phase[:, None])
                    + 0.35 * np.sin(2 * np.pi * f2 * t + phase[:, None] / 2)
                )
                # A slow subject-specific baseline imitates sensor offset.
                signal += rng.normal(0, 0.25, size=(cfg.n_channels, 1))
                rows.append((subject, label, signal.astype(np.float32)))
    subjects = np.array([r[0] for r in rows], dtype=int)
    labels = np.array([r[1] for r in rows], dtype=int)
    trials = np.stack([r[2] for r in rows])
    return trials, labels, subjects


def bandpower_features(trials, sampling_rate, subject_normalize=False):
    """Extract log power in mu (8-13 Hz) and beta (13-30 Hz) bands."""
    x = trials.astype(np.float64).copy()
    if subject_normalize:
        # Per-trial channel normalization removes amplitude differences while
        # preserving relative spectral structure.
        x -= x.mean(axis=2, keepdims=True)
        x /= x.std(axis=2, keepdims=True) + 1e-6
    freqs, psd = periodogram(x, fs=sampling_rate, axis=-1)
    bands = ((8.0, 13.0), (13.0, 30.0))
    features = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        features.append(np.log(psd[..., mask].mean(axis=-1) + 1e-9))
    return np.concatenate(features, axis=1)


def split_cross_subject(subjects, labels, test_subjects=(4, 5), calibration_per_class=0, seed=7):
    rng = np.random.default_rng(seed)
    train_mask = ~np.isin(subjects, test_subjects)
    test_mask = np.isin(subjects, test_subjects)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    calibration_idx = []
    evaluation_idx = []
    for subject in test_subjects:
        for label in np.unique(labels):
            idx = test_idx[(subjects[test_idx] == subject) & (labels[test_idx] == label)]
            idx = idx.copy()
            rng.shuffle(idx)
            calibration_idx.extend(idx[:calibration_per_class])
            evaluation_idx.extend(idx[calibration_per_class:])
    return train_idx, np.array(calibration_idx), np.array(evaluation_idx)


def evaluate_method(name, X, y, train_idx, calibration_idx, test_idx, cfg):
    if name == "global_baseline":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg.seed))
        fit_idx = train_idx
    elif name == "subject_normalized":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg.seed))
        fit_idx = train_idx
    elif name == "few_shot_calibration":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg.seed))
        fit_idx = np.concatenate([train_idx, calibration_idx])
    else:
        raise ValueError(name)
    model.fit(X[fit_idx], y[fit_idx])
    pred = model.predict(X[test_idx])
    return {
        "method": name,
        "accuracy": accuracy_score(y[test_idx], pred),
        "macro_f1": f1_score(y[test_idx], pred, average="macro"),
        "kappa": cohen_kappa_score(y[test_idx], pred),
        "n_train": len(fit_idx),
        "n_test": len(test_idx),
        "pred": pred,
    }


def run_experiment(output_dir: str | Path = "results", cfg: ExperimentConfig | None = None):
    cfg = cfg or ExperimentConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials, labels, subjects = make_synthetic_eeg(cfg)
    train_idx, calibration_idx, test_idx = split_cross_subject(
        subjects, labels, calibration_per_class=cfg.calibration_trials_per_class, seed=cfg.seed
    )
    X_global = bandpower_features(trials, cfg.sampling_rate, subject_normalize=False)
    X_subject = bandpower_features(trials, cfg.sampling_rate, subject_normalize=True)

    # Baseline uses raw spectral features; adapted methods use normalized features.
    results = []
    r = evaluate_method("global_baseline", X_global, labels, train_idx, calibration_idx, test_idx, cfg)
    results.append({k: v for k, v in r.items() if k != "pred"})
    r = evaluate_method("subject_normalized", X_subject, labels, train_idx, calibration_idx, test_idx, cfg)
    results.append({k: v for k, v in r.items() if k != "pred"})
    r = evaluate_method("few_shot_calibration", X_subject, labels, train_idx, calibration_idx, test_idx, cfg)
    results.append({k: v for k, v in r.items() if k != "pred"})
    metrics = pd.DataFrame(results)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    best = metrics.iloc[metrics["accuracy"].argmax()]["method"]
    best_X = X_global if best == "global_baseline" else X_subject
    best_r = evaluate_method(best, best_X, labels, train_idx, calibration_idx, test_idx, cfg)
    cm = confusion_matrix(labels[test_idx], best_r["pred"])
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    summary = {"config": cfg.__dict__, "best_method": best, "metrics": metrics.to_dict(orient="records")}
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics, cm

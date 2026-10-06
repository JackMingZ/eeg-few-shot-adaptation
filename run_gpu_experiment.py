from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eeg_project import ExperimentConfig, make_synthetic_eeg
from src.eegnet_torch import run_gpu_eegnet, torch_status


def main():
    status = torch_status()
    print(status)
    if not status["available"]:
        print("PyTorch is not installed; install from requirements-gpu.txt first.")
        return
    cfg = ExperimentConfig()
    trials, labels, subjects = make_synthetic_eeg(cfg)
    results = run_gpu_eegnet(trials, labels, subjects, Path(__file__).parent / "results", seed=cfg.seed)
    for row in results:
        print(row)
    df = pd.DataFrame(results)
    out = Path(__file__).parent / "results"
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["method"], df["accuracy"], color=["#477998", "#e98a15"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("CUDA EEGNet cross-subject results")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(out / "gpu_accuracy_comparison.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

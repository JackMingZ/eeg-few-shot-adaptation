from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eeg_project import ExperimentConfig, run_experiment


def main():
    metrics, cm = run_experiment(Path(__file__).parent / "results", ExperimentConfig())
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results_dir = Path(__file__).parent / "results"

    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics["method"], metrics["accuracy"], color=["#7aa6c2", "#4d908e", "#f4a261"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Cross-subject EEG classification")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix of best method")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()

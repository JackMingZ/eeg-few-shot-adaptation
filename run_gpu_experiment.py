from pathlib import Path

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


if __name__ == "__main__":
    main()

"""Named input profiles, independent of model and training implementation."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentProfile:
    data_root: str
    samples: int
    sampling_rate: float
    epoch_start: float
    epoch_end: float
    temporal_kernel: int
    dropout: float

PROFILES = {
    'paper': ExperimentProfile('data/processed_paper', 256, 128., 2.5, 4.5, 32, 0.25),
    'legacy': ExperimentProfile('data/processed', 500, 125., 2., 6., 64, 0.5),
}

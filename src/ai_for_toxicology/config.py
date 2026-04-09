"""Shared configuration for model training and evaluation."""

from __future__ import annotations

from pathlib import Path

TOX21_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

DEFAULT_TRAIN_PATH = Path("data/Train/tox21_train_clean.csv")
DEFAULT_VAL_PATH = Path("data/Val/tox21_val_clean.csv")
DEFAULT_TEST_PATH = Path("data/Test/tox21_test_clean.csv")
DEFAULT_CHECKPOINT_PATH = Path(
    "artifacts/end_to_end_checkpoints/e2evae_full_seqconv_ce_phase2_adaptive_best.pt"
)

DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPOCHS = 10
DEFAULT_KL_ANNEAL_EPOCHS = 10
DEFAULT_SEED = 42
DEFAULT_DROPOUT = 0.30
DEFAULT_FOCAL_GAMMA = 4.0

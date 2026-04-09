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

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_CHEMBL_PATH = DEFAULT_DATA_ROOT / "Train" / "chembl_clean.csv"
DEFAULT_ZINC_PATH = DEFAULT_DATA_ROOT / "Train" / "zinc250k_clean.csv"
DEFAULT_TRAIN_PATH = DEFAULT_DATA_ROOT / "Train" / "tox21_train_clean.csv"
DEFAULT_VAL_PATH = DEFAULT_DATA_ROOT / "Val" / "tox21_val_clean.csv"
DEFAULT_TEST_PATH = DEFAULT_DATA_ROOT / "Test" / "tox21_test_clean.csv"
DEFAULT_CHECKPOINT_PATH = Path(
    "artifacts/end_to_end_checkpoints/e2evae_full_seqconv_ce_phase2_adaptive_best.pt"
)

DEFAULT_MAX_LEN = 120
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPOCHS = 10
DEFAULT_KL_ANNEAL_EPOCHS = 10
DEFAULT_SEED = 42
DEFAULT_DROPOUT = 0.10
DEFAULT_FOCAL_GAMMA = 4.0

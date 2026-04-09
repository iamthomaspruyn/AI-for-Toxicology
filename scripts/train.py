#!/usr/bin/env python
"""Train VAEWithPredictor on Tox21 data with notebook-aligned losses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_for_toxicology.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    DEFAULT_WEIGHT_DECAY,
)
from ai_for_toxicology.data import (  # noqa: E402
    build_aligned_dataset,
    load_checkpoint_meta,
    load_tox21_frame,
)
from ai_for_toxicology.model import VAEWithPredictor  # noqa: E402
from ai_for_toxicology.train import TrainConfig, train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("artifacts/end_to_end_checkpoints/submission_train_best.pt"),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for faster local verification runs.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap for faster local verification runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt, meta = load_checkpoint_meta(args.checkpoint)

    train_df = load_tox21_frame(args.train_csv, origin_split="train")
    val_df = load_tox21_frame(args.val_csv, origin_split="val")

    if args.max_train_samples is not None:
        train_df = train_df.head(args.max_train_samples).copy()
    if args.max_val_samples is not None:
        val_df = val_df.head(args.max_val_samples).copy()

    train_data = build_aligned_dataset(train_df, meta)
    val_data = build_aligned_dataset(val_df, meta)

    model = VAEWithPredictor(
        vocab_size=meta.vocab_size,
        seq_len=meta.seq_len,
        latent_dim=meta.latent_dim,
        num_tasks=meta.num_tasks,
        dropout=0.30,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        seed=args.seed,
    )

    train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        cfg=cfg,
        pad_idx=meta.pad_idx,
        checkpoint_out=args.checkpoint_out,
    )

    print("Saved trained checkpoint:", args.checkpoint_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate checkpoint on Tox21 split and export metrics/predictions."""

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
    DEFAULT_DROPOUT,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
)
from ai_for_toxicology.data import (  # noqa: E402
    build_aligned_dataset,
    load_checkpoint_meta,
    load_tox21_frame,
)
from ai_for_toxicology.model import VAEWithPredictor  # noqa: E402
from ai_for_toxicology.test import (  # noqa: E402
    evaluate_split,
    write_metrics_csv,
    write_task_predictions_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("reports/submission_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("reports/submission_predictions.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    ckpt, meta = load_checkpoint_meta(args.checkpoint)

    split_to_csv = {
        "train": args.train_csv,
        "val": args.val_csv,
        "test": args.test_csv,
    }
    split_df = load_tox21_frame(split_to_csv[args.split], origin_split=args.split)
    split_data = build_aligned_dataset(split_df, meta)

    model = VAEWithPredictor(
        vocab_size=meta.vocab_size,
        seq_len=meta.seq_len,
        latent_dim=meta.latent_dim,
        num_tasks=meta.num_tasks,
        dropout=DEFAULT_DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    metrics, probs = evaluate_split(
        model=model,
        split=split_data,
        device=device,
        batch_size=args.batch_size,
    )
    metrics["split"] = args.split

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)

    write_metrics_csv(metrics, str(args.metrics_out))
    write_task_predictions_csv(probs, str(args.predictions_out))

    print("Evaluation metrics:", metrics)
    print("Saved:", args.metrics_out)
    print("Saved:", args.predictions_out)


if __name__ == "__main__":
    main()

"""Evaluation utilities for checkpoint-based toxicity prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import TOX21_TASKS
from .data import Tox21Dataset
from .model import VAEWithPredictor


def _masked_arrays(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool)
    return y_true[valid], y_score[valid]


def _macro_auroc(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for i in range(y_true.shape[1]):
        yt, ys = _masked_arrays(y_true[:, i], y_score[:, i], mask[:, i])
        if len(np.unique(yt)) < 2:
            continue
        vals.append(roc_auc_score(yt, ys))
    return float(np.mean(vals)) if vals else float("nan")


def _macro_auprc(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for i in range(y_true.shape[1]):
        yt, ys = _masked_arrays(y_true[:, i], y_score[:, i], mask[:, i])
        if len(np.unique(yt)) < 2:
            continue
        vals.append(average_precision_score(yt, ys))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_split(
    model: VAEWithPredictor,
    split: dict,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, np.ndarray]:
    loader = DataLoader(
        Tox21Dataset(split["x"], split["y"], split["mask"]),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    probs_chunks = []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(device)
            logits, _, _ = model.predict_logits(xb)
            probs_chunks.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.vstack(probs_chunks)
    metrics = {
        "macro_auroc": _macro_auroc(split["y"], probs, split["mask"]),
        "macro_auprc": _macro_auprc(split["y"], probs, split["mask"]),
        "n_samples": int(len(split["x"])),
    }
    return metrics, probs


def write_metrics_csv(metrics: dict, output_csv: str) -> None:
    row = {"split": metrics.get("split", "unknown")}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(output_csv, index=False)


def write_task_predictions_csv(probs: np.ndarray, output_csv: str) -> None:
    pd.DataFrame(probs, columns=TOX21_TASKS).to_csv(output_csv, index=False)

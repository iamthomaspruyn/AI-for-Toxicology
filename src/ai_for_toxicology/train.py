"""Notebook-aligned multi-phase training utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Tox21Dataset
from .model import VAEWithPredictor


@dataclass
class TrainingLoopConfig:
    """Hyperparameters matching the notebook training loop."""

    seed: int = 42
    batch_size: int = 128
    min_epochs: int = 50
    max_epochs: int = 120
    early_stopping_patience: int = 12
    lr_scheduler_factor: float = 0.25
    lr_scheduler_patience: int = 4
    kl_anneal_epochs: int = 10
    phase1_lr: float = 5e-4
    phase2_lr: float = 5e-5
    phase1_recon_weight: float = 1.0
    phase2_recon_weight: float = 0.0
    phase2_pred_weight: float = 1.0


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    mask: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_tasks: int,
) -> DataLoader:
    """Create loader and mirror notebook defaults for missing labels/masks."""
    if y is None:
        y = np.zeros((len(x), num_tasks), dtype=np.float32)
    if mask is None:
        mask = np.zeros((len(x), num_tasks), dtype=np.float32)
    return DataLoader(Tox21Dataset(x, y, mask), batch_size=batch_size, shuffle=shuffle)


def kl_beta(epoch: int, kl_anneal_epochs: int) -> float:
    return min(1.0, epoch / max(1, kl_anneal_epochs))


def compute_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    pad_idx: int,
    recon_weight: float = 1.0,
    pred_weight: float = 1.0,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute notebook-style summed losses normalized by batch size.

    Returns:
      total, recon_raw_sum, kl_raw_sum, bce_raw_sum
    """
    recon = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        x.view(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum()

    if y is not None and mask.sum() > 0 and pred_weight > 0:
        bce_all = F.binary_cross_entropy_with_logits(
            pred,
            y,
            reduction="none",
            pos_weight=pos_weight,
        )
        bce = (bce_all * mask).sum()
    else:
        bce = torch.tensor(0.0, device=logits.device)

    total = ((recon_weight * recon) + (beta * kl) + (pred_weight * bce)) / x.size(0)
    return total, recon, kl, bce


def run_epoch(
    model: VAEWithPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    pad_idx: int,
    epoch: int,
    cfg: TrainingLoopConfig,
    pretrain_mode: bool,
    optimizer: torch.optim.Optimizer | None = None,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    """Single train/eval epoch matching notebook metrics."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    metrics = {"total": 0.0, "recon": 0.0, "kl": 0.0, "bce": 0.0, "acc": 0.0}
    total_tokens = 0
    beta = kl_beta(epoch, cfg.kl_anneal_epochs)

    if pretrain_mode:
        recon_w, pred_w = cfg.phase1_recon_weight, 0.0
    else:
        recon_w, pred_w = cfg.phase2_recon_weight, cfg.phase2_pred_weight

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        if is_train:
            optimizer.zero_grad()

        logits, mu, logvar, pred = model(x)
        loss, recon_loss, kl_loss, bce_loss = compute_loss(
            logits=logits,
            x=x,
            mu=mu,
            logvar=logvar,
            pred=pred,
            y=y,
            mask=mask,
            beta=beta,
            pad_idx=pad_idx,
            recon_weight=recon_w,
            pred_weight=pred_w,
            pos_weight=pos_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        metrics["total"] += float(loss.item())
        metrics["recon"] += float(recon_loss.item())
        metrics["kl"] += float(kl_loss.item())
        metrics["bce"] += float(bce_loss.item())

        preds = logits.argmax(dim=-1)
        valid = x != pad_idx
        metrics["acc"] += float(((preds == x) & valid).sum().item())
        total_tokens += int(valid.sum().item())

    n_batches = max(1, len(loader))
    n_samples = max(1, len(loader.dataset))
    return {
        "total": metrics["total"] / n_batches,
        "recon_raw": metrics["recon"] / n_samples,
        "recon_per_token": metrics["recon"] / max(1, total_tokens),
        "kl_raw": metrics["kl"] / n_samples,
        "bce_raw": metrics["bce"] / n_samples,
        "token_acc": metrics["acc"] / max(1, total_tokens),
    }


def evaluate(
    model: VAEWithPredictor,
    x: np.ndarray,
    *,
    device: torch.device,
    pad_idx: int,
    num_tasks: int,
    cfg: TrainingLoopConfig,
    y: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epoch: int = 0,
    pretrain_mode: bool = True,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    loader = make_loader(
        x,
        y,
        mask,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_tasks=num_tasks,
    )
    return run_epoch(
        model,
        loader,
        device=device,
        pad_idx=pad_idx,
        epoch=epoch,
        cfg=cfg,
        pretrain_mode=pretrain_mode,
        optimizer=None,
        pos_weight=pos_weight,
    )


def _checkpoint_payload(
    model: nn.Module,
    history: dict[str, list[float]],
    *,
    epoch: int,
    best_epoch: int | None,
    best_val_total: float | None,
    best_val_token_acc: float | None,
    epochs_no_improve: int,
    token_to_idx: dict[str, int],
    seq_len: int,
    vocab_size: int,
    max_len: int,
    pad_idx: int,
    unk_idx: int,
    eos_idx: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    test_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_total": float(best_val_total) if best_val_total is not None else None,
        "best_val_token_acc": (
            float(best_val_token_acc) if best_val_token_acc is not None else None
        ),
        "epochs_no_improve": int(epochs_no_improve),
        "token_to_idx": token_to_idx,
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "max_len": int(max_len),
        "pad_idx": int(pad_idx),
        "unk_idx": int(unk_idx),
        "eos_idx": int(eos_idx),
        "history": history,
        "test_metrics": test_metrics,
        "encoder_layout": "onehot_channels_first_seqconv",
        "decoder_output": "logits",
        "loss_name": "token_cross_entropy_plus_kl",
        "selection_metric": "val_token_acc",
    }


def _save_training_checkpoints(
    model: nn.Module,
    history: dict[str, list[float]],
    *,
    save_dir: Path,
    checkpoint_stem: str,
    epoch: int,
    best_epoch: int | None,
    best_val_total: float | None,
    best_val_token_acc: float | None,
    epochs_no_improve: int,
    token_to_idx: dict[str, int],
    seq_len: int,
    vocab_size: int,
    max_len: int,
    pad_idx: int,
    unk_idx: int,
    eos_idx: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    is_best: bool,
    save_epoch_checkpoints: bool,
) -> None:
    payload = _checkpoint_payload(
        model,
        history,
        epoch=epoch,
        best_epoch=best_epoch,
        best_val_total=best_val_total,
        best_val_token_acc=best_val_token_acc,
        epochs_no_improve=epochs_no_improve,
        token_to_idx=token_to_idx,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_len=max_len,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
        eos_idx=eos_idx,
        optimizer=optimizer,
        scheduler=scheduler,
        test_metrics=None,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / f"{checkpoint_stem}_last.pt"
    torch.save(payload, last_path)

    if save_epoch_checkpoints:
        epoch_path = save_dir / f"{checkpoint_stem}_epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_path)

    if is_best:
        best_path = save_dir / f"{checkpoint_stem}_best.pt"
        torch.save(payload, best_path)


def train_model(
    train_x: np.ndarray,
    val_x: np.ndarray,
    *,
    model: VAEWithPredictor,
    device: torch.device,
    pad_idx: int,
    num_tasks: int,
    cfg: TrainingLoopConfig,
    token_to_idx: dict[str, int],
    seq_len: int,
    vocab_size: int,
    max_len: int,
    unk_idx: int,
    eos_idx: int,
    y_train: np.ndarray | None = None,
    mask_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    mask_val: np.ndarray | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    history: dict[str, list[float]] | None = None,
    start_epoch: int = 0,
    best_epoch: int | None = None,
    best_val_token_acc: float | None = None,
    best_val_total: float | None = None,
    epochs_no_improve: int = 0,
    pretrain_mode: bool = True,
    scheduler_mode: str = "min",
    pos_weight: torch.Tensor | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_stem: str | None = None,
    save_epoch_checkpoints: bool = False,
) -> tuple[VAEWithPredictor, torch.optim.Optimizer, Any, dict[str, list[float]], dict]:
    """Notebook-aligned training loop with checkpointing and early stopping."""
    if optimizer is None:
        initial_lr = cfg.phase1_lr if pretrain_mode else cfg.phase2_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
            min_lr=1e-6,
        )

    train_loader = make_loader(
        train_x, y_train, mask_train, cfg.batch_size, True, num_tasks=num_tasks
    )
    val_loader = make_loader(
        val_x, y_val, mask_val, cfg.batch_size, False, num_tasks=num_tasks
    )

    metrics_to_track = [
        "beta",
        "lr",
        "train_total",
        "val_total",
        "train_recon_per_token",
        "val_recon_per_token",
        "train_recon_raw",
        "val_recon_raw",
        "train_kl",
        "val_kl",
        "train_token_acc",
        "val_token_acc",
        "train_bce",
        "val_bce",
    ]
    if history is None:
        history = {k: [] for k in metrics_to_track}
    else:
        for k in metrics_to_track:
            history.setdefault(k, [])

    if scheduler_mode == "max":
        if best_val_token_acc is None:
            best_val_token_acc = (
                float(np.max(history["val_token_acc"]))
                if history["val_token_acc"]
                else float("-inf")
            )
        if best_val_total is None:
            best_val_total = (
                float(np.min(history["val_total"]))
                if history["val_total"]
                else float("inf")
            )
    else:
        if best_val_total is None:
            best_val_total = (
                float(np.min(history["val_total"]))
                if history["val_total"]
                else float("inf")
            )
        if best_val_token_acc is None:
            best_val_token_acc = (
                float(np.max(history["val_token_acc"]))
                if history["val_token_acc"]
                else float("-inf")
            )

    early_stopped = False
    last_epoch = start_epoch
    for ep in range(start_epoch + 1, cfg.max_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            pad_idx=pad_idx,
            epoch=ep,
            cfg=cfg,
            pretrain_mode=pretrain_mode,
            optimizer=optimizer,
            pos_weight=pos_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            pad_idx=pad_idx,
            epoch=ep,
            cfg=cfg,
            pretrain_mode=pretrain_mode,
            optimizer=None,
            pos_weight=pos_weight,
        )

        if scheduler_mode == "max":
            scheduler.step(val_metrics["token_acc"])
        else:
            scheduler.step(val_metrics["total"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["beta"].append(kl_beta(ep, cfg.kl_anneal_epochs))
        history["lr"].append(current_lr)
        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(val_metrics["total"])
        history["train_recon_per_token"].append(train_metrics["recon_per_token"])
        history["val_recon_per_token"].append(val_metrics["recon_per_token"])
        history["train_recon_raw"].append(train_metrics["recon_raw"])
        history["val_recon_raw"].append(val_metrics["recon_raw"])
        history["train_kl"].append(train_metrics["kl_raw"])
        history["val_kl"].append(val_metrics["kl_raw"])
        history["train_token_acc"].append(train_metrics["token_acc"])
        history["val_token_acc"].append(val_metrics["token_acc"])
        history["train_bce"].append(train_metrics["bce_raw"])
        history["val_bce"].append(val_metrics["bce_raw"])

        if scheduler_mode == "max":
            is_best = val_metrics["token_acc"] > (best_val_token_acc + 1e-12)
        else:
            is_best = val_metrics["total"] < (best_val_total - 1e-12)

        if is_best:
            best_epoch = ep
            best_val_token_acc = float(val_metrics["token_acc"])
            best_val_total = float(val_metrics["total"])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if checkpoint_dir is not None and checkpoint_stem is not None:
            _save_training_checkpoints(
                model,
                history,
                save_dir=checkpoint_dir,
                checkpoint_stem=checkpoint_stem,
                epoch=ep,
                best_epoch=best_epoch,
                best_val_total=best_val_total,
                best_val_token_acc=best_val_token_acc,
                epochs_no_improve=epochs_no_improve,
                token_to_idx=token_to_idx,
                seq_len=seq_len,
                vocab_size=vocab_size,
                max_len=max_len,
                pad_idx=pad_idx,
                unk_idx=unk_idx,
                eos_idx=eos_idx,
                optimizer=optimizer,
                scheduler=scheduler,
                is_best=is_best,
                save_epoch_checkpoints=save_epoch_checkpoints,
            )

        print(
            f"[Epoch {ep:03d}] "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_acc={val_metrics['token_acc']:.4f} "
            f"lr={current_lr:.2e}"
        )

        last_epoch = ep
        if ep >= cfg.min_epochs and epochs_no_improve >= cfg.early_stopping_patience:
            early_stopped = True
            print(
                "Early stopping triggered at "
                f"epoch {ep} (patience={cfg.early_stopping_patience})."
            )
            break

    info = {
        "start_epoch": int(start_epoch),
        "last_epoch": int(last_epoch),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_total": (
            float(best_val_total) if best_val_total is not None else None
        ),
        "best_val_token_acc": (
            float(best_val_token_acc) if best_val_token_acc is not None else None
        ),
        "epochs_no_improve": int(epochs_no_improve),
        "early_stopped": bool(early_stopped),
    }
    return model, optimizer, scheduler, history, info


def load_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> tuple[dict[str, list[float]] | None, int]:
    """Load checkpoint and optionally restore optimizer/scheduler state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("history"), int(checkpoint.get("epoch", 0))

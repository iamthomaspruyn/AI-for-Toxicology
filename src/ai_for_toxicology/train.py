"""Training loop with reconstruction + KL + masked toxicity BCE."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DEFAULT_KL_ANNEAL_EPOCHS
from .data import Tox21Dataset
from .model import VAEWithPredictor


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    weight_decay: float
    max_epochs: int
    kl_anneal_epochs: int = DEFAULT_KL_ANNEAL_EPOCHS
    recon_weight: float = 1.0
    pred_weight: float = 1.0
    seed: int = 42


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_beta(epoch: int, anneal_epochs: int) -> float:
    return min(1.0, epoch / max(1, anneal_epochs))


def compute_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pred_logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    pad_idx: int,
    beta: float,
    recon_weight: float,
    pred_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        x.view(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum()

    if mask.sum() > 0 and pred_weight > 0:
        bce_all = F.binary_cross_entropy_with_logits(pred_logits, y, reduction="none")
        bce = (bce_all * mask).sum()
    else:
        bce = torch.tensor(0.0, device=logits.device)

    batch_size = x.size(0)
    total = ((recon_weight * recon) + (beta * kl) + (pred_weight * bce)) / batch_size
    return total, recon, kl, bce


def _run_epoch(
    model: VAEWithPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    pad_idx: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_bce = 0.0
    total_tokens = 0
    total_correct = 0

    beta = kl_beta(epoch=epoch, anneal_epochs=cfg.kl_anneal_epochs)

    for xb, yb, mb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)

        if is_train:
            optimizer.zero_grad()

        recon_logits, mu, logvar, pred_logits = model(xb)
        loss, recon, kl, bce = compute_loss(
            logits=recon_logits,
            x=xb,
            mu=mu,
            logvar=logvar,
            pred_logits=pred_logits,
            y=yb,
            mask=mb,
            pad_idx=pad_idx,
            beta=beta,
            recon_weight=cfg.recon_weight,
            pred_weight=cfg.pred_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        preds = recon_logits.argmax(dim=-1)
        valid = xb != pad_idx
        total_correct += int(((preds == xb) & valid).sum().item())
        total_tokens += int(valid.sum().item())

        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_kl += float(kl.item())
        total_bce += float(bce.item())

    n_batches = max(1, len(loader))
    n_samples = max(1, len(loader.dataset))
    return {
        "loss": total_loss / n_batches,
        "recon_raw": total_recon / n_samples,
        "kl_raw": total_kl / n_samples,
        "bce_raw": total_bce / n_samples,
        "token_acc": total_correct / max(1, total_tokens),
    }


def train_model(
    model: VAEWithPredictor,
    train_data: dict,
    val_data: dict,
    device: torch.device,
    cfg: TrainConfig,
    pad_idx: int,
    checkpoint_out: Path,
) -> tuple[VAEWithPredictor, dict]:
    set_all_seeds(cfg.seed)

    train_loader = DataLoader(
        Tox21Dataset(train_data["x"], train_data["y"], train_data["mask"]),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Tox21Dataset(val_data["x"], val_data["y"], val_data["mask"]),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.max_epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            pad_idx=pad_idx,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            cfg=cfg,
            pad_idx=pad_idx,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"train_acc={train_metrics['token_acc']:.4f} "
            f"val_acc={val_metrics['token_acc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "train_config": cfg.__dict__,
        },
        checkpoint_out,
    )
    return model, history

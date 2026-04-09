#!/usr/bin/env python
"""Run notebook-aligned multi-phase training for the VAE + predictor model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_for_toxicology.config import (  # noqa: E402
    DEFAULT_CHEMBL_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_MAX_LEN,
    DEFAULT_SEED,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    DEFAULT_ZINC_PATH,
)
from ai_for_toxicology.data import (  # noqa: E402
    prepare_end_to_end_training_data,
)
from ai_for_toxicology.model import VAEWithPredictor  # noqa: E402
from ai_for_toxicology.train import (  # noqa: E402
    TrainingLoopConfig,
    load_checkpoint,
    make_loader,
    run_epoch,
    set_all_seeds,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--chembl-csv", type=Path, default=DEFAULT_CHEMBL_PATH)
    parser.add_argument("--zinc-csv", type=Path, default=DEFAULT_ZINC_PATH)
    parser.add_argument("--tox21-train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--tox21-val-csv", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--tox21-test-csv", type=Path, default=DEFAULT_TEST_PATH)

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/end_to_end_checkpoints"),
    )
    parser.add_argument("--checkpoint-stem", type=str, default="e2evae_full_seqconv_ce")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("reports/phase_training_test_metrics.json"),
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.25)
    parser.add_argument("--lr-scheduler-patience", type=int, default=4)
    parser.add_argument("--kl-anneal-epochs", type=int, default=10)

    parser.add_argument("--phase1-epochs", type=int, default=90)
    parser.add_argument("--phase1-lr", type=float, default=5e-4)
    parser.add_argument("--phase1-recon-weight", type=float, default=1.0)

    parser.add_argument("--phase2-epochs", type=int, default=80)
    parser.add_argument("--phase2-lr", type=float, default=5e-5)
    parser.add_argument("--phase2-pred-weight", type=float, default=1.0)
    parser.add_argument("--phase2-recon-weight", type=float, default=0.0)

    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--warmup-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--base-lr", type=float, default=1e-7)
    parser.add_argument("--phase2-weight-decay", type=float, default=1e-2)

    parser.add_argument("--latent-dim", type=int, default=292)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")

    # Optional limits for shorter validation runs (default keeps full behavior).
    parser.add_argument("--limit-pretrain-train", type=int, default=None)
    parser.add_argument("--limit-pretrain-val", type=int, default=None)
    parser.add_argument("--limit-ft-train", type=int, default=None)
    parser.add_argument("--limit-ft-val", type=int, default=None)
    parser.add_argument("--limit-ft-test", type=int, default=None)
    return parser.parse_args()


def _build_loop_cfg(args: argparse.Namespace, max_epochs: int) -> TrainingLoopConfig:
    return TrainingLoopConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        min_epochs=args.min_epochs,
        max_epochs=max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        kl_anneal_epochs=args.kl_anneal_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        phase1_recon_weight=args.phase1_recon_weight,
        phase2_recon_weight=args.phase2_recon_weight,
        phase2_pred_weight=args.phase2_pred_weight,
    )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    prepared = prepare_end_to_end_training_data(
        chembl_path=args.chembl_csv,
        zinc_path=args.zinc_csv,
        tox21_train_path=args.tox21_train_csv,
        tox21_val_path=args.tox21_val_csv,
        tox21_test_path=args.tox21_test_csv,
        max_len=args.max_len,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        limit_pretrain_train=args.limit_pretrain_train,
        limit_pretrain_val=args.limit_pretrain_val,
        limit_ft_train=args.limit_ft_train,
        limit_ft_val=args.limit_ft_val,
        limit_ft_test=args.limit_ft_test,
    )

    print("Device:", device)
    print("Data stats:")
    for key, value in prepared.stats.items():
        print(f"  {key}: {value}")

    model = VAEWithPredictor(
        vocab_size=prepared.vocab_size,
        seq_len=prepared.seq_len,
        latent_dim=args.latent_dim,
        num_tasks=prepared.y_train_ft.shape[1],
        dropout=args.dropout,
    ).to(device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: dict | None = None
    start_epoch = 0
    current_phase = 1
    info2 = None

    phase2_last = (
        args.checkpoint_dir / f"{args.checkpoint_stem}_phase2_adaptive_last.pt"
    )
    phase1_best = args.checkpoint_dir / f"{args.checkpoint_stem}_phase1_best.pt"

    if args.auto_resume and phase2_last.exists():
        print("Resuming from existing Phase 2 adaptive checkpoint.")
        history, start_epoch = load_checkpoint(phase2_last, model, device=device)
        current_phase = 2
    elif args.skip_phase1 and phase1_best.exists():
        print("Attempting Phase 1 weight load for Phase 2 start.")
        try:
            checkpoint = torch.load(phase1_best, map_location=device)
            if int(checkpoint.get("vocab_size", -1)) == int(prepared.vocab_size):
                state_dict = checkpoint["model_state_dict"]
                filtered_state = {
                    k: v for k, v in state_dict.items() if not k.startswith("pred_head")
                }
                model.load_state_dict(filtered_state, strict=False)
                start_epoch = int(checkpoint.get("epoch", args.phase1_epochs))
                history = checkpoint.get("history")
                current_phase = 2
                print("Loaded Phase 1 weights successfully.")
            else:
                print("Vocab mismatch, restarting from Phase 1.")
                current_phase = 1
        except Exception as exc:  # noqa: BLE001 - parity with notebook fallback
            print(f"Phase 1 load failed ({exc}); restarting from Phase 1.")
            current_phase = 1
    else:
        print("Starting from scratch (Phase 1).")

    if current_phase == 1:
        print("Running Phase 1 pretraining.")
        cfg_phase1 = _build_loop_cfg(args, max_epochs=args.phase1_epochs)
        opt1 = torch.optim.Adam(model.parameters(), lr=args.phase1_lr)
        sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt1,
            mode="max",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
        )

        model, _, _, history, info1 = train_model(
            prepared.pre_train_x,
            prepared.pre_val_x,
            model=model,
            device=device,
            pad_idx=prepared.pad_idx,
            num_tasks=prepared.y_train_ft.shape[1],
            cfg=cfg_phase1,
            token_to_idx=prepared.token_to_idx,
            seq_len=prepared.seq_len,
            vocab_size=prepared.vocab_size,
            max_len=prepared.max_len,
            unk_idx=prepared.unk_idx,
            eos_idx=prepared.eos_idx,
            optimizer=opt1,
            scheduler=sched1,
            history=history,
            start_epoch=start_epoch,
            pretrain_mode=True,
            scheduler_mode="max",
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_stem=f"{args.checkpoint_stem}_phase1",
            save_epoch_checkpoints=args.save_epoch_checkpoints,
        )
        start_epoch = int(info1["last_epoch"])
        current_phase = 2

    if current_phase == 2:
        pos_counts = torch.tensor(
            (prepared.y_train_ft == 1).sum(axis=0),
            dtype=torch.float32,
        )
        neg_counts = torch.tensor(
            (prepared.y_train_ft == 0).sum(axis=0),
            dtype=torch.float32,
        )
        pos_weight_tensor = ((neg_counts / (pos_counts + 1e-6)) * 2.0).to(device)
        print(
            "Task pos_weight range:",
            f"{float(pos_weight_tensor.min()):.2f} to {float(pos_weight_tensor.max()):.2f}",
        )

        print(f"Phase 2 Stage A warmup for {args.warmup_epochs} epochs.")
        for name, param in model.named_parameters():
            param.requires_grad = "pred_head" in name

        opt_warmup = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.warmup_lr,
        )
        cfg_warmup = _build_loop_cfg(
            args,
            max_epochs=start_epoch + args.warmup_epochs,
        )
        model, _, _, history, info_warmup = train_model(
            prepared.ft_train_x,
            prepared.ft_val_x,
            model=model,
            device=device,
            pad_idx=prepared.pad_idx,
            num_tasks=prepared.y_train_ft.shape[1],
            cfg=cfg_warmup,
            token_to_idx=prepared.token_to_idx,
            seq_len=prepared.seq_len,
            vocab_size=prepared.vocab_size,
            max_len=prepared.max_len,
            unk_idx=prepared.unk_idx,
            eos_idx=prepared.eos_idx,
            y_train=prepared.y_train_ft,
            mask_train=prepared.mask_train_ft,
            y_val=prepared.y_val_ft,
            mask_val=prepared.mask_val_ft,
            optimizer=opt_warmup,
            history=history,
            start_epoch=start_epoch,
            pretrain_mode=False,
            scheduler_mode="min",
            pos_weight=pos_weight_tensor,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_stem=f"{args.checkpoint_stem}_phase2_warmup",
            save_epoch_checkpoints=args.save_epoch_checkpoints,
        )

        print("Phase 2 Stage B adaptive full fine-tuning.")
        for param in model.parameters():
            param.requires_grad = True

        opt_groups = [
            {
                "params": [p for n, p in model.named_parameters() if "pred_head" in n],
                "lr": args.head_lr,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "pred_head" not in n
                ],
                "lr": args.base_lr,
            },
        ]
        opt_fine = torch.optim.AdamW(opt_groups, weight_decay=args.phase2_weight_decay)
        sched_fine = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_fine,
            mode="min",
            factor=0.5,
            patience=5,
        )
        cfg_fine = _build_loop_cfg(
            args,
            max_epochs=int(info_warmup["last_epoch"]) + args.phase2_epochs,
        )
        model, _, _, history, info2 = train_model(
            prepared.ft_train_x,
            prepared.ft_val_x,
            model=model,
            device=device,
            pad_idx=prepared.pad_idx,
            num_tasks=prepared.y_train_ft.shape[1],
            cfg=cfg_fine,
            token_to_idx=prepared.token_to_idx,
            seq_len=prepared.seq_len,
            vocab_size=prepared.vocab_size,
            max_len=prepared.max_len,
            unk_idx=prepared.unk_idx,
            eos_idx=prepared.eos_idx,
            y_train=prepared.y_train_ft,
            mask_train=prepared.mask_train_ft,
            y_val=prepared.y_val_ft,
            mask_val=prepared.mask_val_ft,
            optimizer=opt_fine,
            scheduler=sched_fine,
            history=history,
            start_epoch=int(info_warmup["last_epoch"]),
            best_val_total=float("inf"),
            epochs_no_improve=0,
            pretrain_mode=False,
            scheduler_mode="min",
            pos_weight=pos_weight_tensor,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_stem=f"{args.checkpoint_stem}_phase2_adaptive",
            save_epoch_checkpoints=args.save_epoch_checkpoints,
        )

        best_path = (
            args.checkpoint_dir / f"{args.checkpoint_stem}_phase2_adaptive_best.pt"
        )
        if best_path.exists():
            print("Evaluating with best adaptive checkpoint.")
            load_checkpoint(best_path, model, device=device)

        final_epoch = (
            int(info2["last_epoch"]) if info2 is not None else int(start_epoch)
        )
        test_loader = make_loader(
            prepared.ft_test_x,
            prepared.y_test_ft,
            prepared.mask_test_ft,
            batch_size=args.batch_size,
            shuffle=False,
            num_tasks=prepared.y_train_ft.shape[1],
        )
        test_cfg = _build_loop_cfg(args, max_epochs=final_epoch)
        test_metrics = run_epoch(
            model,
            test_loader,
            device=device,
            pad_idx=prepared.pad_idx,
            epoch=final_epoch,
            cfg=test_cfg,
            pretrain_mode=False,
            optimizer=None,
            pos_weight=pos_weight_tensor,
        )

        print("Final Tox21 test metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w") as f:
            json.dump(
                {
                    "test_metrics": test_metrics,
                    "config": vars(args),
                    "training_stats": prepared.stats,
                    "training_loop": asdict(test_cfg),
                },
                f,
                indent=2,
                default=str,
            )
        print("Saved metrics JSON:", args.metrics_out)


if __name__ == "__main__":
    main()

"""Data loading and SELFIES/token alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import selfies as sf
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_MAX_LEN, TOX21_TASKS


@dataclass
class CheckpointMeta:
    token_to_idx: dict[str, int]
    pad_idx: int
    unk_idx: int
    eos_idx: int
    max_len: int
    seq_len: int
    vocab_size: int
    latent_dim: int
    num_tasks: int


@dataclass
class PreparedTrainingData:
    token_to_idx: dict[str, int]
    idx_to_token: dict[int, str]
    pad_idx: int
    unk_idx: int
    eos_idx: int
    max_len: int
    seq_len: int
    vocab_size: int
    pre_train_x: np.ndarray
    pre_val_x: np.ndarray
    ft_train_x: np.ndarray
    y_train_ft: np.ndarray
    mask_train_ft: np.ndarray
    ft_val_x: np.ndarray
    y_val_ft: np.ndarray
    mask_val_ft: np.ndarray
    ft_test_x: np.ndarray
    y_test_ft: np.ndarray
    mask_test_ft: np.ndarray
    stats: dict


class Tox21Dataset(Dataset):
    """Dataset returning token IDs, labels, and label mask."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.x = torch.as_tensor(x).long()
        self.y = torch.as_tensor(y).float()
        self.mask = torch.as_tensor(mask).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.mask[idx]


def load_checkpoint_meta(checkpoint_path: Path) -> tuple[dict, CheckpointMeta]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    token_to_idx = ckpt["token_to_idx"]
    model_sd = ckpt["model_state_dict"]

    meta = CheckpointMeta(
        token_to_idx=token_to_idx,
        pad_idx=int(ckpt["pad_idx"]),
        unk_idx=int(ckpt["unk_idx"]),
        eos_idx=int(ckpt["eos_idx"]),
        max_len=int(ckpt["max_len"]),
        seq_len=int(ckpt["seq_len"]),
        vocab_size=int(ckpt["vocab_size"]),
        latent_dim=int(model_sd["fc_mu.weight"].shape[0]),
        num_tasks=int(model_sd["pred_head.9.bias"].shape[0]),
    )
    return ckpt, meta


def load_tox21_frame(path: Path, origin_split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "canonical_smiles" not in df.columns:
        raise ValueError(f"{path} is missing canonical_smiles")

    missing = [task for task in TOX21_TASKS if task not in df.columns]
    if missing:
        raise ValueError(f"{path} missing Tox21 tasks: {missing}")

    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True).copy()
    df["canonical_smiles"] = df["canonical_smiles"].astype(str)
    df[TOX21_TASKS] = df[TOX21_TASKS].apply(pd.to_numeric, errors="coerce")
    df["origin_split"] = origin_split
    return df


def load_smiles(path: Path) -> list[str]:
    """Load canonical SMILES and preserve first-seen order (deduplicated)."""
    df = pd.read_csv(path)
    if "canonical_smiles" not in df.columns:
        raise ValueError(f"{path} does not contain canonical_smiles")
    smiles = df["canonical_smiles"].dropna().astype(str).tolist()
    return list(dict.fromkeys(smiles))


def load_tox21_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Tox21 labels and validity mask in notebook-compatible format."""
    df = pd.read_csv(path)
    y = df[TOX21_TASKS].values.astype(np.float32)
    mask = ~np.isnan(y)
    y = np.nan_to_num(y, nan=0.0)
    return y, mask.astype(np.float32)


def split_list(
    data: list[str],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split list into train/val/test using the notebook strategy."""
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)

    n = len(data_copy)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_set = data_copy[:n_test]
    val_set = data_copy[n_test : n_test + n_val]
    train_set = data_copy[n_test + n_val :]
    return train_set, val_set, test_set


def process_aligned_data(
    smiles_list: list[str],
    labels: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    max_len: int = DEFAULT_MAX_LEN,
) -> tuple[list[str], np.ndarray | None, np.ndarray | None, dict]:
    """
    Convert SMILES to SELFIES while preserving alignment with labels/masks.

    Mirrors notebook behavior: drop failed encodings and over-length SELFIES.
    """
    clean_selfies: list[str] = []
    clean_labels: list[np.ndarray] = []
    clean_masks: list[np.ndarray] = []

    dropped_encoding = 0
    dropped_length = 0

    for i, smiles in enumerate(smiles_list):
        try:
            sf_str = sf.encoder(smiles)
        except Exception:
            dropped_encoding += 1
            continue

        if len(list(sf.split_selfies(sf_str))) > max_len:
            dropped_length += 1
            continue

        clean_selfies.append(sf_str)
        if labels is not None and masks is not None:
            clean_labels.append(labels[i])
            clean_masks.append(masks[i])

    stats = {
        "input": int(len(smiles_list)),
        "kept": int(len(clean_selfies)),
        "dropped_encoding": int(dropped_encoding),
        "dropped_length": int(dropped_length),
    }

    if labels is not None and masks is not None:
        return (
            clean_selfies,
            np.asarray(clean_labels, dtype=np.float32),
            np.asarray(clean_masks, dtype=np.float32),
            stats,
        )
    return clean_selfies, None, None, stats


def tokenize_selfies(sf_str: str) -> list[str]:
    return list(sf.split_selfies(sf_str))


def build_vocab(
    pre_train_selfies: list[str],
    ft_train_selfies: list[str],
) -> tuple[dict[str, int], dict[int, str], int, int, int]:
    """Build tokenizer vocabulary exactly as in notebook training."""
    pad = "<PAD>"
    unk = "<UNK>"
    eos = "<EOS>"

    train_tokens = [tokenize_selfies(s) for s in (pre_train_selfies + ft_train_selfies)]
    vocab_tokens = sorted({tok for seq in train_tokens for tok in seq})

    all_tokens = [pad, unk, eos] + vocab_tokens
    token_to_idx = {tok: i for i, tok in enumerate(all_tokens)}
    idx_to_token = {i: tok for tok, i in token_to_idx.items()}
    return (
        token_to_idx,
        idx_to_token,
        token_to_idx[pad],
        token_to_idx[unk],
        token_to_idx[eos],
    )


def encode_selfies_with_vocab(
    sf_str: str,
    token_to_idx: dict[str, int],
    max_len: int,
    unk_idx: int,
    eos_idx: int,
) -> list[int]:
    ids = [token_to_idx.get(tok, unk_idx) for tok in tokenize_selfies(sf_str)]
    ids = ids[:max_len]
    ids.append(eos_idx)
    return ids


def encode_list_to_numpy(
    selfies_list: list[str],
    token_to_idx: dict[str, int],
    max_len: int,
    pad_idx: int,
    unk_idx: int,
    eos_idx: int,
) -> np.ndarray:
    seq_len = max_len + 1
    out_x = np.full((len(selfies_list), seq_len), pad_idx, dtype=np.int64)
    for i, sf_str in enumerate(selfies_list):
        ids = encode_selfies_with_vocab(sf_str, token_to_idx, max_len, unk_idx, eos_idx)
        out_x[i, : len(ids)] = ids
    return out_x


def _truncate_ft(
    x_selfies: list[str],
    y: np.ndarray,
    mask: np.ndarray,
    limit: int | None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if limit is None:
        return x_selfies, y, mask
    n = min(limit, len(x_selfies))
    return x_selfies[:n], y[:n], mask[:n]


def prepare_end_to_end_training_data(
    *,
    chembl_path: Path,
    zinc_path: Path,
    tox21_train_path: Path,
    tox21_val_path: Path,
    tox21_test_path: Path,
    max_len: int = DEFAULT_MAX_LEN,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
    limit_pretrain_train: int | None = None,
    limit_pretrain_val: int | None = None,
    limit_ft_train: int | None = None,
    limit_ft_val: int | None = None,
    limit_ft_test: int | None = None,
) -> PreparedTrainingData:
    """Prepare full pretrain + fine-tune matrices following notebook logic."""
    chembl_smiles = load_smiles(chembl_path)
    zinc_smiles = load_smiles(zinc_path)
    tox21_train_smiles = load_smiles(tox21_train_path)
    tox21_val_smiles = load_smiles(tox21_val_path)
    tox21_test_smiles = load_smiles(tox21_test_path)
    pretrain_smiles = list(dict.fromkeys(chembl_smiles + zinc_smiles))

    y_train_tox21, mask_train_tox21 = load_tox21_labels(tox21_train_path)
    y_val_tox21, mask_val_tox21 = load_tox21_labels(tox21_val_path)
    y_test_tox21, mask_test_tox21 = load_tox21_labels(tox21_test_path)

    base_train_smiles, base_val_smiles, _ = split_list(
        pretrain_smiles, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    all_tox21_smiles = (
        set(tox21_train_smiles) | set(tox21_val_smiles) | set(tox21_test_smiles)
    )
    pretrain_train_smiles = [s for s in base_train_smiles if s not in all_tox21_smiles]
    pretrain_val_smiles = [s for s in base_val_smiles if s not in all_tox21_smiles]

    if not set(pretrain_train_smiles).isdisjoint(all_tox21_smiles):
        raise RuntimeError("Pretraining train set overlaps with Tox21 splits.")

    pre_train_selfies, _, _, pre_train_stats = process_aligned_data(
        pretrain_train_smiles, max_len=max_len
    )
    pre_val_selfies, _, _, pre_val_stats = process_aligned_data(
        pretrain_val_smiles, max_len=max_len
    )
    ft_train_selfies, y_train_ft, mask_train_ft, ft_train_stats = process_aligned_data(
        tox21_train_smiles, y_train_tox21, mask_train_tox21, max_len=max_len
    )
    ft_val_selfies, y_val_ft, mask_val_ft, ft_val_stats = process_aligned_data(
        tox21_val_smiles, y_val_tox21, mask_val_tox21, max_len=max_len
    )
    ft_test_selfies, y_test_ft, mask_test_ft, ft_test_stats = process_aligned_data(
        tox21_test_smiles, y_test_tox21, mask_test_tox21, max_len=max_len
    )

    if limit_pretrain_train is not None:
        pre_train_selfies = pre_train_selfies[
            : min(limit_pretrain_train, len(pre_train_selfies))
        ]
    if limit_pretrain_val is not None:
        pre_val_selfies = pre_val_selfies[
            : min(limit_pretrain_val, len(pre_val_selfies))
        ]
    ft_train_selfies, y_train_ft, mask_train_ft = _truncate_ft(
        ft_train_selfies, y_train_ft, mask_train_ft, limit_ft_train
    )
    ft_val_selfies, y_val_ft, mask_val_ft = _truncate_ft(
        ft_val_selfies, y_val_ft, mask_val_ft, limit_ft_val
    )
    ft_test_selfies, y_test_ft, mask_test_ft = _truncate_ft(
        ft_test_selfies, y_test_ft, mask_test_ft, limit_ft_test
    )

    token_to_idx, idx_to_token, pad_idx, unk_idx, eos_idx = build_vocab(
        pre_train_selfies, ft_train_selfies
    )
    seq_len = max_len + 1
    vocab_size = len(token_to_idx)

    pre_train_x = encode_list_to_numpy(
        pre_train_selfies, token_to_idx, max_len, pad_idx, unk_idx, eos_idx
    )
    pre_val_x = encode_list_to_numpy(
        pre_val_selfies, token_to_idx, max_len, pad_idx, unk_idx, eos_idx
    )
    ft_train_x = encode_list_to_numpy(
        ft_train_selfies, token_to_idx, max_len, pad_idx, unk_idx, eos_idx
    )
    ft_val_x = encode_list_to_numpy(
        ft_val_selfies, token_to_idx, max_len, pad_idx, unk_idx, eos_idx
    )
    ft_test_x = encode_list_to_numpy(
        ft_test_selfies, token_to_idx, max_len, pad_idx, unk_idx, eos_idx
    )

    stats = {
        "chembl_unique": int(len(chembl_smiles)),
        "zinc_unique": int(len(zinc_smiles)),
        "base_pretraining_unique": int(len(pretrain_smiles)),
        "tox21_train_unique": int(len(tox21_train_smiles)),
        "tox21_val_unique": int(len(tox21_val_smiles)),
        "tox21_test_unique": int(len(tox21_test_smiles)),
        "pretrain_train_stats": pre_train_stats,
        "pretrain_val_stats": pre_val_stats,
        "ft_train_stats": ft_train_stats,
        "ft_val_stats": ft_val_stats,
        "ft_test_stats": ft_test_stats,
        "pretrain_train_shape": tuple(pre_train_x.shape),
        "pretrain_val_shape": tuple(pre_val_x.shape),
        "ft_train_shape": tuple(ft_train_x.shape),
        "ft_val_shape": tuple(ft_val_x.shape),
        "ft_test_shape": tuple(ft_test_x.shape),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
    }

    return PreparedTrainingData(
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
        eos_idx=eos_idx,
        max_len=max_len,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pre_train_x=pre_train_x,
        pre_val_x=pre_val_x,
        ft_train_x=ft_train_x,
        y_train_ft=y_train_ft,
        mask_train_ft=mask_train_ft,
        ft_val_x=ft_val_x,
        y_val_ft=y_val_ft,
        mask_val_ft=mask_val_ft,
        ft_test_x=ft_test_x,
        y_test_ft=y_test_ft,
        mask_test_ft=mask_test_ft,
        stats=stats,
    )


def _encode_selfies(sf_str: str, meta: CheckpointMeta) -> list[int]:
    ids = [meta.token_to_idx.get(tok, meta.unk_idx) for tok in sf.split_selfies(sf_str)]
    ids = ids[: meta.max_len]
    ids.append(meta.eos_idx)
    return ids


def build_aligned_dataset(df: pd.DataFrame, meta: CheckpointMeta) -> dict:
    """Build aligned tox dataset from a fixed checkpoint tokenizer."""
    ids_rows: list[list[int]] = []
    y_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    dropped_encoding = 0
    dropped_length = 0
    dropped_missing = 0

    for _, row in df.iterrows():
        smiles = row["canonical_smiles"]

        try:
            sf_str = sf.encoder(smiles)
        except Exception:
            dropped_encoding += 1
            continue

        if len(list(sf.split_selfies(sf_str))) > meta.max_len:
            dropped_length += 1
            continue

        labels = row[TOX21_TASKS].to_numpy(dtype=np.float32)
        mask = ~np.isnan(labels)
        if mask.sum() == 0:
            dropped_missing += 1
            continue

        labels = np.nan_to_num(labels, nan=0.0)
        ids = _encode_selfies(sf_str, meta)

        ids_rows.append(ids)
        y_rows.append(labels)
        mask_rows.append(mask.astype(np.float32))

    if not ids_rows:
        raise RuntimeError("No aligned samples left after filtering and encoding.")

    x = np.full((len(ids_rows), meta.seq_len), meta.pad_idx, dtype=np.int64)
    for i, ids in enumerate(ids_rows):
        x[i, : len(ids)] = ids

    y = np.asarray(y_rows, dtype=np.float32)
    mask = np.asarray(mask_rows, dtype=np.float32)

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "counts": {
            "kept": int(len(ids_rows)),
            "dropped_encoding": int(dropped_encoding),
            "dropped_length": int(dropped_length),
            "dropped_missing": int(dropped_missing),
        },
    }

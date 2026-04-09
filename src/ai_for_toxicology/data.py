"""Data loading and SELFIES/token alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch
from torch.utils.data import Dataset

from .config import TOX21_TASKS


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


def _encode_selfies(sf_str: str, meta: CheckpointMeta) -> list[int]:
    ids = [meta.token_to_idx.get(tok, meta.unk_idx) for tok in sf.split_selfies(sf_str)]
    ids = ids[: meta.max_len]
    ids.append(meta.eos_idx)
    return ids


def build_aligned_dataset(df: pd.DataFrame, meta: CheckpointMeta) -> dict:
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

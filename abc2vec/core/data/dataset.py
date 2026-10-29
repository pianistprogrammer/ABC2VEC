"""PyTorch datasets for ABC2Vec."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.tokenizer.patchifier import BarPatchifier
from core.tokenizer.transposer import ABCTransposer


class ABC2VecDataset(Dataset):
    """
    PyTorch Dataset for ABC2Vec pre-training.

    Returns bar-patched tune data with optional transposition augmentation.

    Args:
        df: DataFrame with ABC notation data (must have 'abc_body' column)
        patchifier: BarPatchifier instance
        augment_transpose: Whether to create transposed versions for TI loss
        body_col: Name of column containing ABC body text
        key_col: Name of column containing key signature
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patchifier: BarPatchifier,
        augment_transpose: bool = True,
        body_col: str = "abc_body",
        key_col: str = "key",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.patchifier = patchifier
        self.augment_transpose = augment_transpose
        self.body_col = body_col
        self.key_col = key_col

        # Transposition range (avoid extreme transpositions)
        self.transpose_semitones = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]

    def __len__(self) -> int:
        """Return number of tunes in dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tune sample.

        Returns:
            Dictionary with bar patches and optional transposed version
        """
        row = self.df.iloc[idx]
        abc_body = row[self.body_col]

        # Patchify original tune
        patches = self.patchifier.patchify(abc_body)

        item = {
            "bar_indices": patches["bar_indices"],
            "char_mask": patches["char_mask"],
            "bar_mask": patches["bar_mask"],
            "num_bars": patches["num_bars"],
        }

        # Create transposed version for Transposition Invariance objective
        if self.augment_transpose:
            semitones = np.random.choice(self.transpose_semitones)
            transposed_body = ABCTransposer.transpose_abc_body(
                abc_body, semitones
            )
            trans_patches = self.patchifier.patchify(transposed_body)

            item["trans_bar_indices"] = trans_patches["bar_indices"]
            item["trans_char_mask"] = trans_patches["char_mask"]
            item["trans_bar_mask"] = trans_patches["bar_mask"]
            item["transpose_semitones"] = semitones

        return item


class SectionPairDataset(Dataset):
    """
    Dataset for Section Contrastive Loss (SCL).

    Yields (section_A, section_B) pairs from the same tune.

    Args:
        section_pairs_path: Path to section_pairs.json file
        patchifier: BarPatchifier instance
    """

    def __init__(
        self,
        section_pairs_path: str,
        patchifier: BarPatchifier,
    ) -> None:
        with open(section_pairs_path) as f:
            self.pairs = json.load(f)
        self.patchifier = patchifier

    def __len__(self) -> int:
        """Return number of section pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a section pair sample.

        Returns:
            Dictionary with patches for both sections
        """
        pair = self.pairs[idx]

        # Patchify section A
        patch_a = self.patchifier.patchify(pair["section_a"])

        # Patchify section B
        patch_b = self.patchifier.patchify(pair["section_b"])

        return {
            "a_bar_indices": patch_a["bar_indices"],
            "a_char_mask": patch_a["char_mask"],
            "a_bar_mask": patch_a["bar_mask"],
            "b_bar_indices": patch_b["bar_indices"],
            "b_char_mask": patch_b["char_mask"],
            "b_bar_mask": patch_b["bar_mask"],
            "tune_id": pair["tune_id"],
        }


class VariantDataset(Dataset):
    """
    Dataset for Variant-Aware Contrastive (VAC) learning.

    Groups tunes by variant families (same tune, different settings).

    Args:
        df: DataFrame with 'abc_body' and 'variant_group_id' columns
        patchifier: BarPatchifier instance
        body_col: Name of column containing ABC body text
        group_col: Name of column containing variant group IDs
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patchifier: BarPatchifier,
        body_col: str = "abc_body",
        group_col: str = "variant_group_id",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.patchifier = patchifier
        self.body_col = body_col
        self.group_col = group_col

    def __len__(self) -> int:
        """Return number of tunes."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a tune sample with variant group ID.

        Returns:
            Dictionary with bar patches and group_id
        """
        row = self.df.iloc[idx]
        abc_body = row[self.body_col]
        group_id = row[self.group_col]

        patches = self.patchifier.patchify(abc_body)

        return {
            "bar_indices": patches["bar_indices"],
            "char_mask": patches["char_mask"],
            "bar_mask": patches["bar_mask"],
            "group_id": group_id,
        }


def load_processed_data(
    data_dir: str,
    split: str = "train"
) -> pd.DataFrame:
    """
    Load processed data from disk.

    Args:
        data_dir: Directory containing processed data
        split: Which split to load ('train', 'val', or 'test')

    Returns:
        DataFrame with processed tune data
    """
    data_path = Path(data_dir) / f"{split}.parquet"
    return pd.read_parquet(data_path)

"""Bar-level patchifier for ABC notation."""

import re
from typing import Dict, List

import torch

from abc2vec.tokenizer.vocabulary import ABCVocabulary


class BarPatchifier:
    """
    Converts ABC notation into bar-level patches.

    Following CLaMP and MelodyT5, this class groups all characters within
    a single bar into a "patch". Each bar becomes one token in the sequence,
    reducing sequence length by ~10x compared to character-level tokenization.

    Process:
        1. Split ABC body on bar lines (|)
        2. Encode each bar's characters into indices
        3. Pad/truncate each bar to max_bar_length characters
        4. Create tensors ready for the model

    Args:
        vocab: ABCVocabulary instance for character encoding
        max_bar_length: Maximum characters per bar (default: 64)
        max_bars: Maximum number of bars per tune (default: 64)
    """

    def __init__(
        self,
        vocab: ABCVocabulary,
        max_bar_length: int = 64,
        max_bars: int = 64,
    ) -> None:
        self.vocab = vocab
        self.max_bar_length = max_bar_length
        self.max_bars = max_bars

    def split_into_bars(self, abc_body: str) -> List[str]:
        """
        Split ABC body into individual bars.

        Handles various bar separators in ABC notation:
        - |  (single bar line)
        - |: (repeat start)
        - :| (repeat end)
        - || (double bar line)
        - [| and |] (thick bar lines)

        Args:
            abc_body: ABC notation body (without headers)

        Returns:
            List of bar strings
        """
        body = abc_body.strip()
        bars = []
        current_bar = []
        i = 0

        while i < len(body):
            ch = body[i]

            if ch == "|":
                # Save current bar if non-empty
                bar_str = "".join(current_bar).strip()
                if bar_str:
                    bars.append(bar_str)
                current_bar = []

                # Skip multi-character barlines (|:, |], etc.)
                if i + 1 < len(body) and body[i + 1] in ":|]":
                    i += 2
                    continue

            elif ch == ":" and i + 1 < len(body) and body[i + 1] == "|":
                # Handle :| repeat ending
                bar_str = "".join(current_bar).strip()
                if bar_str:
                    bars.append(bar_str)
                current_bar = []
                i += 2
                continue

            else:
                current_bar.append(ch)

            i += 1

        # Don't forget the last bar
        bar_str = "".join(current_bar).strip()
        if bar_str:
            bars.append(bar_str)

        return bars

    def patchify(self, abc_body: str) -> Dict[str, torch.Tensor]:
        """
        Convert ABC body to bar patches as PyTorch tensors.

        Args:
            abc_body: ABC notation body string

        Returns:
            Dictionary containing:
                - bar_indices: (max_bars, max_bar_length) tensor of character indices
                - bar_mask: (max_bars,) boolean tensor, True for real bars
                - char_mask: (max_bars, max_bar_length) boolean tensor, True for real chars
                - num_bars: Number of actual bars (before padding)
                - bars: List of bar strings (for reference)
        """
        bars = self.split_into_bars(abc_body)[: self.max_bars]
        num_bars = len(bars)

        # Initialize tensors with padding
        bar_indices = torch.full(
            (self.max_bars, self.max_bar_length),
            self.vocab.pad_idx,
            dtype=torch.long,
        )
        char_mask = torch.zeros(
            self.max_bars, self.max_bar_length, dtype=torch.bool
        )
        bar_mask = torch.zeros(self.max_bars, dtype=torch.bool)

        # Fill in actual bar data
        for i, bar in enumerate(bars):
            encoded = self.vocab.encode(bar)[: self.max_bar_length]
            bar_indices[i, : len(encoded)] = torch.tensor(
                encoded, dtype=torch.long
            )
            char_mask[i, : len(encoded)] = True
            bar_mask[i] = True

        return {
            "bar_indices": bar_indices,
            "bar_mask": bar_mask,
            "char_mask": char_mask,
            "num_bars": num_bars,
            "bars": bars,
        }

    def depatchify(self, bar_indices: torch.Tensor, char_mask: torch.Tensor) -> str:
        """
        Convert bar patches back to ABC notation string.

        Args:
            bar_indices: (num_bars, max_bar_length) tensor of character indices
            char_mask: (num_bars, max_bar_length) boolean mask for valid characters

        Returns:
            Reconstructed ABC notation string
        """
        bars = []
        for bar_idx in range(bar_indices.shape[0]):
            # Get valid character indices for this bar
            valid_chars = bar_indices[bar_idx][char_mask[bar_idx]].tolist()
            if not valid_chars:
                break
            # Decode bar
            bar_text = self.vocab.decode(valid_chars, skip_special=True)
            if bar_text:
                bars.append(bar_text)

        # Join bars with bar lines
        return " | ".join(bars) + " |" if bars else ""

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BarPatchifier(vocab_size={self.vocab.size}, "
            f"max_bar_length={self.max_bar_length}, "
            f"max_bars={self.max_bars})"
        )

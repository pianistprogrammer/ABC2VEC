"""Patch embedding layer for ABC2Vec."""

from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Embeds bar patches into dense vectors.

    This is the first layer of the ABC2Vec encoder. It converts
    character-level bar representations into fixed-size embeddings.

    Process:
        1. Character embedding: each char index → d_char vector
        2. Aggregate characters within a bar via mean pooling
        3. Linear projection to d_model dimension
        4. Add positional embeddings for bar positions
        5. Apply layer normalization and dropout

    Args:
        vocab_size: Size of character vocabulary
        d_char: Dimension of character embeddings (default: 64)
        d_model: Model hidden dimension (default: 256)
        max_bar_length: Maximum characters per bar (default: 64)
        max_bars: Maximum bars per tune (default: 64)
        pad_idx: Index of padding token (default: 0)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        d_char: int = 64,
        d_model: int = 256,
        max_bar_length: int = 64,
        max_bars: int = 64,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_char = d_char
        self.d_model = d_model
        self.max_bar_length = max_bar_length
        self.max_bars = max_bars

        # Character-level embedding
        self.char_embed = nn.Embedding(
            vocab_size, d_char, padding_idx=pad_idx
        )

        # Projection from aggregated char embeddings to model dimension
        self.projection = nn.Linear(d_char, d_model)

        # Positional embedding for bar positions
        self.pos_embed = nn.Embedding(max_bars + 1, d_model)  # +1 for [CLS]

        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: convert bar patches to embeddings.

        Args:
            bar_indices: (batch, max_bars, max_bar_length) character indices
            char_mask: (batch, max_bars, max_bar_length) boolean mask for real chars
            bar_mask: (batch, max_bars) boolean mask for real bars

        Returns:
            Tuple of:
                - embeddings: (batch, max_bars, d_model) bar-level embeddings
                - attention_mask: (batch, max_bars) same as bar_mask
        """
        batch_size = bar_indices.shape[0]

        # Character embedding: (batch, max_bars, max_bar_length, d_char)
        char_embeds = self.char_embed(bar_indices)

        # Mean-pool characters within each bar (mask out padding)
        # char_mask: (batch, max_bars, max_bar_length) → expand for d_char
        mask_expanded = char_mask.unsqueeze(-1).float()
        char_embeds_masked = char_embeds * mask_expanded

        # Sum and divide by count of real characters
        char_sum = char_embeds_masked.sum(dim=2)  # (batch, max_bars, d_char)
        char_count = char_mask.sum(dim=2, keepdim=True).float().clamp(min=1)
        bar_embeds = char_sum / char_count  # (batch, max_bars, d_char)

        # Project to model dimension
        bar_embeds = self.projection(bar_embeds)  # (batch, max_bars, d_model)

        # Add positional embeddings
        positions = torch.arange(
            self.max_bars, device=bar_indices.device
        ).unsqueeze(0)
        bar_embeds = bar_embeds + self.pos_embed(positions)

        # Layer norm + dropout
        bar_embeds = self.layer_norm(bar_embeds)
        bar_embeds = self.dropout(bar_embeds)

        return bar_embeds, bar_mask

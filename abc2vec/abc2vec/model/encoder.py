"""ABC2Vec Transformer Encoder Architecture."""

import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc2vec.model.embedding import PatchEmbedding


@dataclass
class ABC2VecConfig:
    """
    Configuration for ABC2Vec model.

    Attributes:
        vocab_size: Size of character vocabulary
        max_bar_length: Maximum characters per bar
        max_bars: Maximum bars per tune
        pad_idx: Padding token index
        mask_idx: Mask token index
        d_char: Character embedding dimension
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        d_embed: Final tune embedding dimension
        mask_ratio: Ratio of bars to mask for MMM
        temperature: Temperature for contrastive losses
    """

    vocab_size: int = 128
    max_bar_length: int = 64
    max_bars: int = 64
    pad_idx: int = 0
    mask_idx: int = 4
    d_char: int = 64
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    d_embed: int = 128
    mask_ratio: float = 0.15
    temperature: float = 0.07

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ABC2VecConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer.

    Uses pre-layer normalization (more stable training than post-norm)
    and GELU activation in the feed-forward network.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len), True for real tokens

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Convert mask for MultiheadAttention: True = ignore, False = attend
        key_padding_mask = ~mask if mask is not None else None

        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)

        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)

        return x


class ABC2VecEncoder(nn.Module):
    """
    ABC2Vec Transformer Encoder.

    Takes bar-patched input and produces:
        - Bar-level embeddings (for masked modeling)
        - Tune-level embedding (for contrastive objectives)

    Architecture:
        ABC Notation → Bar Patchifier → Patch Embedding →
        Transformer Encoder → Pooling → Tune Embedding

    Args:
        config: ABC2VecConfig instance with model hyperparameters
    """

    def __init__(self, config: ABC2VecConfig) -> None:
        super().__init__()
        self.config = config

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            vocab_size=config.vocab_size,
            d_char=config.d_char,
            d_model=config.d_model,
            max_bar_length=config.max_bar_length,
            max_bars=config.max_bars,
            pad_idx=config.pad_idx,
            dropout=config.dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Tune-level projection head
        self.tune_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_embed),
        )

    def encode_bars(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode bar-patched input to bar-level embeddings.

        Args:
            bar_indices: (batch, max_bars, max_bar_length) character indices
            char_mask: (batch, max_bars, max_bar_length) boolean mask
            bar_mask: (batch, max_bars) boolean mask for real bars

        Returns:
            Bar embeddings of shape (batch, max_bars, d_model)
        """
        # Patch embedding
        x, mask = self.patch_embed(bar_indices, char_mask, bar_mask)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        return x

    def get_tune_embedding(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get tune-level embedding via mean pooling + projection.

        Args:
            bar_indices: (batch, max_bars, max_bar_length)
            char_mask: (batch, max_bars, max_bar_length)
            bar_mask: (batch, max_bars)

        Returns:
            Tune embedding of shape (batch, d_embed), L2-normalized
        """
        bar_embeddings = self.encode_bars(bar_indices, char_mask, bar_mask)

        # Mean pool over non-padding bars
        mask_expanded = bar_mask.unsqueeze(-1).float()
        pooled = (bar_embeddings * mask_expanded).sum(dim=1)
        bar_count = bar_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = pooled / bar_count

        # Project to embedding dimension and L2 normalize
        tune_emb = self.tune_projection(pooled)
        tune_emb = F.normalize(tune_emb, p=2, dim=-1)

        return tune_emb

    def forward(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            Dictionary containing:
                - bar_embeddings: (batch, max_bars, d_model)
                - pooled: (batch, d_model) mean-pooled representation
                - tune_embedding: (batch, d_embed) L2-normalized tune embedding
        """
        bar_embeddings = self.encode_bars(bar_indices, char_mask, bar_mask)

        # Mean pooling
        mask_expanded = bar_mask.unsqueeze(-1).float()
        pooled = (bar_embeddings * mask_expanded).sum(dim=1)
        bar_count = bar_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = pooled / bar_count

        # Tune embedding
        tune_emb = F.normalize(self.tune_projection(pooled), p=2, dim=-1)

        return {
            "bar_embeddings": bar_embeddings,
            "pooled": pooled,
            "tune_embedding": tune_emb,
        }


class MaskedMusicModelingHead(nn.Module):
    """
    Head for predicting masked bar patches.

    Takes bar-level embeddings and predicts the characters
    that were in each masked bar.

    Args:
        config: ABC2VecConfig instance
    """

    def __init__(self, config: ABC2VecConfig) -> None:
        super().__init__()
        self.config = config

        # Transform bar embedding to character space
        self.bar_to_chars = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

        # Predict character at each position
        self.char_predictor = nn.Linear(config.d_model, config.vocab_size)

        # Learnable position embeddings for characters within a bar
        self.char_pos = nn.Embedding(config.max_bar_length, config.d_model)

    def forward(
        self,
        bar_embeddings: torch.Tensor,
        masked_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict characters for masked bars.

        Args:
            bar_embeddings: (batch, max_bars, d_model) encoder output
            masked_indices: (batch, num_masked) indices of masked bars

        Returns:
            Character logits (batch, num_masked, max_bar_length, vocab_size)
        """
        # Gather masked bar embeddings
        idx = masked_indices.unsqueeze(-1).expand(
            -1, -1, bar_embeddings.shape[-1]
        )
        masked_embeds = torch.gather(bar_embeddings, dim=1, index=idx)

        # Transform
        hidden = self.bar_to_chars(masked_embeds)

        # Expand to character positions
        char_positions = self.char_pos.weight.unsqueeze(0).unsqueeze(0)
        hidden_expanded = hidden.unsqueeze(2) + char_positions

        # Predict characters
        char_logits = self.char_predictor(hidden_expanded)

        return char_logits


class ABC2VecModel(nn.Module):
    """
    Complete ABC2Vec model.

    Combines:
        - Transformer encoder (bar-level embeddings)
        - Masked Music Modeling head
        - Tune-level embedding (for contrastive objectives)

    Args:
        config: ABC2VecConfig instance
    """

    def __init__(self, config: ABC2VecConfig) -> None:
        super().__init__()
        self.config = config

        # Core encoder
        self.encoder = ABC2VecEncoder(config)

        # MMM head
        self.mmm_head = MaskedMusicModelingHead(config)

    def mask_bars(
        self,
        bar_indices: torch.Tensor,
        bar_mask: torch.Tensor,
        mask_ratio: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random bar masking for MMM objective.

        Args:
            bar_indices: (batch, max_bars, max_bar_length)
            bar_mask: (batch, max_bars) boolean mask
            mask_ratio: Fraction of bars to mask

        Returns:
            Tuple of:
                - masked_bar_indices: Input with masked bars replaced by pad_idx
                - masked_positions: (batch, num_masked) indices of masked bars
                - target_bar_indices: Original content of masked bars
                - mask_labels: (batch, num_masked) boolean mask for valid positions
        """
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio

        batch_size, max_bars, max_bar_length = bar_indices.shape
        device = bar_indices.device

        masked_bar_indices = bar_indices.clone()

        # Determine number of bars to mask per sample
        num_real_bars = bar_mask.sum(dim=1).long()
        num_to_mask = (num_real_bars.float() * mask_ratio).long().clamp(min=1)
        max_masked = num_to_mask.max().item()

        # Initialize output tensors
        masked_positions = torch.zeros(
            batch_size, max_masked, dtype=torch.long, device=device
        )
        target_bar_indices = torch.zeros(
            batch_size, max_masked, max_bar_length,
            dtype=torch.long, device=device
        )
        mask_labels = torch.zeros(
            batch_size, max_masked, dtype=torch.bool, device=device
        )

        # Randomly select bars to mask for each sample
        for b in range(batch_size):
            n_real = num_real_bars[b].item()
            n_mask = num_to_mask[b].item()

            # Random selection
            perm = torch.randperm(n_real, device=device)[:n_mask]

            masked_positions[b, :n_mask] = perm
            mask_labels[b, :n_mask] = True

            # Store targets and replace with padding
            for i, pos in enumerate(perm):
                target_bar_indices[b, i] = bar_indices[b, pos]
                masked_bar_indices[b, pos] = self.config.pad_idx

        return (
            masked_bar_indices,
            masked_positions,
            target_bar_indices,
            mask_labels,
        )

    def forward_mmm(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Masked Music Modeling.

        Args:
            bar_indices: (batch, max_bars, max_bar_length)
            char_mask: (batch, max_bars, max_bar_length)
            bar_mask: (batch, max_bars)

        Returns:
            Dictionary with keys:
                - mmm_logits: (batch, num_masked, max_bar_length, vocab_size)
                - mmm_targets: (batch, num_masked, max_bar_length)
                - mmm_mask: (batch, num_masked)
                - bar_embeddings: (batch, max_bars, d_model)
        """
        # Apply masking
        (
            masked_indices,
            masked_pos,
            targets,
            mask_labels,
        ) = self.mask_bars(bar_indices, bar_mask)

        # Update char_mask to reflect masked positions
        masked_char_mask = char_mask.clone()
        for b in range(bar_indices.shape[0]):
            for pos in masked_pos[b][mask_labels[b]]:
                masked_char_mask[b, pos] = False

        # Encode with masked bars
        bar_embeddings = self.encoder.encode_bars(
            masked_indices, masked_char_mask, bar_mask
        )

        # Predict masked bars
        char_logits = self.mmm_head(bar_embeddings, masked_pos)

        return {
            "mmm_logits": char_logits,
            "mmm_targets": targets,
            "mmm_mask": mask_labels,
            "bar_embeddings": bar_embeddings,
        }

    def forward_contrastive(
        self,
        bar_indices_1: torch.Tensor,
        char_mask_1: torch.Tensor,
        bar_mask_1: torch.Tensor,
        bar_indices_2: torch.Tensor,
        char_mask_2: torch.Tensor,
        bar_mask_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive objectives.

        Used for:
            - Transposition Invariance (original + transposed)
            - Section Contrastive Learning (A-section + B-section)

        Args:
            bar_indices_1: First input
            char_mask_1: First input character mask
            bar_mask_1: First input bar mask
            bar_indices_2: Second input
            char_mask_2: Second input character mask
            bar_mask_2: Second input bar mask

        Returns:
            Tuple of (emb_1, emb_2), each of shape (batch, d_embed)
        """
        emb_1 = self.encoder.get_tune_embedding(
            bar_indices_1, char_mask_1, bar_mask_1
        )
        emb_2 = self.encoder.get_tune_embedding(
            bar_indices_2, char_mask_2, bar_mask_2
        )
        return emb_1, emb_2

    def get_embedding(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-time: get tune-level embedding.

        Args:
            bar_indices: (batch, max_bars, max_bar_length)
            char_mask: (batch, max_bars, max_bar_length)
            bar_mask: (batch, max_bars)

        Returns:
            Tune embedding (batch, d_embed), L2-normalized
        """
        return self.encoder.get_tune_embedding(
            bar_indices, char_mask, bar_mask
        )

    def forward(
        self,
        bar_indices: torch.Tensor,
        char_mask: torch.Tensor,
        bar_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass returning all representations.

        Args:
            bar_indices: (batch, max_bars, max_bar_length)
            char_mask: (batch, max_bars, max_bar_length)
            bar_mask: (batch, max_bars)

        Returns:
            Dictionary with bar_embeddings, pooled, and tune_embedding
        """
        return self.encoder(bar_indices, char_mask, bar_mask)

    def save_pretrained(self, path: str) -> None:
        """
        Save model weights and configuration.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_pretrained(cls, path: str, device: str = "cpu") -> "ABC2VecModel":
        """
        Load pre-trained model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded ABC2VecModel instance
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle config being either dict or JSON string
        config_data = checkpoint["config"]
        if isinstance(config_data, str):
            config_data = json.loads(config_data)

        config = ABC2VecConfig(**config_data)
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

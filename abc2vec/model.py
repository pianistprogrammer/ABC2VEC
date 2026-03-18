
"""ABC2Vec Model Architecture."""

import json, math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PatchEmbedding


@dataclass
class ABC2VecConfig:
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

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        key_padding_mask = ~mask if mask is not None else None
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)
        return x


class ABC2VecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            config.vocab_size, config.d_char, config.d_model,
            config.max_bar_length, config.max_bars, config.pad_idx)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.tune_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_embed))

    def encode_bars(self, bar_indices, char_mask, bar_mask):
        x, mask = self.patch_embed(bar_indices, char_mask, bar_mask)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)

    def get_tune_embedding(self, bar_indices, char_mask, bar_mask):
        bar_embeddings = self.encode_bars(bar_indices, char_mask, bar_mask)
        mask_exp = bar_mask.unsqueeze(-1).float()
        pooled = (bar_embeddings * mask_exp).sum(dim=1) / bar_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return F.normalize(self.tune_projection(pooled), p=2, dim=-1)

    def forward(self, bar_indices, char_mask, bar_mask):
        bar_embeddings = self.encode_bars(bar_indices, char_mask, bar_mask)
        mask_exp = bar_mask.unsqueeze(-1).float()
        pooled = (bar_embeddings * mask_exp).sum(dim=1) / bar_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        tune_emb = F.normalize(self.tune_projection(pooled), p=2, dim=-1)
        return {"bar_embeddings": bar_embeddings, "pooled": pooled, "tune_embedding": tune_emb}


class MaskedMusicModelingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bar_to_chars = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.LayerNorm(config.d_model))
        self.char_predictor = nn.Linear(config.d_model, config.vocab_size)
        self.char_pos = nn.Embedding(config.max_bar_length, config.d_model)

    def forward(self, bar_embeddings, masked_indices):
        idx = masked_indices.unsqueeze(-1).expand(-1, -1, bar_embeddings.shape[-1])
        masked_embeds = torch.gather(bar_embeddings, dim=1, index=idx)
        hidden = self.bar_to_chars(masked_embeds)
        char_positions = self.char_pos.weight.unsqueeze(0).unsqueeze(0)
        hidden_expanded = hidden.unsqueeze(2) + char_positions
        return self.char_predictor(hidden_expanded)


class ABC2VecModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ABC2VecEncoder(config)
        self.mmm_head = MaskedMusicModelingHead(config)

    def mask_bars(self, bar_indices, bar_mask, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio
        batch_size, max_bars, max_bar_length = bar_indices.shape
        device = bar_indices.device
        masked_bar_indices = bar_indices.clone()
        num_real = bar_mask.sum(dim=1).long()
        num_to_mask = (num_real.float() * mask_ratio).long().clamp(min=1)
        max_masked = num_to_mask.max().item()
        masked_pos = torch.zeros(batch_size, max_masked, dtype=torch.long, device=device)
        targets = torch.zeros(batch_size, max_masked, max_bar_length, dtype=torch.long, device=device)
        mask_labels = torch.zeros(batch_size, max_masked, dtype=torch.bool, device=device)
        for b in range(batch_size):
            n = num_to_mask[b].item()
            perm = torch.randperm(num_real[b].item(), device=device)[:n]
            masked_pos[b, :n] = perm
            mask_labels[b, :n] = True
            for i, p in enumerate(perm):
                targets[b, i] = bar_indices[b, p]
                masked_bar_indices[b, p] = self.config.pad_idx
        return masked_bar_indices, masked_pos, targets, mask_labels

    def forward_mmm(self, bar_indices, char_mask, bar_mask):
        masked_indices, masked_pos, targets, mask_labels = self.mask_bars(bar_indices, bar_mask)
        masked_cm = char_mask.clone()
        for b in range(bar_indices.shape[0]):
            for pos in masked_pos[b][mask_labels[b]]:
                masked_cm[b, pos] = False
        bar_embeddings = self.encoder.encode_bars(masked_indices, masked_cm, bar_mask)
        logits = self.mmm_head(bar_embeddings, masked_pos)
        return {"mmm_logits": logits, "mmm_targets": targets, "mmm_mask": mask_labels,
                "bar_embeddings": bar_embeddings}

    def forward_contrastive(self, bi1, cm1, bm1, bi2, cm2, bm2):
        return (self.encoder.get_tune_embedding(bi1, cm1, bm1),
                self.encoder.get_tune_embedding(bi2, cm2, bm2))

    def get_embedding(self, bar_indices, char_mask, bar_mask):
        return self.encoder.get_tune_embedding(bar_indices, char_mask, bar_mask)

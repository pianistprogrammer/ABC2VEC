"""Pre-training objectives for ABC2Vec."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc2vec.model.encoder import ABC2VecConfig


class MaskedMusicModelingLoss(nn.Module):
    """
    Masked Music Modeling (MMM) Loss.

    BERT-style objective: predict masked bar patches from their context.
    Cross-entropy loss over predicted characters in masked bars.

    Args:
        pad_idx: Padding token index to ignore in loss computation
    """

    def __init__(self, pad_idx: int = 0) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_idx, reduction="mean"
        )

    def forward(
        self,
        mmm_logits: torch.Tensor,
        mmm_targets: torch.Tensor,
        mmm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMM loss.

        Args:
            mmm_logits: (batch, num_masked, max_bar_length, vocab_size)
            mmm_targets: (batch, num_masked, max_bar_length) ground truth
            mmm_mask: (batch, num_masked) boolean mask for valid positions

        Returns:
            Scalar loss tensor
        """
        # Only compute loss on valid masked positions
        valid = mmm_mask.unsqueeze(-1).expand_as(mmm_targets)

        logits_flat = mmm_logits[valid].view(-1, mmm_logits.shape[-1])
        targets_flat = mmm_targets[valid].view(-1)

        if logits_flat.shape[0] == 0:
            return torch.tensor(
                0.0, device=mmm_logits.device, requires_grad=True
            )

        return self.loss_fn(logits_flat, targets_flat)


class SectionContrastiveLoss(nn.Module):
    """
    Section Contrastive Loss (SCL) - Novel objective.

    Folk tunes have AABB structure. This objective trains the model
    so that A-section and B-section of the same tune are closer than
    sections from different tunes.

    Uses symmetric InfoNCE (NT-Xent) loss.

    Args:
        temperature: Temperature parameter for contrastive loss
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SCL loss.

        Args:
            emb_a: (batch, d_embed) embeddings of A-sections
            emb_b: (batch, d_embed) embeddings of B-sections (same tune)

        Returns:
            Scalar loss tensor
        """
        batch_size = emb_a.shape[0]
        device = emb_a.device

        # Normalize embeddings
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        # Compute similarity matrix
        sim_ab = torch.matmul(emb_a, emb_b.T) / self.temperature
        sim_ba = sim_ab.T

        # Labels: diagonal entries are positive pairs
        labels = torch.arange(batch_size, device=device)

        # Symmetric InfoNCE
        loss_ab = F.cross_entropy(sim_ab, labels)
        loss_ba = F.cross_entropy(sim_ba, labels)

        return (loss_ab + loss_ba) / 2


class TranspositionInvarianceLoss(nn.Module):
    """
    Transposition Invariance (TI) Loss.

    The same tune in different keys should produce similar embeddings.
    Contrastive loss between original and transposed versions.

    Args:
        temperature: Temperature parameter for contrastive loss
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb_orig: torch.Tensor,
        emb_trans: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TI loss.

        Args:
            emb_orig: (batch, d_embed) embeddings of original tunes
            emb_trans: (batch, d_embed) embeddings of transposed versions

        Returns:
            Scalar loss tensor
        """
        batch_size = emb_orig.shape[0]
        device = emb_orig.device

        # Normalize
        emb_orig = F.normalize(emb_orig, p=2, dim=-1)
        emb_trans = F.normalize(emb_trans, p=2, dim=-1)

        # Concatenate for richer negative set
        emb_all = torch.cat([emb_orig, emb_trans], dim=0)

        # Similarity matrix: (2*batch, 2*batch)
        sim = torch.matmul(emb_all, emb_all.T) / self.temperature

        # Positive pairs: (i, i+batch) and (i+batch, i)
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device),
            ]
        )

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))

        return F.cross_entropy(sim, labels)


class VariantAwareContrastiveLoss(nn.Module):
    """
    Variant-Aware Contrastive (VAC) Loss - Novel objective.

    Uses ground-truth variant labels: same tune family, different settings
    are hard positives. Implements supervised contrastive learning (SupCon).

    Args:
        temperature: Temperature parameter for contrastive loss
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        group_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VAC loss.

        Args:
            embeddings: (batch, d_embed) tune embeddings
            group_ids: (batch,) integer group IDs (same ID = same tune family)

        Returns:
            Scalar loss tensor
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask: group_ids[i] == group_ids[j]
        ids = group_ids.unsqueeze(1)
        pos_mask = (ids == ids.T).float()

        # Remove self from positives
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - self_mask

        # Count positives per anchor
        pos_count = pos_mask.sum(dim=1)
        valid_anchors = pos_count > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Mask self-similarity before log_softmax
        sim = sim.masked_fill(self_mask.bool(), float("-inf"))

        # Log-sum-exp for denominator
        log_denom = torch.logsumexp(sim, dim=1)

        # Log probability for each pair
        log_prob = sim - log_denom.unsqueeze(1)

        # Average log-prob over positive pairs
        pos_log_prob = torch.where(
            pos_mask.bool(), log_prob, torch.zeros_like(log_prob)
        ).sum(dim=1) / pos_count.clamp(min=1)

        # Average over valid anchors
        loss = -pos_log_prob[valid_anchors].mean()

        return loss


class ABC2VecLoss(nn.Module):
    """
    Combined multi-objective loss for ABC2Vec.

    Combines all four pre-training objectives:
        L = λ_mmm * L_MMM + λ_scl * L_SCL + λ_ti * L_TI + λ_vac * L_VAC

    Args:
        config: ABC2VecConfig instance
        lambda_mmm: Weight for Masked Music Modeling loss
        lambda_scl: Weight for Section Contrastive Loss
        lambda_ti: Weight for Transposition Invariance loss
        lambda_vac: Weight for Variant-Aware Contrastive loss
    """

    def __init__(
        self,
        config: ABC2VecConfig,
        lambda_mmm: float = 1.0,
        lambda_scl: float = 0.5,
        lambda_ti: float = 0.5,
        lambda_vac: float = 0.5,
    ) -> None:
        super().__init__()

        self.lambda_mmm = lambda_mmm
        self.lambda_scl = lambda_scl
        self.lambda_ti = lambda_ti
        self.lambda_vac = lambda_vac

        self.mmm_loss = MaskedMusicModelingLoss(pad_idx=config.pad_idx)
        self.scl_loss = SectionContrastiveLoss(temperature=config.temperature)
        self.ti_loss = TranspositionInvarianceLoss(temperature=config.temperature)
        self.vac_loss = VariantAwareContrastiveLoss(temperature=config.temperature)

    def forward(
        self,
        mmm_logits=None,
        mmm_targets=None,
        mmm_mask=None,
        scl_emb_a=None,
        scl_emb_b=None,
        ti_emb_orig=None,
        ti_emb_trans=None,
        vac_embeddings=None,
        vac_group_ids=None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Each objective is optional - only provided objectives are computed.

        Args:
            mmm_logits: MMM predictions
            mmm_targets: MMM targets
            mmm_mask: MMM mask
            scl_emb_a: Section A embeddings
            scl_emb_b: Section B embeddings
            ti_emb_orig: Original tune embeddings
            ti_emb_trans: Transposed tune embeddings
            vac_embeddings: Variant tune embeddings
            vac_group_ids: Variant group IDs

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        total = torch.tensor(0.0, requires_grad=True, device=self._get_device())

        # MMM
        if all(x is not None for x in [mmm_logits, mmm_targets, mmm_mask]):
            l_mmm = self.mmm_loss(mmm_logits, mmm_targets, mmm_mask)
            losses["mmm"] = l_mmm.item()
            total = total + self.lambda_mmm * l_mmm

        # SCL
        if scl_emb_a is not None and scl_emb_b is not None:
            l_scl = self.scl_loss(scl_emb_a, scl_emb_b)
            losses["scl"] = l_scl.item()
            total = total + self.lambda_scl * l_scl

        # TI
        if ti_emb_orig is not None and ti_emb_trans is not None:
            l_ti = self.ti_loss(ti_emb_orig, ti_emb_trans)
            losses["ti"] = l_ti.item()
            total = total + self.lambda_ti * l_ti

        # VAC
        if vac_embeddings is not None and vac_group_ids is not None:
            l_vac = self.vac_loss(vac_embeddings, vac_group_ids)
            losses["vac"] = l_vac.item()
            total = total + self.lambda_vac * l_vac

        losses["total"] = total.item()

        return total, losses

    def _get_device(self) -> torch.device:
        """Get device from first parameter."""
        return next(self.parameters()).device

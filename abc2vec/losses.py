
"""Pre-training objectives for ABC2Vec."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMusicModelingLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")

    def forward(self, mmm_logits, mmm_targets, mmm_mask):
        valid = mmm_mask.unsqueeze(-1).expand_as(mmm_targets)
        logits_flat = mmm_logits[valid].view(-1, mmm_logits.shape[-1])
        targets_flat = mmm_targets[valid].view(-1)
        if logits_flat.shape[0] == 0:
            return torch.tensor(0.0, device=mmm_logits.device, requires_grad=True)
        return self.loss_fn(logits_flat, targets_flat)


class SectionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b):
        batch_size = emb_a.shape[0]
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)
        sim_ab = torch.matmul(emb_a, emb_b.T) / self.temperature
        labels = torch.arange(batch_size, device=emb_a.device)
        return (F.cross_entropy(sim_ab, labels) + F.cross_entropy(sim_ab.T, labels)) / 2


class TranspositionInvarianceLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_orig, emb_trans):
        batch_size = emb_orig.shape[0]
        device = emb_orig.device
        emb_orig = F.normalize(emb_orig, p=2, dim=-1)
        emb_trans = F.normalize(emb_trans, p=2, dim=-1)
        emb_all = torch.cat([emb_orig, emb_trans], dim=0)
        sim = torch.matmul(emb_all, emb_all.T) / self.temperature
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)])
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))
        return F.cross_entropy(sim, labels)


class VariantAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, group_ids):
        batch_size = embeddings.shape[0]
        device = embeddings.device
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        group_ids = group_ids.unsqueeze(0)
        pos_mask = (group_ids == group_ids.T).float()
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - self_mask
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        sim = sim.masked_fill(self_mask.bool(), float("-inf"))
        log_denom = torch.logsumexp(sim, dim=1)
        log_prob = sim - log_denom.unsqueeze(1)
        pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp(min=1)
        return -pos_log_prob[valid].mean()


class ABC2VecLoss(nn.Module):
    def __init__(self, config, lambda_mmm=1.0, lambda_scl=0.5, lambda_ti=0.5, lambda_vac=0.5):
        super().__init__()
        self.lambda_mmm = lambda_mmm
        self.lambda_scl = lambda_scl
        self.lambda_ti = lambda_ti
        self.lambda_vac = lambda_vac
        self.mmm_loss = MaskedMusicModelingLoss(pad_idx=config.pad_idx)
        self.scl_loss = SectionContrastiveLoss(temperature=config.temperature)
        self.ti_loss = TranspositionInvarianceLoss(temperature=config.temperature)
        self.vac_loss = VariantAwareContrastiveLoss(temperature=config.temperature)

    def forward(self, mmm_logits=None, mmm_targets=None, mmm_mask=None,
                scl_emb_a=None, scl_emb_b=None,
                ti_emb_orig=None, ti_emb_trans=None,
                vac_embeddings=None, vac_group_ids=None):
        losses = {}
        total = torch.tensor(0.0, requires_grad=True)
        if mmm_logits is not None:
            l = self.mmm_loss(mmm_logits, mmm_targets, mmm_mask)
            losses["mmm"] = l.item()
            total = total + self.lambda_mmm * l
        if scl_emb_a is not None:
            l = self.scl_loss(scl_emb_a, scl_emb_b)
            losses["scl"] = l.item()
            total = total + self.lambda_scl * l
        if ti_emb_orig is not None:
            l = self.ti_loss(ti_emb_orig, ti_emb_trans)
            losses["ti"] = l.item()
            total = total + self.lambda_ti * l
        if vac_embeddings is not None:
            l = self.vac_loss(vac_embeddings, vac_group_ids)
            losses["vac"] = l.item()
            total = total + self.lambda_vac * l
        losses["total"] = total.item()
        return total, losses

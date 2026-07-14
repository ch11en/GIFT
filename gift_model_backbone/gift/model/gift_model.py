#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GIFT model: utility matching plus fault-tolerance decision layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


NEG_SENTIMENT_ID = 0


class BehaviorSemanticUtilityBackbone(nn.Module):
    """Industrial-style behavior + semantic fusion backbone for GIFT.

    It keeps user/item ID embeddings for collaborative signal, adds train-only
    user/item behavior statistics, and fuses review quadruple semantics. This is
    the default backbone because recommendation systems usually depend on user
    behavior analysis rather than graph neural networks.
    """

    def __init__(self, num_users, num_items, embedding_dim=128, behavior_dim=5, dropout=0.2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.5))
        self.user_behavior_proj = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )
        self.item_behavior_proj = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, user_ids, item_ids, user_behavior_features, item_behavior_features, semantic_summary):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        ub = self.user_behavior_proj(user_behavior_features)
        ib = self.item_behavior_proj(item_behavior_features)
        semantic = self.semantic_proj(semantic_summary)
        user_side = torch.cat([u, ub], dim=-1)
        item_side = torch.cat([i, ib], dim=-1)
        score = self.mlp(torch.cat([user_side, item_side, semantic], dim=-1)).squeeze(-1)
        score = score + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)
        return score + self.global_bias


class NeuMFUtilityBackbone(nn.Module):
    """NeuMF-style utility backbone with behavior and semantic side features."""

    def __init__(self, num_users, num_items, embedding_dim=128, behavior_dim=5, dropout=0.2):
        super().__init__()
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.5))
        self.user_behavior_proj = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )
        self.item_behavior_proj = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        mlp_in = embedding_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, user_ids, item_ids, user_behavior_features, item_behavior_features, semantic_summary):
        gmf = self.gmf_user_emb(user_ids) * self.gmf_item_emb(item_ids)
        ub = self.user_behavior_proj(user_behavior_features)
        ib = self.item_behavior_proj(item_behavior_features)
        semantic = self.semantic_proj(semantic_summary)
        mlp_input = torch.cat(
            [self.mlp_user_emb(user_ids), self.mlp_item_emb(item_ids), ub, ib, semantic],
            dim=-1,
        )
        mlp_out = self.mlp(mlp_input)
        score = self.fusion(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)
        score = score + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)
        return score + self.global_bias


class CrossLayerV2(nn.Module):
    """DCNv2-style full-rank cross layer."""

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, input_dim)

    def forward(self, x0, x):
        return x0 * self.weight(x) + x


class DCNv2UtilityBackbone(nn.Module):
    """DCNv2-style cross/deep utility backbone for dense recommendation features."""

    def __init__(self, num_users, num_items, embedding_dim=128, behavior_dim=5, dropout=0.2, num_cross_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.5))
        self.user_behavior_proj = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )
        self.item_behavior_proj = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.input_dim = embedding_dim * 4
        self.cross_layers = nn.ModuleList([CrossLayerV2(self.input_dim) for _ in range(num_cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(self.input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.Linear(self.input_dim + embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, user_ids, item_ids, user_behavior_features, item_behavior_features, semantic_summary):
        dense = torch.cat(
            [
                self.user_emb(user_ids),
                self.item_emb(item_ids),
                self.user_behavior_proj(user_behavior_features),
                self.item_behavior_proj(item_behavior_features),
                self.semantic_proj(semantic_summary),
            ],
            dim=-1,
        )
        cross = dense
        for layer in self.cross_layers:
            cross = layer(dense, cross)
        deep = self.deep(dense)
        score = self.output(torch.cat([cross, deep], dim=-1)).squeeze(-1)
        score = score + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)
        return score + self.global_bias


def build_utility_backbone(backbone, num_users, num_items, embedding_dim, dropout):
    backbone = str(backbone or "behavior_mlp").lower()
    if backbone == "behavior_mlp":
        return BehaviorSemanticUtilityBackbone(num_users, num_items, embedding_dim, dropout=dropout)
    if backbone == "neumf":
        return NeuMFUtilityBackbone(num_users, num_items, embedding_dim, dropout=dropout)
    if backbone == "dcnv2":
        return DCNv2UtilityBackbone(num_users, num_items, embedding_dim, dropout=dropout)
    raise ValueError(f"Unsupported GIFT utility backbone: {backbone}")


class GIFTModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_aspects,
        embedding_dim=128,
        lambda_risk=0.5,
        use_regret_risk=True,
        use_loss_aversion=True,
        use_max_risk=True,
        risk_pooling="max",
        use_severity=True,
        fixed_negative_penalty=False,
        backbone="behavior_mlp",
        dropout=0.2,
    ):
        super().__init__()
        self.lambda_risk = float(lambda_risk)
        self.use_regret_risk = bool(use_regret_risk)
        self.use_loss_aversion = bool(use_loss_aversion)
        self.risk_pooling = str(risk_pooling or ("max" if use_max_risk else "mean")).lower()
        self.use_max_risk = self.risk_pooling == "max"
        self.use_severity = bool(use_severity)
        self.fixed_negative_penalty = bool(fixed_negative_penalty)
        self.backbone = str(backbone or "behavior_mlp").lower()

        self.utility_backbone = build_utility_backbone(self.backbone, num_users, num_items, embedding_dim, dropout)
        self.aspect_emb = nn.Embedding(num_aspects, embedding_dim, padding_idx=0)
        self.sentiment_emb = nn.Embedding(3, embedding_dim)
        self.quad_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.regret_head = nn.Linear(embedding_dim, 1)
        self.low_regret_head = nn.Linear(embedding_dim, 1)
        self.utility_head = nn.Linear(embedding_dim, 1)
        self.conflict_head = nn.Linear(embedding_dim, 1)
        self.severity_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def encode_quadruples(self, aspect_category_ids, sentiment_ids):
        aspect_h = self.aspect_emb(aspect_category_ids)
        sentiment_h = self.sentiment_emb(sentiment_ids.clamp(0, 2))
        return self.quad_encoder(torch.cat([aspect_h, sentiment_h], dim=-1))

    def forward(self, batch):
        user_ids = batch["user_ids"]
        item_ids = batch["item_ids"]
        aspect_ids = batch["aspect_category_ids"]
        sentiment_ids = batch["sentiment_ids"]
        quad_mask = batch["quad_mask"]

        quad_repr = self.encode_quadruples(aspect_ids, sentiment_ids)
        semantic_summary = (quad_repr * quad_mask.unsqueeze(-1)).sum(dim=1)
        semantic_summary = semantic_summary / quad_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        user_behavior = batch.get("user_behavior_features")
        if user_behavior is None:
            user_behavior = torch.zeros(user_ids.size(0), 5, device=user_ids.device)
        item_behavior = batch.get("item_behavior_features")
        if item_behavior is None:
            item_behavior = torch.zeros(item_ids.size(0), 5, device=item_ids.device)
        utility_score = self.utility_backbone(
            user_ids,
            item_ids,
            user_behavior,
            item_behavior,
            semantic_summary,
        )

        regret_logits = self.regret_head(quad_repr).squeeze(-1)
        low_regret_logits = self.low_regret_head(quad_repr).squeeze(-1)
        utility_logits = self.utility_head(quad_repr).squeeze(-1)
        conflict_logits = self.conflict_head(quad_repr).squeeze(-1)
        severity_logits = self.severity_head(quad_repr).squeeze(-1)

        neg_mask = ((sentiment_ids == NEG_SENTIMENT_ID).float() * quad_mask)
        if self.fixed_negative_penalty:
            regret_prob = neg_mask
        else:
            regret_prob = torch.sigmoid(regret_logits) * neg_mask

        severity_score = torch.sigmoid(severity_logits) * neg_mask
        effective_severity = severity_score if self.use_severity else neg_mask
        user_sensitivity = batch.get("user_aspect_risk_values")
        if user_sensitivity is None:
            user_sensitivity = torch.ones_like(neg_mask)
        per_quad_risk = regret_prob * effective_severity * user_sensitivity * neg_mask

        if self.use_loss_aversion:
            loss_aversion = batch.get("user_aspect_loss_aversion_values")
            if loss_aversion is not None:
                per_quad_risk = per_quad_risk * loss_aversion

        max_regret_risk = per_quad_risk.max(dim=1).values
        mean_regret_risk = per_quad_risk.sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1.0)
        regret_risk_score = max_regret_risk if self.risk_pooling == "max" else mean_regret_risk
        if self.use_regret_risk:
            final_score = utility_score - self.lambda_risk * regret_risk_score
        else:
            final_score = utility_score

        return {
            "final_score": final_score,
            "utility_score": utility_score,
            "regret_risk_score": regret_risk_score,
            "max_regret_risk": max_regret_risk,
            "mean_regret_risk": mean_regret_risk,
            "regret_logits": regret_logits,
            "low_regret_logits": low_regret_logits,
            "utility_logits": utility_logits,
            "conflict_logits": conflict_logits,
            "severity_logits": severity_logits,
            "severity_score": severity_score,
            "effective_severity": effective_severity,
            "regret_prob": regret_prob,
            "weighted_quad_risk": per_quad_risk,
        }


def masked_bce_with_logits(logits, labels, mask):
    mask = mask.float()
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def gift_loss(outputs, batch, args):
    ratings = batch["ratings"]
    loss_rec = F.mse_loss(outputs["final_score"], ratings)
    quad_mask = batch["quad_mask"]
    sentiment_ids = batch["sentiment_ids"]
    regret_mask = (((sentiment_ids == NEG_SENTIMENT_ID).float() + batch["regret_labels"]).clamp_max(1.0)) * quad_mask

    loss_regret = masked_bce_with_logits(outputs["regret_logits"], batch["regret_labels"], regret_mask)
    loss_low = masked_bce_with_logits(outputs["low_regret_logits"], batch["low_regret_labels"], quad_mask)
    loss_utility = masked_bce_with_logits(outputs["utility_logits"], batch["utility_labels"], quad_mask)
    loss_conflict = masked_bce_with_logits(outputs["conflict_logits"], batch["conflict_labels"], quad_mask)
    neg_mask = ((sentiment_ids == NEG_SENTIMENT_ID).float() * quad_mask)
    severity_labels = batch.get("severity_labels")
    if severity_labels is None:
        severity_labels = torch.where(neg_mask > 0, torch.full_like(neg_mask, 0.5), torch.zeros_like(neg_mask))
    if neg_mask.sum().item() > 0:
        loss_severity = F.mse_loss(outputs["severity_score"][neg_mask > 0], severity_labels.float()[neg_mask > 0])
        avg_predicted_severity = outputs["severity_score"][neg_mask > 0].detach().mean()
        avg_label_severity = severity_labels.float()[neg_mask > 0].detach().mean()
    else:
        loss_severity = ratings.sum() * 0.0
        avg_predicted_severity = ratings.sum().detach() * 0.0
        avg_label_severity = ratings.sum().detach() * 0.0

    risk_target_mask = ((ratings <= 2.0) | (ratings >= 4.0)).float()
    risk_target = (ratings <= 2.0).float()
    if risk_target_mask.sum().item() > 0:
        risk_prob = outputs["max_regret_risk"].float().clamp(1e-6, 1.0 - 1e-6)
        with torch.cuda.amp.autocast(enabled=False):
            loss_risk_calib = F.binary_cross_entropy(risk_prob, risk_target.float(), reduction="none")
        loss_risk_calib = (loss_risk_calib * risk_target_mask).sum() / risk_target_mask.sum().clamp_min(1.0)
    else:
        loss_risk_calib = ratings.sum() * 0.0

    total = (
        loss_rec
        + args.alpha_regret * loss_regret
        + (args.alpha_low_regret * loss_low if args.use_low_regret_loss else 0.0)
        + args.alpha_utility * loss_utility
        + (args.alpha_conflict * loss_conflict if args.use_conflict_loss else 0.0)
        + args.alpha_risk_calib * loss_risk_calib
        + (args.alpha_severity * loss_severity if args.use_severity else 0.0)
    )
    avg_weighted_regret_risk = outputs["weighted_quad_risk"].detach().sum(dim=1).mean()
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_regret": float(loss_regret.detach().cpu()),
        "loss_low_regret": float(loss_low.detach().cpu()),
        "loss_utility": float(loss_utility.detach().cpu()),
        "loss_conflict": float(loss_conflict.detach().cpu()),
        "loss_risk_calib": float(loss_risk_calib.detach().cpu()),
        "loss_severity": float(loss_severity.detach().cpu()),
        "avg_predicted_severity": float(avg_predicted_severity.detach().cpu()),
        "avg_label_severity": float(avg_label_severity.detach().cpu()),
        "avg_weighted_regret_risk": float(avg_weighted_regret_risk.detach().cpu()),
    }

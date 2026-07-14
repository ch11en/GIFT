#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TAC-based personalized hard negative sampler for GIFT."""

import math
import random

from gift.samplers.gift_tac_diagnosis import GIFTTargetAwareCalibration


def sigmoid(value):
    value = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _softmax_sample(rows, rng):
    if not rows:
        return None
    max_logit = max(row["sampling_logit"] for row in rows)
    weights = [math.exp(row["sampling_logit"] - max_logit) for row in rows]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(rows)
    target = rng.random() * total
    acc = 0.0
    for row, weight in zip(rows, weights):
        acc += weight
        if acc >= target:
            return row
    return rows[-1]


class GIFTTargetAwareSampler:
    """Sample negatives with TAC hard-negative preference and false-negative suppression."""

    def __init__(
        self,
        user_profiles,
        item_profiles,
        user_interacted_items=None,
        tau_hard=2.0,
        tau_false=1.0,
        risk_threshold=0.5,
        pos_threshold=0.5,
        false_negative_weight=0.1,
        ambiguous_weight=0.5,
        use_false_negative_suppression=True,
        use_severity=True,
        use_user_sensitivity=True,
        rng=None,
    ):
        self.diagnosis = GIFTTargetAwareCalibration(
            user_profiles,
            item_profiles,
            risk_threshold=risk_threshold,
            pos_threshold=pos_threshold,
            use_severity=use_severity,
            use_user_sensitivity=use_user_sensitivity,
        )
        self.user_interacted_items = user_interacted_items or {}
        self.tau_hard = float(tau_hard)
        self.tau_false = float(tau_false)
        self.false_negative_weight = float(false_negative_weight)
        self.ambiguous_weight = float(ambiguous_weight)
        self.use_false_negative_suppression = bool(use_false_negative_suppression)
        self.rng = rng or random.Random()

    def _known_items(self, user_id):
        known = set()
        for key in (user_id, str(user_id)):
            known.update(self.user_interacted_items.get(key, set()) or set())
        try:
            known.update(self.user_interacted_items.get(int(user_id), set()) or set())
        except (TypeError, ValueError):
            pass
        return {int(x) for x in known}

    def score_candidates(self, user_id, candidate_negative_items, base_model_scores=None):
        base_model_scores = base_model_scores or {}
        known = self._known_items(user_id)
        rows = []
        for item_id in candidate_negative_items:
            item_id = int(item_id)
            if item_id in known:
                continue
            base_score = base_model_scores.get(item_id, base_model_scores.get(str(item_id)))
            diag = self.diagnosis.diagnose_sample(user_id, item_id, base_score=base_score)
            false_score = max(diag["false_negative_score"], 0.0)
            false_penalty = self.tau_false * false_score if self.use_false_negative_suppression else 0.0
            sampling_logit = self.tau_hard * diag["hard_negative_score"] - false_penalty
            sample_weight = sigmoid(diag["negative_risk"] - diag["positive_match"])
            if self.use_false_negative_suppression and diag["sample_type"] == "possible_false_negative":
                sample_weight *= self.false_negative_weight
            if diag["sample_type"] == "ambiguous":
                sample_weight *= self.ambiguous_weight
            rows.append({
                "item_id": item_id,
                "sampling_logit": float(sampling_logit),
                "sample_weight": float(max(0.0, min(1.0, sample_weight))),
                **diag,
            })
        return rows

    def sample(self, user_id, positive_item_id, candidate_negative_items, base_model_scores=None):
        rows = self.score_candidates(user_id, candidate_negative_items, base_model_scores=base_model_scores)
        selected = _softmax_sample(rows, self.rng)
        if selected is None:
            raise ValueError("GIFTTargetAwareSampler received no valid negative candidates")
        return {
            "user_id": int(user_id),
            "positive_item_id": int(positive_item_id),
            "sampled_negative_item": int(selected["item_id"]),
            "sample_weight": float(selected["sample_weight"]),
            "negative_risk": float(selected["negative_risk"]),
            "positive_match": float(selected["positive_match"]),
            "false_negative_score": float(selected["false_negative_score"]),
            "hard_negative_score": float(selected["hard_negative_score"]),
            "sample_type": selected["sample_type"],
        }


TACNegativeSampler = GIFTTargetAwareSampler

__all__ = ["GIFTTargetAwareSampler", "TACNegativeSampler", "sigmoid"]

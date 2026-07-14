#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Target-Aware Calibration helpers for GIFT.

TAC converts AEC item evidence and RPE user profiles into reliability signals
used by auxiliary learning and score calibration.
"""


SAMPLE_TYPES = {
    "reliable_hard_negative",
    "possible_false_negative",
    "ordinary_negative",
    "ambiguous",
}


def _lookup(table, key):
    if not table:
        return {}
    if key in table:
        return table[key] or {}
    try:
        ikey = int(key)
        if ikey in table:
            return table[ikey] or {}
    except (TypeError, ValueError):
        pass
    skey = str(key)
    return table.get(skey, {}) or {}


def _profile_bucket(profile, names):
    for name in names:
        value = profile.get(name)
        if value is not None:
            return value or {}
    return {}


def _float_dict(values):
    out = {}
    for key, value in (values or {}).items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            out[str(key)] = float(value)
    return out


def _dot(left, right):
    left = _float_dict(left)
    right = _float_dict(right)
    if len(left) > len(right):
        left, right = right, left
    return sum(value * float(right.get(key, 0.0)) for key, value in left.items())


class GIFTTargetAwareCalibration:
    """Compute TAC positive match, negative risk, and sample type."""

    def __init__(
        self,
        user_profiles,
        item_profiles,
        risk_threshold=0.5,
        pos_threshold=0.5,
        use_severity=True,
        use_user_sensitivity=True,
    ):
        self.user_profiles = user_profiles or {}
        self.item_profiles = item_profiles or {}
        self.risk_threshold = float(risk_threshold)
        self.pos_threshold = float(pos_threshold)
        self.use_severity = bool(use_severity)
        self.use_user_sensitivity = bool(use_user_sensitivity)

    def user_profile(self, user_id):
        return _lookup(self.user_profiles, user_id)

    def item_profile(self, item_id):
        return _lookup(self.item_profiles, item_id)

    def compute_positive_match(self, user_id, item_id):
        user = self.user_profile(user_id)
        item = self.item_profile(item_id)
        user_pos = _profile_bucket(user, ["positive", "P_u", "positive_aspects"])
        item_pos = _profile_bucket(item, ["positive", "Pos_i", "positive_aspects"])
        return float(_dot(user_pos, item_pos))

    def compute_negative_risk(self, user_id, item_id):
        user = self.user_profile(user_id)
        item = self.item_profile(item_id)
        user_neg = _profile_bucket(user, ["negative", "N_u", "negative_sensitivity"])
        item_neg = _profile_bucket(item, ["negative", "Neg_i", "negative_aspects"])
        severity = _profile_bucket(item, ["severity", "Sev_i", "severity_aspects"])
        user_neg = _float_dict(user_neg)
        item_neg = _float_dict(item_neg)
        severity = _float_dict(severity)
        values = []
        for aspect, neg_value in item_neg.items():
            sensitivity = float(user_neg.get(aspect, 0.0)) if self.use_user_sensitivity else 1.0
            sev_value = float(severity.get(aspect, 0.5)) if self.use_severity else 1.0
            values.append(sensitivity * float(neg_value) * sev_value)
        return float(max(values) if values else 0.0)

    def compute_false_negative_score(self, user_id, item_id):
        """Positive values indicate a sampled negative may be target-reliable."""
        return self.compute_positive_match(user_id, item_id) - self.compute_negative_risk(user_id, item_id)

    def compute_hard_negative_score(self, user_id, item_id, base_score=None):
        score = self.compute_negative_risk(user_id, item_id)
        if base_score is not None:
            score += float(base_score)
        return float(score)

    def diagnose_sample(self, user_id, item_id, base_score=None):
        """Return TAC reliability signals for training-time sample weighting."""
        positive_match = self.compute_positive_match(user_id, item_id)
        negative_risk = self.compute_negative_risk(user_id, item_id)
        false_negative_score = positive_match - negative_risk
        hard_negative_score = negative_risk + (float(base_score) if base_score is not None else 0.0)
        if negative_risk >= self.risk_threshold and positive_match < self.pos_threshold:
            sample_type = "reliable_hard_negative"
        elif positive_match >= self.pos_threshold and negative_risk < self.risk_threshold:
            sample_type = "possible_false_negative"
        elif positive_match >= self.pos_threshold and negative_risk >= self.risk_threshold:
            sample_type = "ambiguous"
        else:
            sample_type = "ordinary_negative"
        return {
            "tac_positive_match": float(positive_match),
            "tac_negative_risk": float(negative_risk),
            "tac_reliability_signal": float(false_negative_score),
            "positive_match": float(positive_match),
            "negative_risk": float(negative_risk),
            "false_negative_score": float(false_negative_score),
            "hard_negative_score": float(hard_negative_score),
            "sample_type": sample_type,
        }


TACSampleDiagnosis = GIFTTargetAwareCalibration

__all__ = ["GIFTTargetAwareCalibration", "TACSampleDiagnosis", "SAMPLE_TYPES"]

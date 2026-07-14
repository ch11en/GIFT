#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility-Risk Calibration for GIFT final ranking."""

from gift.samplers.gift_tac_diagnosis import GIFTTargetAwareCalibration


class GIFTUtilityRiskCalibrator:
    """Calibrate final ranking scores with utility and regret-risk signals."""

    def __init__(
        self,
        user_profiles,
        item_profiles,
        alpha_desired=0.2,
        beta_risk=0.5,
        risk_threshold=0.5,
        pos_threshold=0.5,
        use_severity=True,
        use_user_sensitivity=True,
    ):
        self.alpha_desired = float(alpha_desired)
        self.beta_risk = float(beta_risk)
        self.diagnosis = GIFTTargetAwareCalibration(
            user_profiles,
            item_profiles,
            risk_threshold=risk_threshold,
            pos_threshold=pos_threshold,
            use_severity=use_severity,
            use_user_sensitivity=use_user_sensitivity,
        )

    def score_item(self, user_id, item_id, base_score):
        desired = self.diagnosis.compute_positive_match(user_id, item_id)
        risk = self.diagnosis.compute_negative_risk(user_id, item_id)
        rerank_score = float(base_score) + self.alpha_desired * desired - self.beta_risk * risk
        return {
            "user_id": int(user_id),
            "item_id": int(item_id),
            "base_score": float(base_score),
            "urc_score": float(rerank_score),
            "aec_positive_match": float(desired),
            "rpe_negative_risk": float(risk),
            "desired_coverage": float(desired),
            "risk_exposure": float(risk),
            "rerank_score": float(rerank_score),
        }

    def calibrate_candidates(self, user_id, candidate_items, base_scores, top_k=10, mode="score_mode", risk_budget=None):
        scored = [
            self.score_item(user_id, item_id, base_scores.get(item_id, base_scores.get(str(item_id), 0.0)))
            for item_id in candidate_items
        ]
        scored.sort(key=lambda row: row["rerank_score"], reverse=True)
        if mode == "score_mode":
            return scored[:top_k]
        if mode != "constraint_mode":
            raise ValueError(f"Unsupported GIFT rerank mode: {mode}")
        budget = float(risk_budget if risk_budget is not None else 1.0)
        kept = [row for row in scored if row["risk_exposure"] <= budget]
        if len(kept) < top_k:
            used = {row["item_id"] for row in kept}
            fillers = [row for row in sorted(scored, key=lambda r: (r["risk_exposure"], -r["rerank_score"])) if row["item_id"] not in used]
            kept.extend(fillers[: top_k - len(kept)])
        return kept[:top_k]

    def rerank(self, user_id, candidate_items, base_scores, top_k=10, mode="score_mode", risk_budget=None):
        """Compatibility wrapper for older reranking scripts."""
        return self.calibrate_candidates(user_id, candidate_items, base_scores, top_k, mode, risk_budget)


__all__ = ["GIFTUtilityRiskCalibrator"]

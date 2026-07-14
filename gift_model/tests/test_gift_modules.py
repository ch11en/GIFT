#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for GIFT internal scoring behavior."""

import math
import os
import random
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gift.rerankers.gift_urc_calibrator import GIFTUtilityRiskCalibrator
from gift.samplers.gift_tac_diagnosis import GIFTTargetAwareCalibration
from gift.samplers.gift_tac_sampler import GIFTTargetAwareSampler


USER_PROFILES = {
    1: {
        "positive": {3: 1.0, 4: 0.5},
        "negative": {5: 0.9, 6: 0.4},
        "tolerance": {6: 0.7},
    }
}

ITEM_PROFILES = {
    10: {
        "positive": {3: 0.1},
        "negative": {5: 1.0},
        "severity": {5: 0.8},
    },
    11: {
        "positive": {3: 0.9, 4: 0.2},
        "negative": {},
        "severity": {},
    },
    12: {
        "positive": {3: 0.9},
        "negative": {5: 1.0},
        "severity": {5: 0.8},
    },
    13: {
        "positive": {},
        "negative": {6: 0.2},
        "severity": {6: 0.3},
    },
}


class GIFTInternalScoringTest(unittest.TestCase):
    def test_sample_diagnosis_uses_quadruple_profiles(self):
        diagnosis = GIFTTargetAwareCalibration(USER_PROFILES, ITEM_PROFILES, risk_threshold=0.5, pos_threshold=0.5)

        self.assertAlmostEqual(diagnosis.compute_positive_match(1, 11), 1.0)
        self.assertAlmostEqual(diagnosis.compute_negative_risk(1, 10), 0.72)
        self.assertEqual(diagnosis.diagnose_sample(1, 10)["sample_type"], "reliable_hard_negative")
        self.assertEqual(diagnosis.diagnose_sample(1, 11)["sample_type"], "possible_false_negative")
        self.assertEqual(diagnosis.diagnose_sample(1, 12)["sample_type"], "ambiguous")
        self.assertEqual(diagnosis.diagnose_sample(999, 999)["sample_type"], "ordinary_negative")

    def test_sampler_excludes_interacted_items_and_downweights_false_negatives(self):
        sampler = GIFTTargetAwareSampler(
            USER_PROFILES,
            ITEM_PROFILES,
            user_interacted_items={1: {10}},
            risk_threshold=0.5,
            pos_threshold=0.5,
            false_negative_weight=0.1,
            ambiguous_weight=0.5,
            rng=random.Random(7),
        )

        candidates = sampler.score_candidates(1, [10, 11, 12, 13])
        self.assertNotIn(10, [row["item_id"] for row in candidates])

        by_item = {row["item_id"]: row for row in candidates}
        false_weight = by_item[11]["sample_weight"]
        ambiguous_weight = by_item[12]["sample_weight"]
        self.assertLess(false_weight, ambiguous_weight)
        self.assertEqual(by_item[11]["sample_type"], "possible_false_negative")

        sampled = sampler.sample(1, 99, [11, 12, 13])
        self.assertIn(sampled["sampled_negative_item"], {11, 12, 13})
        self.assertGreaterEqual(sampled["sample_weight"], 0.0)
        self.assertLessEqual(sampled["sample_weight"], 1.0)

    def test_reranker_score_and_constraint_modes(self):
        reranker = GIFTUtilityRiskCalibrator(USER_PROFILES, ITEM_PROFILES, alpha_desired=0.5, beta_risk=1.0)

        scored = reranker.rerank(1, [10, 11, 12], {10: 1.0, 11: 0.9, 12: 0.8}, top_k=3, mode="score_mode")
        self.assertEqual(scored[0]["item_id"], 11)
        self.assertTrue(all("rerank_score" in row for row in scored))
        self.assertTrue(all("urc_score" in row for row in scored))

        constrained = reranker.rerank(
            1,
            [10, 11, 12],
            {10: 1.0, 11: 0.9, 12: 0.8},
            top_k=2,
            mode="constraint_mode",
            risk_budget=0.1,
        )
        self.assertEqual(constrained[0]["item_id"], 11)
        self.assertTrue(math.isclose(constrained[0]["risk_exposure"], 0.0))
        self.assertTrue(math.isclose(constrained[0]["rpe_negative_risk"], 0.0))


if __name__ == "__main__":
    unittest.main()

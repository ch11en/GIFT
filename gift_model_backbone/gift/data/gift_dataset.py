#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset loader for GIFT profile-based training and evaluation."""

import json
import math
import os
import pickle

import torch
from torch.utils.data import Dataset


def sentiment_to_class(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        text = str(value).lower()
        if text in {"negative", "neg", "-1"}:
            value = -1
        elif text in {"positive", "pos", "1"}:
            value = 1
        else:
            value = 0
    if value < 0:
        return 0
    if value > 0:
        return 2
    return 1


def load_pickle(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lookup(mapping, key, default=None):
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    try:
        key_int = int(key)
        if key_int in mapping:
            return mapping[key_int]
    except (TypeError, ValueError):
        pass
    return mapping.get(str(key), default)


class GIFTDataset(Dataset):
    """Loads GIFT train, validation, or test splits from a processed directory."""

    def __init__(self, data_dir, split="train", max_quadruples=20):
        self.data_dir = str(data_dir)
        self.split = split
        self.max_quadruples = int(max_quadruples)
        self.rows = load_jsonl(os.path.join(self.data_dir, f"{split}.jsonl"))
        self.stats = load_pickle(os.path.join(self.data_dir, "stats.pkl"), {})
        self.mappings = load_pickle(os.path.join(self.data_dir, "mappings.pkl"), {})
        self.user_aspect_risk = load_pickle(os.path.join(self.data_dir, "user_aspect_risk.pkl"), {})
        self.user_aspect_tolerance = load_pickle(os.path.join(self.data_dir, "user_aspect_tolerance.pkl"), {})
        self.user_aspect_loss_aversion = load_pickle(os.path.join(self.data_dir, "user_aspect_loss_aversion.pkl"), {})
        self.user_behavior_stats = load_pickle(os.path.join(self.data_dir, "user_behavior_stats.pkl"), {})
        self.item_behavior_stats = load_pickle(os.path.join(self.data_dir, "item_behavior_stats.pkl"), {})
        self.num_users = self._count("num_users", "user_id", "user_to_id")
        self.num_items = self._count("num_items", "item_id", "item_to_id")
        self.num_aspects = self._aspect_count()

    def _count(self, stat_key, row_key, map_key):
        candidates = [int(self.stats.get(stat_key, 0) or 0)]
        mapping = self.mappings.get(map_key, {}) if isinstance(self.mappings, dict) else {}
        if mapping:
            candidates.append(max(int(v) for v in mapping.values()) + 1)
        if self.rows:
            candidates.append(max(int(row.get(row_key, 0)) for row in self.rows) + 1)
        return max(candidates + [1])

    def _aspect_count(self):
        candidates = [int(self.stats.get("num_aspects", 0) or 0)]
        aspect_map = self.mappings.get("aspect_to_id", {}) if isinstance(self.mappings, dict) else {}
        if aspect_map:
            candidates.append(max(int(v) for v in aspect_map.values()) + 1)
        for row in self.rows[:10000]:
            for quad in self._select_quadruples(row):
                aspect = quad.get("aspect_category_id", quad.get("aspect_id", 0))
                try:
                    candidates.append(int(aspect) + 1)
                except (TypeError, ValueError):
                    pass
        return max(candidates + [2])

    def _select_quadruples(self, row):
        quads = row.get("quadruples") or []
        if not quads:
            quads = row.get("model_quadruples") or []
        return quads[: self.max_quadruples]

    def _behavior_features(self, table, key):
        stats = lookup(table, key, {}) or {}
        count = float(stats.get("train_interaction_count", 0.0) or 0.0)
        rating_sum = float(stats.get("train_rating_sum", 0.0) or 0.0)
        pos = float(stats.get("train_positive_count", 0.0) or 0.0)
        neg = float(stats.get("train_negative_count", 0.0) or 0.0)
        neu = float(stats.get("train_neutral_count", 0.0) or 0.0)
        denom = max(count, 1.0)
        return [math.log1p(count), rating_sum / (5.0 * denom), pos / denom, neg / denom, neu / denom]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        user_id = int(row["user_id"])
        item_id = int(row["item_id"])
        quads = self._select_quadruples(row)
        aspect_ids, sentiment_ids, mask = [], [], []
        regret, low_regret, utility, conflict, severity = [], [], [], [], []
        for quad in quads:
            aspect = int(quad.get("aspect_category_id", quad.get("aspect_id", 0)) or 0)
            sentiment = quad.get("sentiment_id", quad.get("sentiment", 0))
            sentiment_class = sentiment_to_class(sentiment)
            aspect_ids.append(aspect)
            sentiment_ids.append(sentiment_class)
            mask.append(1.0)
            regret.append(float(quad.get("regret_label", 1.0 if sentiment_class == 0 else 0.0) or 0.0))
            low_regret.append(float(quad.get("low_regret_label", 0.0) or 0.0))
            utility.append(float(quad.get("utility_label", 1.0 if sentiment_class == 2 else 0.0) or 0.0))
            conflict.append(float(quad.get("conflict_label", 0.0) or 0.0))
            severity.append(float(quad.get("severity_label", 0.0) or 0.0))
        pad = self.max_quadruples - len(aspect_ids)
        if pad > 0:
            aspect_ids.extend([0] * pad)
            sentiment_ids.extend([1] * pad)
            mask.extend([0.0] * pad)
            regret.extend([0.0] * pad)
            low_regret.extend([0.0] * pad)
            utility.extend([0.0] * pad)
            conflict.extend([0.0] * pad)
            severity.extend([0.0] * pad)
        user_risk = lookup(self.user_aspect_risk, user_id, {}) or {}
        user_tol = lookup(self.user_aspect_tolerance, user_id, {}) or {}
        user_loss = lookup(self.user_aspect_loss_aversion, user_id, {}) or {}
        risk_values = [float(lookup(user_risk, a, 0.0) or 0.0) for a in aspect_ids]
        tol_values = [float(lookup(user_tol, a, 0.0) or 0.0) for a in aspect_ids]
        loss_values = [float(lookup(user_loss, a, 1.0) or 1.0) for a in aspect_ids]
        return {
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "item_ids": torch.tensor(item_id, dtype=torch.long),
            "ratings": torch.tensor(float(row.get("rating", 0.0) or 0.0), dtype=torch.float32),
            "aspect_category_ids": torch.tensor(aspect_ids, dtype=torch.long),
            "sentiment_ids": torch.tensor(sentiment_ids, dtype=torch.long),
            "quad_mask": torch.tensor(mask, dtype=torch.float32),
            "regret_labels": torch.tensor(regret, dtype=torch.float32),
            "low_regret_labels": torch.tensor(low_regret, dtype=torch.float32),
            "utility_labels": torch.tensor(utility, dtype=torch.float32),
            "conflict_labels": torch.tensor(conflict, dtype=torch.float32),
            "severity_labels": torch.tensor(severity, dtype=torch.float32),
            "user_aspect_risk_values": torch.tensor(risk_values, dtype=torch.float32),
            "user_aspect_tolerance_values": torch.tensor(tol_values, dtype=torch.float32),
            "user_aspect_loss_aversion_values": torch.tensor(loss_values, dtype=torch.float32),
            "user_behavior_features": torch.tensor(self._behavior_features(self.user_behavior_stats, user_id), dtype=torch.float32),
            "item_behavior_features": torch.tensor(self._behavior_features(self.item_behavior_stats, item_id), dtype=torch.float32),
        }

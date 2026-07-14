# GIFT

Guided Interest and Fault Tolerance Recommendation

GIFT is a research codebase for reliable sequential recommendation. It models user interest together with fault tolerance, so the final ranking can prefer useful items while reducing exposure to item aspects that a user is likely to reject.

This branch adds a compact model backbone release under `gift_model_backbone/`. The package keeps the source files needed to read processed GIFT data, run the main model, compute Target-Aware Calibration, and apply Utility-Risk Calibration. It does not include datasets, checkpoints, logs, generated caches, or paper build artifacts.

## Method Overview

GIFT uses review-grounded aspect evidence from the training split only.

1. Aspect Evidence Construction builds item-side positive and negative aspect evidence.
2. Regret Profile Estimation builds user-side sensitivity, tolerance, and loss-aversion profiles.
3. Target-Aware Calibration adjusts training-time sample reliability and negative sampling.
4. Utility-Risk Calibration combines utility and regret-risk signals for final ranking.

The model name should be reported as `GIFT`. The modules above are internal components of the full model, not separate plug-in models.

## Compact Backbone Layout

```text
gift_model_backbone/
  README.md
  pyproject.toml
  requirements.txt
  gift/
    data/gift_dataset.py
    model/gift_model.py
    samplers/gift_tac_diagnosis.py
    samplers/gift_tac_sampler.py
    rerankers/gift_urc_calibrator.py
  tests/test_gift_modules.py
```

## Core Files

`gift_model_backbone/gift/model/gift_model.py` defines `GIFTModel`, the utility backbones, and `gift_loss`.

`gift_model_backbone/gift/data/gift_dataset.py` loads processed JSONL examples and train-only profile files.

`gift_model_backbone/gift/samplers/` contains TAC diagnosis and target-aware negative sampling.

`gift_model_backbone/gift/rerankers/gift_urc_calibrator.py` contains URC scoring for candidate reranking.

## Quick Start

```bash
cd gift_model_backbone
pip install -r requirements.txt
python -m pytest tests
```

Minimal model construction.

```python
from gift.model import GIFTModel

model = GIFTModel(
    num_users=1000,
    num_items=5000,
    num_aspects=128,
    embedding_dim=128,
    backbone="behavior_mlp",
    lambda_risk=0.5,
)
```

## Expected Batch Fields

`GIFTModel.forward` expects a batch dictionary with these fields.

```text
user_ids
item_ids
ratings
aspect_category_ids
sentiment_ids
quad_mask
regret_labels
low_regret_labels
utility_labels
conflict_labels
```

Optional fields include `severity_labels`, `user_aspect_risk_values`, `user_aspect_tolerance_values`, `user_aspect_loss_aversion_values`, `user_behavior_features`, and `item_behavior_features`.

## Evaluation Protocol

For paper results, use the same data split and full-ranking evaluation protocol across all models.

```text
Datasets   Musical Instruments, Industrial and Scientific, Yelp
Protocol   chronological leave-one-out full-ranking
Metrics    HR@5, NDCG@5, HR@10, NDCG@10, MRR
```

Do not mix full-ranking results with 99-negative sampled evaluation results in the same table.

## Repository Notes

Large datasets, preprocessing outputs, checkpoints, external baseline repositories, and experiment logs should stay outside Git commits.

The compact backbone is intended for code reading, review, and lightweight testing. Full experiment reproduction should use the project scripts and run directories prepared on the experiment server.

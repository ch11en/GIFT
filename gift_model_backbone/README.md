# GIFT Model Backbone

This repository contains the compact GIFT model backbone used in the paper experiments. It keeps only the core model code and lightweight helpers needed to read processed GIFT data, compute target-aware calibration, and apply utility-risk calibration.

## Contents

- `gift/model/gift_model.py` contains the main `GIFTModel`, utility backbones, and training loss.
- `gift/data/gift_dataset.py` loads processed train, validation, or test splits.
- `gift/samplers/` contains target-aware calibration and negative sampling helpers.
- `gift/rerankers/` contains utility-risk calibration for final ranking.
- `tests/test_gift_modules.py` checks TAC and URC scoring behavior.

## Main Model

`GIFTModel` combines a utility backbone with review-grounded regret-risk estimation. The utility side supports `behavior_mlp`, `neumf`, and `dcnv2`. The risk side encodes aspect-sentiment quadruples and computes regret-risk signals that can calibrate the final score.

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

`GIFTModel.forward` expects a batch dictionary with these core fields.

- `user_ids`
- `item_ids`
- `ratings`
- `aspect_category_ids`
- `sentiment_ids`
- `quad_mask`
- `regret_labels`
- `low_regret_labels`
- `utility_labels`
- `conflict_labels`

Optional fields include `severity_labels`, `user_aspect_risk_values`, `user_aspect_loss_aversion_values`, `user_behavior_features`, and `item_behavior_features`.

## Quick Check

```bash
pip install -r requirements.txt
python -m pytest tests
```

## Notes

This is a compact code release for reading and reviewing the GIFT model backbone. It does not include datasets, checkpoints, large experiment artifacts, or paper result logs.

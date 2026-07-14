# GIFT

GIFT is a reliable sequential recommendation model. The model uses review-grounded aspect evidence to learn user interest and fault tolerance together, so recommendation scores can reflect both utility and regret risk.

This repository currently provides a compact GIFT model implementation under `gift_model/`. The release is for code reading, lightweight testing, and future experiment integration. It does not include datasets, checkpoints, logs, or generated experiment artifacts.

## Work Summary

GIFT contains four main parts. Aspect Evidence Construction builds item-side aspect evidence from reviews. Regret Profile Estimation builds user-side risk and tolerance profiles from training data. Target-Aware Calibration adjusts training sample reliability. Utility-Risk Calibration combines utility and risk signals for final ranking.

## Environment

Use Python 3.9 or newer. The minimal dependencies are PyTorch and pytest.

```bash
cd gift_model
pip install -r requirements.txt
```

## Run

Run the included tests.

```bash
cd gift_model
python -m pytest tests
```

Create a GIFT model in Python.

```python
from gift.model import GIFTModel

model = GIFTModel(
    num_users=1000,
    num_items=5000,
    num_aspects=128,
    embedding_dim=128,
    utility_arch="behavior_mlp",
    lambda_risk=0.5,
)
```

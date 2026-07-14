# GIFT Model

This folder contains the compact GIFT model implementation. GIFT is a reliable sequential recommendation model that uses review-grounded aspect evidence to model user interest and fault tolerance together.

## Work Summary

The code includes the main `GIFTModel`, the processed-data loader, Target-Aware Calibration, Utility-Risk Calibration, and lightweight tests. It is intended for reading the core model code and checking the main scoring logic without datasets, checkpoints, logs, or experiment artifacts.

## Environment

Use Python 3.9 or newer. Install the minimal dependencies below.

```bash
pip install -r requirements.txt
```

The dependency file currently contains PyTorch and pytest.

## Run

Run the tests from this folder.

```bash
python -m pytest tests
```

Create a model instance.

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

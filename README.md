# GIFT

Guided Interest and Fault Tolerance Recommendation

GIFT is a research codebase for sequential recommendation with aspect level preference and risk signals. The project studies how a recommender can match user interests while reducing exposure to item defects that a user may not tolerate.

The repository contains the core GIFT modules, LARR related compatibility code, RecBole based evaluation scripts, TokenWeighted full sort experiments, candidate reranking experiments, ablation scripts, and paper support files.

## What GIFT Does

GIFT builds two kinds of train split evidence from aspect sentiment information.

1. Guided interest

   User positive aspect preference is matched with item positive aspect evidence.

2. Fault tolerance

   User negative sensitivity and tolerance are matched with item negative aspects and severity evidence.

The resulting risk signal can be used in sampling, training, reranking, and fixed checkpoint evaluation.

## Repository Layout

```text
preprocess/                  Builds GIFT and regret risk profiles
samplers/                    Negative sampling and sample diagnosis modules
rerankers/                   Aspect aware reranking modules
model/                       LARR and compatible model code
scripts/                     Training, evaluation, conversion, and collection scripts
scripts/tokenweighted_protocol/
                              TokenWeighted full sort and TIGER protocol scripts
remote_scripts/              Remote run scripts mirrored from the server workflow
tests/                       Unit and smoke tests for the experiment runners
analysis/                    Metric validation and experiment notes
paper_gift_aaai/             Paper source files
README_GIFT.md               GIFT method note
README_LARR.md               LARR method note
README_QCDR.md               QCDR compatibility note
```

## Main Evaluation Standards

Use the following names when reporting results.

### TokenWeighted FullSort

This is the main fair comparison setting for SASRec, GRU4Rec, Caser, BERT4Rec, GIFTSASRec, and TIGER.

```text
Data       tw_musical, tw_industrial, tw_yelp
Eval       full sort RecBole style evaluation
Metrics    HR@5, NDCG@5, HR@10, NDCG@10, MRR
```

The main scripts are in `scripts/tokenweighted_protocol/`.

```bash
python scripts/tokenweighted_protocol/run_recbole_tokenweighted_protocol.py \
  --recbole_path gift_baselines/repos/RecBole \
  --shim_path gift_baselines/shims \
  --data_root runs/tokenweighted_protocol_formal_20260607/recbole_data \
  --output_root runs/tokenweighted_protocol_formal_20260607/recbole_sasrec_full \
  --models SASRec \
  --datasets tw_musical tw_industrial tw_yelp
```

GIFTSASRec under the same full sort setting is handled by the mirrored script below.

```bash
python remote_scripts/run_giftsasrec_tokenweighted_fullsort.py \
  --recbole_path gift_baselines/repos/RecBole \
  --shim_path gift_baselines/shims \
  --data_root runs/tokenweighted_protocol_formal_20260607/recbole_data \
  --output_root runs/giftsasrec_tokenweighted_fullsort_20260608 \
  --sasrec_checkpoint_root runs/tokenweighted_protocol_formal_20260607/recbole_sasrec_full/checkpoints \
  --datasets tw_musical tw_industrial tw_yelp
```

### Candidate Rerank

This is a separate reranking setting. It should not be mixed with TokenWeighted FullSort in the same main result table.

```text
Data       musical, industrial, yelp
Eval       sampled candidate reranking
Metrics    HR@K, NDCG@K, MRR@K
```

The related scripts include `scripts/run_recbole_gift_sasrec_rerank.py`, `remote_scripts/run_gift_sasrec_ltr_rerank.py`, and `scripts/tokenweighted_protocol/collect_gift_ltr_ablation_results.py`.

## Basic GIFT Workflow

Build train only aspect profiles.

```bash
python preprocess/build_gift_profiles.py \
  --input processed/musical_larr_severity \
  --output_dir processed/musical_gift
```

Train GIFT with the compatibility entry point.

```bash
python train_gift.py \
  --data_dir processed/musical_gift \
  --sampler gift \
  --base_margin 0.1 \
  --lambda_margin 0.5
```

Evaluate top k ranking metrics.

```bash
python evaluate_gift.py \
  --checkpoint checkpoints/gift_musical/gift_best.pt \
  --data_dir processed/musical_gift \
  --results_dir results/gift_musical \
  --ks 10 20
```

## Reproduction Notes

The repository is organized for research experiments. Large datasets, generated RecBole caches, checkpoints, external baseline repositories, and result archives are not meant to be committed.

Before pushing to GitHub, check that local folders such as `tmp/`, `outputs/`, `results/`, external baseline archives, and private server logs are excluded from the commit.

The active formal comparison uses the TokenWeighted FullSort protocol. Candidate rerank results are useful for analysis and ablation, but they should be reported as a separate protocol.

## Tests

Run the focused tests with pytest.

```bash
pytest tests
```

The tests cover runner planning, fixed checkpoint evaluation behavior, and selected compatibility modules.

## Project Status

This repository is under active research development. Scripts under `remote_scripts/` and `scripts/tokenweighted_protocol/` preserve the experiment workflow used for the current paper comparison.


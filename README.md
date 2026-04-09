# AI-for-Toxicology (Course Project Submission)

Toxicity prediction using latent molecular representations learned from SELFIES with a VAE-based model and a descriptor-based XGBoost baseline.

## Team
- Mohammad Taha
- Thomas Pruyn
- Erin Wong

## Submission Contents

### Notebooks
- `Latent Analysis & Ablation Study_updated.ipynb`
- `PreTrained_VAE_Optimization_Analysis.ipynb`
- `Pretrained_VAE_EndtoEnd_attempt_2.ipynb`
- `Pretrained_VAE_EndtoEnd_attempt_2_latentanalysis.ipynb`
- `final_results.ipynb`
- `XGBoost_Prediction.ipynb`

### Model checkpoints
- `artifacts/end_to_end_checkpoints/e2evae_full_seqconv_ce_phase1_best.pt`
- `artifacts/end_to_end_checkpoints/e2evae_full_seqconv_ce_phase2_adaptive_best.pt`

### Python scripts (minimal modular implementation)
- `src/ai_for_toxicology/model.py`: VAE + predictor architecture
- `src/ai_for_toxicology/data.py`: data loading and SELFIES/token alignment
- `src/ai_for_toxicology/train.py`: training loop (reconstruction + KL + masked BCE)
- `src/ai_for_toxicology/test.py`: evaluation metrics and export helpers
- `scripts/train.py`: CLI entrypoint for model training
- `scripts/test.py`: CLI entrypoint for checkpoint evaluation

## Environment Setup

This repo uses `uv`.

```bash
uv venv
uv sync
```

## Running the Notebooks

Launch Jupyter and open any notebook in the submission set:

```bash
uv run jupyter lab
```

Notebook outputs are intentionally preserved to show training traces and results.

## Running the Python Pipeline

### Train

```bash
uv run python scripts/train.py \
  --chembl-csv data/Train/chembl_clean.csv \
  --zinc-csv data/Train/zinc250k_clean.csv \
  --tox21-train-csv data/Train/tox21_train_clean.csv \
  --tox21-val-csv data/Val/tox21_val_clean.csv \
  --tox21-test-csv data/Test/tox21_test_clean.csv \
  --checkpoint-dir artifacts/end_to_end_checkpoints \
  --checkpoint-stem e2evae_full_seqconv_ce \
  --phase1-epochs 90 \
  --warmup-epochs 15 \
  --phase2-epochs 80 \
  --batch-size 128 \
  --metrics-out reports/phase_training_test_metrics.json
```

Training follows the same staged process as `Pretrained_VAE_EndtoEnd_attempt_2.ipynb`:
1. Phase 1 pretraining (reconstruction/KL objective on ChemBL + ZINC).
2. Phase 2 Stage A warmup (prediction head only, base frozen).
3. Phase 2 Stage B adaptive fine-tuning (full unfreeze with differential learning rates).

### Evaluate

```bash
uv run python scripts/test.py \
  --checkpoint artifacts/end_to_end_checkpoints/e2evae_full_seqconv_ce_phase2_adaptive_best.pt \
  --split test \
  --test-csv data/Test/tox21_test_clean.csv \
  --metrics-out reports/submission_metrics.csv \
  --predictions-out reports/submission_predictions.csv
```

## Expected Outputs
- Phase checkpoints from `scripts/train.py` (e.g., `_phase1_best.pt`, `_phase2_warmup_best.pt`, `_phase2_adaptive_best.pt`).
- Metrics CSV and prediction CSV from `scripts/test.py`.
- Notebook outputs include training curves/tables used in the report.

## AI Disclosure

AI tools were used as coding assistance for formatting, refactoring support, and documentation drafting. All model design, experiment choices, and final validation decisions were reviewed and controlled by the project team.

# AI-for-Toxicology (Course Project Submission)

Toxicity prediction using latent molecular representations learned from SELFIES with a VAE-based model and a descriptor-based XGBoost baseline.

## Team
- Mohammad Taha
- Thomas Pruyn
- Erin Wong

## Submission Contents

### Notebooks
- `Ablation_Study.ipynb`
- `Hyperparameter_and_Model_Optimization.ipynb`
- `Final_Model_Training.ipynb`
- `Latent_Analysis_and_Generative_Capability.ipynb`
- `Cross_Validation_Results.ipynb`
- `XGBoost_Prediction.ipynb`

### Model checkpoints
- `artifacts/end_to_end_checkpoints/Phase1_pretrained_model.pt`
- `artifacts/end_to_end_checkpoints/Phase2_posttrained_model.pt`

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

Training follows the same staged process as `Final_Model_Training.ipynb`:
1. Phase 1 pretraining (reconstruction/KL objective on ChemBL + ZINC).
2. Phase 2 Stage A warmup (prediction head only, base frozen).
3. Phase 2 Stage B adaptive fine-tuning (full unfreeze with differential learning rates).

### Evaluate

```bash
uv run python scripts/test.py \
  --checkpoint artifacts/end_to_end_checkpoints/Phase2_posttrained_model.pt \
  --split test \
  --test-csv data/Test/tox21_test_clean.csv \
  --metrics-out reports/submission_metrics.csv \
  --predictions-out reports/submission_predictions.csv
```

## Expected Outputs
- Included submission checkpoints: `artifacts/end_to_end_checkpoints/Phase1_pretrained_model.pt` and `artifacts/end_to_end_checkpoints/Phase2_posttrained_model.pt`.
- If you rerun `scripts/train.py`, it writes phase-suffixed checkpoints (e.g., `_phase1_best.pt`, `_phase2_warmup_best.pt`, `_phase2_adaptive_best.pt`).
- Metrics CSV and prediction CSV from `scripts/test.py`.
- Notebook outputs include training curves/tables used in the report.

## AI Disclosure

AI tools were used as coding assistance for formatting, refactoring support, and documentation drafting. All initial code implementations, model design, experiment choices, and final validation decisions were human-made by the project team.

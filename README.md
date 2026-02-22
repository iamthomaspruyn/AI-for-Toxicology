## AI-for-Toxicology

Toxicity Prediction using Latent Molecular Representation.

- **Team**: AI for Toxicology (AIT) — Mohammad Taha, Thomas Pruyn, Erin Wong
- **Target venue (intended)**: ICLR workshop

### Project overview

Chemical safety evaluation is difficult to scale: tens of thousands of chemicals exist, exposures are often to mixtures, and traditional animal testing raises ethical and practical constraints. This project explores **data-driven toxicity screening** using the **Tox21** in‑vitro assay data as a human-relevant, scalable alternative.

Our core idea is to learn a **latent representation of molecules** from SMILES strings using a **variational autoencoder (VAE)**, then use that latent representation for **multi-label toxicity prediction** across 12 Tox21 assay endpoints.

### Goals / conditions of success

- **Latent space quality**: compounds with similar toxicity labels cluster in the learned latent space (organized representation).
- **Predictive performance**: classifiers trained on VAE latents achieve **AUC-ROC comparable** to strong baselines using conventional featurization and to prior work.

### Data

- **Source**: Tox21 (Toxicology in the 21st Century)
- **Benchmark**: DeepChem Tox21 benchmark (MoleculeNet)
- **Inputs**: SMILES strings (convertible to molecular graphs, descriptors, fingerprints, or treated as text)
- **Outputs**: 12 binary labels (active/inactive) for selected nuclear receptor and stress-response pathways (multi-label classification)
- **Note**: label distribution is imbalanced

### Modeling plan (work in progress)

- **Representation learning**: SMILES-to-SMILES VAE with an encoder/decoder suitable for sequence data (e.g., RNN-based).
- **Downstream prediction**: train a multi-label classifier (e.g., random forest or neural network) on the VAE latent vectors.
- **Training objectives**:
  - reconstruction loss (character-level cross-entropy over SMILES)
  - KL divergence (VAE regularization)
  - multi-label binary cross-entropy (classification)

### Baselines

- Random forest trained on conventional SMILES featurizations, e.g.:
  - RDKit molecular descriptors
  - Morgan fingerprints

### Evaluation plan

- **Metrics**: AUC-ROC, F1-score, accuracy; potentially Hamming loss and exact match ratio (EMR) for multi-label settings.
- **Splits**: random split and stratified split (to study effects of imbalance).

### Stretch goals

- Latent space exploration for molecule discovery targeting specific pathways
- Uncertainty estimation
- Contrastive learning
- Out-of-distribution (OOD) studies

### Repo status

This repository is currently **under active development**; the proposal describes the intended methods and evaluation plan. Code, experiments, and results will be added iteratively.

### Setup (uv)

This repo uses `uv` for Python and dependency management.

```bash
uv venv
uv sync
```

Add packages when you need them (examples):

```bash
uv add numpy pandas scikit-learn matplotlib seaborn tqdm
uv add jupyterlab ipykernel
```

Run commands inside the environment:

```bash
uv run python -c "print('hello')"
uv run jupyter lab
```

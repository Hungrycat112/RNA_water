# RNA_water

# Mixture of Experts for Riboswitch Prediction

This repository contains implementations of machine learning models for predicting engineered riboswitch behavior from sequence-derived, thermodynamic, structural, and experimental features.

## Implemented Models

- **Decision Tree** – baseline classification model.
- **Dense Neural Network** – modified from Joel's original implementation with architectural and training pipeline improvements.
- **Mixture of Experts (MoE)** – original heterogeneous MoE architecture based on Audrey's implementation.
- **Gaussian Naive Bayes** — probabilistic baseline classifier.
- **Random Forest** — ensemble tree-based classifier.

## Overview

The models were evaluated using:

- Stratified 80/20 train-test split
- Stratified 5-fold cross-validation on the training set
- Independent held-out test set
- Fixed random seed (42) for reproducibility

Performance was assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Ablation Analysis

Ablation experiments were performed to determine how numerical and sequence-derived features contribute to model performance.

Three complementary analyses were conducted:

- **Single-feature ablation** — each numerical feature was individually set to zero while all other inputs were retained, allowing the effect of individual features on classification performance to be measured.

- **Cumulative numerical-feature ablation** — numerical features were progressively removed according to their Decision Tree-derived sensitivity ranking. Performance was evaluated as increasingly many numerical features were removed.

- **Input-source ablation** — models were compared under three input conditions:
  - **Full input:** numerical features + nucleotide sequence
  - **Sequence only:** all numerical features removed while retaining the original sequence
  - **Numerical features only:** numerical features retained while the sequence was replaced with the same constant dummy sequence for every sample

An additional all-inputs-zero control was evaluated for the MoE model to verify behavior in the absence of informative input.

Ablation experiments were performed across the MoE, Dense Neural Network, Random Forest, Decision Tree, and Gaussian Naive Bayes models.

---

## Repository Structure

```
RNA_water/
├── notebooks/
│   └── ablation.ipynb
│   └── Bhoomika_kmers_2026 (1).ipynb
│   └── Dense_model.ipynb
│   └── Decision_tree.ipynb
│   └── Joel_OriginalDenseModelForMOE.ipynb
│   └── riboswitch_model.py
│   └── moe.py
├── requirements.txt
└── README.md
```

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset (`balanced_dataset.csv`) is **not included** in this repository.

Place the dataset in your working directory and upload it when prompted by the notebook.

Expected input:

```
balanced_dataset.csv
```

---

## Running the Notebooks

1. Open the desired notebook in Google Colab.
2. Upload `balanced_dataset.csv` when prompted.
3. Run all cells.

The notebook automatically:
- preprocesses the data
- performs stratified 5-fold cross-validation
- retrains the final model on the full training set
- evaluates on the held-out test set
- saves evaluation figures and metrics
---

## Training Configuration

| Parameter | Value |
|-----------|------:|
| Random seed | 42 |
| Epochs | 100 |
| L1 regularization | 0.01 |
| L2 regularization | 5e-4 |
| Cross-validation | 5-fold Stratified |
| Test split | 20% |

---

## Outputs

The notebook automatically saves:

- Training and validation curves
- Cross-validation metrics
- Independent test metrics
- Confusion matrix
- Model predictions

---

## Code Availability

The original research and model-development code for this project is available in this repository:

**RNA_water**  
https://github.com/Hungrycat112/RNA_water

This repository contains the collaborative experimental code used for model development, including baseline models, dense neural network experiments, k-mer analyses, and the original Mixture of Experts (MoE) implementation.

A standardized version of the MoE pipeline, including preprocessing, training and inference scripts, pretrained model artifacts, and a reproducible Docker environment, is available in the companion repository:

**Riboswitch-MoE**  
https://github.com/Hungrycat112/riboswitch-moe

The corresponding pre-built Docker for reproducible inference is available at:

**Docker Hub**  
https://hub.docker.com/r/hungrycat112/riboswitch-moe/tags

## Citation

If you use this repository in your research, please cite the associated manuscript once available.

---

## License

MIT License

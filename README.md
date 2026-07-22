# RNA_water

# Mixture of Experts for Riboswitch Prediction

This repository contains the implementation of a Mixture of Experts (MoE) neural network for predicting engineered riboswitch behavior from sequence-derived, thermodynamic, structural, and experimental features.

## Overview

The model was evaluated using:

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

---

## Repository Structure

```
RNA_water/
├── notebooks/
│   └── Dense_model.ipynb
│   └── Decision_tree.ipynb
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

## Running the Notebook

1. Open `.ipynb` in Google Colab.
2. Run all cells.
3. Upload `balanced_dataset.csv` when prompted.
4. The notebook will:
   - preprocess the data
   - perform stratified 5-fold cross-validation
   - retrain the model on the full training set
   - evaluate on an independent test set
   - generate evaluation plots
   - save all outputs automatically

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

## Citation

If you use this repository in your research, please cite the associated manuscript once available.

---

## License

MIT License

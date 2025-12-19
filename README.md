# Latent Sculpting for Zero-Shot Generalization

### A Manifold Learning Approach to Out-of-Distribution Anomaly Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rajeeb321123/Latent_sculpting_using_two_stage_method/blob/main/DHS_transformer_encoder_final.ipynb) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Paper Status](https://img.shields.io/badge/Status-Preprint-blue.svg)](https://arxiv.org/)

This repository contains the official PyTorch implementation of the research paper: **"Latent Sculpting for Zero-Shot Generalization: A Manifold Learning Approach to Out-of-Distribution Anomaly Detection"**.

## 🚀 Overview

Standard deep learning classifiers often suffer from **"Generalization Collapse"** when facing zero-shot Out-of-Distribution (OOD) anomalies. This project introduces a **Hierarchical Two-Stage Framework** to address this limitation:

1. **Stage 1 (Latent Sculpting):** A hybrid **1D-CNN + Transformer Encoder** trained with a novel **Dual-Centroid Compactness Loss (DCCL)**. This stage actively "sculpts" benign traffic into a low-entropy, hyperspherical cluster while pushing known anomalies away.

2. **Stage 2 (Probabilistic Expert Review):** A **Masked Autoregressive Flow (MAF)** trained exclusively on the structured benign manifold to learn an exact density estimate.

### 📊 Key Results (Sensitivity Analysis)

The table below demonstrates the trade-off between **Internal Accuracy** (on known traffic) and **Generalization** (on zero-shot attacks) across different sensitivity thresholds ($P_{99}, P_{97}, P_{95}$).

| Model / Threshold | Known Attacks (F1) | Zero-Shot / Unseen Attacks (F1) | Infiltration Detection |
| :--- | :---: | :---: | :---: |
| Supervised MLP Baseline | **0.98** | 0.30 | 0.00% |
| Unsupervised OCSVM | 0.76 | 0.76 | 85.71% |
| **Ours ($P_{99}$ - Strict)** | 0.96 | 0.67 | 69.44% |
| **Ours ($P_{97}$ - Balanced)** | 0.95 | 0.74 | 86.11% |
| **Ours ($P_{95}$ - Sensitive)** | 0.94 | **0.87** | **88.89%** |

> **Note:** Our model achieves state-of-the-art zero-shot detection ($P_{95}$) while maintaining competitive internal accuracy.

## 🛠️ Prerequisites & Setup (Google Colab)

This project is optimized for **Google Colab** using a High-RAM runtime.

### 1. Dependencies

The code relies on standard PyTorch and Data Science libraries.

```bash
pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn tqdm umap-learn joblib pyarrow

```

### 2. Dataset Preparation

This project uses the **CIC-IDS-2017** dataset. Due to size constraints, the CSV files are not included in this repo.

1. Download the machine learning CSV files from the [UNB Dataset Page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Upload the following files to your Colab environment:
* `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
* `Friday-WorkingHours-Morning.pcap_ISCX.csv`
* `Monday-WorkingHours.pcap_ISCX.csv`
* `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
* `Tuesday-WorkingHours.pcap_ISCX.csv`
* `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
* `Wednesday-workingHours.pcap_ISCX.csv`
* `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`



## 🏃 Usage

The pipeline is split into two sequential scripts to manage memory efficiently.

### Step 1: Preprocessing (`preprocess.py`)

Loads raw CSVs, cleans data (Infinity/NaN removal), performs feature engineering (Bytes/Packet, Packets/Sec), and splits data into **Seen** (Training) and **Unseen** (Zero-Shot) sets.

* **Output:** Generates `seen_data.feather`, `unseen_data.feather`, and `scaler_and_cols.joblib`.

```python
# Run in Colab cell
!python preprocess.py

```

### Step 2: Training & Evaluation (`train_and_evaluate.py`)

Loads the processed artifacts, trains the Hybrid Encoder (Stage 1), trains the MAF (Stage 2), and performs the final evaluation.

* **Stage 1:** Asymmetric balancing is applied (Benign samples undersampled to match largest Anomaly class).
* **Stage 2:** Calculates dynamic thresholds at 99th, 97th, and 95th percentiles.
* **Evaluation:** Generates AUROC, AUPRC, and per-attack recall tables.

```python
# Run in Colab cell
!python train_and_evaluate.py

```

## 🧠 Model Architecture

### Stage 1: Hybrid Encoder

* **Input:** 71 features (Zero-Variance filtered).
* **CNN Front-End:** 5-layer 1D-CNN for local feature extraction ().
* **Transformer Back-End:** 3-layer Transformer Encoder for global context ( heads).
* **Loss:** DCCL (Compactness + Separation).

### Stage 2: Density Estimator

* **Model:** Masked Autoregressive Flow (MAF).
* **Depth:** 16 MADE layers.
* **Hidden Dimension:** 512.

## 📄 Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@misc{chhetri2025latent,
  title={Latent Sculpting for Zero-Shot Generalization: A Manifold Learning Approach to Out-of-Distribution Anomaly Detection},
  author={Chhetri, Rajeeb Thapa and Thapa, Saurab and Chen, Zhixiong},
  year={2025},
  eprint={Pending},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

```

## 🤝 Acknowledgement

This research was partially funded by the U.S. Department of Homeland Security (DHS).
We acknowledge the use of the CIC-IDS-2017 dataset provided by the Canadian Institute for Cybersecurity.

---

*Maintained by Rajeeb Thapa Chhetri*

```

```

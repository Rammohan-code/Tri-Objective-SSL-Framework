# Tri-Objective SSL Framework

Official implementation of the paper:
**"Tri-Objective Self-Supervised Representation Learning for High-Dimensional Tabular Data" (IEEE PuneCon 2025)**

## 📘 Overview
This repository contains the implementation of the Tri-Objective SSL model that jointly optimizes:
- **Contrastive Loss**
- **Reconstruction Loss**
- **Consistency Regularization**

It supports mixed-type tabular data and handles missing values through implicit denoising.

## ⚙️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt

🚀 Usage

Example run:

python train.py --dataset demo_data/health.csv --epochs 50


To visualize embeddings:
python plot_tsne.py --input outputs/embeddings.npy

📊 Outputs
Trained representations (embeddings)

t-SNE plots
SHAP feature importances

🧩 Citation
If you use this code, please cite:
Prem K., "Tri-Objective Self-Supervised Representation Learning for High-Dimensional Tabular Data," IEEE PuneCon 2025.
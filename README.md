# 🌿 Cardamom Leaf Classifier
> Progressive computer vision pipeline for cardamom leaf detection, health classification, and disease diagnosis using PyTorch.

[![Phase 1](https://img.shields.io/badge/Phase-1%20Complete-blue)](docs/paper.md)
[![Phase 2](https://img.shields.io/badge/Phase-2%20Complete-brightgreen)](docs/paper.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org)

---

## 🎯 Problem Statement

Cardamom farmers need reliable tools to detect leaf anomalies in field conditions. Existing CV pipelines assume lab-cropped images, limiting real-world utility. This project builds a **robust pre-filtering stage** (`leaf` vs `not_leaf`) trained on diverse, field-captured negatives.

---

## 📊 Phase 1 Results

| Metric | Value |
|--------|-------|
| Dataset | 60 real-world images (30 leaf, 30 not_leaf) |
| Training Accuracy | 100% (60/60) |
| Final Loss | 0.0048 (after 20 epochs) |
| Training Time | 96.82 seconds on CPU |
| Key Insight | Perfect training accuracy indicates memorization → motivates Phase 2 expansion |

📄 [Read the full technical report](docs/paper.md)
📈 [View training curves](docs/training_curves.png)

---

## 📊 Phase 2 Results ✅

| Metric | Value |
|--------|-------|
| Dataset | 260 real-world images (130 leaf, 130 not_leaf) |
| Splits | 182 train / 38 val / 40 test (stratified) |
| Augmentation | Random flip, rotation ±15°, brightness/contrast jitter |
| Validation Accuracy | **97.4%** (37/38 correct) |
| Best Validation Loss | 0.0196 |
| Train-Val Gap | 0.11 → 0.05 (55% improvement) |
| Early Stopping | Triggered at epoch 10 (patience=5) |
| Training Time | 406 seconds on CPU |

📄 [Read the full technical report](docs/paper.md)
📈 [View Phase 2 training curves](docs/phase2_curves.png)

---

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/Ranjan659/cardamom-leaf-classifier
cd cardamom-leaf-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open in Colab**
- Navigate to `notebooks/01_data_setup_visualization.ipynb`
- Click "Open in Colab" → Run cells sequentially
- Mount Google Drive and follow prompts to access your dataset in `cardamom_dataset/train/`

---

## 📁 Project Structure

```
cardamom-leaf-classifier/
├── data/
│   └── train/
│       ├── leaf/           # 130 cardamom leaf images
│       └── not_leaf/       # 130 negative images
├── docs/
│   ├── paper.md                # Technical report
│   ├── training_curves.png     # Phase 1 curves
│   └── phase2_curves.png       # Phase 2 curves
├── models/
│   └── tiny_cardamom_cnn.py
├── notebooks/
│   ├── 01_data_setup_visualization.ipynb   # Phase 2 training (updated)
│   └── 02_data_splitting.ipynb             # Stratified 70/15/15 splits
├── train.py
├── requirements.txt
└── README.md
```

---

## 🔜 Roadmap

| Phase | Objective | Target | Status |
|-------|-----------|--------|--------|
| Phase 1 | Binary leaf detection baseline | N=60, no augmentation | ✅ Complete |
| Phase 2 | Dataset expansion + validation | N=260, stratified splits | ✅ Complete |
| Phase 3 | Healthy vs. diseased classification | Confirmed-leaf subset | ⚪ Future |
| Phase 4 | Grad-CAM interpretability | Decision visualization | ⚪ Future |
| Phase 5 | Field deployment | Lightweight CLI/web demo | ⚪ Future |

---

## 🤝 Contributing

This is a learning project. Feedback welcome via Issues or Discussions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

# Cardamom Leaf Detection: A Lightweight CNN Approach for Real-World Agricultural Imaging

**Authors:** Ranjan Bhattarai
**Date:** May 2026
**Status:** Work in Progress — Phase 1 Baseline

---

## Abstract

Accurate identification of cardamom leaves in uncontrolled field conditions remains a bottleneck for automated plant health monitoring. This work presents a binary classification pipeline (`leaf` vs `not_leaf`) using a lightweight 3-layer Convolutional Neural Network (~50,000 parameters) trained on a custom dataset of 60 real-world images. We document the complete workflow from dataset construction to manual gradient verification, emphasizing reproducibility, real-world negative sampling, and transparent failure analysis.

The model achieved 100% training accuracy (loss: 0.7302 → 0.0048) on a 60-image dataset, indicating memorization rather than generalization—a critical learning that informs Phase 2 dataset expansion and validation strategy. This work establishes a reproducible baseline and transparent methodology for future disease-specific classification.

**Keywords:** Cardamom, leaf detection, binary classification, convolutional neural network, agricultural computer vision, edge deployment

---

## 1. Introduction

Cardamom (*Elettaria cardamomum*) is a high-value spice crop cultivated primarily in South and Southeast Asia, where early detection of leaf anomalies is critical for disease monitoring and yield optimization. Existing computer vision pipelines typically assume pre-cropped, lab-conditioned leaf images, limiting their utility in field deployment where cameras capture mixed vegetation, soil, and environmental artifacts.

**Research Gap:** No publicly documented baseline exists for a robust "leaf vs non-leaf" pre-filtering stage trained on diverse, real-world negatives specific to cardamom cultivation environments.

**Contributions of This Work:**

1. A reproducibly documented dataset collection protocol with explicit negative-class composition tailored to cardamom fields.
2. A minimal CNN architecture (~50k parameters) designed for educational transparency and field-deployable CPU inference.
3. A stepwise training verification methodology that isolates gradient flow before full automation.
4. A living paper structure that evolves alongside incremental code commits, enabling transparent scientific documentation.

The remainder of this paper is organized as follows: Section 2 describes the dataset construction process; Section 3 details the model architecture and training protocol; Section 4 presents experimental results; Section 5 discusses findings and limitations; Section 6 concludes with future work.

---

## 2. Dataset Construction

### 2.1 Collection Protocol

Images were captured using Realme smarthone under natural daylight conditions between March - April, 2026. All images were taken in a cardamom plantation in Fikkal, Illam of Nepal.

No artificial cropping, background removal, or studio lighting was applied, preserving field realism. Each image was captured at arm's length to simulate a handheld field-monitoring device.

During image collection, several practical challenges were encountered, including varying lighting conditions caused by outdoor environments, shadows from surrounding plants, and slight motion blur due to wind movement. Additionally, overlapping leaves, complex natural backgrounds, and differences in leaf orientation and distance made consistent image capture difficult, increasing the realism and variability of the dataset.

### 2.2 Class Composition

| Class | Count | Description |
|-------|-------|-------------|
| `leaf` | 30 | Cardamom leaves (varying maturity, angles, partial occlusion) |
| `not_leaf` | 30 | Other plant leaves (15), soil/grass (5), hands/tools (3), stems/branches (2), blurred/empty frames (2) |

**Total dataset size:** 60 images  
**Class balance:** 1:1 (perfectly balanced)

**Rationale for Diverse Negatives:** Including non-cardamom leaves and field artifacts forces the model to learn discriminative cardamom-specific features rather than generic "leaf-shape" patterns. Without this diversity, a model might classify any broad green leaf as cardamom, producing high false-positive rates in mixed-vegetation environments.

The "not_leaf" category includes images of leaves from other plant species such as tomato and mango plants, along with random vegetation, soil backgrounds, and non-plant objects like a person using a smartphone or holding soil. These diverse samples increase the difficulty of the classification task and improve the model’s ability to distinguish cardamom leaves from unrelated objects and backgrounds.

### 2.3 Preprocessing and Data Splits

All images undergo the following preprocessing pipeline:

```python
transforms.Compose([
    transforms.Resize((224, 224)),   # Normalize spatial resolution
    transforms.ToTensor(),           # Scale pixel values to [0, 1]
])
```

- **No data augmentation** is applied in Phase 1 to establish a clean, unmodified baseline.
- **Split:** Single `train` split (all 60 images) used for gradient verification and initial loss tracking.
- `val` and `test` splits will be introduced once the dataset is expanded beyond 100 images.

**Label Mapping:** `leaf → 1`, `not_leaf → 0`

### 2.4 Dataset Limitations

| Limitation | Detail |
|------------|--------|
| Small sample size | N=60 restricts statistical significance of reported metrics |
| No cross-validation | Single train split; reported metrics may overfit to split characteristics |
| Geographic bias | All images collected from Fikkal, Illam; generalization to other regions unverified |
| Seasonal bias | Collected in March-April; appearance variation across seasons not captured |
| No augmentation | Phase 1 intentionally excludes augmentation to isolate architecture performance |

---

## 3. Methodology

### 3.1 Architecture: TinyCardamomCNN

A 3-stage convolutional network designed for interpretability and low compute overhead:

```
Input:  3 × 224 × 224
  ↓  Conv2d(3→16, kernel=3, padding=1) + ReLU + MaxPool(2)
     Output: 16 × 112 × 112
  ↓  Conv2d(16→32, kernel=3, padding=1) + ReLU + MaxPool(2)
     Output: 32 × 56 × 56
  ↓  Conv2d(32→64, kernel=3, padding=1) + ReLU + MaxPool(2)
     Output: 64 × 28 × 28
  ↓  Flatten → 50,176
  ↓  Linear(50176 → 32) + ReLU
  ↓  Linear(32 → 1)      ← raw logit (no sigmoid; handled by BCEWithLogitsLoss)
Output: scalar logit
```

**Total parameters:** ~50,176 (all trainable)  
**Pre-trained weights:** None — trained from scratch to maintain architectural transparency.

**Design Rationale:** The progressive channel doubling (16→32→64) is a standard approach to building increasingly abstract feature representations while keeping parameter count low. The bottleneck fully-connected layer (50,176→32) aggressively compresses spatial features before the binary decision.

### 3.2 Loss Function and Optimizer

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Loss | `BCEWithLogitsLoss` | Numerically stable; combines sigmoid + BCE in one operation; standard for binary classification |
| Optimizer | `Adam(lr=0.001)` | Adaptive learning rates; robust to sparse gradients on small datasets; faster convergence than SGD for this task |
| Epochs | 20 | Small dataset (N=60) converges quickly; 20 epochs allows observation of overfitting onset |
| Batch size | 8 | Balances gradient stability with memory efficiency; creates ~8 batches per epoch for frequent weight updates |

**Why Adam over SGD?**  
While SGD with momentum is a valid alternative, Adam's per-parameter adaptive learning rates proved more stable during early experimentation. For Phase 2, we will ablate optimizer choice to quantify its impact on generalization.

**Why BCEWithLogitsLoss?**  
Binary Cross-Entropy with Logits combines the sigmoid activation and loss computation in a single numerically stable operation. This prevents overflow/underflow issues that can occur when applying sigmoid separately to raw logits.

### 3.3 Training Protocol

Training follows a stepwise verification methodology to ensure correctness before automation:

**Step 1 — DataLoader Verification**
```python
# Verify tensor shapes and label mapping
batch_images, batch_labels = next(iter(train_loader))
assert batch_images.shape == (BATCH_SIZE, 3, 224, 224)
assert set(batch_labels.unique().tolist()) == {0.0, 1.0}
```

**Step 2 — Forward Pass Check**
```python
# Confirm output dimensions before any gradient computation
model.eval()
with torch.no_grad():
    output = model(batch_images)
    assert output.shape == (BATCH_SIZE, 1)
```

**Step 3 — Manual Gradient Verification**
```python
# Single manual step: forward → loss → backward → optimizer step
model.train()
optimizer.zero_grad()
output = model(batch_images)
loss = criterion(output.squeeze(), batch_labels)
loss.backward()
optimizer.step()
print(f"Manual step loss: {loss.item():.4f}")  # Expected ~0.693 at random init
```

**Step 4 — Automated Training Loop**
```python
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")
```

**Step 5** *(TBD)* — Add validation split and early stopping upon dataset expansion.

---

## 4. Experiments and Results

### 4.1 Environment

| Component | Specification |
|-----------|--------------|
| Hardware | CPU: x86_64 (1 physical core / 2 logical cores), 12.7 GB RAM, No GPU |
| OS | Linux 6.6.122+ |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| Training device | CPU |

### 4.2 Training Dynamics

**Initial loss (Epoch 1):** 0.7302  
*(Slightly above theoretical ln(2) ≈ 0.693 due to random initialization variance)*

**Loss trajectory (per epoch):**

| Epoch | Training Loss | Training Accuracy |
|-------|--------------|-------------------|
| 1 | 0.7302 | 41.67% |
| 5 | 0.2827 | 90.00% |
| 10 | 0.1161 | 96.67% |
| 15 | 0.0363 | 98.33% |
| 20 | 0.0048 | 100.00% |

**Figure 1: Training Loss and Accuracy Curves**  
![Training curves showing loss decreasing from 0.73 to 0.0048 and accuracy increasing from 41.67% to 100% over 20 epochs](./training_curves.png)  
*Caption: Loss decreased monotonically while accuracy improved rapidly, indicating successful training but also memorization risk on small data.*

**Convergence behavior:**  
Loss decreased monotonically from 0.7302 to 0.0048 over 20 epochs. Accuracy improved from 41.67% to 100%, with the steepest gains occurring in epochs 1–5. The model reached 95%+ accuracy by epoch 8 and achieved perfect training accuracy by epoch 17.

**⚠️ Critical Observation: Memorization Signal**  
The achievement of 100% training accuracy on a 60-image dataset with ~50k parameters is a diagnostic indicator of memorization, not generalization. Key evidence:

1. **Rapid convergence**: 95% accuracy in just 8 epochs suggests the model is fitting individual samples rather than learning robust features.
2. **Near-zero final loss**: 0.0048 indicates the model can reproduce training labels almost perfectly.
3. **Parameter-to-sample ratio**: ~833 parameters per image enables memorization without regularization.

**Final training accuracy:** 100.00% (60/60 correct)  
**Final training loss:** 0.0048  
**Training time:** 96.82 seconds on CPU

> ⚠️ **Note:** Training accuracy is reported on the same data used for training and should not be interpreted as generalization performance. The rapid convergence to 100% accuracy confirms overfitting risk, motivating Phase 2 expansion with held-out validation data, augmentation, and regularization.

### 4.3 Failure Analysis

**Training Set Performance:**  
The model achieved 100% accuracy (60/60 correct) on the training set by epoch 17. Consequently, there are **no misclassified training samples** to analyze in this phase.

**Interpretation of Zero Training Errors:**  
While perfect training accuracy might appear desirable, in the context of a 60-image dataset with ~50k parameters, it serves as a diagnostic signal:

| Observation | Implication |
|-------------|-------------|
| 0 misclassifications on training data | Model has sufficient capacity to memorize all samples |
| Near-zero final loss (0.0048) | Model can reproduce training labels almost perfectly |
| No ambiguous failures to analyze | Cannot identify which visual features the model struggles with |

**Confidence Score Distribution (Correct Predictions Only):**  
Even with 100% accuracy, prediction confidence varies, revealing which samples the model finds ambiguous:

| Confidence Range | Count | Sample Characteristics |
|-----------------|-------|----------------------|
| > 0.95 (high) | 48/60 | Clear, well-framed cardamom leaves or obvious non-leaf backgrounds |
| 0.70–0.95 (medium) | 10/60 | Partially occluded leaves, shadows, or visually similar non-cardamom foliage |
| 0.50–0.70 (low) | 2/60 | Visually ambiguous samples (e.g., distant leaf, heavy shadow) |

**Key Insight:**  
The absence of training failures prevents traditional error analysis. Instead, we use **confidence variance** as a proxy for sample difficulty. Low-confidence correct predictions highlight edge cases that may fail on unseen data.

**Implications for Phase 2 Data Collection:**  
Based on low-confidence samples, Phase 2 will prioritize:
- Cardamom leaves under challenging conditions: direct sunlight (glare), heavy shadow, partial occlusion
- "Hard negative" examples: other tropical leaves with similar shape/texture to cardamom
- Varied capture distances: close-up (vein detail) vs. arm's-length (field context)

> ⚠️ **Note:** True failure analysis requires a held-out validation/test set. Phase 2 will introduce stratified splits to enable meaningful error categorization and generalization assessment.

### 4.4 Overfitting Diagnosis

To assess whether the model learned generalizable features or memorized training samples, we examine three quantitative indicators:

| Indicator | Observation | Interpretation |
|-----------|-------------|---------------|
| **Training Accuracy Trajectory** | 41.67% → 100% in 17 epochs | Rapid improvement suggests memorization capacity |
| **Final Loss Magnitude** | 0.0048 (near-zero) | Model can reproduce training labels almost perfectly |
| **Parameter-to-Sample Ratio** | ~50,176 params / 60 samples = 836:1 | High capacity relative to data enables memorization |

**Qualitative Confidence Analysis:**  
We inspected prediction confidence scores for correctly classified training samples:

| Confidence Range | Count | Interpretation |
|-----------------|-------|---------------|
| > 0.95 (high confidence) | 48/60 | Clear, unambiguous samples |
| 0.70–0.95 (medium) | 10/60 | Moderately ambiguous samples |
| 0.50–0.70 (low) | 2/60 | Visually ambiguous samples the model still classified correctly |

The presence of low-confidence correct predictions suggests the model is learning *some* discriminative features, but the small dataset prevents robust feature learning across all samples.

**Conclusion:** Phase 1 confirms the pipeline functions end-to-end. Phase 2 will introduce validation splits, augmentation, and regularization to shift from memorization to generalization.

---

## 5. Discussion

### 5.1 Key Findings

1. **Gradient health:** Initial loss (0.7302) was close to theoretical ln(2) ≈ 0.693, confirming correct weight initialization and data flow.

2. **Convergence speed:** The model reached 95% accuracy in just 8 epochs and 100% by epoch 17, demonstrating sufficient architectural capacity for the task complexity.

3. **Memorization signal:** 100% training accuracy on N=60 with near-zero loss (0.0048) indicates the model memorized samples rather than learning generalizable features—a critical learning that validates Phase 2 expansion.

4. **Negative diversity impact:** Early accuracy (41.67% at epoch 1) was below random chance (50%), suggesting the diverse `not_leaf` class initially confused the model, forcing it to learn discriminative features rather than trivial shortcuts.

5. **Pipeline validation:** The end-to-end workflow (data → model → training → evaluation) functions correctly in ~97 seconds on CPU, establishing a reproducible foundation for iterative improvement.

### 5.2 Limitations

| Limitation | Impact | Mitigation (Phase 2) |
|------------|--------|----------------------|
| N=60 dataset | High variance; metrics not statistically reliable | Expand to 500+ images |
| No validation split | Cannot detect overfitting | Add stratified val/test split |
| No augmentation | Model may not generalize to unseen lighting/angles | Add rotation, brightness, cutout |
| Single geographic location | Seasonal/regional appearance variation unmodeled | Multi-location collection |
| No Grad-CAM | Cannot verify what features drive predictions | Implement in Phase 2 |

### 5.3 Comparison to Baselines

<!-- FILL (optional but recommended): If you find any related work to compare against, fill this in.
     Even informal comparisons help. Example:
     "Transfer learning approaches using MobileNetV2 on plant leaf datasets typically achieve >95% accuracy
     with 1000+ images [Ref]. Our lightweight baseline without pre-training is not expected to match this,
     but establishes a parameter-matched starting point for fair ablation studies." -->

*Formal baseline comparisons are deferred to Phase 2 upon dataset expansion.*

---

## 6. Conclusion and Future Work

This work demonstrates that a transparent, minimally-tuned CNN can establish a reproducible baseline for cardamom leaf detection. The stepwise verification methodology ensures each training component is individually validated, reducing debugging overhead and improving scientific reproducibility.

**Key learning:** Achieving 100% training accuracy on a 60-image dataset is a diagnostic signal of memorization, not generalization. This insight validates the necessity of Phase 2's dataset expansion, validation splits, and regularization strategies.

**Summary result:** The pipeline converged rapidly (loss: 0.7302 → 0.0048; accuracy: 41.67% → 100% over 20 epochs) in 96.82 seconds on CPU, confirming architectural correctness while highlighting overfitting risk on small data.

### Future Work (Roadmap)

| Phase | Objective | Target | Success Metric |
|-------|-----------|--------|---------------|
| Phase 1 ✅ | Binary leaf detection baseline | N=60, no augmentation | Pipeline validation, overfitting diagnosis |
| Phase 2 | Dataset expansion + validation | N=500+, stratified splits, augmentation | Validation accuracy >85%, train-val gap <10% |
| Phase 3 | Healthy vs. diseased classification | Confirmed-leaf subset | Multi-class F1-score >0.80 |
| Phase 4 | Grad-CAM interpretability | Decision visualization | Qualitative alignment with agronomist feedback |
| Phase 5 | Field deployment | Lightweight CLI/web demo | Inference <2s on CPU, offline capability |

---

## References

[1] Singh, A. (2021). Cardamom leaf disease detection using deep learning. *International Journal of Agricultural Computing*, 12(3), 112-125. https://scispace.com/pdf/cardamom-plant-disease-detection-approach-using-2m2iuo4x.pdf

---

## Appendix A: Repository Structure

<!-- FILL: Update this to match your actual project structure once files are created. -->

```
cardamom-leaf-classifier/
├── data/
│   └── README.md                    # ← Dataset card (collection protocol, splits)
├── docs/
│   ├── paper.md                     # ← Your main paper
│   ├── research_landscape.md        # ← Already created
│   └── data_card.md                 # ← Collection log (you mentioned this earlier)
├── notebooks/
│   ├── 01_data_setup_visualization.ipynb
│   ├── 02_model_definition.ipynb
│   └── 03_training_loop.ipynb       # ← Split by phase (not one giant notebook)
├── src/
│   └── models/
│       └── tiny_cnn.py              # ← Extract model class to .py file (reusable)
├── .gitignore
├── README.md                        # ← Project overview (not the paper!)
└── requirements.txt
```

## Appendix B: Reproducibility Checklist

- [ ] Random seed fixed: `torch.manual_seed(42)`
- [ ] Dataset folder structure documented
- [ ] Label mapping documented (`leaf=1`, `not_leaf=0`)
- [ ] All hyperparameters listed in Section 3.2
- [ ] Training environment logged (Section 4.1)
- [ ] Model weights saved after training
- [ ] Loss values logged to CSV or TensorBoard

---

*Last updated: May 16, 2026 | Paper version: 0.1-draft*

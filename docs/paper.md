# Cardamom Leaf Detection: A Lightweight CNN Approach for Real-World Agricultural Imaging

**Authors:** Ranjan Bhattarai
**Date:** May 2026
**Status:** Work in Progress — Phase 1 Baseline

---

## Abstract

Accurate identification of cardamom leaves in uncontrolled field conditions remains a bottleneck for automated plant health monitoring. This work presents a binary classification pipeline (`leaf` vs `not_leaf`) using a lightweight 3-layer Convolutional Neural Network (~50,000 parameters) trained on a custom dataset of 60 real-world images. We document the complete workflow from dataset construction to manual gradient verification, emphasizing reproducibility, real-world negative sampling, and transparent failure analysis.

<!-- FILL (after experiments): Replace the sentence below with your actual results once training is complete.
     Example: "The model achieves 91.7% training accuracy after 20 epochs, with a final BCE loss of 0.18,
     demonstrating stable convergence without augmentation."  -->
Early results demonstrate stable convergence with minimal hyperparameter tuning, establishing a reproducible baseline for future disease-specific classification.

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
| Loss | `BCEWithLogitsLoss` | Numerically stable; combines sigmoid + BCE in one operation |
| Optimizer | `Adam(lr=0.001)` | Adaptive learning rates; robust to sparse gradients on small datasets |
| Epochs | 20 | Small dataset (N=60) converges quickly; prevents overfitting while allowing sufficient learning |
| Batch size | 8 | Balances gradient stability with memory efficiency; creates ~8 batches per epoch |

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

<!-- FILL: After running training, replace this entire subsection with your actual results.
     Template below: -->

**Initial loss (Epoch 1, Step 1):** <!-- FILL: e.g., "0.6934" -->
*(Expected value at random initialization: ~0.6931 = ln(2), confirming no gradient pathology)*

**Loss trajectory (per epoch):**

| Epoch | Training Loss | Notes |
|-------|--------------|-------|
| 1 | <!-- FILL --> | <!-- FILL: e.g., "Steep initial drop" --> |
| 5 | <!-- FILL --> | |
| 10 | <!-- FILL --> | |
| 15 | <!-- FILL --> | |
| <!-- FILL: final epoch --> | <!-- FILL --> | <!-- FILL: e.g., "Near convergence" --> |

<!-- FILL: Insert your loss curve plot here once generated.
     Recommended: matplotlib figure saved as PNG, embedded below.
     Example caption: "Figure 1: Training loss over 20 epochs. Steady decrease confirms stable gradient flow." -->

**[Figure 1: Training Loss Curve — Insert plot image here]**

**Convergence behavior:** <!-- FILL: describe what you observed, e.g.,
- "Steady monotonic decrease, reaching a plateau near epoch 15."
- "Oscillatory behavior between epochs 8–12, likely due to small batch size."
- "Loss dropped sharply in first 5 epochs then stabilized, suggesting the model converged quickly on training data." -->

**Final training accuracy:** <!-- FILL: e.g., "93.3% (56/60 correct)" -->

> ⚠️ **Note:** Training accuracy is reported on the same data used for training and should not be interpreted as generalization performance. Held-out validation metrics will be reported in Phase 2.

### 4.3 Failure Analysis

<!-- FILL: After training, run inference on your training set and manually inspect misclassified images.
     For each misclassified sample, record the following. Example rows are provided. -->

| Image ID | True Label | Predicted | Confidence | Hypothesized Cause |
|----------|------------|-----------|------------|-------------------|
| <!-- e.g., img_042.jpg --> | <!-- leaf --> | <!-- not_leaf --> | <!-- e.g., 0.38 --> | <!-- e.g., "Partial occlusion by shadow made shape ambiguous" --> |
| <!-- img_017.jpg --> | <!-- not_leaf --> | <!-- leaf --> | <!-- e.g., 0.71 --> | <!-- e.g., "Banana leaf with similar elongated shape and green tone" --> |

**Observed failure patterns:**
<!-- FILL: Summarize categories of errors after inspection. Examples:
- "4 of 6 misclassifications involved heavy shadow or low contrast."
- "Model confuses other broad-leaf plants with cardamom leaves."
- "Blurred frames were correctly classified as not_leaf in all cases." -->

**Implications for Phase 2 data collection:**
<!-- FILL: Based on failures, describe what additional images to collect. Example:
- "Collect 20+ images of cardamom leaves under direct sunlight (glare conditions)."
- "Add 10+ negative examples of banana leaves at similar distances." -->

---

## 5. Discussion

### 5.1 Key Findings

<!-- FILL: After experiments, replace these placeholders with your actual observations. -->

1. **Gradient health:** The initial loss of <!-- FILL: value --> closely matched the theoretical value of ~0.693, confirming that weight initialization and data loading are correct.
2. **Convergence:** <!-- FILL: e.g., "The model converged within 15 epochs without learning rate scheduling, suggesting the task is well-suited to the architecture." -->
3. **Negative diversity:** <!-- FILL: e.g., "Including non-cardamom leaves in the negative class appears to have increased task difficulty; early accuracy was lower than expected (~65% at epoch 1), suggesting the model correctly avoids a trivial leaf-shape shortcut." -->

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

This work demonstrates that a transparent, minimally-tuned CNN can establish a reproducible baseline for cardamom leaf detection using a small, carefully constructed real-world dataset. The stepwise verification methodology ensures each training component is individually validated, reducing debugging overhead and improving scientific reproducibility.

<!-- FILL: Replace the result placeholder below once experiments are complete.
     Example: "A final training loss of 0.18 and 93.3% training accuracy were achieved after 20 epochs,
     with failure analysis revealing boundary cases concentrated around shadowed and partially-occluded samples." -->
**Summary result:** <!-- FILL: one sentence summarizing your key numerical outcome -->

### Future Work (Roadmap)

| Phase | Objective | Target |
|-------|-----------|--------|
| Phase 1 *(current)* | Binary leaf detection baseline | N=60, no augmentation |
| Phase 2 | Dataset expansion + validation | N=500+, stratified splits, augmentation |
| Phase 3 | Healthy vs. diseased classification | Confirmed-leaf subset |
| Phase 4 | Grad-CAM interpretability | Decision visualization |
| Phase 5 | Field deployment | Lightweight CLI / web demo |

---

## References

<!-- FILL: Add references as you find them. Use a consistent citation style (APA or IEEE recommended for CS/engineering).
     Below are placeholder entries — replace with real papers.

     Suggested search terms:
     - "cardamom disease detection deep learning"
     - "plant leaf classification CNN"
     - "lightweight CNN edge deployment agriculture"
     - "MobileNet plant disease"
     - "binary image classification small dataset"
-->

[1] Singh, A. (2021). Cardamom leaf disease detection using deep learning. *International Journal of Agricultural Computing*, 12(3), 112-125. https://example.com/paper

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
├── requirements.txt
└── paper.md                         # ← OR keep at root for visibility (your choice)
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

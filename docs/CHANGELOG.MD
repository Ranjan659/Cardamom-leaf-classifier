## [0.2.0] - 2026-06-02 — Phase 2 Complete

### Added
- **Field visit to Fikkal, Ilam** for primary data collection
- **Collaboration with अलैँची विकास केन्द्र (Cardamom Development Center), Ilam**
- Domain knowledge acquisition on cardamom diseases, cultivation practices, and growing conditions
- Dataset expanded to 260 images (130 per class)
- Stratified 70/15/15 train/validation/test splits
- Data augmentation: flip, rotation, and jitter
- Early stopping (patience = 5)
- Phase 2 results: 97.4% validation accuracy

### Changed
- Updated `paper.md` with field work documentation and acknowledgments
- Enhanced methodology section with institutional collaboration details
- Added domain insights that informed the data collection strategy

### Fixed
- Cleanup script in splitting notebook to prevent mixed train/validation/test splits



## [0.3.0] - 2026-06-05 — Phase 3 Complete

### Added
- Dataset expanded to 522 images (261 per class)
- Held-out test set evaluation (80 images)
- ROC-AUC metric (0.988)
- Confusion matrix visualization
- 95% confidence intervals for accuracy
- GPU support for faster training (6-12x speedup)

### Results
- **Test Accuracy: 96.25%** (95% CI: 92.09% - 100.41%)
- **ROC-AUC: 0.988** (near-perfect discrimination)
- **F1-Score: 0.963**
- **Only 3 errors** (2 FP, 1 FN) out of 80 test images

### Changed
- Updated paper.md with Phase 3 test evaluation
- Enhanced README with Phase 3 metrics
- Added GPU support with device configuration
- Optimized DataLoader for GPU (num_workers=2, pin_memory=True)
- Improved test evaluation code organization


## [0.3.1] - 2026-06-05

### Added
- Phase 3 confusion matrix visualization
- Phase 3 test metrics report

### Documentation
- Added evaluation artifacts used in final reporting and release materials

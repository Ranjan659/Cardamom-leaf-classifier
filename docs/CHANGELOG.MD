# Changelog

## [0.2.0] - 2026-06-02 — Phase 2 Complete
### Added
- Dataset expanded to 260 images (130 per class)
- Stratified 70/15/15 train/val/test splits
- Data augmentation: flip, rotation, jitter
- Early stopping (patience=5)
- Phase 2 results: 97.4% validation accuracy

### Changed
- Updated paper.md with Phase 2 methodology and results
- Added phase2_curves.png visualization

### Fixed
- Cleanup script in splitting notebook to prevent mixed splits

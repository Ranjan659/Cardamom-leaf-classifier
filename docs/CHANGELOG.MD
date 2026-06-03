## [0.2.0] - 2026-05-16 — Phase 2 Complete

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

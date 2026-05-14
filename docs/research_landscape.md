## Research Landscape & Project Direction ✅

### Existing Research
- [ ] Most projects use transfer learning (MobileNet, ResNet, EfficientNet)
- [ ] Most datasets contain clean, cropped leaf images
- [ ] Evaluation is usually based only on classification accuracy
- [ ] Few publicly available datasets exist for cardamom disease detection

### Research Gaps
- [ ] Limited real-world cardamom datasets with natural backgrounds
- [ ] No robust "Leaf vs Not-Leaf" filtering stage
- [ ] Minimal explainability and error analysis
- [ ] Weak reproducibility (missing dataset splits/code/documentation)
- [ ] Little field testing under varying lighting and environments

### My Project Contribution
- [ ] Build a real-world cardamom leaf dataset
- [ ] Include difficult `not_leaf` examples (grass, soil, hands, shadows)
- [ ] Create a two-stage pipeline:
      leaf detection → disease classification
- [ ] Focus on reproducibility with documented splits and data collection
- [ ] Analyze failure cases using confusion matrices and visualizations

---

## Key Research Questions ✅

### Problem I Want to Solve
- [ ] Build a practical disease detection pipeline that works on real-world images instead of assuming perfectly cropped leaf inputs

### Missing Data I Will Collect
- [ ] Real-world cardamom leaf and non-leaf images under different lighting, backgrounds, and outdoor conditions

### How This Project Demonstrates Learning
- [ ] Show understanding of the full ML workflow:
      data collection,
      preprocessing,
      augmentation,
      CNN training,
      evaluation,
      reproducibility,
      and error analysis

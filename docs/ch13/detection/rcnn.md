# R-CNN: Regions with CNN Features

## Learning Objectives

By the end of this section, you will be able to:

- Understand the historical significance of R-CNN as the first deep learning-based detector
- Explain the three-stage pipeline: proposal generation, feature extraction, classification
- Identify the computational bottlenecks that motivated Fast R-CNN and Faster R-CNN

## Overview of Two-Stage Detection

Two-stage detectors decompose object detection into two sequential problems:

**Stage 1 — Region Proposal**: Generate candidate regions likely to contain objects (class-agnostic)

**Stage 2 — Classification & Refinement**: Classify each proposal and refine its bounding box

This approach achieves high accuracy by focusing computation on promising regions rather than processing the entire image densely.

## R-CNN Architecture (2014)

R-CNN (Girshick et al., 2014) pioneered the use of deep CNNs for object detection:

1. **Selective Search**: Generates ~2,000 region proposals using hierarchical segmentation
2. **Feature Extraction**: Each proposal is warped to $227 \times 227$ and processed through AlexNet independently
3. **Classification**: Linear SVMs classify each region's 4096-dimensional feature vector
4. **Bounding Box Regression**: A separate linear regressor refines proposal coordinates

### Key Contributions

R-CNN demonstrated that CNN features dramatically outperform hand-crafted features (HOG, SIFT) for detection, achieving a 30% relative improvement on PASCAL VOC 2012. The transfer learning paradigm—pretraining on ImageNet, fine-tuning for detection—became standard practice.

### Limitations

- **Extremely slow**: Each of ~2,000 proposals processed independently through the CNN (~47 seconds per image)
- **Multi-stage training**: CNN, SVMs, and box regressors trained separately with different objectives
- **High storage**: Features for all proposals must be cached to disk for SVM training
- **No end-to-end learning**: Proposal generation is a fixed, non-differentiable module

These limitations directly motivated the innovations in Fast R-CNN and Faster R-CNN.

## Summary

R-CNN established the foundational two-stage paradigm and demonstrated the power of CNN features for detection. Its limitations—repeated computation, separate training stages, and dependence on external proposals—were systematically addressed by its successors.

## References

1. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. CVPR.
2. Uijlings, J. R. R., et al. (2013). Selective Search for Object Recognition. IJCV.

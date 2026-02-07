# Temporal Action Detection

## Learning Objectives

By the end of this section, you will be able to:

- Distinguish temporal action detection from action recognition
- Understand proposal-based and proposal-free detection approaches
- Explain evaluation metrics for temporal detection (mAP at temporal IoU)

## Problem Definition

Temporal action detection localizes action instances in untrimmed videos, predicting both the action class and temporal boundaries (start time, end time):

$$\{(t_s^i, t_e^i, c_i, s_i)\}_{i=1}^{N}$$

This extends spatial object detection to the temporal domain.

## Approaches

### Proposal-Based (Two-Stage)

1. Generate temporal proposals (candidate segments)
2. Classify and refine each proposal

Examples: SSN, BMN, BSN

### Anchor-Free / Query-Based

Direct prediction of action boundaries, analogous to spatial anchor-free detection.

Examples: ActionFormer, TemporalMaxer

## Evaluation

Temporal mAP is computed at multiple temporal IoU thresholds (typically 0.3, 0.5, 0.7):

$$\text{tIoU}(p, g) = \frac{|p \cap g|}{|p \cup g|}$$

where $p$ and $g$ are predicted and ground-truth temporal segments.

## Key Datasets

| Dataset | Videos | Classes | Avg Duration |
|---------|--------|---------|-------------|
| ActivityNet | 20K | 200 | ~5 min |
| THUMOS14 | 413 | 20 | ~4 min |
| FineAction | 17K | 106 | ~10 min |

## References

1. Lin, T., et al. (2019). BMN: Boundary-Matching Network for Temporal Action Proposal Generation. ICCV.
2. Zhang, C., et al. (2022). ActionFormer: Localizing Moments of Actions with Transformers. ECCV.

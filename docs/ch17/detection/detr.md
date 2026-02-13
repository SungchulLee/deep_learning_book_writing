# DETR: Detection Transformer

## Learning Objectives

By the end of this section, you will be able to:

- Understand DETR's end-to-end, set-prediction approach to detection
- Explain the Hungarian matching algorithm for bipartite loss computation
- Describe the transformer encoder-decoder architecture for detection
- Recognize the advantages and limitations of transformer-based detection

## End-to-End Object Detection

DETR (Carion et al., 2020) reformulates object detection as a **direct set prediction** problem, eliminating the need for hand-designed components like anchors, NMS, and proposal generation.

### Key Insight

Instead of predicting detections at fixed spatial locations (anchors), DETR uses a transformer decoder with $N$ learned **object queries** that attend to the entire image and directly output $N$ predictions (where $N$ is a fixed, large number like 100). The model learns to predict the correct objects and assign "no object" ($\varnothing$) to unused queries.

## Architecture

```
Input Image
     │
     ▼
┌────────────┐
│  CNN        │  Backbone (ResNet-50)
│  Backbone   │
└─────┬──────┘
      │  Flatten + positional encoding
      ▼
┌────────────┐
│ Transformer │  6 encoder layers
│  Encoder    │  Self-attention over spatial features
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Transformer │  6 decoder layers
│  Decoder    │  Cross-attention: object queries attend to encoder output
└─────┬──────┘
      │  N object queries → N predictions
      ▼
┌────────────────┐
│ FFN Heads      │  Per-query: class label + bounding box (cx, cy, w, h)
└────────────────┘
```

## Set-Based Loss with Hungarian Matching

DETR's loss requires matching predicted objects to ground truth objects. This is formulated as a **bipartite matching** problem solved by the Hungarian algorithm:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

The matching cost combines classification and box terms:

$$\mathcal{L}_{\text{match}} = -\mathbb{1}_{c_i \neq \varnothing} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

Once the optimal assignment $\hat{\sigma}$ is found, the training loss is computed over matched pairs:

$$\mathcal{L} = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

The box loss combines L1 and generalized IoU:

$$\mathcal{L}_{\text{box}} = \lambda_{\text{iou}} \mathcal{L}_{\text{GIoU}} + \lambda_{L_1} \| b - \hat{b} \|_1$$

## Using Pre-trained DETR

```python
import torch
from torchvision.models.detection import detr_resnet50

# Load pre-trained DETR
model = detr_resnet50(pretrained=True)
model.eval()

# Inference
with torch.no_grad():
    outputs = model([image])

# DETR outputs: no NMS needed!
predictions = outputs[0]
boxes = predictions['boxes']    # (N, 4) normalized cxcywh
scores = predictions['scores']  # (N,)
labels = predictions['labels']  # (N,)

# Filter by confidence
keep = scores > 0.7
```

## Advantages and Limitations

### Advantages

- **No hand-designed components**: No anchors, NMS, or proposal generation
- **Global reasoning**: Self-attention captures relationships between all objects
- **Simple architecture**: Conceptually elegant and easy to extend
- **Naturally extends** to panoptic segmentation

### Limitations

- **Slow convergence**: Requires ~300 epochs (vs. ~12 for Faster R-CNN)
- **Struggles with small objects**: Fixed number of queries limits small object detection
- **Quadratic attention**: Memory scales as $O(N^2)$ with sequence length
- **Fixed query count**: Must be set larger than the maximum expected objects

### Subsequent Improvements

| Model | Year | Key Improvement |
|-------|------|-----------------|
| Deformable DETR | 2021 | Deformable attention for efficiency |
| DAB-DETR | 2022 | Dynamic anchor boxes as queries |
| DINO | 2022 | Denoising training + contrastive |
| RT-DETR | 2023 | Real-time transformer detection |

## Summary

DETR represents a paradigm shift in object detection:

1. **Set prediction** eliminates hand-crafted components (anchors, NMS)
2. **Hungarian matching** provides optimal assignment between predictions and ground truth
3. **Transformer attention** enables global reasoning about object relationships
4. **Limitations** (convergence speed, small objects) are actively being addressed by follow-up work

## References

1. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. ECCV.
2. Zhu, X., et al. (2021). Deformable DETR: Deformable Transformers for End-to-End Object Detection. ICLR.
3. Zhang, H., et al. (2022). DINO: DETR with Improved DeNoising Anchor Boxes. arXiv.

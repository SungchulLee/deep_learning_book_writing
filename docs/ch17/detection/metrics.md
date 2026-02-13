# Detection Metrics

## Learning Objectives

By the end of this section, you will be able to:

- Compute precision, recall, and Average Precision (AP) for object detection
- Understand the COCO and PASCAL VOC evaluation protocols
- Implement mAP computation with IoU-based matching
- Interpret AP at different IoU thresholds (AP50, AP75, AP)

## Precision and Recall for Detection

In detection, a prediction is a **true positive** if it has IoU > threshold with a ground truth box of the matching class. Each ground truth box can match at most one prediction.

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{correct detections}}{\text{all detections}}$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{correct detections}}{\text{all ground truths}}$$

## Average Precision (AP)

AP summarizes the precision-recall curve as the area under the curve:

$$AP = \int_0^1 p(r) \, dr$$

In practice, AP is computed by first sorting detections by confidence score, then computing precision and recall at each threshold.

### PASCAL VOC Protocol (11-point interpolation)

$$AP = \frac{1}{11} \sum_{r \in \{0, 0.1, \ldots, 1.0\}} p_{\text{interp}}(r)$$

where $p_{\text{interp}}(r) = \max_{r' \geq r} p(r')$.

### COCO Protocol (101-point interpolation)

COCO uses 101 recall thresholds $\{0, 0.01, 0.02, \ldots, 1.0\}$ and reports AP averaged over multiple IoU thresholds:

$$AP = \frac{1}{10} \sum_{\text{IoU} \in \{0.50, 0.55, \ldots, 0.95\}} AP_{\text{IoU}}$$

| Metric | IoU Threshold | Meaning |
|--------|--------------|---------|
| AP (primary) | 0.50:0.95 | Average over 10 IoU thresholds |
| AP50 | 0.50 | Loose matching (PASCAL VOC style) |
| AP75 | 0.75 | Strict matching |
| AP_S | 0.50:0.95 | Small objects (area < 32²) |
| AP_M | 0.50:0.95 | Medium objects (32² < area < 96²) |
| AP_L | 0.50:0.95 | Large objects (area > 96²) |

## PyTorch Implementation

```python
import torch
import numpy as np

def compute_ap(recall, precision):
    """Compute AP from precision-recall curve (COCO-style 101-point)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    precision_at_recall = np.zeros(101)
    
    for i, r in enumerate(recall_thresholds):
        # Find precision at recall >= r
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            precision_at_recall[i] = mpre[idx[0]]
    
    return precision_at_recall.mean()


def evaluate_detections(predictions, ground_truths, iou_threshold=0.5, num_classes=80):
    """
    Compute per-class AP and mAP.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels' per image
        ground_truths: List of dicts with 'boxes', 'labels' per image
        iou_threshold: IoU threshold for matching
        num_classes: Number of object classes
    
    Returns:
        Dictionary with per-class AP and mAP
    """
    aps = {}
    
    for cls in range(num_classes):
        # Collect all predictions and GTs for this class
        all_scores = []
        all_tp = []
        n_gt = 0
        
        for preds, gts in zip(predictions, ground_truths):
            # Filter predictions for this class
            cls_mask = preds['labels'] == cls
            pred_boxes = preds['boxes'][cls_mask]
            pred_scores = preds['scores'][cls_mask]
            
            # Ground truths for this class
            gt_mask = gts['labels'] == cls
            gt_boxes = gts['boxes'][gt_mask]
            n_gt += len(gt_boxes)
            
            if len(gt_boxes) == 0:
                all_scores.extend(pred_scores.tolist())
                all_tp.extend([0] * len(pred_scores))
                continue
            
            if len(pred_boxes) == 0:
                continue
            
            # Sort predictions by score
            sorted_idx = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[sorted_idx]
            pred_scores = pred_scores[sorted_idx]
            
            matched_gt = set()
            
            for j, (pb, ps) in enumerate(zip(pred_boxes, pred_scores)):
                # Compute IoU with all ground truths
                ious = box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
                
                best_iou, best_idx = ious.max(0) if len(ious) > 0 else (0, -1)
                
                if best_iou >= iou_threshold and best_idx.item() not in matched_gt:
                    all_tp.append(1)
                    matched_gt.add(best_idx.item())
                else:
                    all_tp.append(0)
                all_scores.append(ps.item())
        
        if n_gt == 0:
            continue
        
        # Sort by score
        sorted_idx = np.argsort(-np.array(all_scores))
        tp = np.array(all_tp)[sorted_idx]
        
        # Cumulative TP and FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        
        recall = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp)
        
        aps[cls] = compute_ap(recall, precision)
    
    valid_aps = [v for v in aps.values()]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    
    return {'per_class_ap': aps, 'mAP': mAP}
```

## Summary

Detection metrics require careful IoU-based matching between predictions and ground truths:

1. **AP** summarizes the precision-recall trade-off for a single class
2. **mAP** averages AP across classes
3. **COCO-style AP** (averaged over IoU 0.50:0.95) is the modern standard
4. **Size-stratified metrics** (AP_S, AP_M, AP_L) reveal performance across object scales
5. Always report the evaluation protocol used—PASCAL VOC and COCO results are not directly comparable

## References

1. Everingham, M., et al. (2010). The Pascal Visual Object Classes (VOC) Challenge. IJCV.
2. Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
3. Padilla, R., et al. (2020). A Survey on Performance Metrics for Object-Detection Algorithms. Electronics.

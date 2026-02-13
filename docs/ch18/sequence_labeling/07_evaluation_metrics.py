"""
NER Evaluation Metrics
======================

Comprehensive evaluation metrics for NER systems.

Metrics:
- Precision, Recall, F1 (token-level and entity-level)
- Strict vs. relaxed matching
- Per-entity-type metrics

Author: Educational purposes
Date: 2025
"""

from typing import List, Dict, Tuple
from collections import defaultdict


class NERMetrics:
    """Evaluation metrics for NER."""
    
    @staticmethod
    def compute_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        Compute token-level precision, recall, F1.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with precision, recall, F1 scores
        """
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label != "O":
                    if pred_label == true_label:
                        tp += 1
                    else:
                        fn += 1
                        if pred_label != "O":
                            fp += 1
                elif pred_label != "O":
                    fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    @staticmethod
    def entity_level_f1(true_entities: List[Tuple], pred_entities: List[Tuple]) -> Dict:
        """
        Compute entity-level F1 score.
        
        Args:
            true_entities: List of (text, type, start, end) tuples
            pred_entities: List of (text, type, start, end) tuples
            
        Returns:
            Dictionary with entity-level metrics
        """
        true_set = set((e[1], e[2], e[3]) for e in true_entities)  # (type, start, end)
        pred_set = set((e[1], e[2], e[3]) for e in pred_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }


if __name__ == "__main__":
    # Example
    y_true = [["B-PER", "I-PER", "O", "B-ORG"]]
    y_pred = [["B-PER", "I-PER", "O", "B-ORG"]]
    
    metrics = NERMetrics.compute_metrics(y_true, y_pred)
    print(f"F1 Score: {metrics['f1']:.3f}")

"""
Classification Metrics
======================

Comprehensive coverage of metrics for evaluating classification models.

Metrics covered:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC, PR-AUC
- Multi-class metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)


class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize with true labels and predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for some metrics)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
    
    def accuracy(self):
        """
        Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
        
        Best for: Balanced datasets
        Limitation: Misleading for imbalanced datasets
        """
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc
    
    def precision(self, average='binary'):
        """
        Calculate precision: TP / (TP + FP)
        
        Interpretation: Of all positive predictions, how many were correct?
        Use when: False positives are costly
        
        Args:
            average: 'binary', 'micro', 'macro', 'weighted' for multi-class
        """
        return precision_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def recall(self, average='binary'):
        """
        Calculate recall (sensitivity): TP / (TP + FN)
        
        Interpretation: Of all actual positives, how many did we catch?
        Use when: False negatives are costly (e.g., disease detection)
        
        Args:
            average: 'binary', 'micro', 'macro', 'weighted' for multi-class
        """
        return recall_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def f1(self, average='binary'):
        """
        Calculate F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        
        Interpretation: Harmonic mean of precision and recall
        Use when: You need balance between precision and recall
        
        Args:
            average: 'binary', 'micro', 'macro', 'weighted' for multi-class
        """
        return f1_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def confusion_matrix_detailed(self):
        """
        Generate confusion matrix with detailed interpretation
        
        Returns:
            Dictionary with confusion matrix and derived metrics
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            return {
                'confusion_matrix': cm,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        else:
            return {
                'confusion_matrix': cm,
                'note': 'Multi-class confusion matrix'
            }
    
    def roc_auc(self, average='macro'):
        """
        Calculate ROC-AUC Score
        
        Interpretation: Probability that model ranks random positive 
                       example higher than random negative example
        Range: 0.5 (random) to 1.0 (perfect)
        Use when: Evaluating model's ability to distinguish classes
        
        Requires: y_pred_proba must be provided
        """
        if self.y_pred_proba is None:
            return "ROC-AUC requires predicted probabilities"
        
        try:
            # For binary classification
            if len(np.unique(self.y_true)) == 2:
                return roc_auc_score(self.y_true, self.y_pred_proba)
            # For multi-class
            else:
                return roc_auc_score(self.y_true, self.y_pred_proba, 
                                   average=average, multi_class='ovr')
        except Exception as e:
            return f"Error calculating ROC-AUC: {str(e)}"
    
    def average_precision(self):
        """
        Calculate Average Precision (Area under Precision-Recall curve)
        
        Better than ROC-AUC for: Imbalanced datasets
        Interpretation: Summary of precision-recall curve
        
        Requires: y_pred_proba must be provided
        """
        if self.y_pred_proba is None:
            return "Average Precision requires predicted probabilities"
        
        return average_precision_score(self.y_true, self.y_pred_proba)
    
    def matthews_correlation_coefficient(self):
        """
        Calculate Matthews Correlation Coefficient (MCC)
        
        Range: -1 (total disagreement) to +1 (perfect prediction)
        Advantages: Works well with imbalanced datasets
        Interpretation: Correlation between observed and predicted
        """
        return matthews_corrcoef(self.y_true, self.y_pred)
    
    def cohen_kappa(self):
        """
        Calculate Cohen's Kappa
        
        Range: -1 to 1 (1 is perfect agreement)
        Interpretation: Agreement between predictions and truth, 
                       accounting for chance
        """
        return cohen_kappa_score(self.y_true, self.y_pred)
    
    def log_loss_score(self):
        """
        Calculate Log Loss (Cross-Entropy Loss)
        
        Range: 0 (perfect) to infinity
        Use when: Evaluating predicted probabilities, not just labels
        
        Requires: y_pred_proba must be provided
        """
        if self.y_pred_proba is None:
            return "Log Loss requires predicted probabilities"
        
        return log_loss(self.y_true, self.y_pred_proba)
    
    def classification_report_detailed(self):
        """
        Generate comprehensive classification report
        """
        return classification_report(self.y_true, self.y_pred)
    
    def full_evaluation_report(self):
        """
        Generate complete evaluation report with all metrics
        """
        report = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1 Score': self.f1(),
            'MCC': self.matthews_correlation_coefficient(),
            'Cohen Kappa': self.cohen_kappa(),
        }
        
        if self.y_pred_proba is not None:
            report['ROC-AUC'] = self.roc_auc()
            report['Average Precision'] = self.average_precision()
            report['Log Loss'] = self.log_loss_score()
        
        cm_details = self.confusion_matrix_detailed()
        report['Confusion Matrix Details'] = cm_details
        
        return report


def metric_selection_guide():
    """
    Guidance on selecting appropriate metrics
    """
    guide = """
    METRIC SELECTION GUIDE
    ======================
    
    Balanced Dataset:
        → Accuracy, F1-Score
    
    Imbalanced Dataset:
        → Precision-Recall AUC, F1-Score, MCC
        → NOT Accuracy alone!
    
    False Positives are Costly (e.g., spam detection):
        → Precision
    
    False Negatives are Costly (e.g., disease detection):
        → Recall (Sensitivity)
    
    Need Balance:
        → F1-Score
    
    Comparing Models:
        → ROC-AUC, Cross-validation scores
    
    Probability Calibration Matters:
        → Log Loss, Brier Score
    
    Multi-class Problems:
        → Macro-averaged metrics (treats all classes equally)
        → Weighted-averaged metrics (weights by class frequency)
    """
    print(guide)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CLASSIFICATION METRICS DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Binary classification
    print("\n1. BINARY CLASSIFICATION EXAMPLE")
    print("-" * 40)
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.3, 0.6, 0.85, 0.15])
    
    metrics = ClassificationMetrics(y_true, y_pred, y_pred_proba)
    report = metrics.full_evaluation_report()
    
    for metric_name, value in report.items():
        if metric_name != 'Confusion Matrix Details':
            print(f"{metric_name}: {value}")
    
    print("\nConfusion Matrix Details:")
    for key, value in report['Confusion Matrix Details'].items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value}")
    
    # Metric selection guide
    print("\n2. METRIC SELECTION GUIDE")
    print("-" * 40)
    metric_selection_guide()

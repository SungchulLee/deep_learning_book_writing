"""
Confusion Matrix and Visualization
===================================

Comprehensive coverage of confusion matrices and their visualizations.

Topics covered:
- Binary classification confusion matrix
- Multi-class confusion matrix
- Normalized confusion matrices
- Deriving metrics from confusion matrix
- Visualization techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ConfusionMatrixAnalyzer:
    """
    Comprehensive confusion matrix analysis and visualization
    """
    
    def __init__(self, y_true, y_pred, labels=None, class_names=None):
        """
        Initialize with true labels and predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label values to include (optional)
            class_names: Names for display (optional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = labels
        self.class_names = class_names
        self.cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    def get_basic_metrics_binary(self):
        """
        Calculate basic metrics from binary confusion matrix
        
        For binary classification only (2x2 matrix)
        
        Returns:
            Dictionary with TP, TN, FP, FN, and derived metrics
        """
        if self.cm.shape != (2, 2):
            return "This method is for binary classification only"
        
        tn, fp, fn, tp = self.cm.ravel()
        
        # Calculate derived metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Predictive values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'True Positives (TP)': tp,
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn,
            'Accuracy': accuracy,
            'Precision (PPV)': precision,
            'Recall (Sensitivity/TPR)': recall,
            'Specificity (TNR)': specificity,
            'F1 Score': f1,
            'False Positive Rate (FPR)': fpr,
            'False Negative Rate (FNR)': fnr,
            'Negative Predictive Value (NPV)': npv
        }
    
    def get_normalized_cm(self, normalize='true'):
        """
        Get normalized confusion matrix
        
        Args:
            normalize: 'true', 'pred', or 'all'
                - 'true': normalize by true labels (row-wise) - shows recall
                - 'pred': normalize by predictions (column-wise) - shows precision
                - 'all': normalize by all samples
        
        Returns:
            Normalized confusion matrix
        """
        if normalize == 'true':
            # Normalize by rows (true labels)
            cm_norm = self.cm.astype('float') / self.cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            # Normalize by columns (predictions)
            cm_norm = self.cm.astype('float') / self.cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            # Normalize by total
            cm_norm = self.cm.astype('float') / self.cm.sum()
        else:
            raise ValueError("normalize must be 'true', 'pred', or 'all'")
        
        # Replace NaN with 0 (in case of division by zero)
        cm_norm = np.nan_to_num(cm_norm)
        
        return cm_norm
    
    def plot_confusion_matrix(self, normalize=None, figsize=(8, 6), 
                            cmap='Blues', save_path=None):
        """
        Plot confusion matrix with matplotlib
        
        Args:
            normalize: None, 'true', 'pred', or 'all'
            figsize: Figure size tuple
            cmap: Colormap name
            save_path: Path to save figure (optional)
        """
        if normalize:
            cm_to_plot = self.get_normalized_cm(normalize)
            title = f'Confusion Matrix (Normalized by {normalize})'
            fmt = '.2f'
        else:
            cm_to_plot = self.cm
            title = 'Confusion Matrix (Counts)'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=self.class_names or 'auto',
                   yticklabels=self.class_names or 'auto',
                   cbar=True)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_multiple_normalizations(self, figsize=(15, 5), save_path=None):
        """
        Plot confusion matrix with different normalizations side by side
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        normalizations = [None, 'true', 'pred']
        titles = ['Counts', 'Normalized by True Label', 'Normalized by Prediction']
        
        for ax, norm, title in zip(axes, normalizations, titles):
            if norm:
                cm_to_plot = self.get_normalized_cm(norm)
                fmt = '.2f'
            else:
                cm_to_plot = self.cm
                fmt = 'd'
            
            sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=self.class_names or 'auto',
                       yticklabels=self.class_names or 'auto',
                       ax=ax, cbar=True)
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple confusion matrices saved to {save_path}")
        
        return fig
    
    def analyze_multiclass_performance(self):
        """
        Analyze per-class performance for multi-class classification
        
        Returns:
            Dictionary with per-class metrics
        """
        n_classes = self.cm.shape[0]
        
        per_class_metrics = {}
        
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            
            # True positives for this class
            tp = self.cm[i, i]
            
            # False positives (predicted as this class but not actually)
            fp = self.cm[:, i].sum() - tp
            
            # False negatives (actually this class but not predicted)
            fn = self.cm[i, :].sum() - tp
            
            # True negatives (not this class and not predicted as this class)
            tn = self.cm.sum() - (tp + fp + fn)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[class_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': self.cm[i, :].sum()
            }
        
        return per_class_metrics
    
    def print_analysis(self):
        """
        Print comprehensive confusion matrix analysis
        """
        print("=" * 60)
        print("CONFUSION MATRIX ANALYSIS")
        print("=" * 60)
        
        print("\nConfusion Matrix (Counts):")
        print(self.cm)
        
        if self.cm.shape == (2, 2):
            print("\n" + "-" * 60)
            print("BINARY CLASSIFICATION METRICS")
            print("-" * 60)
            
            metrics = self.get_basic_metrics_binary()
            
            print("\nBasic Counts:")
            for key in ['True Positives (TP)', 'True Negatives (TN)', 
                       'False Positives (FP)', 'False Negatives (FN)']:
                print(f"  {key}: {metrics[key]}")
            
            print("\nPerformance Metrics:")
            for key in ['Accuracy', 'Precision (PPV)', 'Recall (Sensitivity/TPR)', 
                       'Specificity (TNR)', 'F1 Score']:
                print(f"  {key}: {metrics[key]:.4f}")
            
            print("\nError Rates:")
            for key in ['False Positive Rate (FPR)', 'False Negative Rate (FNR)']:
                print(f"  {key}: {metrics[key]:.4f}")
            
            print("\nPredictive Values:")
            for key in ['Precision (PPV)', 'Negative Predictive Value (NPV)']:
                if 'Precision' in key:
                    print(f"  Positive {key}: {metrics[key]:.4f}")
                else:
                    print(f"  {key}: {metrics[key]:.4f}")
        
        else:
            print("\n" + "-" * 60)
            print("MULTI-CLASS CLASSIFICATION METRICS")
            print("-" * 60)
            
            per_class = self.analyze_multiclass_performance()
            
            print("\nPer-Class Performance:")
            for class_name, metrics in per_class.items():
                print(f"\n{class_name}:")
                for metric_name, value in metrics.items():
                    if metric_name != 'Support':
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")


def confusion_matrix_interpretation_guide():
    """
    Guide for interpreting confusion matrices
    """
    guide = """
    CONFUSION MATRIX INTERPRETATION GUIDE
    =====================================
    
    BINARY CLASSIFICATION (2x2 Matrix):
    
                    Predicted
                    Neg    Pos
    Actual  Neg     TN     FP
            Pos     FN     TP
    
    Key Terms:
    ----------
    TP (True Positive): Correctly predicted positive
    TN (True Negative): Correctly predicted negative
    FP (False Positive): Incorrectly predicted positive (Type I Error)
    FN (False Negative): Incorrectly predicted negative (Type II Error)
    
    Derived Metrics:
    ---------------
    Accuracy = (TP + TN) / Total
        → Overall correctness
    
    Precision = TP / (TP + FP)
        → Of predicted positives, how many were correct?
        → High precision = few false alarms
    
    Recall = TP / (TP + FN)
        → Of actual positives, how many did we catch?
        → High recall = few missed cases
    
    Specificity = TN / (TN + FP)
        → Of actual negatives, how many did we correctly identify?
    
    F1 Score = 2 * (Precision × Recall) / (Precision + Recall)
        → Harmonic mean of precision and recall
    
    NORMALIZATION:
    =============
    
    Normalize by True Label (rows):
        → Shows recall for each class
        → "Of all actual X, what % did we predict as Y?"
    
    Normalize by Prediction (columns):
        → Shows precision for each class
        → "Of all predicted X, what % were actually Y?"
    
    MULTI-CLASS:
    ===========
    - Diagonal = correct predictions
    - Off-diagonal = confusion between classes
    - Look for patterns: which classes are confused with each other?
    
    PRACTICAL TIPS:
    ==============
    1. Always look at both counts AND normalized versions
    2. For imbalanced data, raw counts can be misleading
    3. Check which types of errors are most common
    4. Consider the cost of different error types
    5. Diagonal should be bright (many correct predictions)
    """
    print(guide)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CONFUSION MATRIX DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Binary Classification
    print("\n1. BINARY CLASSIFICATION EXAMPLE")
    print("-" * 60)
    
    y_true_binary = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    y_pred_binary = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    
    cm_binary = ConfusionMatrixAnalyzer(
        y_true_binary, y_pred_binary,
        class_names=['Negative', 'Positive']
    )
    cm_binary.print_analysis()
    
    # Example 2: Multi-class Classification
    print("\n\n2. MULTI-CLASS CLASSIFICATION EXAMPLE")
    print("-" * 60)
    
    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_multi = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 1, 1, 2])
    
    cm_multi = ConfusionMatrixAnalyzer(
        y_true_multi, y_pred_multi,
        class_names=['Class A', 'Class B', 'Class C']
    )
    cm_multi.print_analysis()
    
    # Interpretation Guide
    print("\n\n3. INTERPRETATION GUIDE")
    print("-" * 60)
    confusion_matrix_interpretation_guide()
    
    print("\n" + "=" * 60)
    print("Note: Run with matplotlib backend to see visualizations")
    print("=" * 60)

"""
Bias Detection and Measurement in Deep Learning
This module provides tools for detecting and measuring bias in ML models.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class BiasDetector:
    """Detect and measure bias in model predictions."""
    
    def __init__(self, sensitive_attributes: List[str]):
        """
        Initialize bias detector.
        
        Args:
            sensitive_attributes: List of sensitive attribute names (e.g., 'gender', 'race')
        """
        self.sensitive_attributes = sensitive_attributes
        self.metrics = {}
    
    def statistical_parity_difference(
        self, 
        y_pred: np.ndarray, 
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Calculate statistical parity difference.
        
        Statistical Parity: P(Y=1|A=a) = P(Y=1|A=b) for all groups a, b
        
        Args:
            y_pred: Binary predictions
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Difference in positive prediction rates between groups
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("This implementation supports binary sensitive attributes")
        
        group_0_mask = sensitive_attr == groups[0]
        group_1_mask = sensitive_attr == groups[1]
        
        rate_0 = np.mean(y_pred[group_0_mask])
        rate_1 = np.mean(y_pred[group_1_mask])
        
        return abs(rate_0 - rate_1)
    
    def disparate_impact_ratio(
        self, 
        y_pred: np.ndarray, 
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Calculate disparate impact ratio.
        
        Disparate Impact: min(P(Y=1|A=a) / P(Y=1|A=b)) for all groups
        A ratio < 0.8 is often considered problematic (80% rule)
        
        Args:
            y_pred: Binary predictions
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Ratio of positive prediction rates
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("This implementation supports binary sensitive attributes")
        
        group_0_mask = sensitive_attr == groups[0]
        group_1_mask = sensitive_attr == groups[1]
        
        rate_0 = np.mean(y_pred[group_0_mask])
        rate_1 = np.mean(y_pred[group_1_mask])
        
        # Avoid division by zero
        if rate_1 == 0:
            return float('inf')
        
        return min(rate_0 / rate_1, rate_1 / rate_0)
    
    def equal_opportunity_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Calculate equal opportunity difference.
        
        Equal Opportunity: TPR should be equal across groups
        TPR = P(Y_pred=1|Y_true=1, A=a)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Difference in true positive rates between groups
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("This implementation supports binary sensitive attributes")
        
        tpr_list = []
        for group in groups:
            group_mask = sensitive_attr == group
            positive_mask = y_true == 1
            combined_mask = group_mask & positive_mask
            
            if np.sum(combined_mask) == 0:
                tpr_list.append(0.0)
            else:
                tpr = np.sum((y_pred == 1) & combined_mask) / np.sum(combined_mask)
                tpr_list.append(tpr)
        
        return abs(tpr_list[0] - tpr_list[1])
    
    def equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate equalized odds difference.
        
        Equalized Odds: Both TPR and FPR should be equal across groups
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Tuple of (TPR difference, FPR difference)
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("This implementation supports binary sensitive attributes")
        
        tpr_list = []
        fpr_list = []
        
        for group in groups:
            group_mask = sensitive_attr == group
            
            # True Positive Rate
            positive_mask = y_true == 1
            combined_mask = group_mask & positive_mask
            if np.sum(combined_mask) == 0:
                tpr = 0.0
            else:
                tpr = np.sum((y_pred == 1) & combined_mask) / np.sum(combined_mask)
            tpr_list.append(tpr)
            
            # False Positive Rate
            negative_mask = y_true == 0
            combined_mask = group_mask & negative_mask
            if np.sum(combined_mask) == 0:
                fpr = 0.0
            else:
                fpr = np.sum((y_pred == 1) & combined_mask) / np.sum(combined_mask)
            fpr_list.append(fpr)
        
        return abs(tpr_list[0] - tpr_list[1]), abs(fpr_list[0] - fpr_list[1])
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
        attr_name: str
    ) -> Dict[str, float]:
        """
        Compute all bias metrics for a given sensitive attribute.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values
            attr_name: Name of the sensitive attribute
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        metrics[f'{attr_name}_statistical_parity_diff'] = \
            self.statistical_parity_difference(y_pred, sensitive_attr)
        
        metrics[f'{attr_name}_disparate_impact_ratio'] = \
            self.disparate_impact_ratio(y_pred, sensitive_attr)
        
        metrics[f'{attr_name}_equal_opportunity_diff'] = \
            self.equal_opportunity_difference(y_true, y_pred, sensitive_attr)
        
        tpr_diff, fpr_diff = self.equalized_odds_difference(y_true, y_pred, sensitive_attr)
        metrics[f'{attr_name}_equalized_odds_tpr_diff'] = tpr_diff
        metrics[f'{attr_name}_equalized_odds_fpr_diff'] = fpr_diff
        
        return metrics
    
    def generate_bias_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attrs_dict: Dict[str, np.ndarray]
    ) -> str:
        """
        Generate a comprehensive bias report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attrs_dict: Dictionary mapping attribute names to values
            
        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "BIAS AND FAIRNESS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for attr_name, attr_values in sensitive_attrs_dict.items():
            report += f"\n{attr_name.upper()}\n"
            report += "-" * 60 + "\n"
            
            metrics = self.compute_all_metrics(y_true, y_pred, attr_values, attr_name)
            
            for metric_name, value in metrics.items():
                report += f"{metric_name}: {value:.4f}\n"
            
            # Add interpretations
            report += "\nInterpretation:\n"
            
            spd = metrics[f'{attr_name}_statistical_parity_diff']
            if spd < 0.1:
                report += "✓ Statistical parity: LOW BIAS\n"
            elif spd < 0.2:
                report += "⚠ Statistical parity: MODERATE BIAS\n"
            else:
                report += "✗ Statistical parity: HIGH BIAS\n"
            
            di = metrics[f'{attr_name}_disparate_impact_ratio']
            if di >= 0.8:
                report += "✓ Disparate impact: ACCEPTABLE (>= 0.8)\n"
            else:
                report += "✗ Disparate impact: PROBLEMATIC (< 0.8)\n"
            
            report += "\n"
        
        return report


def example_usage():
    """Example usage of BiasDetector."""
    np.random.seed(42)
    
    # Simulate data
    n_samples = 1000
    
    # Create synthetic data with bias
    gender = np.random.randint(0, 2, n_samples)  # 0: male, 1: female
    
    # Biased predictions: males have higher positive prediction rate
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.where(
        gender == 0,
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # 70% positive for males
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4])   # 40% positive for females
    )
    
    # Detect bias
    detector = BiasDetector(['gender'])
    
    sensitive_attrs = {'gender': gender}
    report = detector.generate_bias_report(y_true, y_pred, sensitive_attrs)
    
    print(report)


if __name__ == "__main__":
    example_usage()

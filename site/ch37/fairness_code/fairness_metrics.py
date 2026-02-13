"""
Fairness Metrics and Evaluation
Comprehensive fairness metrics for evaluating ML model fairness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score


class FairnessMetrics:
    """Comprehensive fairness metrics for ML models."""
    
    @staticmethod
    def demographic_parity(
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate demographic parity metrics.
        
        Demographic Parity (Statistical Parity): 
        P(Y_pred=1|A=0) = P(Y_pred=1|A=1)
        
        Args:
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute (binary)
            
        Returns:
            Dictionary with demographic parity metrics
        """
        groups = np.unique(sensitive_attr)
        positive_rates = {}
        
        for group in groups:
            mask = sensitive_attr == group
            positive_rates[f'group_{group}'] = np.mean(y_pred[mask])
        
        max_rate = max(positive_rates.values())
        min_rate = min(positive_rates.values())
        
        return {
            'positive_rates': positive_rates,
            'demographic_parity_difference': max_rate - min_rate,
            'demographic_parity_ratio': min_rate / max_rate if max_rate > 0 else 0
        }
    
    @staticmethod
    def equal_opportunity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate equal opportunity metrics.
        
        Equal Opportunity: TPR should be equal across groups
        TPR = P(Y_pred=1|Y_true=1, A=a)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute
            
        Returns:
            Dictionary with equal opportunity metrics
        """
        groups = np.unique(sensitive_attr)
        tpr_dict = {}
        
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.sum((y_pred == 1) & mask) / np.sum(mask)
                tpr_dict[f'tpr_group_{group}'] = tpr
            else:
                tpr_dict[f'tpr_group_{group}'] = 0.0
        
        tpr_values = list(tpr_dict.values())
        
        return {
            'true_positive_rates': tpr_dict,
            'equal_opportunity_difference': max(tpr_values) - min(tpr_values)
        }
    
    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate equalized odds metrics.
        
        Equalized Odds: Both TPR and FPR equal across groups
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute
            
        Returns:
            Dictionary with equalized odds metrics
        """
        groups = np.unique(sensitive_attr)
        tpr_dict = {}
        fpr_dict = {}
        
        for group in groups:
            group_mask = sensitive_attr == group
            
            # TPR
            pos_mask = group_mask & (y_true == 1)
            if np.sum(pos_mask) > 0:
                tpr = np.sum((y_pred == 1) & pos_mask) / np.sum(pos_mask)
            else:
                tpr = 0.0
            tpr_dict[f'tpr_group_{group}'] = tpr
            
            # FPR
            neg_mask = group_mask & (y_true == 0)
            if np.sum(neg_mask) > 0:
                fpr = np.sum((y_pred == 1) & neg_mask) / np.sum(neg_mask)
            else:
                fpr = 0.0
            fpr_dict[f'fpr_group_{group}'] = fpr
        
        tpr_values = list(tpr_dict.values())
        fpr_values = list(fpr_dict.values())
        
        return {
            'true_positive_rates': tpr_dict,
            'false_positive_rates': fpr_dict,
            'tpr_difference': max(tpr_values) - min(tpr_values),
            'fpr_difference': max(fpr_values) - min(fpr_values),
            'average_odds_difference': (max(tpr_values) - min(tpr_values) + 
                                       max(fpr_values) - min(fpr_values)) / 2
        }
    
    @staticmethod
    def predictive_parity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate predictive parity metrics.
        
        Predictive Parity (Outcome Test): 
        PPV (precision) should be equal across groups
        PPV = P(Y_true=1|Y_pred=1, A=a)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute
            
        Returns:
            Dictionary with predictive parity metrics
        """
        groups = np.unique(sensitive_attr)
        ppv_dict = {}
        
        for group in groups:
            mask = (sensitive_attr == group) & (y_pred == 1)
            if np.sum(mask) > 0:
                ppv = np.sum((y_true == 1) & mask) / np.sum(mask)
                ppv_dict[f'ppv_group_{group}'] = ppv
            else:
                ppv_dict[f'ppv_group_{group}'] = 0.0
        
        ppv_values = list(ppv_dict.values())
        
        return {
            'positive_predictive_values': ppv_dict,
            'predictive_parity_difference': max(ppv_values) - min(ppv_values)
        }
    
    @staticmethod
    def calibration_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_attr: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Dict]:
        """
        Calculate calibration metrics for each group.
        
        A well-calibrated model: among predictions with score s,
        approximately s fraction should be positive.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            sensitive_attr: Sensitive attribute
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration metrics per group
        """
        groups = np.unique(sensitive_attr)
        calibration_dict = {}
        
        for group in groups:
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_proba_group = y_pred_proba[mask]
            
            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_proba_group, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_true_prob = []
            bin_pred_prob = []
            bin_counts = []
            
            for i in range(n_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    bin_true_prob.append(np.mean(y_true_group[bin_mask]))
                    bin_pred_prob.append(np.mean(y_proba_group[bin_mask]))
                    bin_counts.append(np.sum(bin_mask))
                else:
                    bin_true_prob.append(0)
                    bin_pred_prob.append(0)
                    bin_counts.append(0)
            
            # Expected Calibration Error (ECE)
            ece = 0
            total_samples = len(y_true_group)
            for i in range(n_bins):
                if bin_counts[i] > 0:
                    ece += (bin_counts[i] / total_samples) * abs(bin_true_prob[i] - bin_pred_prob[i])
            
            calibration_dict[f'group_{group}'] = {
                'expected_calibration_error': ece,
                'bin_true_probabilities': bin_true_prob,
                'bin_predicted_probabilities': bin_pred_prob,
                'bin_counts': bin_counts
            }
        
        return calibration_dict
    
    @staticmethod
    def group_fairness_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate composite group fairness score.
        
        Combines multiple fairness metrics into a single score.
        Lower score indicates better fairness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute
            weights: Weights for different metrics (optional)
            
        Returns:
            Composite fairness score
        """
        if weights is None:
            weights = {
                'demographic_parity': 1.0,
                'equal_opportunity': 1.0,
                'equalized_odds': 1.0,
                'predictive_parity': 1.0
            }
        
        metrics = FairnessMetrics()
        
        # Get all metrics
        dp = metrics.demographic_parity(y_pred, sensitive_attr)
        eo = metrics.equal_opportunity(y_true, y_pred, sensitive_attr)
        eq = metrics.equalized_odds(y_true, y_pred, sensitive_attr)
        pp = metrics.predictive_parity(y_true, y_pred, sensitive_attr)
        
        # Calculate weighted score
        score = 0
        score += weights['demographic_parity'] * dp['demographic_parity_difference']
        score += weights['equal_opportunity'] * eo['equal_opportunity_difference']
        score += weights['equalized_odds'] * eq['average_odds_difference']
        score += weights['predictive_parity'] * pp['predictive_parity_difference']
        
        total_weight = sum(weights.values())
        return score / total_weight if total_weight > 0 else score


def comprehensive_fairness_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    sensitive_attrs: Dict[str, np.ndarray]
) -> str:
    """
    Perform comprehensive fairness evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        sensitive_attrs: Dictionary of sensitive attributes
        
    Returns:
        Formatted evaluation report
    """
    metrics = FairnessMetrics()
    report = []
    
    report.append("=" * 80)
    report.append("COMPREHENSIVE FAIRNESS EVALUATION")
    report.append("=" * 80)
    
    for attr_name, attr_values in sensitive_attrs.items():
        report.append(f"\n{'=' * 80}")
        report.append(f"SENSITIVE ATTRIBUTE: {attr_name.upper()}")
        report.append(f"{'=' * 80}\n")
        
        # Demographic Parity
        report.append("1. DEMOGRAPHIC PARITY")
        report.append("-" * 40)
        dp = metrics.demographic_parity(y_pred, attr_values)
        for key, value in dp.items():
            report.append(f"   {key}: {value}")
        report.append("")
        
        # Equal Opportunity
        report.append("2. EQUAL OPPORTUNITY")
        report.append("-" * 40)
        eo = metrics.equal_opportunity(y_true, y_pred, attr_values)
        for key, value in eo.items():
            report.append(f"   {key}: {value}")
        report.append("")
        
        # Equalized Odds
        report.append("3. EQUALIZED ODDS")
        report.append("-" * 40)
        eq = metrics.equalized_odds(y_true, y_pred, attr_values)
        for key, value in eq.items():
            report.append(f"   {key}: {value}")
        report.append("")
        
        # Predictive Parity
        report.append("4. PREDICTIVE PARITY")
        report.append("-" * 40)
        pp = metrics.predictive_parity(y_true, y_pred, attr_values)
        for key, value in pp.items():
            report.append(f"   {key}: {value}")
        report.append("")
        
        # Calibration (if probabilities provided)
        if y_pred_proba is not None:
            report.append("5. CALIBRATION")
            report.append("-" * 40)
            cal = metrics.calibration_metrics(y_true, y_pred_proba, attr_values)
            for group, cal_metrics in cal.items():
                report.append(f"   {group}: ECE = {cal_metrics['expected_calibration_error']:.4f}")
            report.append("")
        
        # Composite Score
        report.append("6. COMPOSITE FAIRNESS SCORE")
        report.append("-" * 40)
        score = metrics.group_fairness_score(y_true, y_pred, attr_values)
        report.append(f"   Score: {score:.4f} (lower is better)")
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n_samples = 1000
    gender = np.random.randint(0, 2, n_samples)
    y_true = np.random.randint(0, 2, n_samples)
    
    # Biased predictions
    y_pred = np.where(
        gender == 0,
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    )
    
    y_pred_proba = np.random.rand(n_samples)
    
    report = comprehensive_fairness_evaluation(
        y_true, y_pred, y_pred_proba,
        {'gender': gender}
    )
    
    print(report)

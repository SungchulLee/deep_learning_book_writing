"""
Regression Metrics
==================

Comprehensive coverage of metrics for evaluating regression models.

Metrics covered:
- MAE, MSE, RMSE
- R² Score
- Adjusted R²
- MAPE, SMAPE
- Residual analysis
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    max_error, explained_variance_score
)


class RegressionMetrics:
    """
    Comprehensive regression metrics calculator
    """
    
    def __init__(self, y_true, y_pred):
        """
        Initialize with true values and predictions
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.residuals = self.y_true - self.y_pred
    
    def mae(self):
        """
        Mean Absolute Error (MAE)
        
        Formula: (1/n) * Σ|y_true - y_pred|
        
        Properties:
        - Same units as target variable
        - Less sensitive to outliers than MSE
        - Easier to interpret
        
        Interpretation: Average absolute difference between predictions and actual
        """
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def mse(self):
        """
        Mean Squared Error (MSE)
        
        Formula: (1/n) * Σ(y_true - y_pred)²
        
        Properties:
        - Units are squared
        - Heavily penalizes large errors (quadratic)
        - Differentiable everywhere (good for optimization)
        
        Use when: Large errors are particularly undesirable
        """
        return mean_squared_error(self.y_true, self.y_pred)
    
    def rmse(self):
        """
        Root Mean Squared Error (RMSE)
        
        Formula: √MSE
        
        Properties:
        - Same units as target variable
        - More interpretable than MSE
        - Still penalizes large errors
        
        Interpretation: Standard deviation of prediction errors
        """
        return np.sqrt(self.mse())
    
    def r2(self):
        """
        R² Score (Coefficient of Determination)
        
        Formula: 1 - (SS_res / SS_tot)
        Where:
        - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
        - SS_tot = Σ(y_true - y_mean)² (total sum of squares)
        
        Range: -∞ to 1
        - 1.0: Perfect predictions
        - 0.0: Model as good as predicting mean
        - <0: Model worse than predicting mean
        
        Interpretation: Proportion of variance in target explained by model
        """
        return r2_score(self.y_true, self.y_pred)
    
    def adjusted_r2(self, n_features):
        """
        Adjusted R² Score
        
        Formula: 1 - [(1-R²)(n-1)/(n-p-1)]
        Where:
        - n = number of samples
        - p = number of features
        
        Properties:
        - Penalizes adding features that don't improve model
        - Better for comparing models with different numbers of features
        
        Args:
            n_features: Number of features in the model
        """
        n = len(self.y_true)
        r2 = self.r2()
        
        if n <= n_features + 1:
            return None  # Undefined
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2
    
    def mape(self):
        """
        Mean Absolute Percentage Error (MAPE)
        
        Formula: (100/n) * Σ|(y_true - y_pred) / y_true|
        
        Properties:
        - Scale-independent (percentage)
        - Easy to interpret
        - Undefined when y_true = 0
        - Asymmetric (penalizes positive errors more)
        
        Interpretation: Average percentage error
        """
        # Avoid division by zero
        mask = self.y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
    
    def smape(self):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE)
        
        Formula: (100/n) * Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
        
        Properties:
        - More symmetric than MAPE
        - Range: 0% to 200%
        - Better handles cases where actual is close to zero
        
        Interpretation: Symmetric average percentage error
        """
        numerator = np.abs(self.y_true - self.y_pred)
        denominator = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return 0
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    def median_absolute_error(self):
        """
        Median Absolute Error
        
        Properties:
        - Robust to outliers
        - Same units as target
        
        Use when: Dataset has outliers
        """
        return median_absolute_error(self.y_true, self.y_pred)
    
    def max_error_metric(self):
        """
        Maximum Residual Error
        
        Interpretation: Worst-case prediction error
        Use when: Need to bound worst-case performance
        """
        return max_error(self.y_true, self.y_pred)
    
    def explained_variance(self):
        """
        Explained Variance Score
        
        Range: 0 to 1
        Similar to R² but doesn't account for systematic offsets
        """
        return explained_variance_score(self.y_true, self.y_pred)
    
    def residual_analysis(self):
        """
        Analyze residuals (errors)
        
        Returns:
            Dictionary with residual statistics
        """
        return {
            'mean': np.mean(self.residuals),
            'std': np.std(self.residuals),
            'min': np.min(self.residuals),
            'max': np.max(self.residuals),
            'median': np.median(self.residuals),
            'q25': np.percentile(self.residuals, 25),
            'q75': np.percentile(self.residuals, 75)
        }
    
    def full_evaluation_report(self, n_features=None):
        """
        Generate complete evaluation report with all metrics
        
        Args:
            n_features: Number of features (for adjusted R²)
        """
        report = {
            'MAE': self.mae(),
            'MSE': self.mse(),
            'RMSE': self.rmse(),
            'R² Score': self.r2(),
            'Explained Variance': self.explained_variance(),
            'MAPE (%)': self.mape(),
            'SMAPE (%)': self.smape(),
            'Median Absolute Error': self.median_absolute_error(),
            'Max Error': self.max_error_metric()
        }
        
        if n_features is not None:
            adj_r2 = self.adjusted_r2(n_features)
            if adj_r2 is not None:
                report['Adjusted R²'] = adj_r2
        
        report['Residual Analysis'] = self.residual_analysis()
        
        return report


def metric_interpretation_guide():
    """
    Guide for interpreting regression metrics
    """
    guide = """
    REGRESSION METRIC INTERPRETATION GUIDE
    ======================================
    
    MAE (Mean Absolute Error):
        → Average error in same units as target
        → Easier to interpret, robust to outliers
        → Good for: General understanding of model accuracy
    
    RMSE (Root Mean Squared Error):
        → Standard deviation of errors
        → Penalizes large errors more than MAE
        → Good for: When large errors are particularly bad
    
    R² Score:
        → Proportion of variance explained (0 to 1)
        → 0.7-0.9: Good model
        → >0.9: Excellent (but check for overfitting!)
        → <0.3: Poor model
    
    MAPE/SMAPE:
        → Percentage error (scale-independent)
        → Good for: Comparing models on different datasets
        → <10%: Excellent
        → 10-20%: Good
        → 20-50%: Reasonable
        → >50%: Poor
    
    Adjusted R²:
        → Use for comparing models with different features
        → Penalizes unnecessary features
    
    CHOOSING METRICS:
    =================
    
    Standard Practice:
        → Report RMSE or MAE + R²
    
    Comparing Different Scales:
        → Use MAPE/SMAPE
    
    Business Context:
        → MAE (easier to explain)
    
    With Outliers:
        → Median Absolute Error
    
    Critical Applications:
        → Max Error (worst-case scenario)
    """
    print(guide)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("REGRESSION METRICS DEMONSTRATION")
    print("=" * 60)
    
    # Example: House price prediction
    print("\n1. HOUSE PRICE PREDICTION EXAMPLE")
    print("-" * 40)
    y_true = np.array([300000, 450000, 200000, 550000, 380000, 420000, 290000, 510000])
    y_pred = np.array([290000, 470000, 195000, 530000, 400000, 410000, 305000, 495000])
    
    metrics = RegressionMetrics(y_true, y_pred)
    report = metrics.full_evaluation_report(n_features=5)
    
    print("Target: House prices (in $)")
    print(f"\nTrue values:      {y_true}")
    print(f"Predicted values: {y_pred}")
    print("\nMetrics:")
    
    for metric_name, value in report.items():
        if metric_name != 'Residual Analysis':
            if isinstance(value, float):
                if 'R²' in metric_name or 'Variance' in metric_name:
                    print(f"  {metric_name}: {value:.4f}")
                elif '%' in metric_name:
                    print(f"  {metric_name}: {value:.2f}%")
                else:
                    print(f"  {metric_name}: ${value:,.2f}")
            else:
                print(f"  {metric_name}: {value}")
    
    print("\n  Residual Analysis:")
    for key, value in report['Residual Analysis'].items():
        print(f"    {key}: ${value:,.2f}")
    
    # Interpretation guide
    print("\n2. METRIC INTERPRETATION GUIDE")
    print("-" * 40)
    metric_interpretation_guide()

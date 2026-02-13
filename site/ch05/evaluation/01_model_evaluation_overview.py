"""
Model Evaluation and Metrics - Overview
========================================

This module provides an overview of essential concepts in model evaluation
and performance metrics for machine learning models.

Key Concepts:
- Training vs Testing vs Validation data
- Bias-Variance tradeoff
- Overfitting and Underfitting
- Cross-validation strategies
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ModelEvaluationOverview:
    """
    Demonstrates fundamental concepts in model evaluation
    """
    
    @staticmethod
    def train_test_split_example(X, y, test_size=0.2, random_state=42):
        """
        Splits data into training and testing sets
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of data for testing (default 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        print(f"Training/Test ratio: {len(X_train)/len(X_test):.2f}")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def demonstrate_overfitting():
        """
        Demonstrates the concept of overfitting with a simple example
        """
        # Generate synthetic data with noise
        np.random.seed(42)
        X = np.linspace(0, 10, 50)
        y_true = 2 * X + 1
        y_noisy = y_true + np.random.normal(0, 2, 50)
        
        # Simple model (good fit)
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Linear model
        linear_model = LinearRegression()
        linear_model.fit(X.reshape(-1, 1), y_noisy)
        
        # Overfit model (high degree polynomial)
        poly_features = PolynomialFeatures(degree=15)
        X_poly = poly_features.fit_transform(X.reshape(-1, 1))
        overfit_model = LinearRegression()
        overfit_model.fit(X_poly, y_noisy)
        
        print("Linear Model (Good Fit) - Coefficients:", linear_model.coef_)
        print("Number of features in overfit model:", X_poly.shape[1])
        
        return {
            'X': X,
            'y_true': y_true,
            'y_noisy': y_noisy,
            'linear_model': linear_model,
            'overfit_model': overfit_model,
            'poly_features': poly_features
        }


def bias_variance_tradeoff_explanation():
    """
    Explains the bias-variance tradeoff
    """
    explanation = """
    BIAS-VARIANCE TRADEOFF
    ======================
    
    Bias:
    - Error from overly simplistic assumptions in the learning algorithm
    - High bias → underfitting
    - Model misses relevant relations between features and target
    
    Variance:
    - Error from sensitivity to small fluctuations in training set
    - High variance → overfitting
    - Model captures noise along with underlying pattern
    
    Goal: Find the sweet spot that minimizes BOTH bias and variance
    
    Total Error = Bias² + Variance + Irreducible Error
    """
    print(explanation)


def holdout_vs_cross_validation():
    """
    Compares holdout validation with cross-validation
    """
    comparison = """
    HOLDOUT VALIDATION vs CROSS-VALIDATION
    =======================================
    
    Holdout Validation:
    - Split data once: Train/Test (or Train/Val/Test)
    - Pros: Fast, simple
    - Cons: High variance in performance estimate, wastes data
    
    Cross-Validation:
    - Split data into K folds, train on K-1, test on 1, repeat K times
    - Pros: Better use of data, more reliable performance estimate
    - Cons: Computationally expensive (K times slower)
    
    Best Practice:
    - Use holdout for large datasets or when time-constrained
    - Use cross-validation for smaller datasets or final model evaluation
    """
    print(comparison)


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION AND METRICS - OVERVIEW")
    print("=" * 60)
    
    # Demonstrate train-test split
    print("\n1. TRAIN-TEST SPLIT EXAMPLE")
    print("-" * 40)
    X_sample = np.random.randn(1000, 10)
    y_sample = np.random.randint(0, 2, 1000)
    ModelEvaluationOverview.train_test_split_example(X_sample, y_sample)
    
    # Demonstrate overfitting
    print("\n2. OVERFITTING DEMONSTRATION")
    print("-" * 40)
    ModelEvaluationOverview.demonstrate_overfitting()
    
    # Explain bias-variance tradeoff
    print("\n3. BIAS-VARIANCE TRADEOFF")
    print("-" * 40)
    bias_variance_tradeoff_explanation()
    
    # Compare validation strategies
    print("\n4. VALIDATION STRATEGIES")
    print("-" * 40)
    holdout_vs_cross_validation()

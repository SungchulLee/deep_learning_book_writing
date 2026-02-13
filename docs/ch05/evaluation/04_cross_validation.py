"""
Cross-Validation Techniques
============================

Comprehensive coverage of cross-validation strategies for model evaluation.

Techniques covered:
- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out (LOO)
- Time Series Split
- Group K-Fold
- Repeated K-Fold
"""

import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut,
    TimeSeriesSplit, GroupKFold, RepeatedKFold,
    cross_val_score, cross_validate
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression


class CrossValidationDemo:
    """
    Demonstrates various cross-validation techniques
    """
    
    @staticmethod
    def kfold_cv(X, y, model, n_splits=5, scoring='accuracy'):
        """
        Standard K-Fold Cross-Validation
        
        Process:
        1. Split data into K equal folds
        2. For each fold:
           - Train on K-1 folds
           - Test on remaining fold
        3. Average performance across all folds
        
        Pros:
        - Simple, widely used
        - Good use of data
        
        Cons:
        - May not preserve class distribution (for classification)
        - High variance with small datasets
        
        Args:
            X: Features
            y: Target
            model: Sklearn model
            n_splits: Number of folds (default 5)
            scoring: Metric to evaluate
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        
        print(f"\nK-Fold Cross-Validation (K={n_splits})")
        print(f"Scores for each fold: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Std deviation: {scores.std():.4f}")
        print(f"95% Confidence Interval: [{scores.mean() - 1.96*scores.std():.4f}, "
              f"{scores.mean() + 1.96*scores.std():.4f}]")
        
        return scores
    
    @staticmethod
    def stratified_kfold_cv(X, y, model, n_splits=5, scoring='accuracy'):
        """
        Stratified K-Fold Cross-Validation
        
        Process:
        - Similar to K-Fold BUT preserves class distribution in each fold
        - Ensures each fold has approximately same percentage of samples of each class
        
        Use when:
        - Classification tasks
        - Imbalanced datasets
        - You want consistent class representation across folds
        
        Pros:
        - Better for imbalanced datasets
        - More reliable estimates for classification
        
        Args:
            X: Features
            y: Target (must be classification labels)
            model: Sklearn classifier
            n_splits: Number of folds
            scoring: Metric to evaluate
        """
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skfold, scoring=scoring)
        
        print(f"\nStratified K-Fold Cross-Validation (K={n_splits})")
        print(f"Scores for each fold: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Std deviation: {scores.std():.4f}")
        
        # Check class distribution
        print("\nClass distribution in original data:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")
        
        return scores
    
    @staticmethod
    def leave_one_out_cv(X, y, model, scoring='accuracy'):
        """
        Leave-One-Out Cross-Validation (LOO)
        
        Process:
        - Special case of K-Fold where K = n (number of samples)
        - Each sample is used once as test set
        - Train on n-1 samples, test on 1 sample, repeat n times
        
        Pros:
        - Maximum use of training data
        - Deterministic (no randomness)
        - Low bias
        
        Cons:
        - Computationally expensive (n iterations!)
        - High variance
        - Not suitable for large datasets
        
        Use when:
        - Very small datasets (n < 100)
        - You can afford computation time
        
        Args:
            X: Features
            y: Target
            model: Sklearn model
            scoring: Metric to evaluate
        """
        if len(X) > 100:
            print("\nWarning: LOO is computationally expensive for large datasets!")
            print(f"This will perform {len(X)} iterations.")
        
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)
        
        print(f"\nLeave-One-Out Cross-Validation")
        print(f"Number of iterations: {len(scores)}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Std deviation: {scores.std():.4f}")
        
        return scores
    
    @staticmethod
    def time_series_split_cv(X, y, model, n_splits=5, scoring='neg_mean_squared_error'):
        """
        Time Series Split Cross-Validation
        
        Process:
        - Preserves temporal order (NO shuffling!)
        - Growing training set, rolling test set
        - Fold 1: Train on [0:n], test on [n:2n]
        - Fold 2: Train on [0:2n], test on [2n:3n]
        - etc.
        
        Use when:
        - Time series data
        - Temporal dependencies matter
        - Future should not influence past
        
        CRITICAL: Data must be ordered chronologically!
        
        Args:
            X: Features (ordered chronologically)
            y: Target (ordered chronologically)
            model: Sklearn model
            n_splits: Number of splits
            scoring: Metric to evaluate
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
        
        print(f"\nTime Series Split Cross-Validation (n_splits={n_splits})")
        print(f"Scores for each split: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Std deviation: {scores.std():.4f}")
        
        print("\nSplit details:")
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
        
        return scores
    
    @staticmethod
    def group_kfold_cv(X, y, groups, model, n_splits=5, scoring='accuracy'):
        """
        Group K-Fold Cross-Validation
        
        Process:
        - Ensures samples from same group don't appear in both train and test
        - Splits based on groups, not individual samples
        
        Use when:
        - Data has natural groupings (e.g., patients, companies, experiments)
        - Samples within groups are not independent
        - You want to test generalization to new groups
        
        Example: Medical data from multiple hospitals
        - Don't want same patient in train and test
        - Want to test generalization to new hospitals
        
        Args:
            X: Features
            y: Target
            groups: Array of group labels for each sample
            model: Sklearn model
            n_splits: Number of splits
            scoring: Metric to evaluate
        """
        gkfold = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(model, X, y, groups=groups, cv=gkfold, scoring=scoring)
        
        print(f"\nGroup K-Fold Cross-Validation (K={n_splits})")
        print(f"Number of unique groups: {len(np.unique(groups))}")
        print(f"Scores for each fold: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Std deviation: {scores.std():.4f}")
        
        return scores
    
    @staticmethod
    def repeated_kfold_cv(X, y, model, n_splits=5, n_repeats=3, scoring='accuracy'):
        """
        Repeated K-Fold Cross-Validation
        
        Process:
        - Performs K-Fold CV multiple times with different random splits
        - Reduces variance in performance estimate
        
        Pros:
        - More robust estimate
        - Reduces impact of specific split
        
        Cons:
        - More computationally expensive (K × n_repeats iterations)
        
        Use when:
        - Need more reliable estimate
        - Dataset size allows
        
        Args:
            X: Features
            y: Target
            model: Sklearn model
            n_splits: Number of folds per repeat
            n_repeats: Number of times to repeat K-Fold
            scoring: Metric to evaluate
        """
        rkfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        scores = cross_val_score(model, X, y, cv=rkfold, scoring=scoring)
        
        print(f"\nRepeated K-Fold Cross-Validation")
        print(f"K={n_splits}, Repeats={n_repeats}, Total iterations={n_splits*n_repeats}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Std deviation: {scores.std():.4f}")
        print(f"Min score: {scores.min():.4f}")
        print(f"Max score: {scores.max():.4f}")
        
        return scores
    
    @staticmethod
    def cross_validate_detailed(X, y, model, cv=5, scoring=None):
        """
        Detailed cross-validation with multiple metrics and timing
        
        Returns training scores, test scores, and fit/score times
        
        Args:
            X: Features
            y: Target
            model: Sklearn model
            cv: CV strategy or number of folds
            scoring: Dictionary of metrics or single metric
        """
        results = cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            return_train_score=True,
            return_estimator=False
        )
        
        print("\nDetailed Cross-Validation Results")
        print("-" * 40)
        
        for key, values in results.items():
            if key.startswith('test_') or key.startswith('train_'):
                metric_name = key.replace('test_', '').replace('train_', '')
                print(f"{key}:")
                print(f"  Mean: {values.mean():.4f}")
                print(f"  Std: {values.std():.4f}")
            elif 'time' in key:
                print(f"{key}: {values.mean():.4f}s (avg)")
        
        return results


def cv_strategy_selection_guide():
    """
    Guide for selecting appropriate cross-validation strategy
    """
    guide = """
    CROSS-VALIDATION STRATEGY SELECTION GUIDE
    =========================================
    
    Classification (Balanced):
        → K-Fold (K=5 or 10)
    
    Classification (Imbalanced):
        → Stratified K-Fold (K=5 or 10)
        → REQUIRED for reliable estimates!
    
    Regression:
        → K-Fold (K=5 or 10)
        → Stratified K-Fold with binned targets (advanced)
    
    Small Dataset (n < 100):
        → Leave-One-Out (if computationally feasible)
        → K-Fold with K=10 or even K=n
    
    Large Dataset (n > 10,000):
        → K-Fold with K=3 or 5 (faster)
        → Single train-test split may suffice
    
    Time Series:
        → Time Series Split
        → NEVER use regular K-Fold (violates temporal order!)
    
    Grouped Data (e.g., multiple samples per patient):
        → Group K-Fold
        → Ensures no data leakage between train/test
    
    Need Robust Estimate:
        → Repeated K-Fold (K=5, repeats=3-10)
        → Stratified if classification
    
    GENERAL BEST PRACTICES:
    =======================
    
    Default Choice:
        → Stratified K-Fold (K=5) for classification
        → K-Fold (K=5) for regression
    
    When to use K=5 vs K=10:
        → K=5: Faster, good for most cases
        → K=10: More stable estimates, use if time allows
    
    Always:
        → Set random_state for reproducibility
        → Use shuffle=True (except for time series!)
        → Report mean AND std of scores
    
    Never:
        → Use regular K-Fold for imbalanced classification
        → Shuffle time series data
        → Include test set in any fold
    """
    print(guide)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-VALIDATION TECHNIQUES DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample classification data
    X_clf, y_clf = make_classification(
        n_samples=500, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    
    # Generate sample regression data
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=10, random_state=42
    )
    
    # Models
    clf_model = LogisticRegression(max_iter=1000, random_state=42)
    reg_model = LinearRegression()
    
    demo = CrossValidationDemo()
    
    # 1. K-Fold
    print("\n" + "=" * 60)
    print("1. K-FOLD CROSS-VALIDATION (Classification)")
    print("=" * 60)
    demo.kfold_cv(X_clf, y_clf, clf_model, n_splits=5)
    
    # 2. Stratified K-Fold
    print("\n" + "=" * 60)
    print("2. STRATIFIED K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    demo.stratified_kfold_cv(X_clf, y_clf, clf_model, n_splits=5)
    
    # 3. Time Series Split
    print("\n" + "=" * 60)
    print("3. TIME SERIES SPLIT (Regression)")
    print("=" * 60)
    demo.time_series_split_cv(X_reg, y_reg, reg_model, n_splits=5)
    
    # 4. Strategy Selection Guide
    print("\n" + "=" * 60)
    print("4. STRATEGY SELECTION GUIDE")
    print("=" * 60)
    cv_strategy_selection_guide()

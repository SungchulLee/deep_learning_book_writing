"""
Model Comparison and Selection
===============================

Techniques for comparing and selecting the best model.

Topics covered:
- Comparing multiple models
- Statistical significance tests
- Learning curves
- Validation curves
- Model selection strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    cross_val_score, learning_curve, validation_curve
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy import stats


class ModelComparison:
    """
    Tools for comparing multiple models
    """
    
    @staticmethod
    def compare_models_cv(models_dict, X, y, cv=5, scoring='accuracy'):
        """
        Compare multiple models using cross-validation
        
        Args:
            models_dict: Dictionary of {model_name: model_object}
            X: Features
            y: Target
            cv: Cross-validation strategy
            scoring: Metric to use
        
        Returns:
            Dictionary with results for each model
        """
        results = {}
        
        print("=" * 70)
        print("MODEL COMPARISON USING CROSS-VALIDATION")
        print("=" * 70)
        print(f"\nCross-Validation: {cv}-fold")
        print(f"Scoring Metric: {scoring}")
        print("\n" + "-" * 70)
        
        for name, model in models_dict.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            
            print(f"\n{name}:")
            print(f"  Mean Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            print(f"  Score Range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Individual Scores: {[f'{s:.4f}' for s in scores]}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['mean'])
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best_model[0]}")
        print(f"Mean Score: {best_model[1]['mean']:.4f}")
        print("=" * 70)
        
        return results
    
    @staticmethod
    def paired_ttest(scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """
        Paired t-test to compare two models statistically
        
        Use when: You have paired cross-validation scores from two models
        
        H0 (null hypothesis): The two models have the same performance
        H1 (alternative): The models have different performance
        
        Args:
            scores1: CV scores from model 1
            scores2: CV scores from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
        
        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        print("\n" + "=" * 70)
        print("PAIRED T-TEST FOR MODEL COMPARISON")
        print("=" * 70)
        
        print(f"\nComparing: {model1_name} vs {model2_name}")
        print(f"\n{model1_name} scores: {scores1}")
        print(f"{model2_name} scores: {scores2}")
        
        print(f"\nMean difference: {np.mean(scores1 - scores2):.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"\nConclusion (α={alpha}):")
            print(f"  ✓ Statistically significant difference (p < {alpha})")
            if np.mean(scores1) > np.mean(scores2):
                print(f"  → {model1_name} performs significantly better")
            else:
                print(f"  → {model2_name} performs significantly better")
        else:
            print(f"\nConclusion (α={alpha}):")
            print(f"  ✗ No statistically significant difference (p >= {alpha})")
            print(f"  → Cannot conclude that one model is better")
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_difference': np.mean(scores1 - scores2)
        }
    
    @staticmethod
    def plot_learning_curve(model, X, y, cv=5, scoring='accuracy',
                           train_sizes=np.linspace(0.1, 1.0, 10),
                           title="Learning Curve"):
        """
        Plot learning curve to diagnose bias/variance
        
        Learning curves show:
        - How training and validation scores change with training set size
        - Whether model suffers from high bias or high variance
        
        Interpretation:
        - High bias (underfitting):
            → Training and validation scores are both low
            → Scores plateau early
            → Small gap between curves
            → Solution: More complex model, more features
        
        - High variance (overfitting):
            → Training score is high, validation score is low
            → Large gap between curves
            → Validation score may improve with more data
            → Solution: More data, regularization, simpler model
        
        - Good fit:
            → Both scores are high
            → Small gap between curves
            → Scores converge
        
        Args:
            model: Sklearn model
            X: Features
            y: Target
            cv: Cross-validation strategy
            scoring: Metric to use
            train_sizes: Array of training sizes to use
            title: Plot title
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring,
            train_sizes=train_sizes, n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mean, label='Training score', 
                color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color='blue')
        
        plt.plot(train_sizes, val_mean, label='Validation score',
                color='red', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.15, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring.replace("_", " ").title()}')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Diagnosis
        final_gap = train_mean[-1] - val_mean[-1]
        
        print("\n" + "=" * 70)
        print("LEARNING CURVE DIAGNOSIS")
        print("=" * 70)
        
        print(f"\nFinal training score: {train_mean[-1]:.4f}")
        print(f"Final validation score: {val_mean[-1]:.4f}")
        print(f"Gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print("\n⚠ HIGH VARIANCE (Overfitting)")
            print("  → Large gap between training and validation scores")
            print("  → Consider: More data, regularization, simpler model")
        elif val_mean[-1] < 0.6:
            print("\n⚠ HIGH BIAS (Underfitting)")
            print("  → Both scores are low")
            print("  → Consider: More complex model, more features")
        else:
            print("\n✓ GOOD FIT")
            print("  → Scores are high and close together")
        
        return plt.gcf()
    
    @staticmethod
    def plot_validation_curve(model, X, y, param_name, param_range,
                             cv=5, scoring='accuracy'):
        """
        Plot validation curve for hyperparameter tuning
        
        Shows how model performance changes with a hyperparameter value
        
        Interpretation:
        - Helps find optimal hyperparameter value
        - Shows where model starts to overfit or underfit
        
        Args:
            model: Sklearn model
            X: Features
            y: Target
            param_name: Name of parameter to vary
            param_range: Array of parameter values to try
            cv: Cross-validation strategy
            scoring: Metric to use
        """
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(param_range, train_mean, label='Training score',
                color='blue', marker='o')
        plt.fill_between(param_range, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color='blue')
        
        plt.plot(param_range, val_mean, label='Validation score',
                color='red', marker='s')
        plt.fill_between(param_range, val_mean - val_std,
                        val_mean + val_std, alpha=0.15, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel(f'{scoring.replace("_", " ").title()}')
        plt.title(f'Validation Curve - {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Find optimal value
        optimal_idx = np.argmax(val_mean)
        optimal_value = param_range[optimal_idx]
        optimal_score = val_mean[optimal_idx]
        
        plt.axvline(optimal_value, color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal: {optimal_value}')
        plt.legend(loc='best')
        
        print("\n" + "=" * 70)
        print("VALIDATION CURVE ANALYSIS")
        print("=" * 70)
        
        print(f"\nParameter: {param_name}")
        print(f"Optimal value: {optimal_value}")
        print(f"Validation score at optimal: {optimal_score:.4f}")
        
        return plt.gcf()


def model_selection_strategy_guide():
    """
    Comprehensive guide for model selection
    """
    guide = """
    MODEL SELECTION STRATEGY GUIDE
    ==============================
    
    STEP 1: DEFINE THE PROBLEM
    --------------------------
    □ Classification or Regression?
    □ What metric matters most? (accuracy, precision, recall, etc.)
    □ What are the costs of different types of errors?
    □ Are there interpretability requirements?
    □ Are there speed/resource constraints?
    
    STEP 2: BASELINE MODEL
    ---------------------
    Always start with a simple baseline:
    → Classification: Logistic Regression, Decision Tree
    → Regression: Linear Regression, Ridge
    
    Why? Establishes minimum acceptable performance
    
    STEP 3: TRY MULTIPLE MODELS
    ---------------------------
    Try a diverse set of model types:
    → Linear: Logistic/Linear Regression, SVM
    → Tree-based: Decision Trees, Random Forest, XGBoost
    → Instance-based: KNN
    → Neural: Neural Networks (if data size permits)
    
    STEP 4: CROSS-VALIDATION
    ------------------------
    Use cross-validation to evaluate each model:
    → K=5 or 10 for most cases
    → Stratified K-Fold for classification
    → Time Series Split for time series
    
    STEP 5: STATISTICAL COMPARISON
    ------------------------------
    Use paired t-test to compare top models:
    → Are differences statistically significant?
    → Don't just pick highest mean score
    
    STEP 6: LEARNING CURVES
    -----------------------
    For top 2-3 models, plot learning curves:
    → Check for overfitting/underfitting
    → Estimate if more data would help
    
    STEP 7: HYPERPARAMETER TUNING
    -----------------------------
    For best model(s), tune hyperparameters:
    → Use Grid Search or Random Search
    → Use validation curves to guide search
    → Be careful of overfitting to validation set!
    
    STEP 8: FINAL EVALUATION
    ------------------------
    Evaluate final model on held-out test set:
    → Test set should NEVER be used during model selection
    → This gives unbiased estimate of performance
    
    COMMON PITFALLS TO AVOID:
    ========================
    
    ✗ Selecting model based on single train-test split
       → Use cross-validation instead
    
    ✗ Using test set for model selection
       → Test set is for final evaluation only
    
    ✗ Not checking for data leakage
       → Ensure features don't contain information from future
    
    ✗ Ignoring computational constraints
       → Consider training and prediction time
    
    ✗ Focusing only on overall accuracy
       → Check confusion matrix, per-class performance
    
    ✗ Not validating assumptions
       → E.g., linear models assume linear relationships
    
    ✗ Overfitting to validation set through excessive tuning
       → Nested cross-validation can help
    
    DECISION CRITERIA:
    =================
    
    Choose simpler model if:
    → Performance difference is not statistically significant
    → Interpretability is important
    → Faster predictions needed
    → Less data for training
    
    Choose complex model if:
    → Clear performance improvement
    → Black-box acceptable
    → Sufficient computational resources
    → Large dataset available
    """
    print(guide)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("=" * 70)
    print("MODEL COMPARISON AND SELECTION DEMONSTRATION")
    print("=" * 70)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    # Compare models
    comparison = ModelComparison()
    results = comparison.compare_models_cv(models, X, y, cv=5)
    
    # Statistical comparison of top two models
    print("\n\n" + "=" * 70)
    print("STATISTICAL COMPARISON OF TOP TWO MODELS")
    print("=" * 70)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    top_two = sorted_models[:2]
    
    comparison.paired_ttest(
        top_two[0][1]['scores'],
        top_two[1][1]['scores'],
        top_two[0][0],
        top_two[1][0]
    )
    
    # Strategy guide
    print("\n\n" + "=" * 70)
    print("MODEL SELECTION STRATEGY GUIDE")
    print("=" * 70)
    model_selection_strategy_guide()
    
    print("\n" + "=" * 70)
    print("Note: Run with matplotlib backend to see learning/validation curves")
    print("=" * 70)

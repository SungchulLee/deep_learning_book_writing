# Model Evaluation and Metrics - Comprehensive Python Package

A comprehensive collection of Python files covering essential concepts, metrics, and techniques for evaluating machine learning models.

## 📦 Package Contents

### Core Modules

1. **01_model_evaluation_overview.py**
   - Fundamental concepts in model evaluation
   - Train-test split strategies
   - Overfitting and underfitting demonstrations
   - Bias-variance tradeoff
   - Validation strategies comparison

2. **02_classification_metrics.py**
   - Comprehensive classification metrics
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC, PR-AUC
   - Matthews Correlation Coefficient
   - Cohen's Kappa
   - Log Loss
   - Metric selection guide

3. **03_regression_metrics.py**
   - Complete regression metrics suite
   - MAE, MSE, RMSE
   - R² Score and Adjusted R²
   - MAPE, SMAPE
   - Residual analysis
   - Metric interpretation guide

4. **04_cross_validation.py**
   - Various cross-validation techniques
   - K-Fold Cross-Validation
   - Stratified K-Fold
   - Leave-One-Out (LOO)
   - Time Series Split
   - Group K-Fold
   - Repeated K-Fold
   - Strategy selection guide

5. **05_confusion_matrix.py**
   - Confusion matrix analysis and visualization
   - Binary and multi-class confusion matrices
   - Normalized confusion matrices
   - Deriving metrics from confusion matrices
   - Visualization techniques
   - Interpretation guide

6. **06_model_comparison.py**
   - Model comparison techniques
   - Cross-validation comparison
   - Statistical significance testing (paired t-test)
   - Learning curves (bias-variance diagnosis)
   - Validation curves (hyperparameter tuning)
   - Model selection strategy guide

7. **07_comprehensive_examples.py**
   - End-to-end real-world examples
   - Binary Classification: Credit Card Fraud Detection
   - Multi-class Classification: Iris Species
   - Regression: House Price Prediction
   - Complete evaluation workflows

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scikit-learn matplotlib seaborn scipy pandas
```

### Running the Modules

Each module can be run independently:

```bash
# Overview of model evaluation
python 01_model_evaluation_overview.py

# Classification metrics demonstration
python 02_classification_metrics.py

# Regression metrics demonstration
python 03_regression_metrics.py

# Cross-validation techniques
python 04_cross_validation.py

# Confusion matrix analysis
python 05_confusion_matrix.py

# Model comparison and selection
python 06_model_comparison.py

# Comprehensive end-to-end examples
python 07_comprehensive_examples.py
```

## 📚 Usage Examples

### Example 1: Evaluating a Binary Classifier

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import the classification metrics module
import sys
sys.path.append('path/to/modules')
from classification_metrics import ClassificationMetrics

# Generate data
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Get comprehensive metrics
metrics = ClassificationMetrics(y_test, y_pred, y_pred_proba)
report = metrics.full_evaluation_report()

for metric, value in report.items():
    print(f"{metric}: {value}")
```

### Example 2: Comparing Multiple Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import model comparison module
from model_comparison import ModelComparison

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Compare using cross-validation
comparison = ModelComparison()
results = comparison.compare_models_cv(models, X_train, y_train, cv=5)
```

### Example 3: Cross-Validation with Stratification

```python
from cross_validation import CrossValidationDemo

demo = CrossValidationDemo()

# For classification with imbalanced data
demo.stratified_kfold_cv(X, y, model, n_splits=5)

# For time series data
demo.time_series_split_cv(X_time_series, y_time_series, model, n_splits=5)
```

## 🎯 Key Features

### Classification Metrics
- ✅ Binary and multi-class support
- ✅ Comprehensive metric suite
- ✅ ROC and Precision-Recall curves
- ✅ Metric selection guidance
- ✅ Handles imbalanced datasets

### Regression Metrics
- ✅ All standard regression metrics
- ✅ Scale-dependent and scale-independent metrics
- ✅ Residual analysis
- ✅ Interpretation guidelines
- ✅ Outlier-robust metrics

### Cross-Validation
- ✅ Multiple CV strategies
- ✅ Time series support
- ✅ Group-aware splitting
- ✅ Stratification for classification
- ✅ Strategy selection guide

### Visualization
- ✅ Confusion matrix heatmaps
- ✅ Learning curves
- ✅ Validation curves
- ✅ Multiple normalization options
- ✅ Professional plotting

### Model Comparison
- ✅ Statistical significance testing
- ✅ Cross-validation comparison
- ✅ Bias-variance diagnosis
- ✅ Hyperparameter tuning support
- ✅ Comprehensive selection guide

## 📖 Documentation

Each module includes:
- Detailed docstrings for all functions and classes
- Usage examples in the `if __name__ == "__main__"` block
- Interpretation guides for metrics
- Best practices and common pitfalls
- Real-world examples

## 🎓 Learning Path

Recommended order for learning:

1. **Start with**: `01_model_evaluation_overview.py`
   - Understand fundamental concepts
   - Learn about overfitting and underfitting
   - Understand train-test split

2. **Then**: `02_classification_metrics.py` or `03_regression_metrics.py`
   - Depending on your problem type
   - Learn about appropriate metrics
   - Understand when to use each metric

3. **Next**: `04_cross_validation.py`
   - Learn robust evaluation techniques
   - Understand different CV strategies
   - Apply appropriate strategy to your problem

4. **After that**: `05_confusion_matrix.py`
   - Deep dive into classification errors
   - Visualize model performance
   - Understand error patterns

5. **Then**: `06_model_comparison.py`
   - Compare multiple models systematically
   - Use statistical tests
   - Diagnose bias-variance issues

6. **Finally**: `07_comprehensive_examples.py`
   - See everything put together
   - Learn end-to-end workflows
   - Apply to real-world scenarios

## 🔍 Best Practices

### For Classification:
1. Always use stratified splitting for imbalanced data
2. Don't rely solely on accuracy for imbalanced datasets
3. Choose metrics based on business context (false positives vs false negatives)
4. Always examine the confusion matrix
5. Use ROC-AUC for comparing models, but check PR-AUC for imbalanced data

### For Regression:
1. Report multiple metrics (MAE, RMSE, R²)
2. Use MAPE/SMAPE for scale-independent comparison
3. Always perform residual analysis
4. Check for outliers using max error and median absolute error
5. Use adjusted R² when comparing models with different features

### For Cross-Validation:
1. Use stratified K-fold for classification
2. Never shuffle time series data
3. Use group K-fold when samples are not independent
4. Set random_state for reproducibility
5. Report both mean and standard deviation of scores

### For Model Selection:
1. Start with simple baseline models
2. Use cross-validation, not single train-test split
3. Apply statistical tests when comparing models
4. Check learning curves for bias-variance issues
5. Never use test set for model selection

## ⚠️ Common Pitfalls to Avoid

1. **Using accuracy alone for imbalanced datasets**
   - Solution: Use precision, recall, F1, or ROC-AUC

2. **Not using stratification for classification**
   - Solution: Always use StratifiedKFold

3. **Shuffling time series data**
   - Solution: Use TimeSeriesSplit

4. **Using test set for model selection**
   - Solution: Use validation set or cross-validation

5. **Ignoring class imbalance**
   - Solution: Use stratified sampling and appropriate metrics

6. **Overfitting to validation set**
   - Solution: Use nested cross-validation for hyperparameter tuning

7. **Not checking for data leakage**
   - Solution: Carefully review feature engineering pipeline

## 🔧 Customization

All modules are designed to be modular and extensible:

- Add custom metrics by extending the metric classes
- Implement new cross-validation strategies
- Create custom visualizations using the base plotting functions
- Combine modules for your specific workflow

## 📊 Example Output

Running the comprehensive examples produces detailed output including:

- Cross-validation scores with confidence intervals
- Detailed confusion matrices
- Per-class performance metrics
- Statistical significance tests
- Visualizations (when matplotlib is available)
- Business interpretations

## 🤝 Contributing

These modules are designed to be educational and comprehensive. Feel free to:
- Add more metrics
- Implement additional visualization techniques
- Add more real-world examples
- Improve documentation

## 📝 License

This code is provided for educational purposes. Feel free to use and modify as needed.

## 📧 Questions?

Each module includes extensive documentation and examples. Read through the docstrings and run the examples to understand the functionality.

## 🎯 Summary

This package provides everything you need to properly evaluate machine learning models:

✅ Comprehensive metrics for classification and regression
✅ Multiple cross-validation strategies
✅ Statistical model comparison
✅ Visualization tools
✅ Best practices and guidelines
✅ Real-world examples
✅ Complete end-to-end workflows

Happy evaluating! 🚀

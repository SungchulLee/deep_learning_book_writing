# Overfitting, Underfitting, and Bias-Variance Tradeoff

A comprehensive collection of Python scripts demonstrating machine learning concepts related to model fitting, bias-variance tradeoff, and learning curves.

## 📚 Contents

### 1. `overfitting_underfitting_demo.py`
**Main Topics**: Overfitting and Underfitting

Demonstrates the concepts of overfitting and underfitting using polynomial regression with different complexity levels.

**Key Features**:
- Visual comparison of models with different polynomial degrees
- Training and test error analysis
- Clear identification of underfitting, good fit, and overfitting regions
- Synthetic data generation with controllable noise

**Run**: `python overfitting_underfitting_demo.py`

**Output**: 
- Console metrics for each model
- Visualization showing 6 different model complexities
- Image file: `overfitting_underfitting.png`

---

### 2. `bias_variance_tradeoff.py`
**Main Topics**: Bias-Variance Decomposition

Deep dive into the bias-variance tradeoff with empirical decomposition of prediction error.

**Key Features**:
- Bootstrap sampling to estimate bias and variance
- Decomposition: Expected Error = Bias² + Variance + Irreducible Error
- Analysis across different model complexities
- Visualization of how bias decreases and variance increases with complexity

**Run**: `python bias_variance_tradeoff.py`

**Output**:
- Detailed bias-variance analysis table
- Multiple visualizations showing:
  - Bias-variance tradeoff curve
  - Stacked error components
  - Individual model predictions showing variability
- Images: `bias_variance_tradeoff.png`, `bias_variance_predictions.png`

---

### 3. `learning_curves_demo.py`
**Main Topics**: Learning Curves and Model Diagnosis

Demonstrates how to use learning curves to diagnose overfitting and underfitting problems.

**Key Features**:
- Learning curves for various model types
- Automatic diagnosis of underfitting vs overfitting
- Comparison of 6 different models
- Practical interpretation guidelines

**Run**: `python learning_curves_demo.py`

**Output**:
- Learning curves for individual models
- Comprehensive comparison plot
- Diagnosis and recommendations for each model
- Images: `learning_curve_underfitting.png`, `learning_curve_overfitting.png`, 
  `learning_curve_good_fit.png`, `learning_curves_comparison.png`

---

### 4. `practical_solutions.py`
**Main Topics**: Solutions and Best Practices

Practical techniques to address overfitting and underfitting problems.

**Key Features**:
- **Regularization**: Ridge (L2), Lasso (L1), ElasticNet
- **Validation Curves**: Hyperparameter tuning
- **Cross-Validation**: Model selection and evaluation
- **Feature Engineering**: Impact analysis

**Run**: `python practical_solutions.py`

**Output**:
- Regularization comparison across 7 techniques
- Optimal hyperparameter identification
- Cross-validation results
- Feature engineering impact analysis
- Images: `regularization_comparison.png`, `validation_curve_ridge.png`, 
  `cross_validation_comparison.png`

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
- numpy
- matplotlib
- scikit-learn

### Quick Start
Run any script independently:
```bash
python overfitting_underfitting_demo.py
python bias_variance_tradeoff.py
python learning_curves_demo.py
python practical_solutions.py
```

---

## 📊 Key Concepts Covered

### Underfitting (High Bias)
**Symptoms**:
- High training error
- High test error
- Small gap between training and test errors
- Learning curves plateau at high error

**Solutions**:
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer

### Overfitting (High Variance)
**Symptoms**:
- Low training error
- High test error
- Large gap between training and test errors
- Model performs well on training data but poorly on new data

**Solutions**:
- Get more training data
- Reduce model complexity
- Add regularization (L1/L2)
- Use cross-validation
- Feature selection
- Ensemble methods

### Bias-Variance Tradeoff
```
Expected Error = Bias² + Variance + Irreducible Error
```

- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Goal**: Find the sweet spot that minimizes total error

### Learning Curves
Learning curves plot training and validation error vs. training set size:

- **Converging curves at low error**: Good fit
- **High plateau for both curves**: Underfitting (more data won't help)
- **Large gap between curves**: Overfitting (more data can help)

---

## 🎯 Learning Path

**Recommended Order**:
1. Start with `overfitting_underfitting_demo.py` - Visual introduction
2. Move to `learning_curves_demo.py` - Learn to diagnose problems
3. Study `bias_variance_tradeoff.py` - Understand the theory
4. Apply `practical_solutions.py` - Learn solutions

---

## 📈 Example Outputs

Each script generates:
- **Console Output**: Detailed metrics and analysis
- **Visualizations**: High-quality plots saved as PNG files
- **Interpretations**: Automatic diagnosis and recommendations

---

## 🔧 Customization

All scripts use synthetic data with controllable parameters:

```python
# Modify data generation
X, y = generate_data(
    n_samples=200,    # Number of samples
    noise=0.3         # Noise level
)

# Adjust model complexity
degrees = [1, 2, 3, 5, 10, 15]  # Polynomial degrees to test
```

---

## 📖 Additional Resources

### Understanding the Metrics

**Mean Squared Error (MSE)**:
- Lower is better
- Measures average squared difference between predictions and actual values
- Penalizes large errors more heavily

**R² Score**:
- Ranges from -∞ to 1
- 1.0 = Perfect predictions
- 0.0 = Model performs as well as predicting the mean
- Negative = Model performs worse than predicting the mean

### Regularization Parameters

**Ridge (L2) - alpha**:
- alpha = 0: No regularization (same as Linear Regression)
- alpha → ∞: All coefficients → 0
- Typical range: 0.01 to 100

**Lasso (L1) - alpha**:
- Performs feature selection (sets some coefficients to 0)
- Typical range: 0.001 to 10

**ElasticNet**:
- Combines L1 and L2
- l1_ratio: 0 = Ridge, 1 = Lasso, 0.5 = Equal mix

---

## 🎓 Key Takeaways

1. **Always use cross-validation** for model selection
2. **Plot learning curves** to diagnose problems early
3. **Start simple** and increase complexity gradually
4. **Regularization** is essential for high-dimensional problems
5. **More data** often helps with overfitting
6. **Feature engineering** can be more important than model choice
7. **No free lunch**: The best model depends on your data and problem

---

## 🐛 Troubleshooting

**Script runs but no plots appear**:
- Make sure you're not in a headless environment
- Plots are saved as PNG files even if display fails

**Memory errors with large datasets**:
- Reduce `n_samples` in data generation
- Reduce `n_iterations` in bias-variance analysis
- Use smaller polynomial degrees

**Long execution times**:
- Reduce cross-validation folds (cv=3 instead of cv=5)
- Reduce bootstrap iterations in bias-variance analysis
- Use fewer models in comparison scripts

---

## 📝 License

These scripts are provided for educational purposes. Feel free to modify and use them for learning and teaching.

---

## 🤝 Contributing

Suggestions for improvements:
- Add more algorithms (SVM, Neural Networks)
- Include real-world datasets
- Add interactive visualizations
- Implement early stopping demonstrations

---

## 📧 Questions?

For questions about the concepts:
- Review the console output - it includes detailed explanations
- Check the generated visualizations
- Read through the inline comments in the code

Happy Learning! 🎉

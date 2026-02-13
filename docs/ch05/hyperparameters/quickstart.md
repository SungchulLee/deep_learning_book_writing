# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
# Minimum installation
pip install scikit-learn numpy pandas matplotlib seaborn scipy

# Recommended: Also install Optuna for Bayesian optimization
pip install optuna

# Optional: Install AutoML libraries
pip install tpot  # For genetic programming AutoML
```

### 2. Run Your First Example

```bash
# Grid Search Example
python 01_grid_search.py

# Random Search Example  
python 02_random_search.py

# Bayesian Optimization (requires optuna)
python 03_bayesian_optimization.py

# AutoML Introduction
python 04_automl_intro.py

# Comprehensive Comparison
python 05_comparison.py
```

### 3. Understanding the Output

Each script will:
- Load a sample dataset
- Perform hyperparameter tuning
- Print the best parameters found
- Show cross-validation and test scores
- Display timing information
- Create visualizations (when applicable)

## What Each File Does

### `01_grid_search.py`
- **What:** Exhaustive search over parameter grid
- **Best for:** Small parameter spaces
- **Time:** Slowest but most thorough
- **Output:** Best parameters from all combinations

### `02_random_search.py`
- **What:** Random sampling of parameter space
- **Best for:** Large parameter spaces
- **Time:** Faster than grid search
- **Output:** Best parameters from random samples

### `03_bayesian_optimization.py`
- **What:** Intelligent search using past results
- **Best for:** Expensive model evaluations
- **Time:** Most sample-efficient
- **Output:** Best parameters with fewer evaluations
- **Requires:** `pip install optuna`

### `04_automl_intro.py`
- **What:** Automated model selection and tuning
- **Best for:** Quick baselines, beginners
- **Time:** Varies by configuration
- **Output:** Best model and parameters
- **Optional:** `pip install tpot` for full features

### `05_comparison.py`
- **What:** Compares all methods side-by-side
- **Best for:** Understanding trade-offs
- **Time:** ~5-10 minutes
- **Output:** Comparison table and visualization

## Quick Decision Guide

**Choose Grid Search if:**
- ✓ You have < 100 parameter combinations
- ✓ You want to try every option
- ✓ You have time/compute resources

**Choose Random Search if:**
- ✓ You have > 100 parameter combinations
- ✓ You want faster results
- ✓ You're doing initial exploration

**Choose Bayesian Optimization if:**
- ✓ Model training is slow/expensive
- ✓ You want the most efficient search
- ✓ You're willing to install optuna

**Choose AutoML if:**
- ✓ You're starting a new project
- ✓ You want to compare multiple models
- ✓ You need a quick baseline

## Common Use Cases

### 1. Quick Baseline
```bash
python 04_automl_intro.py
# Runs multiple models, picks the best one
```

### 2. Optimize Specific Model
```bash
python 02_random_search.py
# Fast parameter tuning for one model
```

### 3. Final Fine-Tuning
```bash
python 01_grid_search.py
# Exhaustive search in small range
```

### 4. Compare Approaches
```bash
python 05_comparison.py
# See which method works best for your data
```

## Customization Tips

### Use Your Own Data
Replace this in any script:
```python
X_train, X_test, y_train, y_test = load_sample_dataset('iris')
```

With:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    your_X, your_y, test_size=0.2, random_state=42
)
```

### Adjust Parameter Ranges
Modify the parameter dictionaries in each script:
```python
param_grid = {
    'n_estimators': [50, 100, 200],  # Change these values
    'max_depth': [10, 20, 30],       # Add/remove options
}
```

### Change Number of Iterations
For random/bayesian search:
```python
# In the script, find and modify:
n_iter=50  # Increase for better results (slower)
n_trials=100  # Increase for better results (slower)
```

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Scripts running slowly
- Reduce cv=5 to cv=3 for faster results
- Reduce n_iter or n_trials
- Use fewer parameter combinations

### Out of memory
- Reduce parameter space
- Set n_jobs=1 instead of n_jobs=-1
- Use smaller dataset

## Next Steps

1. **Start with**: `05_comparison.py` to see all methods
2. **Then try**: Modify examples with your own data
3. **Learn more**: Read comments in each script
4. **Explore**: Try different parameter ranges
5. **Optimize**: Focus on the method that works best for you

## Resources

- scikit-learn docs: https://scikit-learn.org/stable/modules/grid_search.html
- Optuna docs: https://optuna.readthedocs.io/
- TPOT docs: http://epistasislab.github.io/tpot/

## Questions?

Each script has detailed comments explaining:
- Why to use each method
- When to use it
- Pros and cons
- Parameter explanations

Read through the code for more insights!

---

**Happy Tuning! 🎯**

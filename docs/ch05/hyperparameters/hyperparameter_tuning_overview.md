# Hyperparameter Tuning Tutorial

This package contains Python scripts demonstrating various hyperparameter tuning techniques.

## Contents

1. **01_grid_search.py** - Grid Search implementation with scikit-learn
2. **02_random_search.py** - Random Search implementation
3. **03_bayesian_optimization.py** - Bayesian optimization example
4. **04_automl_intro.py** - Basic AutoML introduction
5. **05_comparison.py** - Comparison of different tuning methods
6. **utils.py** - Utility functions for data loading and visualization

## Installation

```bash
pip install scikit-learn numpy pandas matplotlib seaborn optuna
```

For AutoML features:
```bash
pip install auto-sklearn  # or
pip install tpot
```

## Usage

Run each script individually:
```bash
python 01_grid_search.py
python 02_random_search.py
# etc.
```

## Topics Covered

- Grid Search: Exhaustive search over specified parameter values
- Random Search: Random sampling of hyperparameter space
- Bayesian Optimization: Smart search using probabilistic models
- AutoML: Automated machine learning pipelines
- Comparison: Performance and time comparisons

## Requirements

- Python 3.7+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- optuna (for Bayesian optimization)

"""
Grid Search for Hyperparameter Tuning

Grid Search performs an exhaustive search over a specified parameter grid.
It tries every possible combination of parameters.

Pros:
- Guarantees finding the best combination within the grid
- Easy to understand and implement
- Good for small parameter spaces

Cons:
- Computationally expensive for large grids
- Suffers from curse of dimensionality
- May miss optimal values between grid points
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time
from utils import (load_sample_dataset, print_results, 
                   create_param_grid_summary, plot_parameter_importance)


def grid_search_random_forest():
    """
    Demonstrate Grid Search with Random Forest Classifier
    """
    print("\n" + "="*60)
    print("GRID SEARCH - RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    
    # Show grid summary
    total_combinations = create_param_grid_summary(param_grid)
    
    # Create the model
    rf = RandomForestClassifier(random_state=42)
    
    # Create Grid Search object
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1,
        return_train_score=True
    )
    
    # Perform the search
    print(f"\nSearching through {total_combinations} combinations...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Get best model and evaluate
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    print_results(
        method_name="Grid Search (Random Forest)",
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot parameter importance
    plot_parameter_importance(grid_search.cv_results_, 'n_estimators')
    plot_parameter_importance(grid_search.cv_results_, 'max_depth')
    
    return grid_search


def grid_search_svm():
    """
    Demonstrate Grid Search with Support Vector Machine
    """
    print("\n" + "="*60)
    print("GRID SEARCH - SUPPORT VECTOR MACHINE")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Show grid summary
    total_combinations = create_param_grid_summary(param_grid)
    
    # Create the model
    svm = SVC(random_state=42)
    
    # Create Grid Search object
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Perform the search
    print(f"\nSearching through {total_combinations} combinations...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Get best model and evaluate
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    print_results(
        method_name="Grid Search (SVM)",
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot parameter importance
    plot_parameter_importance(grid_search.cv_results_, 'C')
    plot_parameter_importance(grid_search.cv_results_, 'gamma')
    
    return grid_search


def nested_grid_search():
    """
    Demonstrate nested cross-validation with Grid Search
    This provides a more robust estimate of model performance
    """
    print("\n" + "="*60)
    print("NESTED GRID SEARCH (More Robust Evaluation)")
    print("="*60)
    
    from sklearn.model_selection import cross_val_score
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Inner CV for hyperparameter tuning
    inner_cv = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Outer CV for model evaluation
    print("\nPerforming nested cross-validation...")
    start_time = time.time()
    outer_scores = cross_val_score(
        inner_cv, X_train, y_train, 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    search_time = time.time() - start_time
    
    print(f"\nOuter CV Scores: {outer_scores}")
    print(f"Mean Score: {outer_scores.mean():.4f} (+/- {outer_scores.std() * 2:.4f})")
    print(f"Search Time: {search_time:.2f} seconds")
    
    # Fit on entire training set to get best model
    inner_cv.fit(X_train, y_train)
    test_score = inner_cv.score(X_test, y_test)
    
    print(f"\nBest Parameters: {inner_cv.best_params_}")
    print(f"Test Set Score: {test_score:.4f}")
    
    return inner_cv


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: GRID SEARCH")
    print("="*60)
    
    print("\nGrid Search systematically works through multiple combinations")
    print("of parameter values, cross-validating as it goes to determine")
    print("which combination gives the best performance.")
    
    # Run examples
    print("\n\n### Example 1: Random Forest ###")
    gs_rf = grid_search_random_forest()
    
    print("\n\n### Example 2: Support Vector Machine ###")
    gs_svm = grid_search_svm()
    
    print("\n\n### Example 3: Nested Cross-Validation ###")
    gs_nested = nested_grid_search()
    
    print("\n\nGrid Search completed! Check the results above.")
    print("\nKey Takeaways:")
    print("- Grid Search is exhaustive and guaranteed to find the best")
    print("  combination within your specified grid")
    print("- Computational cost grows exponentially with parameters")
    print("- Use nested CV for unbiased performance estimates")
    print("- Start with coarse grid, then refine around best values")

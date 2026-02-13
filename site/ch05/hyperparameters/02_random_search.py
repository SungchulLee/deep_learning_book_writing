"""
Random Search for Hyperparameter Tuning

Random Search samples parameter values from a distribution randomly.
It doesn't try every combination but samples a fixed number of settings.

Pros:
- More efficient than Grid Search for large parameter spaces
- Can explore more diverse parameter combinations
- Often finds good parameters faster than Grid Search
- Can use continuous distributions

Cons:
- No guarantee of finding the absolute best combination
- Results can vary between runs (unless random_state is set)
- May need more iterations for complex spaces
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform
import time
from utils import (load_sample_dataset, print_results, 
                   plot_parameter_importance)


def random_search_random_forest():
    """
    Demonstrate Random Search with Random Forest Classifier
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH - RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Define parameter distributions
    # Note: we use scipy.stats distributions for continuous parameters
    param_distributions = {
        'n_estimators': randint(50, 500),  # Random integers from 50 to 499
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    print("\nParameter Distributions:")
    print("-" * 40)
    for param, dist in param_distributions.items():
        print(f"{param}: {dist}")
    print("-" * 40)
    
    # Create the model
    rf = RandomForestClassifier(random_state=42)
    
    # Create Random Search object
    n_iter = 100  # Number of random combinations to try
    print(f"\nWill try {n_iter} random combinations...")
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    # Perform the search
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Get best model and evaluate
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    print_results(
        method_name="Random Search (Random Forest)",
        best_params=random_search.best_params_,
        best_score=random_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Show top 5 parameter combinations
    import pandas as pd
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print("\nTop 5 Parameter Combinations:")
    print("-" * 60)
    top_5_params = results_df[['params', 'mean_test_score']].head()
    for idx, row in top_5_params.iterrows():
        print(f"\nRank {row.name + 1}: Score = {row['mean_test_score']:.4f}")
        for param, value in row['params'].items():
            print(f"  {param}: {value}")
    
    return random_search


def random_search_gradient_boosting():
    """
    Demonstrate Random Search with Gradient Boosting Classifier
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH - GRADIENT BOOSTING CLASSIFIER")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),  # Uniform from 0.01 to 0.31
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),  # Uniform from 0.6 to 1.0
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("\nParameter Distributions:")
    print("-" * 40)
    for param, dist in param_distributions.items():
        print(f"{param}: {dist}")
    print("-" * 40)
    
    # Create the model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Create Random Search object
    n_iter = 50
    print(f"\nWill try {n_iter} random combinations...")
    
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Perform the search
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Get best model and evaluate
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    print_results(
        method_name="Random Search (Gradient Boosting)",
        best_params=random_search.best_params_,
        best_score=random_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return random_search


def compare_n_iter():
    """
    Compare different numbers of iterations in Random Search
    """
    print("\n" + "="*60)
    print("COMPARING DIFFERENT NUMBER OF ITERATIONS")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 15),
        'max_features': ['sqrt', 'log2']
    }
    
    # Try different numbers of iterations
    n_iters = [10, 25, 50, 100]
    results = []
    
    print("\nTrying different numbers of random combinations...\n")
    
    for n_iter in n_iters:
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        test_score = random_search.score(X_test, y_test)
        
        results.append({
            'n_iter': n_iter,
            'cv_score': random_search.best_score_,
            'test_score': test_score,
            'time': search_time
        })
        
        print(f"n_iter={n_iter:3d}: CV Score={random_search.best_score_:.4f}, "
              f"Test Score={test_score:.4f}, Time={search_time:.2f}s")
    
    print("\n" + "="*60)
    print("Observations:")
    print("- More iterations generally lead to better scores")
    print("- But returns diminish after a certain point")
    print("- Balance between computational cost and performance")
    print("="*60)
    
    return results


def random_vs_grid_comparison():
    """
    Direct comparison between Random Search and Grid Search
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH VS GRID SEARCH")
    print("="*60)
    
    from sklearn.model_selection import GridSearchCV
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Define a moderate parameter space
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    # Calculate total combinations for grid search
    total_combinations = 4 * 4 * 3 * 2  # 96 combinations
    
    print(f"\nGrid Search will try all {total_combinations} combinations")
    print(f"Random Search will try 50 random combinations")
    
    # Grid Search
    print("\n--- Running Grid Search ---")
    rf_grid = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_grid,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time
    grid_score = grid_search.best_score_
    grid_test = grid_search.score(X_test, y_test)
    
    # Random Search
    print("\n--- Running Random Search ---")
    param_distributions = {
        'n_estimators': randint(50, 350),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 15),
        'max_features': ['sqrt', 'log2']
    }
    
    rf_random = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf_random,
        param_distributions=param_distributions,
        n_iter=50,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time
    random_score = random_search.best_score_
    random_test = random_search.score(X_test, y_test)
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nGrid Search:")
    print(f"  Best CV Score: {grid_score:.4f}")
    print(f"  Test Score: {grid_test:.4f}")
    print(f"  Time: {grid_time:.2f} seconds")
    print(f"  Combinations tried: {total_combinations}")
    
    print(f"\nRandom Search:")
    print(f"  Best CV Score: {random_score:.4f}")
    print(f"  Test Score: {random_test:.4f}")
    print(f"  Time: {random_time:.2f} seconds")
    print(f"  Combinations tried: 50")
    
    print(f"\nTime Savings: {((grid_time - random_time) / grid_time * 100):.1f}%")
    print(f"Score Difference: {abs(grid_score - random_score):.4f}")
    
    return grid_search, random_search


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: RANDOM SEARCH")
    print("="*60)
    
    print("\nRandom Search samples parameter settings from specified")
    print("distributions for a fixed number of iterations. It's often")
    print("more efficient than Grid Search, especially for large parameter")
    print("spaces.")
    
    # Run examples
    print("\n\n### Example 1: Random Forest with Random Search ###")
    rs_rf = random_search_random_forest()
    
    print("\n\n### Example 2: Gradient Boosting with Random Search ###")
    rs_gb = random_search_gradient_boosting()
    
    print("\n\n### Example 3: Effect of Number of Iterations ###")
    iter_results = compare_n_iter()
    
    print("\n\n### Example 4: Random vs Grid Search ###")
    grid, random = random_vs_grid_comparison()
    
    print("\n\nRandom Search completed! Check the results above.")
    print("\nKey Takeaways:")
    print("- Random Search is more efficient for large parameter spaces")
    print("- Can use continuous distributions (not just discrete grids)")
    print("- Often finds good parameters with fewer iterations")
    print("- Trade-off between number of iterations and computation time")

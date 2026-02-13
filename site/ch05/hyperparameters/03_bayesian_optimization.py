"""
Bayesian Optimization for Hyperparameter Tuning

Bayesian Optimization uses probabilistic models to guide the search for
optimal hyperparameters. It builds a surrogate model of the objective 
function and uses it to decide where to sample next.

Pros:
- More efficient than Random Search for expensive evaluations
- Learns from previous iterations
- Can find better parameters with fewer evaluations
- Handles noisy objectives well

Cons:
- More complex to implement
- Can be slower per iteration
- May get stuck in local optima
- Requires additional libraries (e.g., Optuna, scikit-optimize)

This example uses Optuna, a modern hyperparameter optimization framework.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import time

# Note: Install with: pip install optuna
try:
    import optuna
    from optuna.visualization import (plot_optimization_history, 
                                       plot_param_importances,
                                       plot_parallel_coordinate)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

from utils import load_sample_dataset, print_results


def bayesian_optimization_rf():
    """
    Bayesian Optimization with Random Forest using Optuna
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION - RANDOM FOREST")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Define the objective function
    def objective(trial):
        """
        Objective function for Optuna to optimize
        
        Parameters:
        -----------
        trial : optuna.Trial
            A trial object that suggests parameter values
            
        Returns:
        --------
        float : Cross-validation score to maximize
        """
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
        
        # Create and evaluate model
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
        
        return score
    
    # Create a study
    print("\nCreating optimization study...")
    study = optuna.create_study(
        direction='maximize',  # We want to maximize accuracy
        sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
    )
    
    # Optimize
    n_trials = 100
    print(f"Running {n_trials} optimization trials...")
    start_time = time.time()
    
    # Use a callback to show progress
    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: Best score = {study.best_value:.4f}")
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    search_time = time.time() - start_time
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    # Train final model with best parameters
    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    test_score = best_rf.score(X_test, y_test)
    
    # Print results
    print_results(
        method_name="Bayesian Optimization (Random Forest)",
        best_params=best_params,
        best_score=best_score,
        search_time=search_time,
        test_score=test_score
    )
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Show optimization history
    print("\nOptimization Progress:")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Trials that completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Visualizations (optional - requires plotly)
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.show()
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.show()
        
    except ImportError:
        print("\nInstall plotly for visualizations: pip install plotly")
    
    return study


def bayesian_optimization_gb():
    """
    Bayesian Optimization with Gradient Boosting
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION - GRADIENT BOOSTING")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # Define the objective function
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        
        gb = GradientBoostingClassifier(**params, random_state=42)
        score = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
        
        return score
    
    # Create and run study
    print("\nRunning optimization...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    start_time = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    search_time = time.time() - start_time
    
    # Train final model
    best_gb = GradientBoostingClassifier(**study.best_params, random_state=42)
    best_gb.fit(X_train, y_train)
    test_score = best_gb.score(X_test, y_test)
    
    # Print results
    print_results(
        method_name="Bayesian Optimization (Gradient Boosting)",
        best_params=study.best_params,
        best_score=study.best_value,
        search_time=search_time,
        test_score=test_score
    )
    
    return study


def pruning_example():
    """
    Demonstrate early stopping (pruning) of unpromising trials
    This can significantly speed up optimization
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION WITH PRUNING")
    print("="*60)
    
    print("\nPruning stops unpromising trials early to save computation time.")
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    from sklearn.model_selection import cross_validate
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        }
        
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        # Perform cross-validation with intermediate reports
        cv_results = cross_validate(
            rf, X_train, y_train, cv=5, scoring='accuracy', 
            return_train_score=False, n_jobs=-1
        )
        
        # Report intermediate values for pruning
        for i, score in enumerate(cv_results['test_score']):
            trial.report(score, i)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return cv_results['test_score'].mean()
    
    # Create study with pruner
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    print("\nRunning optimization with pruning...")
    start_time = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    search_time = time.time() - start_time
    
    # Statistics
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"\nOptimization Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len(complete_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Time saved: ~{len(pruned_trials) / len(study.trials) * 100:.1f}%")
    print(f"  Total time: {search_time:.2f} seconds")
    print(f"\nBest score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study


def compare_samplers():
    """
    Compare different sampling strategies in Bayesian Optimization
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("COMPARING DIFFERENT SAMPLERS")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        }
        
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
        return score
    
    # Test different samplers
    samplers = {
        'TPE': optuna.samplers.TPESampler(seed=42),
        'Random': optuna.samplers.RandomSampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
    }
    
    results = {}
    
    print("\nTesting different sampling strategies...")
    for name, sampler in samplers.items():
        print(f"\n--- {name} Sampler ---")
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        start_time = time.time()
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        elapsed_time = time.time() - start_time
        
        results[name] = {
            'best_score': study.best_value,
            'time': elapsed_time
        }
        
        print(f"Best score: {study.best_value:.4f}")
        print(f"Time: {elapsed_time:.2f} seconds")
    
    # Summary
    print("\n" + "="*60)
    print("SAMPLER COMPARISON SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{name:10s}: Score={result['best_score']:.4f}, Time={result['time']:.2f}s")
    
    return results


if __name__ == "__main__":
    if not OPTUNA_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: Optuna not installed")
        print("="*60)
        print("\nPlease install Optuna to run this example:")
        print("  pip install optuna")
        print("\nOptuna is a powerful hyperparameter optimization framework")
        print("that implements Bayesian optimization and other advanced techniques.")
        exit(1)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: BAYESIAN OPTIMIZATION")
    print("="*60)
    
    print("\nBayesian Optimization uses past evaluation results to build")
    print("a probabilistic model and intelligently choose the next set of")
    print("hyperparameters to evaluate. It's more efficient than random")
    print("search for expensive evaluations.")
    
    # Run examples
    print("\n\n### Example 1: Random Forest with Bayesian Optimization ###")
    study_rf = bayesian_optimization_rf()
    
    print("\n\n### Example 2: Gradient Boosting with Bayesian Optimization ###")
    study_gb = bayesian_optimization_gb()
    
    print("\n\n### Example 3: Optimization with Pruning ###")
    study_pruned = pruning_example()
    
    print("\n\n### Example 4: Comparing Different Samplers ###")
    sampler_results = compare_samplers()
    
    print("\n\nBayesian Optimization completed! Check the results above.")
    print("\nKey Takeaways:")
    print("- Bayesian Optimization is sample-efficient")
    print("- TPE sampler works well for most problems")
    print("- Pruning can significantly reduce computation time")
    print("- More sophisticated than random/grid search")
    print("- Great for expensive model evaluations")

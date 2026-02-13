"""
Comprehensive Comparison of Hyperparameter Tuning Methods

This script compares all the hyperparameter tuning methods:
1. Grid Search
2. Random Search
3. Bayesian Optimization (Optuna)
4. Simple AutoML

We'll evaluate them on:
- Best score achieved
- Time taken
- Number of model evaluations
- Ease of use
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import time

# Optuna for Bayesian optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

from utils import load_sample_dataset, plot_search_results


def run_grid_search(X_train, y_train):
    """Run Grid Search"""
    print("\n--- Running Grid Search ---")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    return {
        'method': 'Grid Search',
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'time': elapsed_time,
        'n_iterations': len(grid_search.cv_results_['params']),
        'estimator': grid_search.best_estimator_
    }


def run_random_search(X_train, y_train, n_iter=50):
    """Run Random Search"""
    print("\n--- Running Random Search ---")
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [10, 20, 30, None],
        'min_samples_split': randint(2, 15),
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions, n_iter=n_iter, 
        cv=5, scoring='accuracy', n_jobs=-1, 
        random_state=42, verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    return {
        'method': 'Random Search',
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'time': elapsed_time,
        'n_iterations': n_iter,
        'estimator': random_search.best_estimator_
    }


def run_bayesian_optimization(X_train, y_train, n_trials=50):
    """Run Bayesian Optimization with Optuna"""
    if not OPTUNA_AVAILABLE:
        return None
    
    print("\n--- Running Bayesian Optimization ---")
    
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        }
        
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1).mean()
        return score
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed_time = time.time() - start_time
    
    # Train final model
    best_rf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    
    return {
        'method': 'Bayesian Opt',
        'best_params': study.best_params,
        'best_score': study.best_value,
        'time': elapsed_time,
        'n_iterations': n_trials,
        'estimator': best_rf
    }


def run_simple_automl(X_train, y_train):
    """Run Simple AutoML (model selection)"""
    print("\n--- Running Simple AutoML ---")
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    start_time = time.time()
    best_score = 0
    best_model = None
    best_name = None
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    # Fit best model
    best_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    return {
        'method': 'Simple AutoML',
        'best_params': {'model': best_name},
        'best_score': best_score,
        'time': elapsed_time,
        'n_iterations': len(models) * 5,  # Each model x CV folds
        'estimator': best_model
    }


def comprehensive_comparison():
    """
    Run a comprehensive comparison of all methods
    """
    print("="*70)
    print("COMPREHENSIVE HYPERPARAMETER TUNING COMPARISON")
    print("="*70)
    
    # Load data
    print("\nLoading dataset...")
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Run all methods
    results = []
    
    # Grid Search
    result = run_grid_search(X_train, y_train)
    if result:
        result['test_score'] = result['estimator'].score(X_test, y_test)
        results.append(result)
        print(f"Grid Search - CV: {result['best_score']:.4f}, "
              f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    # Random Search
    result = run_random_search(X_train, y_train, n_iter=50)
    if result:
        result['test_score'] = result['estimator'].score(X_test, y_test)
        results.append(result)
        print(f"Random Search - CV: {result['best_score']:.4f}, "
              f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    # Bayesian Optimization
    if OPTUNA_AVAILABLE:
        result = run_bayesian_optimization(X_train, y_train, n_trials=50)
        if result:
            result['test_score'] = result['estimator'].score(X_test, y_test)
            results.append(result)
            print(f"Bayesian Opt - CV: {result['best_score']:.4f}, "
                  f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    # Simple AutoML
    result = run_simple_automl(X_train, y_train)
    if result:
        result['test_score'] = result['estimator'].score(X_test, y_test)
        results.append(result)
        print(f"Simple AutoML - CV: {result['best_score']:.4f}, "
              f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    return results


def visualize_comparison(results):
    """
    Create visualizations comparing all methods
    """
    print("\n\nCreating comparison visualizations...")
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'Method': r['method'],
            'CV Score': r['best_score'],
            'Test Score': r['test_score'],
            'Time (s)': r['time'],
            'Iterations': r['n_iterations']
        }
        for r in results
    ])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Tuning Methods Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. CV Scores
    ax1 = axes[0, 0]
    bars = ax1.bar(df['Method'], df['CV Score'], color='steelblue', alpha=0.8)
    ax1.set_ylabel('Cross-Validation Score', fontsize=11)
    ax1.set_title('Best CV Score by Method', fontsize=12, fontweight='bold')
    ax1.set_ylim([df['CV Score'].min() * 0.95, df['CV Score'].max() * 1.02])
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Test Scores
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Method'], df['Test Score'], color='coral', alpha=0.8)
    ax2.set_ylabel('Test Score', fontsize=11)
    ax2.set_title('Test Set Score by Method', fontsize=12, fontweight='bold')
    ax2.set_ylim([df['Test Score'].min() * 0.95, df['Test Score'].max() * 1.02])
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Time Comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(df['Method'], df['Time (s)'], color='lightgreen', alpha=0.8)
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Computation Time by Method', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 4. Efficiency (Score per second)
    ax4 = axes[1, 1]
    efficiency = df['CV Score'] / df['Time (s)']
    bars = ax4.bar(df['Method'], efficiency, color='plum', alpha=0.8)
    ax4.set_ylabel('CV Score / Time', fontsize=11)
    ax4.set_title('Efficiency (Score per Second)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/hyperparameter_tuning/comparison_results.png', 
                dpi=300, bbox_inches='tight')
    print("Saved visualization to 'comparison_results.png'")
    plt.show()
    
    return df


def print_summary_table(df):
    """
    Print a nice summary table
    """
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Find winners
    best_cv = df.loc[df['CV Score'].idxmax()]
    best_test = df.loc[df['Test Score'].idxmax()]
    fastest = df.loc[df['Time (s)'].idxmin()]
    most_efficient = df.loc[(df['CV Score'] / df['Time (s)']).idxmax()]
    
    print("\n📊 WINNERS:")
    print(f"  🏆 Best CV Score:    {best_cv['Method']} ({best_cv['CV Score']:.4f})")
    print(f"  🎯 Best Test Score:  {best_test['Method']} ({best_test['Test Score']:.4f})")
    print(f"  ⚡ Fastest:          {fastest['Method']} ({fastest['Time (s)']:.2f}s)")
    print(f"  💡 Most Efficient:   {most_efficient['Method']} "
          f"({(most_efficient['CV Score']/most_efficient['Time (s)']):.4f})")


def recommendations():
    """
    Print recommendations for when to use each method
    """
    print("\n" + "="*70)
    print("RECOMMENDATIONS - WHEN TO USE EACH METHOD")
    print("="*70)
    
    recommendations_text = """
    
📋 GRID SEARCH
   ✅ When to use:
      - Small parameter space (< 100 combinations)
      - Need to try every combination
      - Interpretability is important
      - Have computational resources
   ❌ When NOT to use:
      - Large parameter spaces
      - Many hyperparameters
      - Limited time/resources

🎲 RANDOM SEARCH
   ✅ When to use:
      - Large parameter spaces
      - Continuous parameter distributions
      - Limited computational budget
      - Quick initial exploration
   ❌ When NOT to use:
      - Very small parameter spaces
      - Need exhaustive search

🧠 BAYESIAN OPTIMIZATION
   ✅ When to use:
      - Expensive model evaluations
      - Want sample efficiency
      - Medium-sized parameter spaces
      - Can afford setup complexity
   ❌ When NOT to use:
      - Very fast model training
      - Extremely large spaces
      - Need simplicity

🤖 AUTOML
   ✅ When to use:
      - Starting a new project
      - Need baseline quickly
      - Limited ML expertise
      - Want to try many models
   ❌ When NOT to use:
      - Need full control
      - Very specific requirements
      - Production critical systems (without validation)

💡 GENERAL TIPS:
   1. Start with Random Search for quick exploration
   2. Use Bayesian Optimization for expensive models
   3. Grid Search for final fine-tuning in small ranges
   4. AutoML for baselines and model selection
   5. Always validate on separate test set
   6. Consider nested CV for robust estimates
"""
    print(recommendations_text)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE COMPARISON")
    print("="*70)
    print("\nThis will run all hyperparameter tuning methods and compare them.")
    print("This may take a few minutes...\n")
    
    # Run comparison
    results = comprehensive_comparison()
    
    # Visualize
    df = visualize_comparison(results)
    
    # Print summary
    print_summary_table(df)
    
    # Print recommendations
    recommendations()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nKey Insights:")
    print("- Different methods have different trade-offs")
    print("- No single 'best' method for all situations")
    print("- Consider your constraints: time, resources, accuracy needs")
    print("- Start simple, increase complexity as needed")
    print("\nVisualization saved to 'comparison_results.png'")

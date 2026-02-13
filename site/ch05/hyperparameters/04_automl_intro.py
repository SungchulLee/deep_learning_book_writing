"""
AutoML - Automated Machine Learning

AutoML automates the entire machine learning pipeline including:
- Feature preprocessing
- Model selection
- Hyperparameter tuning
- Ensemble creation

Popular AutoML Libraries:
- TPOT: Uses genetic programming
- Auto-sklearn: Extends scikit-learn with automated model selection
- H2O AutoML: Enterprise-focused AutoML
- PyCaret: Low-code ML library
- AutoKeras: AutoML for deep learning

This example demonstrates basic AutoML concepts using TPOT (simpler to install)
and shows how to implement a basic AutoML-like workflow manually.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time

# TPOT is optional - we'll show manual AutoML if not available
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    print("TPOT not installed. Will demonstrate manual AutoML approach.")
    TPOT_AVAILABLE = False

from utils import load_sample_dataset, print_results


def simple_automl():
    """
    Simple AutoML implementation that tries multiple models and selects the best
    """
    print("\n" + "="*60)
    print("SIMPLE AUTOML - MODEL SELECTION")
    print("="*60)
    
    print("\nThis example automatically tries multiple models and")
    print("selects the best one based on cross-validation scores.")
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Define candidate models with some parameter options
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }
    
    print(f"\nEvaluating {len(models)} different models...")
    
    results = []
    start_time = time.time()
    
    for name, model in models.items():
        print(f"\nTrying {name}...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        results.append({
            'model': name,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'pipeline': pipeline
        })
        
        print(f"  CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
    
    total_time = time.time() - start_time
    
    # Sort by score
    results.sort(key=lambda x: x['mean_cv_score'], reverse=True)
    
    # Select best model
    best_result = results[0]
    best_pipeline = best_result['pipeline']
    
    # Train on full training set and evaluate
    best_pipeline.fit(X_train, y_train)
    test_score = best_pipeline.score(X_test, y_test)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL SELECTION RESULTS")
    print("="*60)
    
    print("\nAll Models (sorted by performance):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['model']}: {result['mean_cv_score']:.4f} "
              f"(+/- {result['std_cv_score']:.4f})")
    
    print(f"\n{'Best Model:':<20} {best_result['model']}")
    print(f"{'Best CV Score:':<20} {best_result['mean_cv_score']:.4f}")
    print(f"{'Test Score:':<20} {test_score:.4f}")
    print(f"{'Total Time:':<20} {total_time:.2f} seconds")
    
    # Predictions
    y_pred = best_pipeline.predict(X_test)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred))
    
    return best_pipeline, results


def automl_with_hyperparameter_tuning():
    """
    More advanced AutoML that includes hyperparameter tuning for each model
    """
    print("\n" + "="*60)
    print("AUTOML WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # Define models with their parameter distributions
    model_configs = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators': randint(50, 300),
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': randint(2, 10),
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model__n_estimators': randint(50, 200),
                'model__learning_rate': uniform(0.01, 0.2),
                'model__max_depth': randint(3, 10),
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'model__C': uniform(0.1, 100),
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto'],
            }
        }
    }
    
    print(f"\nTuning {len(model_configs)} models with hyperparameter search...")
    
    results = []
    start_time = time.time()
    
    for name, config in model_configs.items():
        print(f"\n--- Tuning {name} ---")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])
        
        # Random search for hyperparameters
        random_search = RandomizedSearchCV(
            pipeline,
            config['params'],
            n_iter=20,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        results.append({
            'model': name,
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'estimator': random_search.best_estimator_
        })
        
        print(f"Best CV Score: {random_search.best_score_:.4f}")
    
    total_time = time.time() - start_time
    
    # Select best model
    results.sort(key=lambda x: x['best_score'], reverse=True)
    best_result = results[0]
    
    # Evaluate on test set
    test_score = best_result['estimator'].score(X_test, y_test)
    
    # Print results
    print("\n" + "="*60)
    print("AUTOML RESULTS")
    print("="*60)
    
    print("\nAll Models (sorted by performance):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['model']}")
        print(f"   CV Score: {result['best_score']:.4f}")
        print(f"   Best Parameters:")
        for param, value in result['best_params'].items():
            print(f"     {param}: {value}")
    
    print(f"\n{'='*60}")
    print(f"{'Best Model:':<25} {best_result['model']}")
    print(f"{'Best CV Score:':<25} {best_result['best_score']:.4f}")
    print(f"{'Test Score:':<25} {test_score:.4f}")
    print(f"{'Total Time:':<25} {total_time:.2f} seconds")
    print(f"{'='*60}")
    
    return best_result['estimator'], results


def tpot_automl_example():
    """
    Example using TPOT - a true AutoML library
    """
    if not TPOT_AVAILABLE:
        print("\n" + "="*60)
        print("TPOT NOT AVAILABLE")
        print("="*60)
        print("\nTPOT is not installed. To use TPOT AutoML:")
        print("  pip install tpot")
        print("\nTPOT uses genetic programming to automatically design")
        print("and optimize machine learning pipelines.")
        return None
    
    print("\n" + "="*60)
    print("TPOT AUTOML EXAMPLE")
    print("="*60)
    
    print("\nTPOT uses genetic programming to automatically discover")
    print("the best machine learning pipeline for your data.")
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    print("\nInitializing TPOT...")
    print("This will take a few minutes as it evolves pipelines...")
    
    # Create TPOT classifier
    tpot = TPOTClassifier(
        generations=5,  # Number of iterations to run
        population_size=20,  # Number of pipelines in each generation
        cv=5,
        random_state=42,
        verbosity=2,
        n_jobs=-1,
        max_time_mins=3,  # Maximum time in minutes
        max_eval_time_mins=0.5,  # Maximum time for a single pipeline
    )
    
    # Run TPOT
    start_time = time.time()
    tpot.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Evaluate
    train_score = tpot.score(X_train, y_train)
    test_score = tpot.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("TPOT RESULTS")
    print("="*60)
    print(f"\nBest Pipeline Score (CV): {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Search Time: {search_time:.2f} seconds")
    
    # Export the best pipeline
    print("\nExporting best pipeline to 'best_pipeline.py'...")
    tpot.export('/home/claude/hyperparameter_tuning/tpot_best_pipeline.py')
    
    # Show the best pipeline
    print("\nBest Pipeline:")
    print(tpot.fitted_pipeline_)
    
    # Make predictions
    y_pred = tpot.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return tpot


def ensemble_automl():
    """
    Create an ensemble of the best models found by AutoML
    """
    print("\n" + "="*60)
    print("ENSEMBLE AUTOML")
    print("="*60)
    
    from sklearn.ensemble import VotingClassifier
    
    print("\nCombining multiple good models into an ensemble...")
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # Define models
    models = {
        'rf': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        'svm': SVC(kernel='rbf', C=10, probability=True, random_state=42),
    }
    
    # Evaluate individual models
    print("\nIndividual model performance:")
    individual_results = []
    
    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        individual_results.append((name, mean_score))
        print(f"  {name}: {mean_score:.4f}")
    
    # Create voting ensemble
    voting_clf = VotingClassifier(
        estimators=[(name, Pipeline([('scaler', StandardScaler()), ('model', model)])) 
                    for name, model in models.items()],
        voting='soft'  # Use probability predictions
    )
    
    # Evaluate ensemble
    print("\nTraining ensemble...")
    start_time = time.time()
    ensemble_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
    ensemble_time = time.time() - start_time
    
    # Train on full data and test
    voting_clf.fit(X_train, y_train)
    test_score = voting_clf.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"\nEnsemble CV Score: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")
    print(f"Ensemble Test Score: {test_score:.4f}")
    print(f"Training Time: {ensemble_time:.2f} seconds")
    
    # Compare to best individual model
    best_individual = max(individual_results, key=lambda x: x[1])
    print(f"\nBest Individual Model: {best_individual[0]} ({best_individual[1]:.4f})")
    print(f"Ensemble Improvement: {(ensemble_scores.mean() - best_individual[1]):.4f}")
    
    return voting_clf


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUTOMATED MACHINE LEARNING (AutoML)")
    print("="*60)
    
    print("\nAutoML automates the machine learning pipeline including")
    print("feature engineering, model selection, and hyperparameter tuning.")
    print("It makes ML accessible and efficient by automating repetitive tasks.")
    
    # Run examples
    print("\n\n### Example 1: Simple Model Selection ###")
    best_model, all_results = simple_automl()
    
    print("\n\n### Example 2: AutoML with Hyperparameter Tuning ###")
    tuned_model, tuning_results = automl_with_hyperparameter_tuning()
    
    print("\n\n### Example 3: TPOT AutoML ###")
    tpot_model = tpot_automl_example()
    
    print("\n\n### Example 4: Ensemble AutoML ###")
    ensemble = ensemble_automl()
    
    print("\n\nAutoML examples completed!")
    print("\nKey Takeaways:")
    print("- AutoML automates model selection and tuning")
    print("- Can save significant time in model development")
    print("- TPOT and Auto-sklearn are powerful AutoML tools")
    print("- Ensembles often improve over individual models")
    print("- Great for baseline models and non-experts")
    print("\nAutoML libraries to explore:")
    print("  - TPOT: pip install tpot")
    print("  - Auto-sklearn: pip install auto-sklearn")
    print("  - PyCaret: pip install pycaret")
    print("  - H2O AutoML: pip install h2o")

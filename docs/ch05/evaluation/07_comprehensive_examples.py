"""
Comprehensive Examples - End-to-End Model Evaluation
=====================================================

Complete examples demonstrating model evaluation workflow from start to finish.

Examples:
1. Binary Classification: Credit Card Fraud Detection
2. Multi-class Classification: Iris Species
3. Regression: House Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


class Example1_BinaryClassification:
    """
    Example: Credit Card Fraud Detection (Binary Classification)
    
    Business Context:
    - Detecting fraudulent credit card transactions
    - False Negatives (missed fraud) are costly
    - False Positives (blocking legitimate transactions) harm customer experience
    - Need high recall to catch fraud, but also maintain reasonable precision
    """
    
    @staticmethod
    def run():
        print("=" * 80)
        print("EXAMPLE 1: BINARY CLASSIFICATION - CREDIT CARD FRAUD DETECTION")
        print("=" * 80)
        
        # Generate synthetic imbalanced dataset
        print("\n1. GENERATING DATA")
        print("-" * 80)
        X, y = make_classification(
            n_samples=10000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, weights=[0.95, 0.05],  # 5% fraud
            random_state=42
        )
        
        print(f"Total samples: {len(X)}")
        print(f"Fraudulent transactions: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
        print(f"Legitimate transactions: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.2f}%)")
        print("⚠ HIGHLY IMBALANCED DATASET")
        
        # Split data
        print("\n2. SPLITTING DATA")
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        print("\n3. TRAINING MODELS")
        print("-" * 80)
        
        # Model 1: Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        print("✓ Logistic Regression trained")
        
        # Model 2: Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        print("✓ Random Forest trained")
        
        # Cross-validation
        print("\n4. CROSS-VALIDATION")
        print("-" * 80)
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        lr_scores = cross_val_score(lr_model, X_train_scaled, y_train, 
                                   cv=skfold, scoring='f1')
        rf_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                                   cv=skfold, scoring='f1')
        
        print(f"Logistic Regression F1 Score: {lr_scores.mean():.4f} (+/- {lr_scores.std()*2:.4f})")
        print(f"Random Forest F1 Score: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
        
        # Predictions
        print("\n5. TEST SET EVALUATION")
        print("-" * 80)
        
        y_pred_lr = lr_model.predict(X_test_scaled)
        y_pred_rf = rf_model.predict(X_test_scaled)
        
        y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation
        print("\nLOGISTIC REGRESSION:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred_lr):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
        
        print("\nRANDOM FOREST:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred_rf):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
        
        # Confusion Matrix
        print("\n6. CONFUSION MATRIX (Random Forest)")
        print("-" * 80)
        cm = confusion_matrix(y_test, y_pred_rf)
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives (Legitimate correctly identified): {tn}")
        print(f"False Positives (Legitimate flagged as fraud): {fp}")
        print(f"False Negatives (Fraud missed): {fn}")
        print(f"True Positives (Fraud caught): {tp}")
        
        # Business interpretation
        print("\n7. BUSINESS INTERPRETATION")
        print("-" * 80)
        print(f"Out of {sum(y_test)} fraudulent transactions:")
        print(f"  ✓ Caught: {tp} ({tp/sum(y_test)*100:.1f}%)")
        print(f"  ✗ Missed: {fn} ({fn/sum(y_test)*100:.1f}%)")
        print(f"\nOut of {len(y_test)-sum(y_test)} legitimate transactions:")
        print(f"  ✓ Approved: {tn} ({tn/(len(y_test)-sum(y_test))*100:.1f}%)")
        print(f"  ✗ Incorrectly flagged: {fp} ({fp/(len(y_test)-sum(y_test))*100:.1f}%)")
        
        print("\n" + "=" * 80)


class Example2_MulticlassClassification:
    """
    Example: Iris Species Classification (Multi-class)
    
    Problem: Classify iris flowers into three species based on measurements
    """
    
    @staticmethod
    def run():
        print("\n\n" + "=" * 80)
        print("EXAMPLE 2: MULTI-CLASS CLASSIFICATION - IRIS SPECIES")
        print("=" * 80)
        
        # Load data
        print("\n1. LOADING DATA")
        print("-" * 80)
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {iris.feature_names}")
        print(f"Classes: {class_names}")
        
        # Split data
        print("\n2. SPLITTING DATA")
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Train model
        print("\n3. TRAINING MODEL")
        print("-" * 80)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("✓ Random Forest trained")
        
        # Cross-validation
        print("\n4. STRATIFIED CROSS-VALIDATION")
        print("-" * 80)
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skfold, scoring='accuracy')
        print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation
        print("\n5. TEST SET EVALUATION")
        print("-" * 80)
        print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        print("\nPer-Class Metrics:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion Matrix
        print("\n6. CONFUSION MATRIX")
        print("-" * 80)
        cm = confusion_matrix(y_test, y_pred)
        
        print("             Predicted")
        print("             ", "  ".join([f"{name[:4]:>5}" for name in class_names]))
        print("Actual")
        for i, row in enumerate(cm):
            print(f"{class_names[i][:10]:10}", "  ".join([f"{val:5}" for val in row]))
        
        print("\n" + "=" * 80)


class Example3_Regression:
    """
    Example: House Price Prediction (Regression)
    
    Problem: Predict house prices based on features
    """
    
    @staticmethod
    def run():
        print("\n\n" + "=" * 80)
        print("EXAMPLE 3: REGRESSION - HOUSE PRICE PREDICTION")
        print("=" * 80)
        
        # Generate synthetic data
        print("\n1. GENERATING DATA")
        print("-" * 80)
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=1000, n_features=10, n_informative=8,
            noise=10, random_state=42
        )
        
        # Scale to realistic house prices
        y = (y - y.min()) / (y.max() - y.min()) * 500000 + 200000
        
        print(f"Total samples: {len(X)}")
        print(f"Features: 10 (square feet, bedrooms, location score, etc.)")
        print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"Mean price: ${y.mean():,.0f}")
        
        # Split data
        print("\n2. SPLITTING DATA")
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        print("\n3. TRAINING MODELS")
        print("-" * 80)
        
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        print("✓ Ridge Regression trained")
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        print("✓ Random Forest trained")
        
        # Cross-validation
        print("\n4. CROSS-VALIDATION")
        print("-" * 80)
        
        ridge_scores = cross_val_score(ridge_model, X_train_scaled, y_train,
                                      cv=5, scoring='r2')
        rf_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                                   cv=5, scoring='r2')
        
        print(f"Ridge R² Score: {ridge_scores.mean():.4f} (+/- {ridge_scores.std()*2:.4f})")
        print(f"Random Forest R² Score: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
        
        # Predictions
        y_pred_ridge = ridge_model.predict(X_test_scaled)
        y_pred_rf = rf_model.predict(X_test_scaled)
        
        # Evaluation
        print("\n5. TEST SET EVALUATION")
        print("-" * 80)
        
        print("\nRIDGE REGRESSION:")
        print(f"  MAE: ${mean_absolute_error(y_test, y_pred_ridge):,.2f}")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_ridge)):,.2f}")
        print(f"  R² Score: {r2_score(y_test, y_pred_ridge):.4f}")
        
        print("\nRANDOM FOREST:")
        print(f"  MAE: ${mean_absolute_error(y_test, y_pred_rf):,.2f}")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.2f}")
        print(f"  R² Score: {r2_score(y_test, y_pred_rf):.4f}")
        
        # Sample predictions
        print("\n6. SAMPLE PREDICTIONS (Random Forest)")
        print("-" * 80)
        print(f"{'Actual Price':>15} {'Predicted Price':>17} {'Error':>12}")
        print("-" * 50)
        
        for i in range(min(10, len(y_test))):
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            predicted = y_pred_rf[i]
            error = actual - predicted
            print(f"${actual:>14,.0f} ${predicted:>16,.0f} ${error:>11,.0f}")
        
        print("\n" + "=" * 80)


def run_all_examples():
    """
    Run all comprehensive examples
    """
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "COMPREHENSIVE EXAMPLES" + " " * 37 + "#")
    print("#" + " " * 15 + "End-to-End Model Evaluation Workflows" + " " * 28 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # Run all examples
    Example1_BinaryClassification.run()
    Example2_MulticlassClassification.run()
    Example3_Regression.run()
    
    print("\n\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 25 + "ALL EXAMPLES COMPLETED" + " " * 31 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    print("\n📚 KEY TAKEAWAYS:")
    print("=" * 80)
    print("1. Always use appropriate metrics for your problem type and business context")
    print("2. Use stratified splitting for classification, especially with imbalanced data")
    print("3. Cross-validation provides more reliable performance estimates than single split")
    print("4. Consider both model performance AND business implications")
    print("5. Confusion matrices reveal which errors your model makes")
    print("6. Compare multiple models before settling on one")
    print("7. Document your evaluation methodology for reproducibility")
    print("=" * 80)


if __name__ == "__main__":
    run_all_examples()

"""
Logistic Regression with Scikit-learn
======================================

Binary classification on an insurance purchase dataset using
sklearn's LogisticRegression with the L-BFGS solver.

Demonstrates:
- Loading data from a remote CSV
- Train/test splitting
- Fitting sklearn LogisticRegression
- Predicted classes vs predicted probabilities
- Visualization of data, predictions, and probability curves

Source: codebasics ML tutorial series
    https://www.youtube.com/watch?v=zM4VZR0px8E

Author: Deep Learning Foundations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="logistic_regression_sklearn")
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load insurance purchase dataset and split into train/test."""
    url = (
        "https://raw.githubusercontent.com/codebasics/py/master/"
        "ML/7_logistic_reg/insurance_data.csv"
    )
    df = pd.read_csv(url)
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    x = df[["age"]].values.reshape((-1, 1))
    y = df.bought_insurance.values.reshape((-1,))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.5, random_state=ARGS.seed
    )
    print(f"\nTrain: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")
    return x_train, x_test, y_train, y_test

# ============================================================================
# Visualization
# ============================================================================

def plot_data(x, y):
    """Scatter plot of raw data."""
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, "o", ms=5, label="data")
    ax.set_xlabel("Age")
    ax.set_ylabel("Bought Insurance")
    ax.set_title("Insurance Purchase Data")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_result(x, y, y_pred, y_pred_prob):
    """Overlay true labels, predicted classes, and predicted probabilities."""
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, "oy", alpha=0.9, ms=10, label="data")
    ax.plot(x, y_pred, "+r", ms=10, label="pred")
    ax.plot(x, y_pred_prob, "*b", alpha=0.3, ms=10, label="pred_prob")
    ax.set_xlabel("Age")
    ax.set_ylabel("Bought Insurance")
    ax.set_title("Logistic Regression (Sklearn) — Predictions vs Data")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ============================================================================
# Main
# ============================================================================

def main():
    x_train, x_test, y_train, y_test = load_data()
    plot_data(x_train, y_train)

    # Fit model
    model = LogisticRegression(solver="lbfgs")
    model.fit(x_train, y_train)
    print(f"\nCoefficient: {model.coef_[0][0]:.4f}")
    print(f"Intercept:   {model.intercept_[0]:.4f}")
    print(f"Train accuracy: {model.score(x_train, y_train):.4f}")
    print(f"Test accuracy:  {model.score(x_test, y_test):.4f}")

    # Predictions
    y_test_pred = model.predict(x_test)
    y_test_pred_prob = model.predict_proba(x_test)  # shape (n, 2)
    plot_result(x_test, y_test, y_test_pred, y_test_pred_prob[:, 1])


if __name__ == "__main__":
    main()

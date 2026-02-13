"""
Softmax Regression with Scikit-learn
======================================

Multi-class classification on the sklearn digits dataset (8×8 images,
10 classes) using sklearn's LogisticRegression with the L-BFGS solver
and multinomial objective.

Demonstrates:
- Loading the digits dataset
- Train/test splitting
- Multi-class logistic regression (softmax)
- Confusion matrix evaluation

Source: codebasics ML tutorial series
    https://www.youtube.com/watch?v=J5bXOOmkopc

Author: Deep Learning Foundations
"""

import argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="softmax_regression_sklearn")
parser.add_argument(
    "--max_iter", type=int, default=10_000,
    help="maximum iterations for solver (default: 10000)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S",
    help="random seed (default: 1)",
)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

# ============================================================================
# Main
# ============================================================================

def main():
    # Load data
    digits = load_digits()
    print(f"Data shape: {digits.data.shape}, Target shape: {digits.target.shape}")
    print(f"Data dtype: {digits.data.dtype}, Target dtype: {digits.target.dtype}")

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=ARGS.seed
    )
    print(f"Train: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")

    # Train
    model = LogisticRegression(solver="lbfgs", max_iter=ARGS.max_iter)
    model.fit(x_train, y_train)
    print(f"\nTrain accuracy: {model.score(x_train, y_train):.4f}")
    print(f"Test accuracy:  {model.score(x_test, y_test):.4f}")

    # Predictions
    y_test_pred = model.predict(x_test)
    y_test_pred_prob = model.predict_proba(x_test)
    print(f"Prediction probabilities shape: {y_test_pred_prob.shape}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()

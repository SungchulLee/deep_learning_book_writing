"""
Logistic Regression with Gradient Descent (from scratch)
=========================================================

Binary classification on an insurance purchase dataset using
a NumPy implementation of logistic regression trained with
batch gradient descent.

Demonstrates:
- Design matrix construction (prepending ones for bias)
- Sigmoid function
- Cross-entropy loss with numerical stability
- Gradient computation: g = X^T (p - y)
- Manual gradient descent training loop
- Comparison of from-scratch predictions with sklearn

Source: codebasics ML tutorial series
    https://www.youtube.com/watch?v=-Z2a_mzl9LM (parts 1-5)

Author: Deep Learning Foundations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(
    description="logistic_regression_gradient_descent"
)
parser.add_argument(
    "--lr", type=float, default=2e-4, metavar="LR",
    help="learning rate (default: 2e-4)",
)
parser.add_argument(
    "--epochs", type=int, default=100_000, metavar="N",
    help="number of epochs to train (default: 100000)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S",
    help="random seed (default: 1)",
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

    x = df[["age"]].values.reshape((-1, 1))
    y = df.bought_insurance.values.reshape((-1,))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.5, random_state=ARGS.seed
    )
    print(f"Train: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")
    return x_train, x_test, y_train, y_test

# ============================================================================
# Model
# ============================================================================

class LogisticRegression:
    """
    Logistic regression trained with batch gradient descent.

    Parameters are stored as theta of shape (d+1, 1), where d is the
    number of input features.  The design matrix prepends a column of
    ones so that theta[0] acts as the bias/intercept.
    """

    def __init__(self, x, y, theta=None):
        self.x = x
        self.y = y
        self.lr = ARGS.lr
        self.epoch = ARGS.epochs

        if theta is None:
            self.theta = np.random.normal(size=(x.shape[1] + 1, 1))
        else:
            self.theta = theta

    @staticmethod
    def design_matrix(x):
        """Prepend a column of ones: A = [1 | x]."""
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    @staticmethod
    def sigmoid(z):
        """Element-wise sigmoid: σ(z) = 1 / (1 + exp(-z))."""
        return 1.0 / (1.0 + np.exp(-z))

    def gradient(self):
        """Compute gradient: g = A^T (p - y)."""
        A = self.design_matrix(self.x)
        p = self.predict_proba(self.x).reshape((-1, 1))
        y = self.y.reshape((-1, 1))
        return A.T @ (p - y)

    def train(self):
        """Run batch gradient descent for self.epoch iterations."""
        for i in range(self.epoch):
            grad = self.gradient()
            self.theta -= self.lr * grad

    def loss(self):
        """Compute cross-entropy loss with numerical stability."""
        p = self.predict_proba(self.x)
        eps = 1e-6
        return -np.mean(
            self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps)
        )

    def predict_proba(self, x):
        """Return P(y=1 | x) for each sample."""
        A = self.design_matrix(x)
        z = A @ self.theta
        return self.sigmoid(z).reshape((-1,))

    def predict(self, x):
        """Return binary predictions with threshold 0.5."""
        p = self.predict_proba(x)
        decision = np.zeros((x.shape[0],))
        decision[p > 0.5] = 1
        return decision

# ============================================================================
# Visualization
# ============================================================================

def plot_data(x, y):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, "o", ms=5, label="data")
    ax.set_xlabel("Age")
    ax.set_ylabel("Bought Insurance")
    ax.set_title("Insurance Purchase Data")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_result(x, y, y_pred, y_pred_prob):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, "oy", alpha=0.9, ms=10, label="data")
    ax.plot(x, y_pred, "+r", ms=10, label="pred")
    ax.plot(x, y_pred_prob, "*b", alpha=0.3, ms=10, label="pred_prob")
    ax.set_xlabel("Age")
    ax.set_ylabel("Bought Insurance")
    ax.set_title("Logistic Regression (Gradient Descent) — Predictions vs Data")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ============================================================================
# Main
# ============================================================================

def main():
    x_train, x_test, y_train, y_test = load_data()
    plot_data(x_train, y_train)

    model = LogisticRegression(x_train, y_train)
    print(f"\nInitial loss: {model.loss():.4f}")

    model.train()
    print(f"Final loss:   {model.loss():.4f}")
    print(f"Learned theta: {model.theta.flatten()}")

    y_test_pred = model.predict(x_test)
    y_test_pred_prob = model.predict_proba(x_test)
    accuracy = np.mean(y_test_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    plot_result(x_test, y_test, y_test_pred, y_test_pred_prob)


if __name__ == "__main__":
    main()

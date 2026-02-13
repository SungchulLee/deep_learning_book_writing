"""
Softmax Regression with Gradient Descent (from scratch)
=========================================================

Two-layer neural network (784 → 100 → 10) with logistic hidden
activation and softmax output, trained with manual backpropagation
on MNIST using mini-batch gradient descent.

Architecture:
    Z_h = X W_h + b_h          (n × 100)
    H   = logistic(Z_h)        (n × 100)
    Z_o = H W_o + b_o          (n × 10)
    Yhat = softmax(Z_o)        (n × 10)
    J   = -sum(Y * log(Yhat))

Gradients (see Section 2 Theory):
    dJ/dW_o = H^T (Yhat - Y)
    dJ/db_o = 1^T (Yhat - Y)
    dJ/dW_h = X^T [H*(1-H)*(Yhat-Y) W_o^T]
    dJ/db_h = 1^T [H*(1-H)*(Yhat-Y) W_o^T]

Demonstrates:
- One-hot encoding via np.eye
- Numerically stable softmax
- Full backpropagation through a hidden layer
- Mini-batch SGD training loop
- Confusion matrix and wrong-prediction visualization

Author: Deep Learning Foundations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(
    description="softmax_regression_gradient_descent"
)
parser.add_argument(
    "--lr", type=float, default=1e-2, metavar="LR",
    help="learning rate (default: 1e-2)",
)
parser.add_argument(
    "--epochs", type=int, default=50, metavar="N",
    help="number of epochs to train (default: 50)",
)
parser.add_argument(
    "--batch_size", type=int, default=100, metavar="N",
    help="mini-batch size (default: 100)",
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
    """
    Load MNIST via TensorFlow/Keras, normalize to [0,1],
    flatten to 784, and one-hot encode labels.
    """
    import tensorflow as tf

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and flatten
    x_train = (x_train / 255.0).reshape((-1, 784)).astype(np.float32)
    x_test = (x_test / 255.0).reshape((-1, 784)).astype(np.float32)

    # Integer class labels
    y_train_cls = y_train.astype(np.int32)
    y_test_cls = y_test.astype(np.int32)

    # One-hot encoded labels
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    print(f"Train: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")
    print(f"Features: {x_train.shape[1]}, Classes: {y_train.shape[1]}")
    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls

# ============================================================================
# Model Functions
# ============================================================================

logistic = lambda z: 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """Row-wise softmax with numerical stability."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def initialize_weights():
    """Initialize weights with standard normal (small scale implicit)."""
    w_h = np.random.randn(784, 100)
    b_h = np.random.randn(1, 100)
    w_o = np.random.randn(100, 10)
    b_o = np.random.randn(1, 10)
    return w_h, b_h, w_o, b_o


def feed_forward(x, y, y_cls, w_h, b_h, w_o, b_o):
    """Forward pass through the two-layer network."""
    z_h = x @ w_h + b_h
    h = logistic(z_h)
    z_o = h @ w_o + b_o
    y_hat = softmax(z_o)
    y_hat_cls = np.argmax(y_hat, axis=1)
    loss = -(y * np.log(y_hat + 1e-12)).sum()
    accuracy = (y_cls == y_hat_cls).sum() / y_cls.size
    return h, y_hat, y_hat_cls, loss, accuracy


def back_propagation(x, y, h, y_hat, w_o):
    """
    Backward pass computing gradients for all parameters.

    Key derivations:
        dJ/dZ_o = Yhat - Y
        dJ/dW_o = H^T @ (Yhat - Y)
        dJ/dH   = (Yhat - Y) @ W_o^T
        dJ/dZ_h = H * (1 - H) * dJ/dH
        dJ/dW_h = X^T @ dJ/dZ_h
    """
    loss_grad = y_hat - y  # dJ/dZ_o: (n, 10)
    w_o_grad = h.T @ loss_grad  # (100, 10)
    b_o_grad = np.sum(loss_grad, axis=0, keepdims=True)  # (1, 10)
    h_grad = h * (1 - h) * (loss_grad @ w_o.T)  # dJ/dZ_h: (n, 100)
    w_h_grad = x.T @ h_grad  # (784, 100)
    b_h_grad = np.sum(h_grad, axis=0, keepdims=True)  # (1, 100)
    return w_h_grad, b_h_grad, w_o_grad, b_o_grad

# ============================================================================
# Visualization
# ============================================================================

def print_cm(y_test_cls, y_test_cls_pred):
    cm = confusion_matrix(y_true=y_test_cls, y_pred=y_test_cls_pred)
    print(f"\nConfusion Matrix:\n{cm}")


def draw_loss_and_accuracy_trace(loss_trace, accuracy_trace):
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))
    for ax, trace, title in zip(
        (ax0, ax1), (loss_trace, accuracy_trace), ("Loss", "Accuracy")
    ):
        ax.plot(trace)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def draw_10_wrong_preds(x_test, y_test_cls, y_test_cls_pred):
    _, axes = plt.subplots(1, 10, figsize=(12, 3))
    idx = 0
    for ax in axes:
        while True:
            if y_test_cls[idx] == y_test_cls_pred[idx]:
                idx += 1
            else:
                ax.imshow(x_test[idx].reshape((28, 28)), cmap="binary")
                ax.set_title(
                    f"True: {y_test_cls[idx]}\nPred: {y_test_cls_pred[idx]}",
                    fontsize=10,
                )
                ax.axis("off")
                idx += 1
                break
    plt.tight_layout()
    plt.show()

# ============================================================================
# Training
# ============================================================================

def train_step(x, y, y_cls, w_h, b_h, w_o, b_o):
    """Single forward + backward + parameter update."""
    h, y_hat, _, loss, accuracy = feed_forward(x, y, y_cls, w_h, b_h, w_o, b_o)
    grads = back_propagation(x, y, h, y_hat, w_o)
    for para, grad in zip((w_h, b_h, w_o, b_o), grads):
        para -= ARGS.lr * grad
    return loss, accuracy, w_h, b_h, w_o, b_o


def run_train_loop(x_train, y_train, y_train_cls, w_h, b_h, w_o, b_o):
    """Mini-batch SGD training over multiple epochs."""
    loss_trace = []
    accuracy_trace = []

    for i in range(ARGS.epochs):
        # Shuffle data each epoch
        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)
        x_epoch = x_train[idx]
        y_epoch = y_train[idx]
        y_cls_epoch = y_train_cls[idx]

        loss_temp = []
        accuracy_temp = []

        for k in range(x_train.shape[0] // ARGS.batch_size):
            s = k * ARGS.batch_size
            e = (k + 1) * ARGS.batch_size
            x = x_epoch[s:e]
            y = y_epoch[s:e]
            y_cls = y_cls_epoch[s:e]

            loss_run, accuracy_run, w_h, b_h, w_o, b_o = train_step(
                x, y, y_cls, w_h, b_h, w_o, b_o
            )
            loss_temp.append(loss_run)
            accuracy_temp.append(accuracy_run)

        loss_avg = np.mean(loss_temp)
        accuracy_avg = np.mean(accuracy_temp)
        loss_trace.append(loss_avg)
        accuracy_trace.append(accuracy_avg)
        print(
            f"  {i + 1}/{ARGS.epochs}  loss {loss_avg:.2f}  "
            f"accuracy {accuracy_avg:.4f}"
        )

    return w_h, b_h, w_o, b_o, loss_trace, accuracy_trace

# ============================================================================
# Main
# ============================================================================

def main():
    x_train, y_train, y_train_cls, x_test, y_test, y_test_cls = load_data()

    w_h, b_h, w_o, b_o = initialize_weights()

    print(f"\nTraining for {ARGS.epochs} epochs, lr={ARGS.lr}, "
          f"batch_size={ARGS.batch_size}")
    w_h, b_h, w_o, b_o, loss_trace, accuracy_trace = run_train_loop(
        x_train, y_train, y_train_cls, w_h, b_h, w_o, b_o
    )
    draw_loss_and_accuracy_trace(loss_trace, accuracy_trace)

    # Evaluate on test set
    _, _, y_test_cls_pred, _, test_accuracy = feed_forward(
        x_test, y_test, y_test_cls, w_h, b_h, w_o, b_o
    )
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print_cm(y_test_cls, y_test_cls_pred)
    draw_10_wrong_preds(x_test, y_test_cls, y_test_cls_pred)


if __name__ == "__main__":
    main()

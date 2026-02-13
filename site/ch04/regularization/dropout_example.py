"""
Dropout Regularization Example
===============================
Demonstrates how dropout prevents overfitting by randomly dropping neurons during training.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model_without_dropout(input_dim):
    """Create a neural network without dropout."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_model_with_dropout(input_dim, dropout_rate=0.5):
    """Create a neural network with dropout layers."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=5, random_state=42)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model without dropout
    print("Training model WITHOUT dropout...")
    model_no_dropout = create_model_without_dropout(X_train.shape[1])
    history_no_dropout = model_no_dropout.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    # Train model with dropout
    print("Training model WITH dropout...")
    model_with_dropout = create_model_with_dropout(X_train.shape[1], dropout_rate=0.5)
    history_with_dropout = model_with_dropout.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate models
    _, test_acc_no_dropout = model_no_dropout.evaluate(X_test, y_test, verbose=0)
    _, test_acc_with_dropout = model_with_dropout.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Accuracy without Dropout: {test_acc_no_dropout:.4f}")
    print(f"Test Accuracy with Dropout: {test_acc_with_dropout:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_no_dropout.history['loss'], label='Train Loss (No Dropout)')
    plt.plot(history_no_dropout.history['val_loss'], label='Val Loss (No Dropout)')
    plt.plot(history_with_dropout.history['loss'], label='Train Loss (With Dropout)', linestyle='--')
    plt.plot(history_with_dropout.history['val_loss'], label='Val Loss (With Dropout)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_no_dropout.history['accuracy'], label='Train Acc (No Dropout)')
    plt.plot(history_no_dropout.history['val_accuracy'], label='Val Acc (No Dropout)')
    plt.plot(history_with_dropout.history['accuracy'], label='Train Acc (With Dropout)', linestyle='--')
    plt.plot(history_with_dropout.history['val_accuracy'], label='Val Acc (With Dropout)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dropout_comparison.png')
    print("\nPlot saved as 'dropout_comparison.png'")


if __name__ == "__main__":
    main()

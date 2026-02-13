"""
L1 and L2 Regularization Example
=================================
Demonstrates how L1 (Lasso) and L2 (Ridge) regularization prevent overfitting
by adding penalty terms to the loss function.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def create_model_no_regularization(input_dim):
    """Create a neural network without regularization."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_model_l1_regularization(input_dim, l1_factor=0.01):
    """Create a neural network with L1 regularization."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim,
                    kernel_regularizer=regularizers.l1(l1_factor)),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l1(l1_factor)),
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l1(l1_factor)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_model_l2_regularization(input_dim, l2_factor=0.01):
    """Create a neural network with L2 regularization."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim,
                    kernel_regularizer=regularizers.l2(l2_factor)),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_factor)),
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_factor)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_model_l1_l2_regularization(input_dim, l1_factor=0.01, l2_factor=0.01):
    """Create a neural network with both L1 and L2 regularization (Elastic Net)."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim,
                    kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def sklearn_regularization_demo():
    """Demonstrate L1/L2 regularization using scikit-learn."""
    print("="*60)
    print("Scikit-learn Regularization Demo")
    print("="*60)
    
    # Generate dataset
    X, y = make_regression(n_samples=200, n_features=50, n_informative=20,
                          noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'No Regularization': LinearRegression(),
        'L2 (Ridge)': Ridge(alpha=1.0),
        'L1 (Lasso)': Lasso(alpha=0.1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Count non-zero coefficients
        non_zero = np.sum(np.abs(model.coef_) > 1e-5)
        
        results[name] = {
            'train_r2': train_score,
            'test_r2': test_score,
            'non_zero_coefs': non_zero
        }
        
        print(f"\n{name}:")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²: {test_score:.4f}")
        print(f"  Non-zero coefficients: {non_zero}/{len(model.coef_)}")
    
    # Visualize coefficients
    plt.figure(figsize=(15, 4))
    for idx, (name, model) in enumerate(models.items(), 1):
        plt.subplot(1, 3, idx)
        plt.bar(range(len(model.coef_)), model.coef_)
        plt.title(f'{name}\nCoefficients')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('l1_l2_coefficients.png')
    print("\nCoefficients plot saved as 'l1_l2_coefficients.png'")


def neural_network_regularization_demo():
    """Demonstrate L1/L2 regularization in neural networks."""
    print("\n" + "="*60)
    print("Neural Network Regularization Demo")
    print("="*60)
    
    # Generate dataset
    X, y = make_regression(n_samples=500, n_features=20, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'No Regularization': create_model_no_regularization(X_train_scaled.shape[1]),
        'L1 Regularization': create_model_l1_regularization(X_train_scaled.shape[1], 0.001),
        'L2 Regularization': create_model_l2_regularization(X_train_scaled.shape[1], 0.001),
        'L1+L2 (Elastic Net)': create_model_l1_l2_regularization(X_train_scaled.shape[1], 0.001, 0.001)
    }
    
    histories = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            verbose=0
        )
        histories[name] = history
        
        # Evaluate
        train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
        test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} (train)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['val_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('nn_regularization_comparison.png')
    print("\nNeural network plot saved as 'nn_regularization_comparison.png'")


def main():
    # Run both demonstrations
    sklearn_regularization_demo()
    neural_network_regularization_demo()
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("• L1 (Lasso): Pushes coefficients to zero → Feature selection")
    print("• L2 (Ridge): Shrinks coefficients smoothly → Prevents large weights")
    print("• L1+L2 (Elastic Net): Combines benefits of both approaches")
    print("• Regularization helps prevent overfitting and improves generalization")


if __name__ == "__main__":
    main()

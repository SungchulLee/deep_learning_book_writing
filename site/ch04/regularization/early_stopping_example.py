"""
Early Stopping Example
=======================
Demonstrates how early stopping prevents overfitting by stopping training
when validation performance stops improving.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def create_model(input_dim):
    """Create a neural network model."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model


def train_without_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """Train model without early stopping."""
    print("Training WITHOUT early stopping...")
    model = create_model(X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )
    
    return model, history


def train_with_early_stopping(X_train, y_train, X_val, y_val, 
                               patience=10, epochs=200):
    """Train model with early stopping."""
    print(f"Training WITH early stopping (patience={patience})...")
    model = create_model(X_train.shape[1])
    
    # Create early stopping callback
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',          # Metric to monitor
        patience=patience,            # Number of epochs with no improvement
        restore_best_weights=True,   # Restore weights from best epoch
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history


def train_with_advanced_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """Train with multiple callbacks including model checkpointing."""
    print("Training with ADVANCED early stopping (multiple callbacks)...")
    model = create_model(X_train.shape[1])
    
    # Multiple callbacks
    callback_list = [
        # Early stopping based on validation loss
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model
        callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callback_list,
        verbose=0
    )
    
    return model, history


def plot_comparison(histories, labels):
    """Plot training histories for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for history, label in zip(histories, labels):
            if metric in history.history:
                epochs = range(1, len(history.history[metric]) + 1)
                ax.plot(epochs, history.history[metric], label=label, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png', dpi=150)
    print("\nPlot saved as 'early_stopping_comparison.png'")


def main():
    # Generate synthetic dataset
    print("Generating dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # Train models
    model_no_es, history_no_es = train_without_early_stopping(
        X_train, y_train, X_val, y_val, epochs=200
    )
    
    model_es_10, history_es_10 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, patience=10, epochs=200
    )
    
    model_es_20, history_es_20 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, patience=20, epochs=200
    )
    
    model_advanced, history_advanced = train_with_advanced_early_stopping(
        X_train, y_train, X_val, y_val, epochs=200
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Performance")
    print("="*60)
    
    models = {
        'No Early Stopping': model_no_es,
        'Early Stop (patience=10)': model_es_10,
        'Early Stop (patience=20)': model_es_20,
        'Advanced Early Stop': model_advanced
    }
    
    for name, model in models.items():
        loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{name}:")
        print(f"  Test Loss: {loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test AUC: {auc:.4f}\n")
    
    # Plot comparisons
    histories = [history_no_es, history_es_10, history_es_20, history_advanced]
    labels = [
        'No Early Stopping',
        'Early Stop (p=10)',
        'Early Stop (p=20)',
        'Advanced'
    ]
    plot_comparison(histories, labels)
    
    # Print summary
    print("\n" + "="*60)
    print("Key Insights")
    print("="*60)
    print(f"• Without early stopping: Trained for {len(history_no_es.history['loss'])} epochs")
    print(f"• With early stopping (p=10): Stopped at epoch {len(history_es_10.history['loss'])}")
    print(f"• With early stopping (p=20): Stopped at epoch {len(history_es_20.history['loss'])}")
    print(f"• Advanced callbacks: Stopped at epoch {len(history_advanced.history['loss'])}")
    print("\nBenefits of Early Stopping:")
    print("  1. Prevents overfitting by stopping before validation loss increases")
    print("  2. Saves computational time by not training unnecessary epochs")
    print("  3. Automatically finds optimal number of training epochs")
    print("  4. Can be combined with other techniques (learning rate reduction, etc.)")


if __name__ == "__main__":
    main()

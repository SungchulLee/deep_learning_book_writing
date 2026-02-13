"""
Module 56: Zero-Shot Learning - Beginner Level
==============================================

Introduction to Zero-Shot Learning Concepts

This script introduces the fundamental concept of zero-shot learning through
simple, intuitive examples. We'll explore:
1. What zero-shot learning is and why it's useful
2. Attribute-based classification
3. Nearest neighbor in semantic space
4. Simple synthetic examples

Author: Deep Learning Curriculum
Level: Beginner
Estimated Time: 45-60 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# PART 1: Understanding Zero-Shot Learning
# ============================================================================

def demonstrate_traditional_vs_zsl():
    """
    Demonstrate the difference between traditional supervised learning
    and zero-shot learning.
    
    Traditional Learning:
    - Training: See examples of cats, dogs, birds
    - Testing: Classify cats, dogs, birds (same classes)
    
    Zero-Shot Learning:
    - Training: See examples of cats, dogs, birds
    - Testing: Classify horses, zebras, elephants (NEW classes!)
    
    The key: We use auxiliary information (attributes, descriptions) to
    generalize to unseen classes.
    """
    print("="*70)
    print("TRADITIONAL SUPERVISED LEARNING vs ZERO-SHOT LEARNING")
    print("="*70)
    
    # Traditional supervised learning
    print("\n[TRADITIONAL LEARNING]")
    print("Training classes: {cat, dog, bird}")
    print("Test classes:     {cat, dog, bird}")
    print("Can the model classify a 'horse'? NO - never seen it!")
    
    # Zero-shot learning
    print("\n[ZERO-SHOT LEARNING]")
    print("Training classes:  {cat, dog, bird}")
    print("Test classes:      {horse, zebra, elephant}")
    print("Can the model classify a 'horse'? YES! How?")
    print("→ Use descriptions/attributes shared between seen and unseen classes")
    
    # Example attributes
    print("\n[ATTRIBUTE BRIDGE]")
    attributes = {
        'cat':   {'has_fur': 1, 'has_4_legs': 1, 'is_large': 0, 'has_stripes': 0},
        'dog':   {'has_fur': 1, 'has_4_legs': 1, 'is_large': 0, 'has_stripes': 0},
        'bird':  {'has_fur': 0, 'has_4_legs': 0, 'is_large': 0, 'has_stripes': 0},
        'horse': {'has_fur': 1, 'has_4_legs': 1, 'is_large': 1, 'has_stripes': 0},
        'zebra': {'has_fur': 1, 'has_4_legs': 1, 'is_large': 1, 'has_stripes': 1},
        'elephant': {'has_fur': 0, 'has_4_legs': 1, 'is_large': 1, 'has_stripes': 0},
    }
    
    for animal, attrs in attributes.items():
        seen_marker = "[SEEN]" if animal in ['cat', 'dog', 'bird'] else "[UNSEEN]"
        print(f"{animal:10s} {seen_marker:10s}: {attrs}")
    
    print("\nKey insight: Unseen animals share attributes with seen animals!")
    print("→ 'horse' is similar to 'dog' (both have fur, 4 legs)")
    print("→ 'zebra' is unique with stripes, but shares other attributes")
    print("="*70)


# ============================================================================
# PART 2: Attribute-Based Zero-Shot Classification
# ============================================================================

class SimpleAttributeZSL:
    """
    A simple zero-shot learning classifier using attributes.
    
    The model:
    1. Learns to predict attributes from input features during training
    2. At test time, classifies unseen instances by comparing predicted
       attributes to known attribute descriptions of unseen classes
    
    Mathematical formulation:
    - Training: Learn f: X → A (input to attributes)
    - Testing: For test instance x with unseen class c*
              1. Predict attributes: a_pred = f(x)
              2. Compare to class attributes: a_c for c in unseen_classes
              3. Assign to most similar: c* = argmin_c ||a_pred - a_c||²
    """
    
    def __init__(self, n_features: int, n_attributes: int):
        """
        Initialize the zero-shot learning model.
        
        Args:
            n_features: Dimensionality of input features
            n_attributes: Number of binary or continuous attributes
        """
        self.n_features = n_features
        self.n_attributes = n_attributes
        
        # Simple linear model: predict each attribute independently
        # W shape: (n_attributes, n_features)
        # For each attribute i: a_i = sigmoid(W_i · x)
        self.W = np.random.randn(n_attributes, n_features) * 0.1
        
        # Store attribute descriptions for all classes (seen and unseen)
        self.class_attributes = {}  # class_name -> attribute_vector
        
        # Track which classes were seen during training
        self.seen_classes = set()
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict_attributes(self, X):
        """
        Predict attribute probabilities for input features.
        
        Args:
            X: Input features of shape (n_samples, n_features)
        
        Returns:
            Predicted attributes of shape (n_samples, n_attributes)
            Each value in [0, 1] indicating probability of attribute presence
        """
        # Linear transformation followed by sigmoid
        # Shape: (n_samples, n_attributes)
        logits = X @ self.W.T  # Matrix multiplication: (n, f) @ (a, f).T = (n, a)
        attributes = self.sigmoid(logits)
        return attributes
    
    def fit(self, X_train, y_train, class_attributes, learning_rate=0.01, epochs=100):
        """
        Train the attribute prediction model on seen classes.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels (class names)
            class_attributes: Dictionary mapping class names to attribute vectors
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
        """
        # Store class attributes
        self.class_attributes = class_attributes
        
        # Track seen classes
        self.seen_classes = set(y_train)
        
        # Convert labels to attribute vectors for supervised learning
        # For each training instance, we know its true class and thus its true attributes
        y_attributes = np.array([class_attributes[label] for label in y_train])
        
        # Training loop: optimize attribute prediction
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Forward pass: predict attributes
            pred_attributes = self.predict_attributes(X_train)
            
            # Compute loss: binary cross-entropy for each attribute
            # L = -∑_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
            # Clip predictions to avoid log(0)
            pred_clipped = np.clip(pred_attributes, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                y_attributes * np.log(pred_clipped) + 
                (1 - y_attributes) * np.log(1 - pred_clipped)
            )
            
            # Backward pass: compute gradients
            # Gradient of binary cross-entropy w.r.t. predictions
            grad_pred = (pred_attributes - y_attributes) / n_samples
            
            # Gradient w.r.t. weights using chain rule
            # dL/dW = dL/dpred * dpred/dlogits * dlogits/dW
            # For sigmoid: dpred/dlogits = pred * (1 - pred)
            grad_logits = grad_pred * pred_attributes * (1 - pred_attributes)
            grad_W = grad_logits.T @ X_train  # Shape: (n_attributes, n_features)
            
            # Update weights
            self.W -= learning_rate * grad_W
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f}")
    
    def predict(self, X_test, candidate_classes):
        """
        Predict classes for test instances using zero-shot learning.
        
        Args:
            X_test: Test features of shape (n_samples, n_features)
            candidate_classes: List of class names to consider (typically unseen)
        
        Returns:
            Predicted class labels
        """
        # Step 1: Predict attributes for test instances
        pred_attributes = self.predict_attributes(X_test)
        
        # Step 2: Get attribute vectors for candidate classes
        candidate_attr = np.array([
            self.class_attributes[c] for c in candidate_classes
        ])  # Shape: (n_classes, n_attributes)
        
        # Step 3: Find nearest class in attribute space
        # Use Euclidean distance in attribute space
        predictions = []
        for attr in pred_attributes:
            # Compute distance to each candidate class
            # ||attr - candidate_attr[i]||²
            distances = np.sum((candidate_attr - attr)**2, axis=1)
            
            # Assign to nearest class
            best_idx = np.argmin(distances)
            predictions.append(candidate_classes[best_idx])
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test, candidate_classes):
        """
        Evaluate zero-shot learning performance.
        
        Args:
            X_test: Test features
            y_test: True test labels
            candidate_classes: Classes to consider for prediction
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X_test, candidate_classes)
        accuracy = np.mean(predictions == y_test)
        return accuracy


# ============================================================================
# PART 3: Synthetic Example - Animal Classification
# ============================================================================

def generate_synthetic_animal_data():
    """
    Generate synthetic data for animal classification.
    
    We'll create:
    - Seen classes during training: cat, dog, bird
    - Unseen classes for testing: horse, zebra, elephant
    
    Each animal is represented by visual features (simulated) and
    semantic attributes (fur, legs, size, etc.)
    """
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC ANIMAL DATA")
    print("="*70)
    
    # Define attributes for each animal class
    # Attributes: [has_fur, has_4_legs, is_large, has_stripes, can_fly]
    class_attributes = {
        # Seen classes (training)
        'cat':   np.array([1.0, 1.0, 0.0, 0.0, 0.0]),  # Small, furry, 4 legs
        'dog':   np.array([1.0, 1.0, 0.1, 0.0, 0.0]),  # Small-medium, furry, 4 legs
        'bird':  np.array([0.0, 0.0, 0.0, 0.0, 1.0]),  # Small, can fly, 2 legs
        
        # Unseen classes (testing)
        'horse': np.array([1.0, 1.0, 0.9, 0.0, 0.0]),  # Large, furry, 4 legs
        'zebra': np.array([1.0, 1.0, 0.9, 1.0, 0.0]),  # Large, furry, 4 legs, striped!
        'elephant': np.array([0.0, 1.0, 1.0, 0.0, 0.0]),  # Large, 4 legs, no fur
    }
    
    # Feature dimensionality (simulated visual features)
    n_features = 50
    
    # Generate synthetic visual features for each class
    # Features are correlated with attributes
    def generate_class_features(attributes, n_samples=100):
        """Generate features correlated with attributes."""
        # Each attribute contributes to certain feature dimensions
        features = np.zeros((n_samples, n_features))
        
        # Add attribute-correlated components
        for i, attr_val in enumerate(attributes):
            # Each attribute affects 10 dimensions
            start_idx = i * 10
            end_idx = start_idx + 10
            features[:, start_idx:end_idx] = (
                attr_val + np.random.randn(n_samples, 10) * 0.3
            )
        
        # Add some noise to remaining dimensions
        features[:, len(attributes)*10:] = np.random.randn(n_samples, n_features - len(attributes)*10) * 0.1
        
        return features
    
    # Generate training data (seen classes)
    n_train_per_class = 100
    X_train_list = []
    y_train_list = []
    
    seen_classes = ['cat', 'dog', 'bird']
    for class_name in seen_classes:
        features = generate_class_features(class_attributes[class_name], n_train_per_class)
        labels = [class_name] * n_train_per_class
        X_train_list.append(features)
        y_train_list.extend(labels)
    
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    
    # Generate test data (unseen classes)
    n_test_per_class = 50
    X_test_list = []
    y_test_list = []
    
    unseen_classes = ['horse', 'zebra', 'elephant']
    for class_name in unseen_classes:
        features = generate_class_features(class_attributes[class_name], n_test_per_class)
        labels = [class_name] * n_test_per_class
        X_test_list.append(features)
        y_test_list.extend(labels)
    
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)
    
    print(f"\nTraining set:")
    print(f"  Seen classes: {seen_classes}")
    print(f"  Shape: {X_train.shape}")
    print(f"  Samples per class: {n_train_per_class}")
    
    print(f"\nTest set:")
    print(f"  Unseen classes: {unseen_classes}")
    print(f"  Shape: {X_test.shape}")
    print(f"  Samples per class: {n_test_per_class}")
    
    print(f"\nAttribute descriptions:")
    print(f"  Attributes: [has_fur, has_4_legs, is_large, has_stripes, can_fly]")
    for class_name, attrs in class_attributes.items():
        marker = "[SEEN]  " if class_name in seen_classes else "[UNSEEN]"
        print(f"  {marker} {class_name:10s}: {attrs}")
    
    return X_train, y_train, X_test, y_test, class_attributes, seen_classes, unseen_classes


# ============================================================================
# PART 4: Training and Evaluation
# ============================================================================

def train_and_evaluate_zsl():
    """
    Complete pipeline for zero-shot learning:
    1. Generate synthetic data
    2. Train attribute predictors on seen classes
    3. Test on unseen classes using zero-shot learning
    """
    print("\n" + "="*70)
    print("ZERO-SHOT LEARNING PIPELINE")
    print("="*70)
    
    # Generate data
    X_train, y_train, X_test, y_test, class_attributes, seen_classes, unseen_classes = \
        generate_synthetic_animal_data()
    
    # Initialize and train model
    print("\n[TRAINING PHASE]")
    print("Training attribute predictors on SEEN classes...")
    
    n_features = X_train.shape[1]
    n_attributes = len(next(iter(class_attributes.values())))
    
    model = SimpleAttributeZSL(n_features, n_attributes)
    model.fit(X_train, y_train, class_attributes, learning_rate=0.1, epochs=100)
    
    # Test on unseen classes
    print("\n[TESTING PHASE]")
    print("Predicting UNSEEN classes using zero-shot learning...")
    
    accuracy = model.evaluate(X_test, y_test, unseen_classes)
    print(f"\nZero-shot accuracy on unseen classes: {accuracy*100:.2f}%")
    
    # Detailed per-class analysis
    print("\n[PER-CLASS ANALYSIS]")
    predictions = model.predict(X_test, unseen_classes)
    
    for class_name in unseen_classes:
        # Get predictions for this class
        class_mask = y_test == class_name
        class_preds = predictions[class_mask]
        class_acc = np.mean(class_preds == class_name)
        
        print(f"\n{class_name:10s}: {class_acc*100:.2f}% accuracy")
        
        # Show confusion
        from collections import Counter
        pred_counts = Counter(class_preds)
        print(f"  Predicted as: {dict(pred_counts)}")
    
    # Visualize attribute predictions
    visualize_attribute_predictions(model, X_test, y_test, class_attributes, unseen_classes)
    
    return model, accuracy


def visualize_attribute_predictions(model, X_test, y_test, class_attributes, unseen_classes):
    """
    Visualize how well the model predicts attributes for unseen classes.
    """
    # Predict attributes for test samples
    pred_attributes = model.predict_attributes(X_test)
    
    # Average predictions per class
    avg_pred_per_class = {}
    true_attr_per_class = {}
    
    for class_name in unseen_classes:
        class_mask = y_test == class_name
        avg_pred_per_class[class_name] = pred_attributes[class_mask].mean(axis=0)
        true_attr_per_class[class_name] = class_attributes[class_name]
    
    # Create visualization
    fig, axes = plt.subplots(1, len(unseen_classes), figsize=(15, 4))
    attribute_names = ['has_fur', '4_legs', 'large', 'stripes', 'fly']
    
    for idx, (ax, class_name) in enumerate(zip(axes, unseen_classes)):
        predicted = avg_pred_per_class[class_name]
        true = true_attr_per_class[class_name]
        
        x = np.arange(len(attribute_names))
        width = 0.35
        
        ax.bar(x - width/2, true, width, label='True', alpha=0.8, color='green')
        ax.bar(x + width/2, predicted, width, label='Predicted', alpha=0.8, color='blue')
        
        ax.set_xlabel('Attributes')
        ax.set_ylabel('Value')
        ax.set_title(f'{class_name.capitalize()}')
        ax.set_xticks(x)
        ax.set_xticklabels(attribute_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/attribute_predictions.png', dpi=150, bbox_inches='tight')
    print(f"\n[VISUALIZATION]")
    print(f"Saved attribute prediction comparison to: attribute_predictions.png")
    plt.close()


# ============================================================================
# PART 5: Comparison with Random Baseline
# ============================================================================

def compare_with_baseline(X_test, y_test, unseen_classes):
    """
    Compare zero-shot learning with random chance baseline.
    """
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    
    # Random baseline: random guessing among unseen classes
    n_test = len(y_test)
    random_predictions = np.random.choice(unseen_classes, size=n_test)
    random_accuracy = np.mean(random_predictions == y_test)
    
    print(f"\nRandom baseline accuracy: {random_accuracy*100:.2f}%")
    print(f"Expected for {len(unseen_classes)} classes: {100/len(unseen_classes):.2f}%")
    
    # Run our zero-shot model
    X_train, y_train, _, _, class_attributes, _, _ = generate_synthetic_animal_data()
    
    model = SimpleAttributeZSL(X_train.shape[1], 5)
    model.fit(X_train, y_train, class_attributes, learning_rate=0.1, epochs=50)
    zsl_accuracy = model.evaluate(X_test, y_test, unseen_classes)
    
    print(f"Zero-shot learning accuracy: {zsl_accuracy*100:.2f}%")
    print(f"\nImprovement over random: {(zsl_accuracy - random_accuracy)*100:.2f} percentage points")
    print(f"Relative improvement: {(zsl_accuracy / random_accuracy - 1)*100:.2f}%")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run all demonstrations."""
    print("\n" + "="*70)
    print("MODULE 56: ZERO-SHOT LEARNING - BEGINNER LEVEL")
    print("="*70)
    
    # Part 1: Conceptual introduction
    demonstrate_traditional_vs_zsl()
    
    # Part 2 & 3: Generate data and train model
    model, accuracy = train_and_evaluate_zsl()
    
    # Part 4: Baseline comparison
    X_train, y_train, X_test, y_test, class_attributes, seen_classes, unseen_classes = \
        generate_synthetic_animal_data()
    compare_with_baseline(X_test, y_test, unseen_classes)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    Key takeaways:
    1. ✓ Zero-shot learning allows classification of unseen classes
    2. ✓ Requires auxiliary information (attributes, embeddings)
    3. ✓ Learns to predict attributes from features during training
    4. ✓ Matches predicted attributes to class descriptions at test time
    5. ✓ Performance significantly better than random guessing
    
    Limitations of this simple approach:
    - Assumes attributes are binary and independent
    - Linear attribute prediction may be too simple
    - Sensitive to quality of attribute annotations
    - Domain shift between seen and unseen classes
    
    Next steps:
    - More sophisticated attribute prediction (neural networks)
    - Continuous semantic embeddings (word vectors)
    - Visual-semantic embedding spaces
    - Generalized zero-shot learning (mixing seen/unseen at test time)
    """)
    print("="*70)


if __name__ == "__main__":
    main()

# Gradient Derivation for Softmax Regression

## Learning Objectives

By the end of this section, you will be able to:

- Derive the complete gradient of cross-entropy loss with respect to model parameters
- Understand the elegant simplification that emerges from softmax + cross-entropy
- Implement gradient computation from scratch in NumPy
- Verify gradients using PyTorch autograd
- Apply gradient descent to train softmax regression models

---

## The Softmax Regression Model

### Model Setup

For multi-class classification with $K$ classes and input features $\mathbf{x} \in \mathbb{R}^D$:

**Logits (linear scores):**
$$z_k = \mathbf{w}_k^T \mathbf{x} + b_k = \sum_{d=1}^{D} w_{kd} x_d + b_k$$

**Predicted probabilities (softmax):**
$$\hat{\pi}_k = \sigma(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Parameters:** $\boldsymbol{\theta} = \{\mathbf{W}, \mathbf{b}\}$ where $\mathbf{W} \in \mathbb{R}^{K \times D}$ and $\mathbf{b} \in \mathbb{R}^K$

### Loss Function

For a single sample with true class $c$ (one-hot encoded as $\mathbf{y}$):

$$\mathcal{L} = -\log \hat{\pi}_c = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

---

## Gradient Derivation: Step by Step

### Step 1: Gradient w.r.t. Logits

We first compute $\frac{\partial \mathcal{L}}{\partial z_j}$ using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{k=1}^{K} \frac{\partial \mathcal{L}}{\partial \hat{\pi}_k} \cdot \frac{\partial \hat{\pi}_k}{\partial z_j}$$

**Computing $\frac{\partial \mathcal{L}}{\partial \hat{\pi}_k}$:**

$$\mathcal{L} = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

$$\frac{\partial \mathcal{L}}{\partial \hat{\pi}_k} = -\frac{y_k}{\hat{\pi}_k}$$

**Using the softmax Jacobian $\frac{\partial \hat{\pi}_k}{\partial z_j}$:**

From our Jacobian derivation:
$$\frac{\partial \hat{\pi}_k}{\partial z_j} = \hat{\pi}_k(\delta_{kj} - \hat{\pi}_j)$$

**Combining:**

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{k=1}^{K} \left(-\frac{y_k}{\hat{\pi}_k}\right) \cdot \hat{\pi}_k(\delta_{kj} - \hat{\pi}_j)$$

$$= -\sum_{k=1}^{K} y_k(\delta_{kj} - \hat{\pi}_j)$$

$$= -\sum_{k=1}^{K} y_k \delta_{kj} + \hat{\pi}_j \sum_{k=1}^{K} y_k$$

Since $\sum_k y_k = 1$ (one-hot) and $\sum_k y_k \delta_{kj} = y_j$:

$$\frac{\partial \mathcal{L}}{\partial z_j} = -y_j + \hat{\pi}_j = \hat{\pi}_j - y_j$$

### The Beautiful Result

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

This elegant formula shows that the gradient is simply the **difference between predicted probabilities and true labels**!

**Interpretation:**
- If $\hat{\pi}_c$ is close to 1 (correct prediction), gradient ≈ 0 (small update)
- If $\hat{\pi}_c$ is close to 0 (wrong prediction), gradient is large (big update)
- The gradient "pushes" predictions toward the true label

---

### Step 2: Gradient w.r.t. Weights

Using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial w_{kd}} = \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial w_{kd}}$$

Since $z_k = \sum_d w_{kd} x_d + b_k$:

$$\frac{\partial z_k}{\partial w_{kd}} = x_d$$

Therefore:

$$\frac{\partial \mathcal{L}}{\partial w_{kd}} = (\hat{\pi}_k - y_k) \cdot x_d$$

**In matrix form:**

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = (\hat{\boldsymbol{\pi}} - \mathbf{y}) \mathbf{x}^T}$$

where $(\hat{\boldsymbol{\pi}} - \mathbf{y}) \in \mathbb{R}^K$ and $\mathbf{x} \in \mathbb{R}^D$, giving $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \in \mathbb{R}^{K \times D}$.

---

### Step 3: Gradient w.r.t. Biases

$$\frac{\partial \mathcal{L}}{\partial b_k} = \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial b_k} = (\hat{\pi}_k - y_k) \cdot 1$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

---

## Batch Gradient Computation

### For N Samples

Given a batch of $N$ samples $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$:

**Total loss:**
$$\mathcal{L}_{total} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}^{(i)}$$

**Gradient w.r.t. weights:**
$$\frac{\partial \mathcal{L}_{total}}{\partial \mathbf{W}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{\boldsymbol{\pi}}^{(i)} - \mathbf{y}^{(i)}) (\mathbf{x}^{(i)})^T$$

**In matrix notation:**

Let $\mathbf{X} \in \mathbb{R}^{N \times D}$ (samples as rows), $\hat{\mathbf{P}} \in \mathbb{R}^{N \times K}$ (predicted probs), $\mathbf{Y} \in \mathbb{R}^{N \times K}$ (one-hot labels):

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{N} (\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{X}}$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{\boldsymbol{\pi}}^{(i)} - \mathbf{y}^{(i)}) = \frac{1}{N} (\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{1}}$$

---

## NumPy Implementation from Scratch

```python
import numpy as np

class SoftmaxRegressionNumPy:
    """
    Softmax regression implemented from scratch.
    Demonstrates the gradient derivations in code.
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize weights with small random values.
        
        Args:
            input_dim: Number of input features D
            num_classes: Number of classes K
        """
        self.W = np.random.randn(num_classes, input_dim) * 0.01
        self.b = np.zeros(num_classes)
        
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.
        
        Args:
            z: Logits of shape (N, K)
        
        Returns:
            Probabilities of shape (N, K)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: X → logits → probabilities.
        
        Args:
            X: Input features of shape (N, D)
        
        Returns:
            Predicted probabilities of shape (N, K)
        """
        # Logits: z = XW^T + b
        self.z = X @ self.W.T + self.b  # (N, K)
        
        # Probabilities: π = softmax(z)
        self.probs = self.softmax(self.z)  # (N, K)
        
        return self.probs
    
    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            probs: Predicted probabilities (N, K)
            y: True class indices (N,)
        
        Returns:
            Average cross-entropy loss
        """
        N = len(y)
        # Select probability of true class for each sample
        correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-10)
        return np.mean(correct_log_probs)
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Backward pass: compute gradients.
        
        The key insight: ∂L/∂z = π - y (one-hot)
        
        Args:
            X: Input features (N, D)
            y: True class indices (N,)
        
        Returns:
            Tuple of (dW, db) gradients
        """
        N = len(y)
        
        # Convert y to one-hot encoding
        y_onehot = np.zeros_like(self.probs)
        y_onehot[np.arange(N), y] = 1
        
        # Gradient w.r.t. logits: dL/dz = π - y
        dz = self.probs - y_onehot  # (N, K)
        
        # Gradient w.r.t. weights: dL/dW = (1/N) * dz^T @ X
        dW = (1/N) * dz.T @ X  # (K, D)
        
        # Gradient w.r.t. biases: dL/db = (1/N) * sum(dz)
        db = (1/N) * np.sum(dz, axis=0)  # (K,)
        
        return dW, db
    
    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """
        Single training step: forward, backward, update.
        
        Args:
            X: Input features (N, D)
            y: True class indices (N,)
            lr: Learning rate
        
        Returns:
            Loss value
        """
        # Forward pass
        probs = self.forward(X)
        loss = self.compute_loss(probs, y)
        
        # Backward pass
        dW, db = self.backward(X, y)
        
        # Update parameters (gradient descent)
        self.W -= lr * dW
        self.b -= lr * db
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = SoftmaxRegressionNumPy(input_dim=20, num_classes=3)
    
    print("Training Softmax Regression from Scratch")
    print("=" * 50)
    
    for epoch in range(100):
        loss = model.train_step(X_train, y_train, lr=0.1)
        
        if (epoch + 1) % 20 == 0:
            train_acc = model.accuracy(X_train, y_train)
            test_acc = model.accuracy(X_test, y_test)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}, "
                  f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    
    print(f"\nFinal Test Accuracy: {model.accuracy(X_test, y_test):.2%}")
```

---

## PyTorch Verification

### Verifying Gradients with Autograd

```python
import torch
import torch.nn.functional as F

def verify_gradient_derivation():
    """
    Verify our analytical gradients match PyTorch autograd.
    """
    torch.manual_seed(42)
    
    # Setup
    N, D, K = 4, 5, 3  # 4 samples, 5 features, 3 classes
    
    X = torch.randn(N, D)
    y = torch.randint(0, K, (N,))
    
    W = torch.randn(K, D, requires_grad=True)
    b = torch.randn(K, requires_grad=True)
    
    # Forward pass
    logits = X @ W.T + b  # (N, K)
    probs = F.softmax(logits, dim=1)
    
    # Loss (cross-entropy)
    loss = F.cross_entropy(logits, y)
    
    # Autograd backward
    loss.backward()
    
    # Our analytical gradients
    y_onehot = F.one_hot(y, K).float()
    dz = probs.detach() - y_onehot  # (N, K)
    
    dW_analytical = (1/N) * dz.T @ X  # (K, D)
    db_analytical = (1/N) * dz.sum(dim=0)  # (K,)
    
    # Compare
    print("Gradient Verification")
    print("=" * 50)
    print(f"dW max error: {(W.grad - dW_analytical).abs().max().item():.2e}")
    print(f"db max error: {(b.grad - db_analytical).abs().max().item():.2e}")
    print(f"Gradients match: {torch.allclose(W.grad, dW_analytical, atol=1e-5)}")

verify_gradient_derivation()
```

### Understanding the Gradient Flow

```python
def visualize_gradient_flow():
    """
    Show how gradients flow through softmax + cross-entropy.
    """
    torch.manual_seed(42)
    
    # Single sample for clarity
    logits = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    true_class = 0
    
    # Forward
    probs = F.softmax(logits, dim=0)
    loss = -torch.log(probs[true_class])
    
    # Backward
    loss.backward()
    
    print("Gradient Flow Visualization")
    print("=" * 50)
    print(f"Logits z:        {logits.detach().numpy().round(4)}")
    print(f"Probabilities π: {probs.detach().numpy().round(4)}")
    print(f"True class:      {true_class}")
    print(f"Loss:            {loss.item():.4f}")
    print()
    print(f"∂L/∂z (autograd):   {logits.grad.numpy().round(4)}")
    
    # Analytical: π - y
    y_onehot = torch.zeros(3)
    y_onehot[true_class] = 1
    grad_analytical = probs.detach() - y_onehot
    print(f"∂L/∂z (analytical): {grad_analytical.numpy().round(4)}")
    print()
    print("Note: ∂L/∂z = π - y (predicted minus true)")

visualize_gradient_flow()
```

---

## Gradient Properties and Intuition

### Property 1: Gradient Magnitude

The gradient magnitude depends on prediction confidence:

| True Class Prob $\hat{\pi}_c$ | Gradient Magnitude | Interpretation |
|-------------------------------|-------------------|----------------|
| 0.99 | Small (≈0.01) | Confident correct → small update |
| 0.50 | Medium (≈0.50) | Uncertain → moderate update |
| 0.01 | Large (≈0.99) | Confident wrong → large update |

### Property 2: Gradient Direction

- **True class:** Gradient is negative (pushes logit up)
- **Other classes:** Gradient is positive (pushes logits down)
- Net effect: Increases separation between true class and others

### Property 3: Gradient Boundedness

$$\|\nabla_\mathbf{z} \mathcal{L}\|_2 = \|\hat{\boldsymbol{\pi}} - \mathbf{y}\|_2 \leq \sqrt{2}$$

The gradient is bounded, which helps training stability.

---

## With L2 Regularization

### Regularized Loss

$$\mathcal{L}_{reg} = \mathcal{L}_{CE} + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

### Regularized Gradient

$$\frac{\partial \mathcal{L}_{reg}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}_{CE}}{\partial \mathbf{W}} + \lambda \mathbf{W}$$

```python
def train_step_with_regularization(model, X, y, lr, lambda_reg):
    """Training step with L2 regularization."""
    # Forward
    probs = model.forward(X)
    loss = model.compute_loss(probs, y)
    
    # Add regularization to loss (for monitoring)
    reg_loss = 0.5 * lambda_reg * np.sum(model.W ** 2)
    total_loss = loss + reg_loss
    
    # Backward (with regularization gradient)
    dW, db = model.backward(X, y)
    dW += lambda_reg * model.W  # Add regularization gradient
    
    # Update
    model.W -= lr * dW
    model.b -= lr * db
    
    return total_loss
```

---

## Summary of Gradients

### Single Sample

| Quantity | Gradient Formula |
|----------|-----------------|
| Loss w.r.t. logits | $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\boldsymbol{\pi}} - \mathbf{y}$ |
| Loss w.r.t. weights | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = (\hat{\boldsymbol{\pi}} - \mathbf{y})\mathbf{x}^T$ |
| Loss w.r.t. biases | $\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \hat{\boldsymbol{\pi}} - \mathbf{y}$ |

### Batch (N samples)

| Quantity | Gradient Formula |
|----------|-----------------|
| Loss w.r.t. weights | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{N}(\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{X}$ |
| Loss w.r.t. biases | $\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{1}{N}\mathbf{1}^T(\hat{\mathbf{P}} - \mathbf{Y})$ |

### The Key Insight

$$\boxed{\text{Gradient} = \text{Predicted} - \text{True}}$$

This simple formula is why softmax + cross-entropy is so widely used!

---

## Next Steps

With gradient derivation complete, you're ready for:

1. **PyTorch Implementation** — Building complete classifiers
2. **Optimization Algorithms** — SGD, Adam, and beyond
3. **Regularization Techniques** — Dropout, weight decay, batch normalization

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.3
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 8

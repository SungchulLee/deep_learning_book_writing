# MLE for Classification

## Introduction

Classification loss functions are negative log-likelihoods of discrete probability distributions. **Binary cross-entropy is the NLL of a Bernoulli model**, and **categorical cross-entropy is the NLL of a categorical model**. This section makes these connections precise and shows how the softmax and sigmoid functions arise naturally from the MLE framework.

!!! success "Key Insight"
    $$\text{Cross-Entropy Loss} = \text{Negative Log-Likelihood of Categorical Distribution}$$
    
    When you train a classifier with cross-entropy loss, you are performing maximum likelihood estimation under a categorical likelihood.

## Binary Classification: BCE from Bernoulli MLE

### The Probabilistic Model

For binary classification, model the label as a Bernoulli random variable:

$$
y | x \sim \text{Bernoulli}(\sigma(f_\theta(x)))
$$

where $f_\theta(x)$ is the model's logit output and $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid function that maps logits to probabilities.

### Derivation

**Likelihood** of a single observation with $\hat{p} = \sigma(f_\theta(x))$:

$$
p(y | x, \theta) = \hat{p}^{\,y} \cdot (1 - \hat{p})^{1-y}
$$

**Negative log-likelihood**:

$$
-\log p(y | x, \theta) = -y \log \hat{p} - (1-y) \log(1 - \hat{p})
$$

Averaging over the dataset gives the **Binary Cross-Entropy (BCE)** loss:

$$
\boxed{\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]}
$$

### Why Sigmoid?

The sigmoid function arises naturally from the log-odds (logit) parameterization. If we model the log-odds as a linear function:

$$
\log \frac{p}{1-p} = f_\theta(x) \implies p = \sigma(f_\theta(x))
$$

This ensures $p \in (0, 1)$ for any real-valued $f_\theta(x)$, which is why logistic regression uses sigmoid — it is the canonical link function for the Bernoulli distribution.

### Numerical Stability

Computing BCE via $\sigma(z)$ followed by $\log(\sigma(z))$ is numerically unstable. The **log-sum-exp** trick gives a stable formulation:

$$
\text{BCE}(y, z) = \max(z, 0) - yz + \log(1 + e^{-|z|})
$$

where $z = f_\theta(x)$ is the logit. This is exactly what `nn.BCEWithLogitsLoss` implements.

## Multi-Class Classification: Cross-Entropy from Categorical MLE

### The Probabilistic Model

For $K$-class classification:

$$
y | x \sim \text{Categorical}(\text{softmax}(f_\theta(x)))
$$

**Softmax function**:

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, \ldots, K
$$

### Derivation

For one-hot encoded label $\mathbf{y} = (y_1, \ldots, y_K)$ where $y_k = 1$ for the correct class:

$$
p(\mathbf{y} | x, \theta) = \prod_{k=1}^{K} \hat{p}_k^{y_k}
$$

**Negative log-likelihood**:

$$
-\log p(\mathbf{y} | x, \theta) = -\sum_{k=1}^{K} y_k \log \hat{p}_k
$$

For hard labels where $c_i$ is the correct class, this simplifies to $-\log \hat{p}_{c_i}$. The **Categorical Cross-Entropy** loss is:

$$
\boxed{\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n} \log \hat{p}_{i, c_i}}
$$

### Why Softmax?

Just as sigmoid is the canonical link for Bernoulli, softmax is the canonical link for the categorical distribution. It arises from the multinomial logit model:

$$
\log \frac{p_k}{p_K} = z_k \quad (k = 1, \ldots, K-1)
$$

Solving for $p_k$ with the constraint $\sum p_k = 1$ yields exactly the softmax function.

### Numerical Stability

Direct computation of $\exp(z_k) / \sum \exp(z_j)$ overflows for large logits. The **log-softmax** trick subtracts the maximum:

$$
\log \text{softmax}(z)_k = z_k - \log\sum_{j=1}^{K} e^{z_j} = z_k - \left(m + \log\sum_{j=1}^{K} e^{z_j - m}\right)
$$

where $m = \max_j z_j$. PyTorch's `nn.CrossEntropyLoss` combines log-softmax and NLL loss in a single numerically stable operation.

## Label Smoothing

### Motivation

Hard one-hot labels encourage the model to produce extreme logits (pushing softmax outputs toward 0 or 1). **Label smoothing** replaces the hard label with a mixture:

$$
y_k^{\text{smooth}} = (1 - \epsilon) \cdot y_k + \frac{\epsilon}{K}
$$

where $\epsilon$ is the smoothing parameter (typically 0.1).

### MLE Interpretation

Label smoothing is equivalent to MLE under a mixture distribution: with probability $(1 - \epsilon)$ the label is correct, and with probability $\epsilon$ the label is drawn uniformly from all classes. This regularizes the model toward less confident predictions.

## Multi-Label Classification

When each sample can belong to multiple classes independently, the model is a product of independent Bernoulli distributions:

$$
p(\mathbf{y} | x, \theta) = \prod_{k=1}^{K} \hat{p}_k^{y_k}(1 - \hat{p}_k)^{1-y_k}
$$

The loss is the sum of $K$ independent BCE losses:

$$
\mathcal{L}_{\text{multi-label}} = -\frac{1}{nK}\sum_{i=1}^{n}\sum_{k=1}^{K}\left[y_{ik}\log\hat{p}_{ik} + (1-y_{ik})\log(1-\hat{p}_{ik})\right]
$$

Each output uses sigmoid (not softmax), since the probabilities are independent and need not sum to 1.

## PyTorch Implementation

### Binary Cross-Entropy Verification

```python
import torch
import torch.nn as nn
import numpy as np

def bce_from_nll():
    """Verify that manual BCE matches PyTorch implementations."""
    torch.manual_seed(42)
    
    n = 100
    x = torch.randn(n, 2)
    true_w = torch.tensor([2.0, -1.0])
    y = (x @ true_w > 0).float()
    
    # Model prediction
    w = torch.randn(2, requires_grad=True)
    logits = x @ w
    probs = torch.sigmoid(logits)
    
    # Manual BCE
    eps = 1e-7
    manual_bce = -torch.mean(y * torch.log(probs + eps) + 
                              (1 - y) * torch.log(1 - probs + eps))
    
    # PyTorch BCE (from probabilities)
    pytorch_bce = nn.BCELoss()(probs, y)
    
    # PyTorch BCE with logits (numerically stable)
    pytorch_bce_logits = nn.BCEWithLogitsLoss()(logits, y)
    
    print("Binary Cross-Entropy Verification")
    print(f"Manual BCE:       {manual_bce.item():.6f}")
    print(f"PyTorch BCE:      {pytorch_bce.item():.6f}")
    print(f"BCE with Logits:  {pytorch_bce_logits.item():.6f}")
```

### Categorical Cross-Entropy Verification

```python
def cross_entropy_from_nll():
    """Verify that manual cross-entropy matches PyTorch."""
    torch.manual_seed(42)
    
    n = 100
    K = 5
    x = torch.randn(n, 10)
    y = torch.randint(0, K, (n,))
    
    # Model (simple linear)
    W = torch.randn(10, K, requires_grad=True)
    logits = x @ W  # Shape: (n, K)
    
    # Manual softmax and cross-entropy
    exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    manual_ce = -torch.mean(torch.log(probs[range(n), y] + 1e-7))
    
    # PyTorch CrossEntropyLoss (combines LogSoftmax + NLLLoss)
    pytorch_ce = nn.CrossEntropyLoss()(logits, y)
    
    print("Cross-Entropy Verification")
    print(f"Manual CE:  {manual_ce.item():.6f}")
    print(f"PyTorch CE: {pytorch_ce.item():.6f}")
```

### Complete Classification Training

```python
def train_classification_mle_perspective():
    """Complete example showing MLE interpretation of classification training."""
    torch.manual_seed(42)
    
    # Generate 3-class data
    n_per_class = 100
    X0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, 0.0])
    X1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 0.0])
    X2 = torch.randn(n_per_class, 2) + torch.tensor([0.0, 3.0])
    
    X = torch.cat([X0, X1, X2])
    y = torch.cat([torch.zeros(n_per_class), 
                   torch.ones(n_per_class), 
                   2*torch.ones(n_per_class)]).long()
    
    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    
    # Model
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 3)  # Output: logits for 3 classes
    )
    
    # CrossEntropy = Categorical NLL
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Classifier (MLE with Categorical likelihood)")
    print("-" * 50)
    
    for epoch in range(300):
        logits = model(X)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 60 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                accuracy = (pred == y).float().mean().item()
            
            nll_bits = loss.item() / np.log(2)
            print(f"Epoch {epoch+1}: NLL = {loss.item():.4f} nats "
                  f"({nll_bits:.4f} bits), Accuracy = {accuracy:.2%}")
```

## Exercises

1. **Focal Loss**: Derive the focal loss $\text{FL}(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$ and explain its probabilistic interpretation. When does it reduce to standard cross-entropy?

2. **Multi-Task Classification**: Implement a multi-task learning loss where one output head is binary classification (Bernoulli) and another is multi-class (categorical).

3. **Label Smoothing Analysis**: Empirically compare training with hard labels vs. label smoothing ($\epsilon = 0.1$) on a simple classification task. Measure calibration (reliability diagram) in addition to accuracy.

4. **Ordinal Regression**: Design a loss function for ordinal classification (e.g., ratings 1–5) that respects the ordering. Hint: use cumulative probabilities.

5. **Temperature Scaling**: Implement temperature scaling for post-hoc calibration and show that it corresponds to adjusting the categorical distribution's concentration.

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 4
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Chapter 6
- Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*
- Szegedy, C. et al. (2016). "Rethinking the Inception Architecture for Computer Vision." *CVPR*

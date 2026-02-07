# Energy-Based OOD Detection

## Overview

The energy score (Liu et al., 2020) uses the log-sum-exp of logits as an OOD score, providing theoretically motivated detection that outperforms softmax baselines.

## Energy Score

The free energy of input $\mathbf{x}$ is:

$$E(\mathbf{x}) = -T \cdot \log \sum_{c=1}^C \exp(f_c(\mathbf{x}) / T)$$

In-distribution inputs have lower (more negative) energy; OOD inputs have higher energy.

## Connection to Softmax

The energy is related to the log-partition function. While softmax normalizes and loses magnitude information, the energy preserves the total logit magnitude, which is informative for OOD detection.

## Implementation

```python
import torch


def energy_ood_score(model, x, temperature=1.0):
    """Energy-based OOD score (higher = more likely OOD)."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return energy  # Higher energy = more likely OOD


def energy_with_fine_tuning(model, train_loader, m_in=-25, m_out=-7,
                            epochs=10, lr=0.001):
    """
    Fine-tune model with energy-regularized loss.
    Encourages low energy for in-distribution, high for OOD.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for x, y in train_loader:
            logits = model(x)
            ce_loss = criterion(logits, y)
            
            # Energy regularization
            energy = -torch.logsumexp(logits, dim=-1)
            energy_loss = torch.pow(F.relu(energy - m_in), 2).mean()
            
            loss = ce_loss + 0.1 * energy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## References

- Liu, W., et al. (2020). "Energy-based Out-of-distribution Detection." NeurIPS.

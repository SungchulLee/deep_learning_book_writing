# Missing Data Imputation

Using VAEs to fill in missing values in financial datasets by leveraging learned data distributions.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the probabilistic approach to missing data imputation via VAEs
- Implement gradient-based and sampling-based imputation methods
- Apply VAE imputation to common financial data missingness patterns
- Validate imputation quality against ground truth

---

## The Missing Data Problem in Finance

Financial datasets frequently contain missing values due to asset delistings, trading halts, different market calendars, data vendor issues, and newly listed instruments. Traditional approaches like forward-fill, mean imputation, or dropping incomplete rows either introduce bias or discard valuable information.

---

## VAE-Based Imputation

### Probabilistic Approach

A trained VAE has learned the joint distribution $p_\theta(x)$ over complete data. For a partially observed sample $x = (x_{\text{obs}}, x_{\text{miss}})$, we seek:

$$p(x_{\text{miss}} | x_{\text{obs}}) = \int p_\theta(x_{\text{miss}} | z) \, p(z | x_{\text{obs}}) \, dz$$

The VAE provides an approximate mechanism for this: find a latent code $z$ consistent with the observed values, then decode to fill in the missing values.

---

## Method 1: Optimization-Based Imputation

Find the latent code that best explains the observed data:

```python
import torch
import torch.nn.functional as F

def impute_optimization(model, x_partial, mask, num_iters=200, lr=0.01, device='cpu'):
    """
    Impute missing values by optimizing the latent code.
    
    Args:
        model: Trained VAE
        x_partial: Partially observed data [1, D] (missing values can be any value)
        mask: Binary mask [1, D], 1 = observed, 0 = missing
        num_iters: Optimization steps
        lr: Learning rate
    
    Returns:
        Imputed complete data [1, D]
    """
    model.eval()
    x_partial = x_partial.to(device)
    mask = mask.to(device)
    
    # Initialize z from encoder (using observed values)
    with torch.no_grad():
        mu, logvar = model.encode(x_partial * mask)
    
    z = mu.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        recon = model.decode(z)
        
        # Loss only on observed values
        loss = ((recon - x_partial).pow(2) * mask).sum()
        
        # Optional: prior regularization
        prior_loss = 0.5 * z.pow(2).sum()
        total_loss = loss + 0.01 * prior_loss
        
        total_loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        imputed = model.decode(z)
    
    # Combine: keep observed values, use decoded for missing
    result = x_partial * mask + imputed * (1 - mask)
    return result
```

---

## Method 2: Sampling-Based Imputation

Generate multiple imputation candidates for uncertainty quantification:

```python
def impute_sampling(model, x_partial, mask, num_samples=50, device='cpu'):
    """
    Multiple imputation via posterior sampling.
    
    Returns multiple imputed versions for uncertainty estimation.
    """
    model.eval()
    x_partial = x_partial.to(device)
    mask = mask.to(device)
    
    imputations = []
    
    with torch.no_grad():
        mu, logvar = model.encode(x_partial * mask)
        
        for _ in range(num_samples):
            # Sample from approximate posterior
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            
            recon = model.decode(z)
            imputed = x_partial * mask + recon * (1 - mask)
            imputations.append(imputed.cpu())
    
    imputations = torch.stack(imputations)
    
    return {
        'mean': imputations.mean(dim=0),
        'std': imputations.std(dim=0),
        'samples': imputations
    }
```

The standard deviation across samples provides a natural **uncertainty estimate** for each imputed value.

---

## Financial Missingness Patterns

| Pattern | Description | VAE Approach |
|---------|-------------|-------------|
| **Random missing** | Random cells missing | Standard imputation |
| **Block missing** | Entire asset missing for a period | Leverage cross-asset correlations |
| **Monotone missing** | New assets have no early history | CVAE conditioned on time |
| **Systematic** | Markets closed on different days | Calendar-aware masking |

---

## Validation

### Hold-Out Evaluation

Artificially mask known values and evaluate imputation quality:

```python
def evaluate_imputation(model, complete_data, missing_rate=0.2):
    """
    Evaluate imputation by artificially masking data.
    """
    # Create random mask
    mask = (torch.rand_like(complete_data) > missing_rate).float()
    x_partial = complete_data * mask
    
    # Impute
    result = impute_optimization(model, x_partial, mask)
    
    # Evaluate on masked positions only
    missing_mask = 1 - mask
    mse = ((result - complete_data).pow(2) * missing_mask).sum() / missing_mask.sum()
    mae = ((result - complete_data).abs() * missing_mask).sum() / missing_mask.sum()
    
    return {'mse': mse.item(), 'mae': mae.item()}
```

---

## Summary

| Method | Speed | Uncertainty | Quality |
|--------|-------|-------------|---------|
| **Optimization** | Slower (iterative) | No | Higher |
| **Sampling** | Fast (forward pass) | Yes (multiple samples) | Good |
| **Mean decoding** | Fastest | No | Baseline |

---

## Exercises

### Exercise 1

Train a VAE on complete financial return data. Mask 10%, 20%, and 50% of values. Compare imputation MSE for each missing rate.

### Exercise 2

Compare VAE imputation against forward-fill and mean imputation on a real dataset with natural missing values.

---

## What's Next

The final section covers [Scenario Generation](scenarios.md) for risk management and stress testing.

# Bias-Variance Decomposition

## Overview

This section provides the formal decomposition and empirical estimation of bias and variance components.

## Decomposition

Let $y = f(x) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$, $\text{Var}(\varepsilon) = \sigma^2$. Denote $\bar{f}(x) = \mathbb{E}_\mathcal{D}[\hat{f}(x)]$. Then:

$$\mathbb{E}\left[(y - \hat{f})^2\right] = \underbrace{(f - \bar{f})^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \bar{f})^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

## Empirical Estimation

Estimate bias and variance via bootstrap:

```python
def estimate_bias_variance(model_fn, X_train, y_train, X_test, y_test,
                           n_bootstrap=50):
    predictions = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        model = model_fn()
        model.fit(X_train[idx], y_train[idx])
        predictions.append(model.predict(X_test))

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)

    bias_sq = ((mean_pred - y_test) ** 2).mean()
    variance = predictions.var(axis=0).mean()
    mse = ((predictions - y_test) ** 2).mean()
    noise = mse - bias_sq - variance

    return {'bias_sq': bias_sq, 'variance': variance, 'noise': noise}
```

## Key Takeaways

- The decomposition is exact for squared error loss.
- Bias and variance are properties of the learning algorithm over all possible training sets.
- Empirical estimation via bootstrap provides practical approximations.

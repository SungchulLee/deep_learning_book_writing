# Mathematical Derivation

## Overview

This section provides a rigorous step-by-step derivation of the bias-variance decomposition from first principles.

## Setup

Let $y = f(x) + \varepsilon$ where $\mathbb{E}[\varepsilon] = 0$ and $\text{Var}(\varepsilon) = \sigma^2$. Let $\hat{f}$ be a model trained on a random training set $\mathcal{D}$, and define $\bar{f}(x) = \mathbb{E}_\mathcal{D}[\hat{f}(x)]$.

## Derivation

Start with the expected squared error:

$$\mathbb{E}\left[(y - \hat{f})^2\right] = \mathbb{E}\left[(f + \varepsilon - \hat{f})^2\right]$$

Add and subtract $\bar{f}$:

$$= \mathbb{E}\left[((f - \bar{f}) + (\bar{f} - \hat{f}) + \varepsilon)^2\right]$$

Expand the square:

$$= (f - \bar{f})^2 + \mathbb{E}[(\bar{f} - \hat{f})^2] + \mathbb{E}[\varepsilon^2] + 2(f - \bar{f})\mathbb{E}[\bar{f} - \hat{f}] + 2(f - \bar{f})\mathbb{E}[\varepsilon] + 2\mathbb{E}[(\bar{f} - \hat{f})\varepsilon]$$

The cross terms vanish:

- $\mathbb{E}[\bar{f} - \hat{f}] = \bar{f} - \bar{f} = 0$ by definition of $\bar{f}$.
- $\mathbb{E}[\varepsilon] = 0$ by assumption.
- $\mathbb{E}[(\bar{f} - \hat{f})\varepsilon] = 0$ because the model and test noise are independent.

This yields the decomposition:

$$\mathbb{E}\left[(y - \hat{f})^2\right] = \underbrace{(f - \bar{f})^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \bar{f})^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

## Interpretation

The decomposition holds pointwise for each test input $x$. The total expected error is obtained by integrating over the test distribution:

$$\text{Expected Error} = \int \left[\text{Bias}^2(x) + \text{Var}(x) + \sigma^2(x)\right] p(x) \, dx$$

## Beyond Squared Error

The standard decomposition applies to squared error loss. For other losses (cross-entropy, 0-1 loss), analogous but more complex decompositions exist. The intuition remains: simpler models err systematically (bias), complex models err erratically (variance).

## Key Takeaways

- The derivation relies on adding and subtracting $\bar{f} = \mathbb{E}[\hat{f}]$ and exploiting independence of noise from the model.
- All cross terms vanish, yielding an exact additive decomposition.
- The result is fundamental to understanding model selection and regularization.

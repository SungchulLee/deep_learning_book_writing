# Bayesian Neural Network Methods: Comparison Guide

This page provides a comprehensive comparison of Bayesian neural network methods to help practitioners choose the right approach for their application.

---

## Method Overview

### Taxonomy

```
BNN Methods
├── Sampling Methods
│   ├── HMC (exact, expensive)
│   └── SGLD/SGHMC (scalable)
│
├── Variational Methods
│   ├── Bayes by Backprop (mean-field)
│   └── Natural gradient VI
│
├── Laplace Approximation
│   ├── Full/Diagonal Laplace
│   └── Last-layer Laplace
│
└── Implicit Methods
    ├── MC Dropout
    ├── Deep Ensembles
    └── SWAG
```

---

## Quick Comparison Table

| Method | Training | Inference | Accuracy | Scalability | Implementation |
|--------|----------|-----------|----------|-------------|----------------|
| **HMC** | Very High | High | ★★★★★ | ★☆☆☆☆ | Complex |
| **SGLD** | Medium | High | ★★★★☆ | ★★★★☆ | Moderate |
| **Bayes by Backprop** | High | Medium | ★★★☆☆ | ★★★★☆ | Moderate |
| **MC Dropout** | Low | Medium | ★★☆☆☆ | ★★★★★ | Simple |
| **Deep Ensembles** | High (M×) | Low (M×) | ★★★★☆ | ★★★☆☆ | Simple |
| **SWAG** | Low+ | Medium | ★★★☆☆ | ★★★★☆ | Moderate |
| **Laplace** | Low+ | Low | ★★★☆☆ | ★★★☆☆ | Moderate |

**Legend**: ★ = Low/Poor, ★★★★★ = High/Excellent

---

## Detailed Comparison

### Posterior Approximation Quality

| Method | Posterior Form | Multimodal? | Correlations? |
|--------|---------------|-------------|---------------|
| HMC | Exact samples | Yes | Yes |
| SGLD | Approximate samples | Partially | Yes |
| Mean-field VI | Factorized Gaussian | No | No |
| MC Dropout | Implicit (Bernoulli) | No | Limited |
| Deep Ensembles | Mixture of points | Yes | No |
| SWAG | Low-rank Gaussian | No | Partial |
| Laplace | Gaussian at MAP | No | Partial |

### Computational Cost

| Method | Training | Inference (T samples) | Memory |
|--------|----------|----------------------|--------|
| HMC | $O(TNd)$ | $O(T)$ | $O(Td)$ |
| SGLD | $O(Ed)$ | $O(T)$ | $O(Td)$ |
| Mean-field VI | $O(Ed)$ | $O(T)$ | $O(2d)$ |
| MC Dropout | $O(Ed)$ | $O(T)$ | $O(d)$ |
| Deep Ensembles | $O(MEd)$ | $O(M)$ | $O(Md)$ |
| SWAG | $O(Ed) + O(Kd)$ | $O(T)$ | $O((K+2)d)$ |
| Laplace | $O(Ed) + O(d^2)$ | $O(T)$ | $O(d^2)$ or $O(d)$ |

where $d$ = parameters, $N$ = data size, $E$ = epochs, $T$ = samples, $M$ = ensemble size, $K$ = SWAG rank.

---

## When to Use Each Method

### MC Dropout

**Best for**:
- ✅ Quick uncertainty from existing dropout models
- ✅ Resource-constrained deployment
- ✅ When simplicity is paramount

**Avoid when**:
- ❌ Architecture doesn't have dropout
- ❌ Need accurate uncertainty quantification
- ❌ High-stakes decisions

### Deep Ensembles

**Best for**:
- ✅ Best overall uncertainty quality
- ✅ Embarrassingly parallel training
- ✅ When computational budget allows

**Avoid when**:
- ❌ Memory-constrained deployment
- ❌ Need single model
- ❌ Very limited training budget

### SWAG

**Best for**:
- ✅ Post-hoc uncertainty from trained model
- ✅ Large models where VI is expensive
- ✅ When Gaussian approximation suffices

**Avoid when**:
- ❌ Need multimodal posterior
- ❌ Cannot modify training schedule
- ❌ Need real-time inference

### Laplace Approximation

**Best for**:
- ✅ Post-hoc uncertainty (no retraining)
- ✅ Theoretical interpretability
- ✅ When linearization is accurate

**Avoid when**:
- ❌ Far from Gaussian posterior
- ❌ Need multimodal uncertainty
- ❌ Full Hessian is intractable

### Bayes by Backprop

**Best for**:
- ✅ Principled Bayesian treatment
- ✅ Flexible prior specification
- ✅ Online learning settings

**Avoid when**:
- ❌ Mean-field is too restrictive
- ❌ KL optimization is unstable
- ❌ Simpler methods suffice

### SGLD / SGHMC

**Best for**:
- ✅ Need samples from true posterior
- ✅ Large datasets (scalable MCMC)
- ✅ Multimodal exploration

**Avoid when**:
- ❌ Need rapid convergence
- ❌ Hyperparameter tuning is difficult
- ❌ Memory for samples is limited

---

## Decision Flowchart

```
Start
  │
  ▼
Have pre-trained model? ─── Yes ──► Want to retrain? ─── No ──► Laplace or SWAG
  │                                        │
  No                                      Yes
  │                                        │
  ▼                                        ▼
Have dropout? ─── Yes ──► MC Dropout       │
  │                                        │
  No                                       │
  │                                        │
  ▼                                        ▼
Need best uncertainty? ─── Yes ──► Budget for M models? ─── Yes ──► Deep Ensembles
  │                                        │
  No                                      No
  │                                        │
  ▼                                        ▼
Scalability crucial? ─── Yes ──► SGLD     Bayes by Backprop
  │
  No
  │
  ▼
Small model? ─── Yes ──► HMC
  │
  No
  │
  ▼
SWAG or Laplace
```

---

## Empirical Performance

### Uncertainty Quality (Typical Rankings)

**Out-of-Distribution Detection**:
1. Deep Ensembles
2. SGLD
3. SWAG
4. Bayes by Backprop
5. MC Dropout
6. Laplace

**Calibration** (Expected Calibration Error):
1. Deep Ensembles
2. SWAG
3. SGLD
4. Laplace (with temperature)
5. Bayes by Backprop
6. MC Dropout

**Active Learning**:
1. SGLD
2. Deep Ensembles
3. Bayes by Backprop
4. SWAG
5. MC Dropout

*Note: Rankings vary by dataset and architecture.*

---

## Implementation Complexity

### From Scratch

| Method | Code Complexity | Hyperparameters | Debugging |
|--------|-----------------|-----------------|-----------|
| MC Dropout | ★☆☆☆☆ | Few | Easy |
| Deep Ensembles | ★☆☆☆☆ | Few | Easy |
| Laplace | ★★☆☆☆ | Few | Moderate |
| SWAG | ★★☆☆☆ | Few | Moderate |
| Bayes by Backprop | ★★★☆☆ | Several | Moderate |
| SGLD | ★★★☆☆ | Several | Hard |
| HMC | ★★★★☆ | Many | Hard |

### Using Libraries

| Method | PyTorch | TensorFlow | JAX |
|--------|---------|------------|-----|
| MC Dropout | Built-in | Built-in | Simple |
| Deep Ensembles | Easy | Easy | Easy |
| Laplace | laplace-torch | tfp | Simple |
| SWAG | swa_utils | tf-swag | Simple |
| Bayes by Backprop | pyro, blitz | tfp | NumPyro |
| SGLD | pyro | tfp | BlackJAX |
| HMC | pyro | tfp | BlackJAX |

---

## Combining Methods

### Effective Combinations

1. **Multi-SWAG**: Multiple SWAG from different initializations
   - Captures multiple modes
   - Better than single SWAG

2. **Ensemble + MC Dropout**: Ensemble of dropout models with MC at inference
   - Diversity from both sources
   - Expensive but effective

3. **Deep Ensembles + Last-layer Laplace**: Ensemble with Laplace per member
   - Post-hoc improvement
   - Cheap addition to ensembles

4. **SWAG + Temperature Scaling**: SWAG with calibrated temperature
   - Improved calibration
   - Simple post-processing

---

## Summary Recommendations

| Scenario | Recommended Method |
|----------|-------------------|
| **Production, accuracy matters** | Deep Ensembles |
| **Quick prototype** | MC Dropout |
| **Post-hoc uncertainty** | SWAG or Laplace |
| **Research, best posterior** | SGLD or HMC |
| **Principled Bayesian** | Bayes by Backprop |
| **Memory constrained** | MC Dropout |
| **Very large models** | MC Dropout or Last-layer Laplace |

### Key References

- Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? *NeurIPS*.
- Ashukha, A., et al. (2020). Pitfalls of in-domain uncertainty estimation. *ICLR*.
- Wilson, A. G. (2020). The case for Bayesian deep learning. *arXiv*.

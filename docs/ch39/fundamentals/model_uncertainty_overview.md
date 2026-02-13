# Module 63: Model Uncertainty

## Overview
This module provides a comprehensive introduction to uncertainty quantification in deep learning models. Understanding and measuring model uncertainty is crucial for deploying neural networks in real-world applications, especially in high-stakes domains like healthcare, autonomous driving, and financial systems.

## Learning Objectives
By the end of this module, students will be able to:
- Distinguish between epistemic and aleatoric uncertainty
- Implement Monte Carlo Dropout for uncertainty estimation
- Build and train deep ensembles
- Apply Bayesian Neural Networks for uncertainty quantification
- Calibrate model predictions using temperature scaling
- Evaluate uncertainty estimates quantitatively
- Visualize and interpret model uncertainty

## Prerequisites
- Module 20: Feedforward Networks
- Module 14: Loss Functions
- Module 15: Optimizers
- Module 21: Regularization Techniques
- Basic understanding of probability and statistics

## Module Structure

### 01_beginner_uncertainty_basics.py
**Topics Covered:**
- Types of uncertainty (epistemic vs aleatoric)
- Prediction intervals vs confidence intervals
- Basic probabilistic predictions
- Softmax temperature and calibration
- Simple ensemble methods

**Key Concepts:**
- Epistemic uncertainty: model uncertainty (reducible with more data)
- Aleatoric uncertainty: data uncertainty (irreducible noise)
- Calibration: alignment between predicted probabilities and actual frequencies

**Time Estimate:** 2-3 hours

---

### 02_intermediate_mc_dropout_ensembles.py
**Topics Covered:**
- Monte Carlo Dropout implementation
- Deep ensemble construction
- Variance-based uncertainty
- Uncertainty decomposition
- Bootstrap aggregating

**Key Concepts:**
- MC Dropout: using dropout at test time for uncertainty estimation
- Deep Ensembles: training multiple models with different initializations
- Predictive variance as uncertainty measure
- Mean-variance decomposition

**Time Estimate:** 3-4 hours

---

### 03_advanced_bayesian_uncertainty.py
**Topics Covered:**
- Bayesian Neural Networks (BNN)
- Variational Inference for BNNs
- Bayes by Backprop algorithm
- Stochastic Weight Averaging Gaussian (SWAG)
- Laplace approximation

**Key Concepts:**
- Weight distributions instead of point estimates
- KL divergence in variational inference
- Posterior approximation
- Gaussian weight distributions

**Time Estimate:** 4-5 hours

---

### 04_calibration_evaluation.py
**Topics Covered:**
- Temperature scaling
- Platt scaling
- Isotonic regression
- Calibration metrics (ECE, MCE, Brier score)
- Reliability diagrams
- Uncertainty evaluation metrics

**Key Concepts:**
- Post-hoc calibration methods
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Negative Log-Likelihood (NLL)
- Proper scoring rules

**Time Estimate:** 3-4 hours

---

### 05_practical_applications.py
**Topics Covered:**
- Out-of-distribution detection
- Active learning with uncertainty
- Selective prediction
- Confidence-based decision making
- Uncertainty in regression tasks
- Medical diagnosis example

**Key Concepts:**
- Using uncertainty for OOD detection
- Rejection option in classification
- Uncertainty-guided data acquisition
- Heteroscedastic regression

**Time Estimate:** 3-4 hours

---

## Mathematical Foundations

### Epistemic vs Aleatoric Uncertainty

**Epistemic Uncertainty (Model Uncertainty):**
- Captures ignorance about model parameters
- Can be reduced with more training data
- Represented by distributions over model weights: p(w|D)

**Aleatoric Uncertainty (Data Uncertainty):**
- Captures inherent noise in observations
- Cannot be reduced with more data
- Represented by observation noise: p(y|x,w)

**Total Predictive Uncertainty:**
```
Var[y|x,D] = E[Var[y|x,w]] + Var[E[y|x,w]]
             \_____________/   \____________/
              Aleatoric         Epistemic
```

### Bayesian Neural Networks

**Posterior Distribution:**
```
p(w|D) = p(D|w)p(w) / p(D)
```

**Predictive Distribution:**
```
p(y*|x*,D) = ∫ p(y*|x*,w) p(w|D) dw
```

### Monte Carlo Dropout

**Approximate Predictive Distribution:**
```
p(y|x) ≈ (1/T) Σ p(y|x,w_t)  where w_t ~ dropout
```

### Expected Calibration Error

```
ECE = Σ (|B_m|/n) |acc(B_m) - conf(B_m)|
```
where B_m are bins of predictions grouped by confidence.

## Installation Requirements

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn seaborn
```

## Usage Examples

### Quick Start: Monte Carlo Dropout
```python
from module_63 import MCDropoutModel, estimate_uncertainty

# Create model with dropout
model = MCDropoutModel(input_dim=784, hidden_dim=256, output_dim=10)

# Get predictions with uncertainty
predictions, uncertainty = estimate_uncertainty(model, test_data, n_samples=100)
```

### Quick Start: Deep Ensemble
```python
from module_63 import DeepEnsemble, train_ensemble

# Train ensemble
ensemble = DeepEnsemble(n_models=5, input_dim=784, output_dim=10)
train_ensemble(ensemble, train_loader, epochs=10)

# Get ensemble predictions
mean_pred, uncertainty = ensemble.predict_with_uncertainty(test_data)
```

## Key Takeaways

1. **Why Uncertainty Matters:**
   - Critical for safety-critical applications
   - Enables informed decision-making
   - Identifies when model is unreliable
   - Guides active learning and data collection

2. **Practical Methods:**
   - MC Dropout: Simple, no retraining needed
   - Deep Ensembles: High performance, higher computational cost
   - Bayesian NNs: Principled framework, complex to implement

3. **Calibration is Essential:**
   - Neural networks are often overconfident
   - Post-hoc calibration improves reliability
   - Always evaluate calibration metrics

4. **Trade-offs:**
   - Accuracy vs Uncertainty Quality
   - Computational Cost vs Uncertainty Precision
   - Complexity vs Interpretability

## Further Reading

### Papers:
1. Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
2. Lakshminarayanan et al. (2017) - "Simple and Scalable Predictive Uncertainty"
3. Blundell et al. (2015) - "Weight Uncertainty in Neural Networks"
4. Guo et al. (2017) - "On Calibration of Modern Neural Networks"
5. Maddox et al. (2019) - "A Simple Baseline for Bayesian Uncertainty"

### Resources:
- "Bayesian Deep Learning" - Yarin Gal's Thesis
- "Uncertainty in Deep Learning" - Kendall & Gal (2017)
- PyTorch Uncertainty Estimation Libraries

## Assessment Ideas

1. **Coding Exercise:** Implement MC Dropout on MNIST
2. **Analysis Task:** Compare uncertainty estimates across methods
3. **Application Project:** Build a medical diagnosis system with uncertainty
4. **Research Task:** Evaluate calibration on out-of-distribution data
5. **Discussion:** When is uncertainty quantification critical?

## Common Pitfalls

1. Confusing prediction confidence with uncertainty
2. Not evaluating calibration separately from accuracy
3. Using too few MC samples (use 50-100 minimum)
4. Ignoring computational costs in production
5. Not considering both types of uncertainty

## Connections to Other Modules

- **Module 45: Bayesian Neural Networks** - Theoretical foundation
- **Module 21: Regularization** - Dropout as regularization
- **Module 54: Self-Supervised Learning** - Uncertainty for pseudo-labeling
- **Module 61: Bias and Fairness** - Uncertainty in fairness metrics
- **Module 64: Model Deployment** - Uncertainty in production systems

---

**Author:** Deep Learning Curriculum Team  
**Version:** 1.0  
**Last Updated:** November 2025

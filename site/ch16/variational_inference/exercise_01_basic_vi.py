"""
Variational Inference - Exercises
==================================

This file contains exercises for practicing VI concepts and implementations.
Solutions are provided in the solutions/ directory.

Author: Prof. Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")


# ============================================================================
# EXERCISE 1: Basic VI - Gaussian Mean Estimation (Beginner)
# ============================================================================

print("=" * 80)
print("EXERCISE 1: Gaussian Mean Estimation with VI")
print("=" * 80)

exercise_1_description = """
PROBLEM:
-------
You observe n data points from N(θ, σ²) where σ² is known.
Your prior on θ is N(μ₀, σ₀²).

TASK:
----
1. Derive the ELBO for variational family q(θ) = N(m, s²)
2. Derive the optimal variational parameters m* and s*
3. Implement gradient ascent to maximize ELBO
4. Compare with the exact posterior (which is also Gaussian)
5. Visualize the convergence of variational parameters

GIVEN:
-----
- Data: [2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0]
- Known σ² = 0.25
- Prior: μ₀ = 0, σ₀² = 4.0

DELIVERABLES:
------------
1. Mathematical derivation (written)
2. Python implementation
3. Plots showing:
   a) Prior, likelihood, posterior, and variational approximation
   b) ELBO convergence
   c) Parameter convergence (m and s)

HINTS:
-----
- Start with ELBO = E_q[log p(D,θ)] - E_q[log q(θ)]
- For Gaussian q, expectations have closed forms
- Check your answer against the conjugate posterior
"""

print(exercise_1_description)

# Generate data for students
np.random.seed(42)
data_ex1 = np.array([2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0])
sigma_sq_ex1 = 0.25
mu_0_ex1 = 0.0
sigma_0_sq_ex1 = 4.0

print("\nData provided:")
print(f"  x = {data_ex1}")
print(f"  σ² = {sigma_sq_ex1}")
print(f"  μ₀ = {mu_0_ex1}, σ₀² = {sigma_0_sq_ex1}")

# TODO: Students implement here
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# EXERCISE 2: Mean-Field VI - Bayesian Linear Regression (Intermediate)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 2: Bayesian Linear Regression with Mean-Field VI")
print("=" * 80)

exercise_2_description = """
PROBLEM:
-------
Linear regression: y = Xw + ε where ε ~ N(0, σ²)

Bayesian model:
- Likelihood: y | X, w, σ² ~ N(Xw, σ²I)
- Prior on weights: w ~ N(0, λ⁻¹I)
- Prior on precision: τ = 1/σ² ~ Gamma(a₀, b₀)

TASK:
----
1. Derive mean-field approximation: q(w,τ) = q(w)q(τ)
2. Derive CAVI updates for q(w) and q(τ)
3. Implement the full CAVI algorithm
4. Compare with exact posterior (which exists for this model!)
5. Analyze the effect of prior strength λ

DATA GENERATION:
---------------
True model: y = 2x₁ + 3x₂ - 1 + ε, ε ~ N(0, 0.5²)
Generate n=50 samples with x₁, x₂ ~ U(0, 1)

DELIVERABLES:
------------
1. CAVI update equations (derived mathematically)
2. Complete Python implementation
3. Plots:
   a) Posterior over weights w
   b) Posterior over precision τ
   c) Predictive distribution
   d) ELBO convergence
4. Comparison with exact posterior
5. Analysis of prior sensitivity

HINTS:
-----
- Both q(w) and q(τ) will be in conjugate families
- q(w) = N(m_w, Σ_w)
- q(τ) = Gamma(a_n, b_n)
- Use the general CAVI update: q*_j ∝ exp{E_{-j}[log p(all)]}
"""

print(exercise_2_description)

# Generate data for students
np.random.seed(42)
n_ex2 = 50
X_ex2 = np.random.rand(n_ex2, 2)
w_true_ex2 = np.array([2.0, 3.0])
y_true_ex2 = X_ex2 @ w_true_ex2 - 1.0
y_ex2 = y_true_ex2 + np.random.normal(0, 0.5, n_ex2)

print("\nData generated:")
print(f"  n = {n_ex2} samples")
print(f"  True weights: w = {w_true_ex2}")
print(f"  True intercept: b = -1.0")
print(f"  Noise std: σ = 0.5")

# TODO: Students implement here
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# EXERCISE 3: Variational Bayes for Mixture Models (Advanced)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 3: Univariate Gaussian Mixture with Full Bayesian Treatment")
print("=" * 80)

exercise_3_description = """
PROBLEM:
-------
Fit a K-component Gaussian mixture model to 1D data with
FULL Bayesian treatment (priors on all parameters).

Model:
- Data: x_i ~ Σ_k π_k N(μ_k, τ_k⁻¹)
- Priors:
  * π ~ Dirichlet(α)
  * μ_k ~ N(m, (βτ_k)⁻¹)
  * τ_k ~ Gamma(a, b)

Mean-field approximation:
q(Z, π, μ, τ) = q(Z) q(π) ∏_k q(μ_k, τ_k)

TASK:
----
1. Derive complete CAVI updates for all variational factors
2. Implement the full algorithm from scratch
3. Handle the coupling between μ_k and τ_k
4. Implement model selection (choose K using ELBO)
5. Compare with sklearn's GMM

CHALLENGES:
----------
- Conjugate updates involve Gaussian-Gamma coupling
- Need to handle empty clusters gracefully
- Label switching problem in mixture models
- Initialization is critical

DATA:
----
Generate mixture of 3 Gaussians:
- Component 1: μ=-5, σ=1, weight=0.3
- Component 2: μ=0, σ=1.5, weight=0.5
- Component 3: μ=5, σ=0.8, weight=0.2
n=300 samples

DELIVERABLES:
------------
1. Complete derivation of CAVI updates
2. Full implementation with:
   - Proper initialization
   - Convergence checking
   - Empty cluster handling
3. Plots:
   a) Data with learned components
   b) Responsibilities (soft assignments)
   c) ELBO convergence
   d) Model selection curve (ELBO vs K)
4. Comparison with sklearn
5. Analysis of posterior uncertainty

BONUS:
-----
- Implement automatic relevance determination (ARD) for K
- Visualize posterior correlations between parameters
- Implement diagonal plus low-rank covariance
"""

print(exercise_3_description)

# Generate complex mixture data
np.random.seed(42)
n_ex3 = 300
mixture_params = [
    {'mean': -5.0, 'std': 1.0, 'weight': 0.3},
    {'mean': 0.0, 'std': 1.5, 'weight': 0.5},
    {'mean': 5.0, 'std': 0.8, 'weight': 0.2},
]

data_ex3 = []
labels_ex3 = []
for k, params in enumerate(mixture_params):
    n_k = int(n_ex3 * params['weight'])
    data_k = np.random.normal(params['mean'], params['std'], n_k)
    data_ex3.extend(data_k)
    labels_ex3.extend([k] * n_k)

data_ex3 = np.array(data_ex3)
labels_ex3 = np.array(labels_ex3)

print("\nMixture data generated:")
print(f"  n = {len(data_ex3)} samples")
print(f"  K = {len(mixture_params)} components")
for k, params in enumerate(mixture_params):
    print(f"  Component {k+1}: μ={params['mean']}, σ={params['std']}, π={params['weight']}")

# Visualize the data
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(data_ex3, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('x', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title('Mixture Data Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
colors = ['red', 'green', 'blue']
for k in range(len(mixture_params)):
    mask = labels_ex3 == k
    plt.hist(data_ex3[mask], bins=20, density=True, alpha=0.6, 
            color=colors[k], label=f'Component {k+1}', edgecolor='black')
plt.xlabel('x', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title('True Component Separation', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/variational_inference/exercises/exercise_03_data.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n[Data visualization saved: exercise_03_data.png]")

# TODO: Students implement here
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# EXERCISE 4: Stochastic Variational Inference (Advanced)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 4: Implement Stochastic Variational Inference")
print("=" * 80)

exercise_4_description = """
PROBLEM:
-------
Implement SVI for Bayesian logistic regression on a large dataset
that doesn't fit in memory at once.

Model:
- Binary classification: p(y=1|x,w) = σ(w^T x)
- Prior: w ~ N(0, λ⁻¹I)
- Variational family: q(w) = N(m, S)

TASK:
----
1. Derive the ELBO with data subsampling
2. Implement stochastic natural gradient updates
3. Handle mini-batch variance reduction
4. Implement learning rate schedules
5. Compare with full-batch VI

STOCHASTIC GRADIENT:
-------------------
∇_θ ELBO ≈ (N/M) Σ_{i∈M} ∇_θ E_q[log p(y_i|x_i,w)] + ∇_θ E_q[log p(w)/q(w)]

where M is mini-batch size, N is full data size.

DATA:
----
Generate binary classification data:
- N = 10,000 samples
- d = 20 features
- True weights: sparse (only 5 non-zero)
- Class balance: 50-50

DELIVERABLES:
------------
1. SVI implementation with:
   - Mini-batch processing
   - Natural gradient updates
   - Adaptive learning rates
   - Convergence diagnostics
2. Plots:
   a) ELBO vs iterations
   b) Test accuracy vs iterations
   c) Parameter estimates vs truth
   d) Learning rate schedule
3. Comparison: SVI vs full-batch VI
4. Memory usage analysis
5. Scaling experiments (vary N, d, batch size)

HINTS:
-----
- Use natural gradient: ∇_nat = S ∇_m ELBO
- Implement Robbins-Monro learning rate: ρ_t = (t + τ)^{-κ}
- Use Rao-Blackwellization for variance reduction
- Monitor ELBO on held-out validation set
"""

print(exercise_4_description)

# Generate large-scale data
np.random.seed(42)
n_ex4 = 10000
d_ex4 = 20
w_true_ex4 = np.zeros(d_ex4)
w_true_ex4[:5] = np.random.randn(5) * 2  # Only first 5 features are relevant

X_ex4 = np.random.randn(n_ex4, d_ex4)
logits = X_ex4 @ w_true_ex4
probs = 1 / (1 + np.exp(-logits))
y_ex4 = (np.random.rand(n_ex4) < probs).astype(int)

print("\nLarge-scale data generated:")
print(f"  n = {n_ex4} samples")
print(f"  d = {d_ex4} features")
print(f"  Sparsity: {np.sum(w_true_ex4 != 0)} / {d_ex4} non-zero weights")
print(f"  Class balance: {np.mean(y_ex4):.2%} positive")

# TODO: Students implement here
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# BONUS EXERCISE: Black-Box Variational Inference
# ============================================================================

print("\n" + "=" * 80)
print("BONUS EXERCISE: Black-Box Variational Inference")
print("=" * 80)

bonus_exercise_description = """
PROBLEM:
-------
Implement black-box VI using the score function gradient estimator
(REINFORCE) for a model without conjugacy.

Model: Bayesian Probit Regression
- y_i | x_i, w ~ Bernoulli(Φ(w^T x_i))
- where Φ is the standard normal CDF
- Prior: w ~ N(0, I)

This model is NON-CONJUGATE, so standard CAVI doesn't apply!

TASK:
----
1. Implement score function gradient estimator
2. Implement variance reduction techniques:
   - Control variates
   - Rao-Blackwellization
3. Compare with reparameterization trick (if possible)
4. Analyze gradient variance over training

SCORE FUNCTION GRADIENT:
-----------------------
∇_θ E_q[f(z)] = E_q[(∇_θ log q(z)) f(z)]

Can be estimated via Monte Carlo without knowing ∇_z f!

DELIVERABLES:
------------
1. Black-box VI implementation
2. Variance reduction analysis
3. Comparison with model-specific VI
4. Gradient variance plots
5. Final predictive accuracy

This is challenging but teaches you VI for arbitrary models!
"""

print(bonus_exercise_description)


# ============================================================================
# INSTRUCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("INSTRUCTIONS FOR EXERCISES")
print("=" * 80)

instructions = """
HOW TO COMPLETE EXERCISES:
=========================

1. READ THE PROBLEM CAREFULLY
   - Understand the model
   - Identify what needs to be derived vs implemented
   - Note any hints or simplifications

2. MATHEMATICAL DERIVATION
   - Start from first principles
   - Write out the ELBO explicitly
   - Derive update equations step-by-step
   - Check your math with simple cases

3. IMPLEMENTATION
   - Start with initialization
   - Implement each CAVI update carefully
   - Add convergence checks
   - Include visualization code

4. TESTING
   - Test on simple known cases first
   - Compare with exact solutions when available
   - Check gradient computations numerically
   - Verify ELBO increases monotonically

5. ANALYSIS
   - Create required plots
   - Analyze sensitivity to hyperparameters
   - Compare with alternative methods
   - Write up findings

SUBMISSION:
==========
- Python notebook with code and derivations
- PDF with mathematical derivations
- All required plots
- Brief writeup (1-2 pages) per exercise

RESOURCES:
=========
- Module notes and code
- Bishop's PRML Chapter 10
- Blei et al. (2017) VI review
- Office hours: [Schedule TBD]

GRADING:
=======
- Mathematical derivation: 40%
- Code implementation: 30%
- Plots and visualization: 15%
- Analysis and writeup: 15%

HAVE FUN LEARNING VI!
"""

print(instructions)

print("\n" + "=" * 80)
print("END OF EXERCISES")
print("=" * 80)

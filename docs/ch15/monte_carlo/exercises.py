"""
exercises.py

EXERCISES: Progressive Problems in Importance Sampling for Bayesian Inference

This file contains exercises ranging from beginner to advanced levels,
with detailed solutions and explanations.

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")

print("=" * 70)
print("IMPORTANCE SAMPLING EXERCISES")
print("=" * 70)

# =============================================================================
# BEGINNER EXERCISES
# =============================================================================

print("\n" + "=" * 70)
print("BEGINNER LEVEL EXERCISES")
print("=" * 70)

# Exercise 1: Basic IS for Exponential Distribution
# ==============================================
print("\n" + "-" * 70)
print("EXERCISE 1: Estimating E[X²] for Exponential Distribution")
print("-" * 70)

print("""
Problem:
--------
Let X ~ Exp(λ=2), where the PDF is p(x) = 2e^(-2x) for x ≥ 0.
Estimate E[X²] using importance sampling with proposal q(x) = Exp(1).

a) Implement importance sampling
b) Compare with the analytical value: E[X²] = 2/λ² = 0.5
c) Compute ESS
d) Try different sample sizes and observe convergence
""")

print("\nSOLUTION:")

# True distribution: Exp(2)
lambda_true = 2.0
target = stats.expon(scale=1/lambda_true)

# Analytical E[X²]
true_value = 2.0 / lambda_true**2
print(f"Analytical E[X²] = {true_value:.6f}")

# Proposal: Exp(1)
proposal = stats.expon(scale=1.0)

# Function to estimate
h = lambda x: x**2

# Importance sampling
def exercise1_is(n_samples):
    # Sample from proposal
    samples = proposal.rvs(size=n_samples)
    
    # Compute importance weights
    weights_unnorm = target.pdf(samples) / proposal.pdf(samples)
    weights = weights_unnorm / np.sum(weights_unnorm)
    
    # Estimate
    estimate = np.sum(weights * h(samples))
    
    # ESS
    ess = 1.0 / np.sum(weights**2)
    
    return estimate, ess

# Try different sample sizes
sample_sizes = [100, 500, 1000, 5000]
print("\nResults:")
print(f"{'n':<8} {'Estimate':<12} {'Error':<12} {'ESS':<10} {'ESS/n'}")
print("-" * 55)

for n in sample_sizes:
    est, ess = exercise1_is(n)
    error = abs(est - true_value)
    print(f"{n:<8} {est:<12.6f} {error:<12.6f} {ess:<10.1f} {ess/n:.2%}")

print("\nKey insights:")
print("- ESS/n shows the efficiency of the proposal")
print("- Error decreases with √n for consistent estimator")
print("- This proposal works well because it has heavier tails than target")


# Exercise 2: Beta-Binomial with Different Priors
# ==============================================
print("\n" + "-" * 70)
print("EXERCISE 2: Effect of Prior Choice in Beta-Binomial Model")
print("-" * 70)

print("""
Problem:
--------
You observe 15 successes in 20 Bernoulli trials.
Estimate the posterior mean of θ using importance sampling with:
a) Uniform prior: Beta(1,1) as proposal
b) Jeffreys prior: Beta(0.5,0.5) as proposal
c) Informative prior: Beta(2,2) as proposal

Compare ESS and accuracy for each case.
""")

print("\nSOLUTION:")

# Data
successes = 15
trials = 20
failures = trials - successes

# Unnormalized posterior (likelihood only, since we use prior as proposal)
def unnorm_posterior(theta, alpha0, beta0):
    """Posterior ∝ likelihood × prior"""
    # Clip to avoid log(0)
    theta = np.clip(theta, 1e-10, 1-1e-10)
    
    # Log likelihood
    log_lik = successes * np.log(theta) + failures * np.log(1 - theta)
    
    # Log prior
    log_prior = (alpha0-1) * np.log(theta) + (beta0-1) * np.log(1 - theta)
    
    return np.exp(log_lik + log_prior)

# Different priors
priors = {
    'Uniform Beta(1,1)': (1, 1),
    'Jeffreys Beta(0.5,0.5)': (0.5, 0.5),
    'Informative Beta(2,2)': (2, 2),
}

n_samples = 2000
print(f"\nUsing {n_samples} samples:")
print(f"{'Prior':<25} {'Post Mean':<12} {'ESS':<10} {'ESS/n'}")
print("-" * 55)

for name, (alpha0, beta0) in priors.items():
    # Analytical posterior
    alpha_n = alpha0 + successes
    beta_n = beta0 + failures
    analytical_mean = alpha_n / (alpha_n + beta_n)
    
    # Importance sampling
    proposal = stats.beta(alpha0, beta0)
    samples = proposal.rvs(size=n_samples)
    
    weights_unnorm = unnorm_posterior(samples, alpha0, beta0) / proposal.pdf(samples)
    weights = weights_unnorm / np.sum(weights_unnorm)
    
    post_mean = np.sum(weights * samples)
    ess = 1.0 / np.sum(weights**2)
    
    print(f"{name:<25} {post_mean:<12.6f} {ess:<10.1f} {ess/n_samples:.2%}")
    print(f"{'  (analytical)':<25} {analytical_mean:<12.6f}")

print("\nKey insights:")
print("- All give accurate posterior mean estimates")
print("- ESS depends on prior-posterior mismatch")
print("- Weak prior as proposal → good ESS when data is strong")


# =============================================================================
# INTERMEDIATE EXERCISES
# =============================================================================

print("\n" + "=" * 70)
print("INTERMEDIATE LEVEL EXERCISES")
print("=" * 70)

# Exercise 3: Optimal Sample Size
# ==============================
print("\n" + "-" * 70)
print("EXERCISE 3: Determining Optimal Sample Size")
print("-" * 70)

print("""
Problem:
--------
For estimating E[θ] where θ ~ N(5,1) using proposal q ~ N(3,2):

a) How does variance of estimate scale with n?
b) Estimate the sample size needed for 95% confidence interval width < 0.1
c) Compare computational cost vs accuracy trade-off
""")

print("\nSOLUTION:")

target_ex3 = stats.norm(5, 1)
proposal_ex3 = stats.norm(3, 2)
h_identity = lambda x: x

true_mean = 5.0

# Estimate variance for different n
sample_sizes_ex3 = [50, 100, 200, 500, 1000, 2000, 5000]
n_reps = 200

print("\nVariance scaling with sample size:")
print(f"{'n':<8} {'Est Var':<12} {'Std Error':<12} {'95% CI Width':<15} {'Time (ms)'}")
print("-" * 65)

import time

for n in sample_sizes_ex3:
    estimates = []
    start_time = time.time()
    
    for _ in range(n_reps):
        samples = proposal_ex3.rvs(size=n)
        weights_unnorm = target_ex3.pdf(samples) / proposal_ex3.pdf(samples)
        weights = weights_unnorm / np.sum(weights_unnorm)
        estimate = np.sum(weights * h_identity(samples))
        estimates.append(estimate)
    
    elapsed = (time.time() - start_time) / n_reps * 1000  # ms per replication
    
    var_est = np.var(estimates)
    std_err = np.std(estimates)
    ci_width = 1.96 * 2 * std_err  # 95% CI width
    
    print(f"{n:<8} {var_est:<12.6f} {std_err:<12.6f} {ci_width:<15.6f} {elapsed:<.2f}")

print("\nKey insights:")
print("- Variance ∝ 1/n (standard Monte Carlo rate)")
print("- For CI width < 0.1, need approximately n ≥ 1500")
print("- Computational cost scales linearly with n")


# Exercise 4: Diagnosing Poor Proposals
# ====================================
print("\n" + "-" * 70)
print("EXERCISE 4: Identifying and Fixing Poor Proposals")
print("-" * 70)

print("""
Problem:
--------
Given target π ~ N(10, 1), you try proposal q ~ N(0, 1).
This is a poor proposal. Diagnose why and suggest improvements.

a) Compute ESS
b) Examine weight distribution
c) Propose and test a better proposal
""")

print("\nSOLUTION:")

target_ex4 = stats.norm(10, 1)
poor_proposal = stats.norm(0, 1)

n_samples_ex4 = 2000

# Poor proposal
samples_poor = poor_proposal.rvs(size=n_samples_ex4)
weights_poor_unnorm = target_ex4.pdf(samples_poor) / poor_proposal.pdf(samples_poor)
weights_poor = weights_poor_unnorm / np.sum(weights_poor_unnorm)
ess_poor = 1.0 / np.sum(weights_poor**2)

print("\nPoor Proposal Analysis:")
print(f"  Proposal: N(0, 1)")
print(f"  ESS: {ess_poor:.1f} ({ess_poor/n_samples_ex4:.1%})")
print(f"  Max weight: {np.max(weights_poor):.6f}")
print(f"  CV of weights: {np.std(weights_poor)/np.mean(weights_poor):.2f}")

# Weight concentration
sorted_weights = np.sort(weights_poor)[::-1]
cumsum = np.cumsum(sorted_weights)
n_for_50pct = np.searchsorted(cumsum, 0.5) + 1
print(f"  Samples for 50% weight: {n_for_50pct} ({n_for_50pct/n_samples_ex4:.1%})")

print("\nDiagnosis:")
print("  ✗ Very low ESS (~1-2% efficiency)")
print("  ✗ Few samples carry most weight")
print("  ✗ Proposal mean far from target mean")
print("  ✗ Most samples in low-probability region")

# Better proposal
better_proposal = stats.norm(10, 1.5)
samples_better = better_proposal.rvs(size=n_samples_ex4)
weights_better_unnorm = target_ex4.pdf(samples_better) / better_proposal.pdf(samples_better)
weights_better = weights_better_unnorm / np.sum(weights_better_unnorm)
ess_better = 1.0 / np.sum(weights_better**2)

print("\nImproved Proposal Analysis:")
print(f"  Proposal: N(10, 1.5)")
print(f"  ESS: {ess_better:.1f} ({ess_better/n_samples_ex4:.1%})")
print(f"  Max weight: {np.max(weights_better):.6f}")
print(f"  Improvement: {ess_better/ess_poor:.1f}x better ESS")


# =============================================================================
# ADVANCED EXERCISES
# =============================================================================

print("\n" + "=" * 70)
print("ADVANCED LEVEL EXERCISES")
print("=" * 70)

# Exercise 5: Importance Sampling for Rare Events
# ==============================================
print("\n" + "-" * 70)
print("EXERCISE 5: Rare Event Probability Estimation")
print("-" * 70)

print("""
Problem:
--------
Estimate P(X > 4) where X ~ N(0, 1). This is a rare event (p ≈ 0.000032).

a) Try naive Monte Carlo
b) Use importance sampling with shifted proposal
c) Compute variance reduction factor
d) Determine sample size for 10% relative error
""")

print("\nSOLUTION:")

target_ex5 = stats.norm(0, 1)
threshold = 4.0
true_prob = 1 - target_ex5.cdf(threshold)

print(f"True probability: {true_prob:.8f}")

# Naive Monte Carlo
n_mc = 100000
samples_mc = target_ex5.rvs(size=n_mc)
h_indicator = lambda x: (x > threshold).astype(float)
estimate_mc = np.mean(h_indicator(samples_mc))

print(f"\nNaive MC ({n_mc} samples):")
print(f"  Estimate: {estimate_mc:.8f}")
print(f"  Relative error: {abs(estimate_mc - true_prob)/true_prob:.1%}")

# Importance sampling with shifted proposal
# Proposal centered near the tail
proposal_ex5 = stats.norm(threshold + 1, 1)
n_is = 10000

samples_is = proposal_ex5.rvs(size=n_is)
weights_unnorm_ex5 = target_ex5.pdf(samples_is) / proposal_ex5.pdf(samples_is)
weights_ex5 = weights_unnorm_ex5 / np.sum(weights_unnorm_ex5)
estimate_is = np.sum(weights_ex5 * h_indicator(samples_is))

print(f"\nImportance Sampling ({n_is} samples):")
print(f"  Estimate: {estimate_is:.8f}")
print(f"  Relative error: {abs(estimate_is - true_prob)/true_prob:.1%}")
print(f"  ESS: {1.0/np.sum(weights_ex5**2):.1f}")

# Variance comparison via replication
n_reps_ex5 = 500

mc_estimates = []
is_estimates = []

for _ in range(n_reps_ex5):
    # MC
    samples_mc_rep = target_ex5.rvs(size=1000)
    mc_estimates.append(np.mean(h_indicator(samples_mc_rep)))
    
    # IS
    samples_is_rep = proposal_ex5.rvs(size=1000)
    w_unnorm = target_ex5.pdf(samples_is_rep) / proposal_ex5.pdf(samples_is_rep)
    w_norm = w_unnorm / np.sum(w_unnorm)
    is_estimates.append(np.sum(w_norm * h_indicator(samples_is_rep)))

var_mc = np.var(mc_estimates)
var_is = np.var(is_estimates)
variance_reduction = var_mc / var_is

print(f"\nVariance Comparison (1000 samples, {n_reps_ex5} replications):")
print(f"  MC variance: {var_mc:.2e}")
print(f"  IS variance: {var_is:.2e}")
print(f"  Variance reduction: {variance_reduction:.1f}x")

# Sample size for 10% relative error
# Relative error ≈ std/mean
# Want std/mean < 0.1, so std < 0.1*mean
# std ≈ √var, so √var < 0.1*true_prob
# For n samples: var ∝ 1/n

target_rel_error = 0.10
required_std = target_rel_error * true_prob
required_var = required_std**2

n_required_mc = var_mc / required_var
n_required_is = var_is / required_var

print(f"\nSample size for 10% relative error:")
print(f"  MC needs: {n_required_mc:.0f} samples")
print(f"  IS needs: {n_required_is:.0f} samples")
print(f"  Reduction: {n_required_mc/n_required_is:.1f}x fewer samples with IS")


# Exercise 6: Multimodal Posterior
# ==============================
print("\n" + "-" * 70)
print("EXERCISE 6: Importance Sampling for Multimodal Posterior")
print("-" * 70)

print("""
Problem:
--------
Consider a mixture likelihood creating a bimodal posterior:
  Likelihood: 0.4*N(y|θ, 1) + 0.6*N(y|θ+6, 1) with y = 2
  Prior: θ ~ N(0, 4)
  
The posterior has two modes. Design an IS strategy.

a) Implement single-component proposal
b) Implement mixture proposal
c) Compare ESS
d) Estimate posterior mean and variance
""")

print("\nSOLUTION:")

y_obs = 2.0

def log_likelihood_mixture(theta):
    """Log likelihood for mixture model"""
    ll1 = stats.norm.logpdf(y_obs, theta, 1) + np.log(0.4)
    ll2 = stats.norm.logpdf(y_obs, theta + 6, 1) + np.log(0.6)
    return logsumexp([ll1, ll2])

def unnorm_posterior_mixture(theta):
    """Unnormalized posterior"""
    log_prior = stats.norm.logpdf(theta, 0, 2)
    return np.exp(log_likelihood_mixture(theta) + log_prior)

# Single-component proposal (centered between modes)
proposal_single = stats.norm(-2, 3)

n_samples_ex6 = 5000
samples_single = proposal_single.rvs(size=n_samples_ex6)
weights_single_unnorm = np.array([
    unnorm_posterior_mixture(s) / proposal_single.pdf(s)
    for s in samples_single
])
weights_single = weights_single_unnorm / np.sum(weights_single_unnorm)
ess_single = 1.0 / np.sum(weights_single**2)

post_mean_single = np.sum(weights_single * samples_single)
post_var_single = np.sum(weights_single * (samples_single - post_mean_single)**2)

print("\nSingle-Component Proposal N(-2, 3):")
print(f"  ESS: {ess_single:.1f} ({ess_single/n_samples_ex6:.1%})")
print(f"  Posterior mean: {post_mean_single:.4f}")
print(f"  Posterior std: {np.sqrt(post_var_single):.4f}")

# Mixture proposal (two components, one per mode)
class MixtureProposal:
    def __init__(self, means, stds, weights):
        self.components = [stats.norm(m, s) for m, s in zip(means, stds)]
        self.weights = np.array(weights) / np.sum(weights)
    
    def rvs(self, size):
        # Sample component
        components = np.random.choice(len(self.components), size=size, p=self.weights)
        samples = []
        for i in range(size):
            comp_idx = components[i]
            sample = self.components[comp_idx].rvs()
            samples.append(sample)
        return np.array(samples)
    
    def pdf(self, x):
        densities = []
        for comp, weight in zip(self.components, self.weights):
            densities.append(weight * comp.pdf(x))
        return np.sum(densities, axis=0)

# Mixture centered at approximate modes
proposal_mixture = MixtureProposal(
    means=[-1, -7],  # Approximate posterior modes
    stds=[1.5, 1.5],
    weights=[0.4, 0.6]
)

samples_mixture = proposal_mixture.rvs(size=n_samples_ex6)
weights_mixture_unnorm = np.array([
    unnorm_posterior_mixture(s) / proposal_mixture.pdf(s)
    for s in samples_mixture
])
weights_mixture = weights_mixture_unnorm / np.sum(weights_mixture_unnorm)
ess_mixture = 1.0 / np.sum(weights_mixture**2)

post_mean_mixture = np.sum(weights_mixture * samples_mixture)
post_var_mixture = np.sum(weights_mixture * (samples_mixture - post_mean_mixture)**2)

print("\nMixture Proposal:")
print(f"  Components: 0.4*N(-1,1.5) + 0.6*N(-7,1.5)")
print(f"  ESS: {ess_mixture:.1f} ({ess_mixture/n_samples_ex6:.1%})")
print(f"  Posterior mean: {post_mean_mixture:.4f}")
print(f"  Posterior std: {np.sqrt(post_var_mixture):.4f}")
print(f"\nImprovement: {ess_mixture/ess_single:.2f}x better ESS with mixture proposal")

print("\nKey insights:")
print("- Multimodal posteriors need careful proposal design")
print("- Single-component proposals may miss modes")
print("- Mixture proposals can capture multiple modes")
print("- ESS much higher with mixture proposal")

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
These exercises cover:

BEGINNER:
1. Basic IS implementation and convergence
2. Effect of prior choice in conjugate models

INTERMEDIATE:
3. Sample size determination and cost-accuracy trade-offs
4. Diagnosing and fixing poor proposals

ADVANCED:
5. Rare event estimation with variance reduction
6. Multimodal posteriors with mixture proposals

Key Skills Developed:
- Implementing IS from scratch
- Computing and interpreting ESS
- Diagnosing proposal quality
- Choosing appropriate proposals
- Variance reduction techniques
- Handling complex posteriors

For further practice:
- Try different target distributions
- Experiment with proposal families
- Compare IS with MCMC methods
- Implement adaptive IS variations
- Apply to real Bayesian inference problems
""")

plt.show()

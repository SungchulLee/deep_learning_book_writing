"""
Bayesian Inference - Module 3: Conjugate Priors
Level: Intermediate
Topics: Conjugate families, analytical posteriors, computational advantages

Conjugate priors allow analytical solutions to Bayesian inference problems,
avoiding the need for numerical integration or sampling methods.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# CONJUGATE PRIOR THEORY
# ============================================================================

"""
CONJUGATE PRIOR DEFINITION:

A prior distribution p(θ) is conjugate to a likelihood function p(D|θ) if
the posterior distribution p(θ|D) is in the same parametric family as the prior.

Mathematically:
If p(θ) ∈ F and p(θ|D) ∈ F for all data D, then the prior family F is conjugate.

ADVANTAGES OF CONJUGATE PRIORS:
1. Analytical posteriors (closed-form solutions)
2. Computational efficiency (no numerical integration)
3. Interpretable updates (simple parameter transformations)
4. Sequential updating is straightforward
5. Mathematical elegance and insight

COMMON CONJUGATE FAMILIES:
1. Beta-Binomial: Beta prior + Binomial/Bernoulli likelihood → Beta posterior
2. Normal-Normal: Normal prior + Normal likelihood (known variance) → Normal posterior
3. Gamma-Poisson: Gamma prior + Poisson likelihood → Gamma posterior
4. Gamma-Exponential: Gamma prior + Exponential likelihood → Gamma posterior
5. Dirichlet-Multinomial: Dirichlet prior + Multinomial likelihood → Dirichlet posterior
6. Normal-Inverse-Gamma: For Normal with unknown mean and variance
"""

# ============================================================================
# CONJUGATE PAIR 1: BETA-BINOMIAL
# ============================================================================

class BetaBinomialModel:
    """
    Beta-Binomial conjugate model for binary data.
    
    Prior: Beta(α, β)
    Likelihood: Binomial(n, θ)
    Posterior: Beta(α + k, β + n - k)
    
    where k is the number of successes in n trials.
    """
    
    def __init__(self, alpha=1, beta=1):
        """
        Initialize with prior parameters.
        
        Parameters:
        -----------
        alpha, beta : float
            Parameters of Beta prior (both > 0)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive")
        
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.posterior_alpha = alpha
        self.posterior_beta = beta
        self.data_history = []
    
    def update(self, successes, trials):
        """
        Update posterior with new data.
        
        Parameters:
        -----------
        successes : int
            Number of successes observed
        trials : int
            Number of trials conducted
        """
        failures = trials - successes
        self.posterior_alpha += successes
        self.posterior_beta += failures
        self.data_history.append((successes, trials))
    
    def posterior_predictive(self, n_trials=1):
        """
        Compute posterior predictive distribution for future observations.
        
        The posterior predictive distribution is:
        P(y successes in n trials | data) = 
            ∫ Binomial(y|n,θ) * Beta(θ|α',β') dθ
        = BetaBinomial(y|n,α',β')
        
        Parameters:
        -----------
        n_trials : int
            Number of future trials
        
        Returns:
        --------
        probs : numpy array
            Probability of 0, 1, ..., n_trials successes
        """
        y_values = np.arange(n_trials + 1)
        probs = []
        
        for y in y_values:
            # Beta-Binomial formula
            prob = (stats.binom.comb(n_trials, y) * 
                   stats.beta.beta_func(y + self.posterior_alpha, 
                                       n_trials - y + self.posterior_beta) / 
                   stats.beta.beta_func(self.posterior_alpha, self.posterior_beta))
            probs.append(prob)
        
        return np.array(probs)
    
    def summary(self):
        """Print summary statistics."""
        prior_dist = stats.beta(self.prior_alpha, self.prior_beta)
        post_dist = stats.beta(self.posterior_alpha, self.posterior_beta)
        
        print("Beta-Binomial Model Summary")
        print("="*60)
        print(f"Prior: Beta({self.prior_alpha}, {self.prior_beta})")
        print(f"  Mean: {prior_dist.mean():.4f}")
        print(f"  Std:  {prior_dist.std():.4f}")
        print(f"\nPosterior: Beta({self.posterior_alpha}, {self.posterior_beta})")
        print(f"  Mean: {post_dist.mean():.4f}")
        print(f"  Std:  {post_dist.std():.4f}")
        print(f"  95% Credible Interval: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
        print(f"\nTotal observations: {sum([t for _, t in self.data_history])}")

# ============================================================================
# CONJUGATE PAIR 2: GAMMA-POISSON
# ============================================================================

"""
GAMMA-POISSON MODEL:

Used for count data (e.g., number of events in fixed time periods).

Prior: Gamma(α, β)
  p(λ) = (β^α / Γ(α)) λ^(α-1) e^(-βλ)
  Mean: α/β
  Variance: α/β²

Likelihood: Poisson(λ)
  p(x|λ) = (λ^x / x!) e^(-λ)

Posterior: Gamma(α + Σx, β + n)
  where Σx is the sum of observed counts, n is number of observations

INTERPRETATION:
- α: prior "pseudo-counts" (total events)
- β: prior "pseudo-observations" (number of time periods)
- After observing data, add actual counts to α and number of periods to β
"""

class GammaPoissonModel:
    """
    Gamma-Poisson conjugate model for count data.
    
    Prior: Gamma(α, β)
    Likelihood: Poisson(λ)
    Posterior: Gamma(α + Σx, β + n)
    """
    
    def __init__(self, alpha=1, beta=1):
        """
        Initialize with prior parameters.
        
        Parameters:
        -----------
        alpha : float
            Shape parameter (pseudo-counts)
        beta : float
            Rate parameter (pseudo-observations)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive")
        
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.posterior_alpha = alpha
        self.posterior_beta = beta
        self.data = []
    
    def update(self, counts):
        """
        Update posterior with new count data.
        
        Parameters:
        -----------
        counts : array-like
            Observed counts (one per time period)
        """
        counts = np.asarray(counts)
        self.posterior_alpha += np.sum(counts)
        self.posterior_beta += len(counts)
        self.data.extend(counts)
    
    def posterior_predictive(self):
        """
        Compute posterior predictive distribution.
        
        The posterior predictive for Gamma-Poisson is a Negative Binomial:
        P(x|data) = NegativeBinomial(x|α', β'/(β'+1))
        """
        # Negative Binomial parameters
        n = self.posterior_alpha
        p = self.posterior_beta / (self.posterior_beta + 1)
        
        return stats.nbinom(n, p)
    
    def summary(self):
        """Print summary statistics."""
        prior_dist = stats.gamma(self.prior_alpha, scale=1/self.prior_beta)
        post_dist = stats.gamma(self.posterior_alpha, scale=1/self.posterior_beta)
        
        print("Gamma-Poisson Model Summary")
        print("="*60)
        print(f"Prior: Gamma({self.prior_alpha}, {self.prior_beta})")
        print(f"  Mean (rate): {prior_dist.mean():.4f}")
        print(f"  Std:         {prior_dist.std():.4f}")
        print(f"\nPosterior: Gamma({self.posterior_alpha}, {self.posterior_beta})")
        print(f"  Mean (rate): {post_dist.mean():.4f}")
        print(f"  Std:         {post_dist.std():.4f}")
        print(f"  95% Credible Interval: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
        print(f"\nTotal observations: {len(self.data)}")
        print(f"Total count: {sum(self.data)}")

# ============================================================================
# CONJUGATE PAIR 3: NORMAL-NORMAL (Known Variance)
# ============================================================================

"""
NORMAL-NORMAL MODEL (Known Variance):

Used for continuous data when we know the variance but want to infer the mean.

Prior: N(μ₀, σ₀²)
Likelihood: N(μ, σ²) with known σ²
Posterior: N(μₙ, σₙ²)

Posterior parameters:
  Precision (inverse variance): τₙ = τ₀ + nτ
  where τ₀ = 1/σ₀², τ = 1/σ²
  
  Mean: μₙ = (τ₀μ₀ + nτx̄) / τₙ
  Variance: σₙ² = 1/τₙ

INTERPRETATION:
The posterior mean is a precision-weighted average of prior mean and sample mean.
As n increases, the data dominates and μₙ → x̄.
"""

class NormalNormalModel:
    """
    Normal-Normal conjugate model (known variance).
    
    Prior: N(μ₀, σ₀²)
    Likelihood: N(μ, σ²) with known σ²
    Posterior: N(μₙ, σₙ²)
    """
    
    def __init__(self, prior_mean=0, prior_std=1, known_std=1):
        """
        Initialize with prior parameters and known data standard deviation.
        
        Parameters:
        -----------
        prior_mean : float
            Mean of prior distribution
        prior_std : float
            Standard deviation of prior
        known_std : float
            Known standard deviation of data
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.known_std = known_std
        
        # Initialize posterior to prior
        self.posterior_mean = prior_mean
        self.posterior_std = prior_std
        
        self.data = []
    
    def update(self, observations):
        """
        Update posterior with new observations.
        
        Parameters:
        -----------
        observations : array-like
            New data points
        """
        observations = np.asarray(observations)
        n = len(observations)
        x_bar = np.mean(observations)
        
        # Precision calculations
        prior_precision = 1 / (self.prior_std ** 2)
        data_precision = n / (self.known_std ** 2)
        posterior_precision = prior_precision + data_precision
        
        # Update parameters
        self.posterior_mean = ((prior_precision * self.prior_mean + 
                               data_precision * x_bar) / posterior_precision)
        self.posterior_std = np.sqrt(1 / posterior_precision)
        
        # For next update, current posterior becomes new prior
        self.prior_mean = self.posterior_mean
        self.prior_std = self.posterior_std
        
        self.data.extend(observations)
    
    def summary(self):
        """Print summary statistics."""
        print("Normal-Normal Model Summary (Known Variance)")
        print("="*60)
        print(f"Prior: N({self.prior_mean:.4f}, {self.prior_std:.4f})")
        print(f"\nKnown data std: {self.known_std:.4f}")
        print(f"\nPosterior: N({self.posterior_mean:.4f}, {self.posterior_std:.4f})")
        
        post_dist = stats.norm(self.posterior_mean, self.posterior_std)
        print(f"  95% Credible Interval: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
        print(f"\nTotal observations: {len(self.data)}")
        if self.data:
            print(f"Sample mean: {np.mean(self.data):.4f}")
            print(f"Sample std:  {np.std(self.data, ddof=1):.4f}")

# ============================================================================
# EXAMPLE: COMPARING ALL THREE CONJUGATE MODELS
# ============================================================================

def demonstrate_conjugate_priors():
    """
    Demonstrate all three conjugate prior families with examples.
    """
    print("\n" + "="*70)
    print("CONJUGATE PRIORS DEMONSTRATION")
    print("="*70)
    
    # Example 1: Beta-Binomial (coin flipping)
    print("\n" + "-"*70)
    print("EXAMPLE 1: Beta-Binomial Model")
    print("-"*70)
    
    bb_model = BetaBinomialModel(alpha=2, beta=2)  # Weak prior belief in fairness
    bb_model.update(successes=17, trials=20)
    bb_model.summary()
    
    # Visualization
    theta = np.linspace(0, 1, 1000)
    prior_pdf = stats.beta(bb_model.prior_alpha, bb_model.prior_beta).pdf(theta)
    post_pdf = stats.beta(bb_model.posterior_alpha, bb_model.posterior_beta).pdf(theta)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(theta, prior_pdf, 'b--', linewidth=2, label='Prior')
    plt.plot(theta, post_pdf, 'r-', linewidth=2, label='Posterior')
    plt.xlabel('θ (Success Probability)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title('Beta-Binomial: Prior and Posterior', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Posterior predictive
    plt.subplot(1, 2, 2)
    pred_probs = bb_model.posterior_predictive(n_trials=10)
    plt.bar(range(11), pred_probs, alpha=0.7, color='green')
    plt.xlabel('Number of Successes in 10 Trials', fontsize=11)
    plt.ylabel('Probability', fontsize=11)
    plt.title('Posterior Predictive Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('conjugate_beta_binomial.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 2: Gamma-Poisson (website visits)
    print("\n" + "-"*70)
    print("EXAMPLE 2: Gamma-Poisson Model")
    print("-"*70)
    
    gp_model = GammaPoissonModel(alpha=2, beta=1)  # Prior: expect ~2 events per period
    # Observed daily website visits
    daily_visits = [5, 3, 7, 4, 6, 5, 8, 3, 4, 6]
    gp_model.update(daily_visits)
    gp_model.summary()
    
    # Visualization
    lambda_vals = np.linspace(0, 15, 1000)
    prior_pdf = stats.gamma(gp_model.prior_alpha, scale=1/gp_model.prior_beta).pdf(lambda_vals)
    post_pdf = stats.gamma(gp_model.posterior_alpha, scale=1/gp_model.posterior_beta).pdf(lambda_vals)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lambda_vals, prior_pdf, 'b--', linewidth=2, label='Prior')
    plt.plot(lambda_vals, post_pdf, 'r-', linewidth=2, label='Posterior')
    plt.axvline(np.mean(daily_visits), color='green', linestyle=':', linewidth=2, 
                label=f'Sample mean={np.mean(daily_visits):.1f}')
    plt.xlabel('λ (Rate Parameter)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title('Gamma-Poisson: Prior and Posterior', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Data histogram
    plt.subplot(1, 2, 2)
    plt.hist(daily_visits, bins=10, alpha=0.7, color='gray', edgecolor='black')
    plt.xlabel('Daily Visits', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Observed Data Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('conjugate_gamma_poisson.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 3: Normal-Normal (measurement)
    print("\n" + "-"*70)
    print("EXAMPLE 3: Normal-Normal Model")
    print("-"*70)
    
    nn_model = NormalNormalModel(prior_mean=100, prior_std=10, known_std=5)
    # Observed measurements
    measurements = [102, 98, 105, 101, 99, 103, 97, 104, 100, 102]
    nn_model.update(measurements)
    nn_model.summary()
    
    # Visualization
    mu_vals = np.linspace(85, 115, 1000)
    prior_pdf = stats.norm(100, 10).pdf(mu_vals)
    post_pdf = stats.norm(nn_model.posterior_mean, nn_model.posterior_std).pdf(mu_vals)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mu_vals, prior_pdf, 'b--', linewidth=2, label='Prior')
    plt.plot(mu_vals, post_pdf, 'r-', linewidth=2, label='Posterior')
    plt.axvline(np.mean(measurements), color='green', linestyle=':', linewidth=2, 
                label=f'Sample mean={np.mean(measurements):.1f}')
    plt.xlabel('μ (Mean)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title('Normal-Normal: Prior and Posterior', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Data histogram
    plt.subplot(1, 2, 2)
    plt.hist(measurements, bins=8, alpha=0.7, color='gray', edgecolor='black', density=True)
    x = np.linspace(min(measurements)-3, max(measurements)+3, 100)
    plt.plot(x, stats.norm(nn_model.posterior_mean, nn_model.known_std).pdf(x), 
             'r-', linewidth=2, label='Data model with posterior mean')
    plt.xlabel('Measurements', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title('Observed Data with Model', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('conjugate_normal_normal.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 3: CONJUGATE PRIORS")
    print("="*70)
    
    demonstrate_conjugate_priors()
    
    print("\n" + "="*70)
    print("MODULE 3 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Conjugate priors yield analytical posterior distributions")
    print("2. Beta-Binomial: for binary/proportion data")
    print("3. Gamma-Poisson: for count data")
    print("4. Normal-Normal: for continuous data (known variance)")
    print("5. Conjugacy makes sequential updating computationally efficient")
    print("\nNext: Module 4 - MAP Estimation")
    print("="*70)

# ============================================================================
# EXERCISES
# ============================================================================

"""
EXERCISE 1: Conjugate Pair Verification
Mathematically verify that Beta is conjugate to Binomial by working through
the algebra of Bayes' theorem. Show that p(θ|data) is indeed Beta.

EXERCISE 2: Prior Elicitation
You believe a coin is fair but aren't certain. Express this as Beta(α, β).
What values of α and β capture "weak belief" vs "strong belief"?

EXERCISE 3: Exponential-Gamma
Research the Gamma-Exponential conjugate pair (for waiting times/lifetimes).
Implement a class similar to GammaPoissonModel for this pair.

EXERCISE 4: Dirichlet-Multinomial
Extend Beta-Binomial to multiple categories using Dirichlet-Multinomial.
Implement inference for a 3-outcome dice rolling problem.

EXERCISE 5: Non-conjugate Prior
What happens when the prior is not conjugate? Compare analytical Beta-Binomial
with numerical integration using a different prior (e.g., Uniform on log(θ)).
"""

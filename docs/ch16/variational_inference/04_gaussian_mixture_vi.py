"""
Variational Inference - Module 4: Gaussian Mixture Models with VI
==================================================================

Learning Objectives:
-------------------
1. Understand mixture models and latent variable models
2. Derive VI updates for Gaussian Mixture Models
3. Compare VI with EM algorithm
4. Implement complete GMM-VI algorithm
5. Analyze convergence and model selection

Prerequisites:
-------------
- Modules 01-03: VI basics, ELBO, mean-field, CAVI
- Understanding of mixture models
- Familiarity with EM algorithm (helpful but not required)

Author: Prof. Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import digamma, gammaln, logsumexp
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# ============================================================================
# SECTION 1: Gaussian Mixture Model - Problem Setup
# ============================================================================

def gmm_model_introduction():
    """
    Introduce the Gaussian Mixture Model and variational inference approach.
    
    GAUSSIAN MIXTURE MODEL (GMM):
    ============================
    
    A GMM represents data as a mixture of K Gaussian distributions:
    
    p(x | π, μ, Σ) = Σₖ πₖ N(x | μₖ, Σₖ)
    
    where:
    - π = (π₁, ..., π_K) are mixture weights (π_k ≥ 0, Σ_k π_k = 1)
    - μ = (μ₁, ..., μ_K) are component means
    - Σ = (Σ₁, ..., Σ_K) are component covariances
    
    LATENT VARIABLE FORMULATION:
    ----------------------------
    Introduce latent indicators z_i ∈ {1,...,K} for each data point x_i:
    
    p(z_i = k) = πₖ
    p(x_i | z_i = k) = N(x_i | μₖ, Σₖ)
    
    Complete data likelihood:
    p(D, Z | θ) = ∏ᵢ ∏ₖ [πₖ N(x_i | μₖ, Σₖ)]^{z_{ik}}
    
    where z_{ik} = 1 if z_i = k, 0 otherwise (one-hot encoding)
    
    BAYESIAN GMM:
    ------------
    Add priors on parameters:
    
    - π ~ Dirichlet(α)
    - μₖ ~ N(m, (βΣₖ)⁻¹) for each k
    - Σₖ ~ InverseWishart(ν, W) for each k
    
    VARIATIONAL INFERENCE FOR GMM:
    =============================
    
    Parameters: θ = {π, μ, Σ}
    Latent variables: Z = {z₁, ..., z_n}
    
    Mean-field approximation:
    q(Z, θ) = q(Z) ∏ₖ q(πₖ) q(μₖ) q(Σₖ)
    
    Or more simply (conjugate priors):
    q(Z, θ) = q(Z) q(π) q(μ, Σ)
    
    KEY INSIGHT:
    -----------
    VI for GMM is closely related to EM algorithm!
    - E-step ↔ Update q(Z)
    - M-step ↔ Update q(θ)
    
    But VI maintains uncertainty over parameters via distributions,
    while EM uses point estimates.
    """
    
    print("=" * 80)
    print("GAUSSIAN MIXTURE MODEL WITH VARIATIONAL INFERENCE")
    print("=" * 80)
    
    print("""
PROBLEM SETUP:
=============

Data: x₁, x₂, ..., x_n ∈ ℝᵈ

Generative Model:
    For each data point i:
    1. Sample cluster assignment: z_i ~ Categorical(π)
    2. Sample data point: x_i ~ N(μ_{z_i}, Σ_{z_i})

Inference Goal:
    Given observed X, infer:
    - Cluster assignments Z
    - Component parameters {π, μ, Σ}

VI Approach:
    Approximate p(Z, θ | X) with q(Z)q(θ)
    
MODEL COMPLEXITY:
================

Parameters per component:
- Mean μₖ: d parameters
- Covariance Σₖ: d(d+1)/2 parameters (symmetric)
- Mixture weight πₖ: 1 parameter (with Σπₖ = 1 constraint)

Total: K × [d + d(d+1)/2] + (K-1) parameters

Example (d=2, K=3):
    Each component: 2 + 3 = 5 parameters
    Total: 15 + 2 = 17 parameters
    
This high dimensionality makes full Bayesian inference challenging!
""")


# ============================================================================
# SECTION 2: Simplified GMM with Fixed Covariance
# ============================================================================

class GaussianMixtureVI:
    """
    Gaussian Mixture Model with Variational Inference.
    
    Simplified version: Fixed, shared, spherical covariance σ²I.
    This makes derivations clearer while maintaining key concepts.
    
    Model:
    -----
    x_i | z_i, θ ~ N(μ_{z_i}, σ²I)
    z_i | π ~ Categorical(π)
    π ~ Dirichlet(α)
    μₖ ~ N(m₀, s₀²I)
    
    Variational approximation:
    -------------------------
    q(Z, π, μ) = q(Z) q(π) q(μ)
    
    where:
    - q(Z) = ∏ᵢ Categorical(φᵢ)  [responsibilities]
    - q(π) = Dirichlet(α*)        [mixture weights]
    - q(μₖ) = N(mₖ*, sₖ²I)         [component means]
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        """
        Initialize GMM-VI model.
        
        Parameters:
        ----------
        n_components : int
            Number of mixture components
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance for ELBO
        """
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        # Model hyperparameters (set later during fit)
        self.alpha_0 = None
        self.m_0 = None
        self.s_0_sq = None
        self.sigma_sq = None
        
        # Variational parameters
        self.phi = None      # Responsibilities (n × K)
        self.alpha = None    # Dirichlet parameters (K)
        self.m = None        # Mean parameters (K × d)
        self.s_sq = None     # Variance parameters (K)
        
        # Monitoring
        self.elbo_history = []
        
    def initialize(self, X):
        """
        Initialize variational parameters.
        
        Strategy: Use k-means++ initialization
        """
        n, d = X.shape
        
        # Initialize responsibilities with k-means++
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.K, init='k-means++', n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Convert labels to responsibilities (add small noise)
        self.phi = np.zeros((n, self.K))
        for i in range(n):
            self.phi[i, labels[i]] = 0.9
            self.phi[i, :] += 0.1 / self.K
        self.phi /= self.phi.sum(axis=1, keepdims=True)
        
        # Initialize mixture weights
        self.alpha = self.alpha_0 + self.phi.sum(axis=0)
        
        # Initialize component means
        self.m = np.zeros((self.K, d))
        self.s_sq = np.zeros(self.K)
        for k in range(self.K):
            # Weighted mean
            Nₖ = self.phi[:, k].sum()
            self.m[k] = (self.phi[:, k] @ X) / Nₖ if Nₖ > 0 else X[k]
            # Initial variance
            self.s_sq[k] = self.s_0_sq
    
    def fit(self, X, alpha_0=1.0, m_0=None, s_0_sq=10.0, sigma_sq=1.0):
        """
        Fit GMM using variational inference.
        
        Parameters:
        ----------
        X : ndarray (n × d)
            Data matrix
        alpha_0 : float or array
            Dirichlet prior concentration
        m_0 : ndarray (d,) or None
            Prior mean (if None, use data mean)
        s_0_sq : float
            Prior variance for component means
        sigma_sq : float
            Fixed observation noise variance
        """
        n, d = X.shape
        
        # Set hyperparameters
        self.alpha_0 = np.ones(self.K) * alpha_0 if np.isscalar(alpha_0) else alpha_0
        self.m_0 = np.mean(X, axis=0) if m_0 is None else m_0
        self.s_0_sq = s_0_sq
        self.sigma_sq = sigma_sq
        
        print(f"\nInitializing GMM-VI with K={self.K} components...")
        print(f"Data: n={n}, d={d}")
        print(f"Hyperparameters: α₀={alpha_0}, s₀²={s_0_sq}, σ²={sigma_sq}")
        
        # Initialize
        self.initialize(X)
        
        # CAVI loop
        print(f"\nRunning CAVI (max_iter={self.max_iter})...")
        print("-" * 80)
        
        for iteration in range(self.max_iter):
            # Update q(μₖ) for each component
            self.update_means(X)
            
            # Update q(π)
            self.update_weights()
            
            # Update q(Z) - responsibilities
            self.update_responsibilities(X)
            
            # Compute ELBO
            elbo = self.compute_elbo(X)
            self.elbo_history.append(elbo)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d}: ELBO = {elbo:12.4f}")
            
            # Check convergence
            if iteration > 0:
                elbo_change = abs(elbo - self.elbo_history[-2])
                if elbo_change < self.tol:
                    print(f"\nConverged at iteration {iteration}")
                    print(f"Final ELBO: {elbo:.4f}")
                    break
        
        return self
    
    def update_means(self, X):
        """
        Update q(μₖ) for each component k.
        
        Derivation:
        ----------
        q*(μₖ) ∝ exp{E_{q₋μₖ}[log p(X, Z, π, μ)]}
        
        Result: q(μₖ) = N(mₖ, sₖ²I)
        
        where:
            Nₖ = Σᵢ φᵢₖ
            sₖ² = 1 / (1/s₀² + Nₖ/σ²)
            mₖ = sₖ² × (m₀/s₀² + (Σᵢ φᵢₖ xᵢ)/σ²)
        """
        n, d = X.shape
        
        for k in range(self.K):
            # Effective sample size for component k
            N_k = self.phi[:, k].sum()
            
            if N_k < 1e-10:  # Empty component
                self.m[k] = self.m_0
                self.s_sq[k] = self.s_0_sq
                continue
            
            # Weighted sum of data points
            x_k_bar = (self.phi[:, k] @ X) / N_k
            
            # Posterior variance
            precision_posterior = 1 / self.s_0_sq + N_k / self.sigma_sq
            self.s_sq[k] = 1 / precision_posterior
            
            # Posterior mean
            self.m[k] = self.s_sq[k] * (self.m_0 / self.s_0_sq + N_k * x_k_bar / self.sigma_sq)
    
    def update_weights(self):
        """
        Update q(π).
        
        Derivation:
        ----------
        q*(π) ∝ exp{E_{q₋π}[log p(Z, π)]}
        
        Result: q(π) = Dirichlet(α*)
        
        where:
            αₖ* = α₀ₖ + Σᵢ φᵢₖ
        """
        self.alpha = self.alpha_0 + self.phi.sum(axis=0)
    
    def update_responsibilities(self, X):
        """
        Update q(Z) - the responsibilities φᵢₖ = q(z_i = k).
        
        Derivation:
        ----------
        q*(z_i = k) ∝ exp{E_{q₋z}[log p(x_i, z_i = k | θ)]}
        
        Result:
            log φᵢₖ = E[log πₖ] + E[log p(x_i | μₖ)] + const
        
        where:
            E[log πₖ] = ψ(αₖ) - ψ(Σⱼ αⱼ)
            E[log p(x_i | μₖ)] = -d/2 log(2πσ²) - ||x_i - mₖ||²/(2σ²) - d·sₖ²/(2σ²)
        """
        n, d = X.shape
        
        # Compute log responsibilities
        log_phi = np.zeros((n, self.K))
        
        # E[log π_k]
        E_log_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        
        for k in range(self.K):
            # E[log p(x_i | μ_k)]
            diff = X - self.m[k]
            sq_dist = np.sum(diff**2, axis=1)
            
            E_log_lik = -0.5 * d * np.log(2 * np.pi * self.sigma_sq)
            E_log_lik -= sq_dist / (2 * self.sigma_sq)
            E_log_lik -= d * self.s_sq[k] / (2 * self.sigma_sq)
            
            log_phi[:, k] = E_log_pi[k] + E_log_lik
        
        # Normalize (log-sum-exp trick for numerical stability)
        log_phi_norm = logsumexp(log_phi, axis=1, keepdims=True)
        self.phi = np.exp(log_phi - log_phi_norm)
    
    def compute_elbo(self, X):
        """
        Compute the Evidence Lower Bound.
        
        ELBO = E_q[log p(X, Z, θ)] - E_q[log q(Z, θ)]
        
        Components:
        ----------
        1. E_q[log p(X | Z, μ)]
        2. E_q[log p(Z | π)]
        3. E_q[log p(π)]
        4. E_q[log p(μ)]
        5. -E_q[log q(Z)]
        6. -E_q[log q(π)]
        7. -E_q[log q(μ)]
        """
        n, d = X.shape
        elbo = 0.0
        
        # 1. E_q[log p(X | Z, μ)]
        for k in range(self.K):
            diff = X - self.m[k]
            sq_dist = np.sum(diff**2, axis=1)
            
            log_lik = -0.5 * d * np.log(2 * np.pi * self.sigma_sq)
            log_lik -= sq_dist / (2 * self.sigma_sq)
            log_lik -= d * self.s_sq[k] / (2 * self.sigma_sq)
            
            elbo += np.sum(self.phi[:, k] * log_lik)
        
        # 2. E_q[log p(Z | π)]
        E_log_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        elbo += np.sum(self.phi * E_log_pi)
        
        # 3. E_q[log p(π)]
        elbo += gammaln(self.alpha_0.sum()) - np.sum(gammaln(self.alpha_0))
        elbo += np.sum((self.alpha_0 - 1) * E_log_pi)
        
        # 4. E_q[log p(μ)]
        for k in range(self.K):
            diff = self.m[k] - self.m_0
            elbo += -0.5 * d * np.log(2 * np.pi * self.s_0_sq)
            elbo += -np.sum(diff**2) / (2 * self.s_0_sq)
            elbo += -d * self.s_sq[k] / (2 * self.s_0_sq)
        
        # 5. -E_q[log q(Z)]
        elbo -= np.sum(self.phi * np.log(self.phi + 1e-10))
        
        # 6. -E_q[log q(π)]
        elbo -= gammaln(self.alpha.sum()) - np.sum(gammaln(self.alpha))
        elbo -= np.sum((self.alpha - 1) * E_log_pi)
        
        # 7. -E_q[log q(μ)]
        for k in range(self.K):
            elbo += 0.5 * d * (1 + np.log(2 * np.pi * self.s_sq[k]))
        
        return elbo
    
    def predict(self, X):
        """Predict cluster assignments (hard assignments)."""
        self.update_responsibilities(X)
        return np.argmax(self.phi, axis=1)
    
    def predict_proba(self, X):
        """Predict cluster probabilities (soft assignments)."""
        self.update_responsibilities(X)
        return self.phi


def demonstrate_gmm_vi():
    """
    Demonstrate GMM-VI on synthetic data.
    """
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION: GMM with Variational Inference")
    print("=" * 80)
    
    # Generate synthetic mixture data
    np.random.seed(42)
    n_samples = 300
    
    # True parameters
    true_means = np.array([[-3, -3], [0, 4], [3, -2]])
    true_covs = [np.eye(2) * 0.5, np.eye(2) * 0.8, np.eye(2) * 0.6]
    true_weights = np.array([0.3, 0.4, 0.3])
    
    # Generate data
    X = []
    true_labels = []
    for k in range(3):
        n_k = int(n_samples * true_weights[k])
        X_k = np.random.multivariate_normal(true_means[k], true_covs[k], n_k)
        X.append(X_k)
        true_labels.extend([k] * n_k)
    
    X = np.vstack(X)
    true_labels = np.array(true_labels)
    
    print(f"\nGenerated synthetic mixture data:")
    print(f"  Samples: {len(X)}")
    print(f"  True components: {len(true_means)}")
    print(f"  True weights: {true_weights}")
    
    # Fit GMM-VI
    gmm = GaussianMixtureVI(n_components=3, max_iter=100, tol=1e-4)
    gmm.fit(X, alpha_0=1.0, s_0_sq=10.0, sigma_sq=0.8)
    
    # Get predictions
    pred_labels = gmm.predict(X)
    pred_proba = gmm.predict_proba(X)
    
    # Visualize results
    visualize_gmm_results(X, true_labels, pred_labels, pred_proba, 
                         gmm, true_means, true_weights)
    
    return gmm, X, true_labels


def visualize_gmm_results(X, true_labels, pred_labels, pred_proba,
                          gmm, true_means, true_weights):
    """
    Visualize GMM-VI results.
    """
    
    print("\n[Generating GMM-VI results visualization...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: True clusters
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', 
                        alpha=0.6, s=30)
    ax.scatter(true_means[:, 0], true_means[:, 1], c='red', marker='X', 
              s=200, edgecolors='black', linewidths=2, label='True means')
    ax.set_xlabel('x₁', fontsize=11)
    ax.set_ylabel('x₂', fontsize=11)
    ax.set_title('(a) True Cluster Labels', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax)
    
    # Plot 2: Predicted clusters
    ax = axes[0, 1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis', 
                        alpha=0.6, s=30)
    ax.scatter(gmm.m[:, 0], gmm.m[:, 1], c='red', marker='X', 
              s=200, edgecolors='black', linewidths=2, label='Learned means')
    ax.set_xlabel('x₁', fontsize=11)
    ax.set_ylabel('x₂', fontsize=11)
    ax.set_title('(b) Predicted Clusters (Hard)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax)
    
    # Plot 3: Soft assignments (entropy)
    ax = axes[0, 2]
    entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-10), axis=1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=entropy, cmap='RdYlGn_r', 
                        alpha=0.6, s=30)
    ax.scatter(gmm.m[:, 0], gmm.m[:, 1], c='black', marker='X', 
              s=200, edgecolors='white', linewidths=2)
    ax.set_xlabel('x₁', fontsize=11)
    ax.set_ylabel('x₂', fontsize=11)
    ax.set_title('(c) Assignment Uncertainty\n(Entropy)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Entropy', fontsize=10)
    
    # Plot 4: ELBO convergence
    ax = axes[1, 0]
    ax.plot(gmm.elbo_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(d) ELBO Convergence', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Component responsibilities
    ax = axes[1, 1]
    N_k = gmm.phi.sum(axis=0)
    E_pi = gmm.alpha / gmm.alpha.sum()
    
    x_pos = np.arange(gmm.K)
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, true_weights, width, label='True weights', 
                   color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, E_pi, width, label='Learned weights', 
                   color='blue', alpha=0.7)
    
    ax.set_xlabel('Component', fontsize=11)
    ax.set_ylabel('Weight', fontsize=11)
    ax.set_title('(e) Mixture Weights', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k+1}' for k in range(gmm.K)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Comparison table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Compute metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    # Create summary text
    summary_text = f"""
GMM-VI Results Summary
{'=' * 35}

Model Configuration:
  • Components: {gmm.K}
  • Iterations: {len(gmm.elbo_history)}
  • Final ELBO: {gmm.elbo_history[-1]:.2f}

Clustering Performance:
  • Adjusted Rand Index: {ari:.3f}
  • Normalized MI: {nmi:.3f}

Component Statistics:
  • Total samples: {len(X)}
  • Avg. assignment entropy: {entropy.mean():.3f}

Learned Parameters:
  • Mixture weights: {E_pi}
  • Mean vectors: 
    {gmm.m}
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/tmp/variational_inference/intermediate/figures/11_gmm_vi_results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[Figure saved: 11_gmm_vi_results.png]")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    
    print("=" * 80)
    print("VARIATIONAL INFERENCE - MODULE 4")
    print("Gaussian Mixture Models with VI")
    print("=" * 80)
    print("\nAuthor: Prof. Sungchul")
    print("Institution: Yonsei University")
    print("Email: sungchulyonsei@gmail.com")
    print("=" * 80)
    
    # Create directory
    import os
    os.makedirs('/tmp/variational_inference/intermediate/figures', exist_ok=True)
    
    # Run demonstration
    gmm_model_introduction()
    demonstrate_gmm_vi()
    
    print("\n" + "=" * 80)
    print("MODULE COMPLETE!")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  • 11_gmm_vi_results.png")
    print("\nNext: Module 05 - Exponential Family VI")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

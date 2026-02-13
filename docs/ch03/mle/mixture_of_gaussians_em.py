#!/usr/bin/env python3
"""
================================================================================
MIXTURE OF GAUSSIANS MLE - Expectation-Maximization Algorithm
================================================================================

DIFFICULTY: ⭐⭐⭐ Advanced (Level 3)

LEARNING OBJECTIVES:
- Understand latent variable models
- Learn the Expectation-Maximization (EM) algorithm
- Implement soft clustering
- See how MLE handles missing/latent data

PROBLEM: We observe data from K different Gaussian distributions, but don't
know which point came from which distribution. Estimate all parameters!

MODEL:
- K Gaussian components with means μₖ, covariances Σₖ
- Mixing weights πₖ (probability of component k)
- Latent variable zᵢ indicates which component generated xᵢ

LIKELIHOOD (with latent variables):
P(x, z | θ) = ∏ [πₖ N(xᵢ | μₖ, Σₖ)]^{z_ik}

But we don't observe z! So we use EM algorithm:

E-STEP: Compute posterior probabilities (responsibilities)
γᵢₖ = P(z_ik = 1 | xᵢ, θ)

M-STEP: Update parameters to maximize expected complete-data log-likelihood
πₖ = (1/N) Σ γᵢₖ
μₖ = (Σ γᵢₖ xᵢ) / (Σ γᵢₖ)
Σₖ = (Σ γᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ) / (Σ γᵢₖ)

APPLICATIONS:
- Clustering (soft k-means)
- Anomaly detection
- Image segmentation  
- Speech recognition
- Topic modeling

AUTHOR: PyTorch MLE Tutorial
DATE: 2025
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Tuple, List


def generate_gmm_data(n_samples: int = 300, n_components: int = 3, seed: int = 42):
    """Generate data from a Gaussian Mixture Model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # True parameters
    true_means = [
        torch.tensor([-3.0, -3.0]),
        torch.tensor([3.0, 3.0]),
        torch.tensor([0.0, 5.0])
    ]
    
    true_covs = [
        torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
        torch.tensor([[1.5, -0.3], [-0.3, 1.5]]),
        torch.tensor([[0.8, 0.0], [0.0, 0.8]])
    ]
    
    true_weights = torch.tensor([0.3, 0.4, 0.3])
    
    # Generate data
    X_list = []
    labels_list = []
    
    for k in range(n_components):
        n_k = int(n_samples * true_weights[k])
        
        # Sample from multivariate Gaussian
        mean = true_means[k].numpy()
        cov = true_covs[k].numpy()
        X_k = np.random.multivariate_normal(mean, cov, n_k)
        
        X_list.append(torch.tensor(X_k, dtype=torch.float32))
        labels_list.append(torch.full((n_k,), k, dtype=torch.long))
    
    X = torch.cat(X_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(len(X))
    X, labels = X[perm], labels[perm]
    
    return X, labels, true_means, true_covs, true_weights


class GaussianMixture:
    """Gaussian Mixture Model with EM algorithm"""
    
    def __init__(self, n_components=3, n_iterations=50):
        self.n_components = n_components
        self.n_iterations = n_iterations
        
        self.means = None
        self.covs = None
        self.weights = None
        self.history = []
    
    def initialize_parameters(self, X):
        """Initialize parameters using k-means++"""
        n_samples, n_features = X.shape
        
        # Initialize means using k-means++ strategy
        indices = torch.randperm(n_samples)[:self.n_components]
        self.means = X[indices].clone()
        
        # Initialize covariances as identity matrices
        self.covs = [torch.eye(n_features) for _ in range(self.n_components)]
        
        # Initialize weights uniformly
        self.weights = torch.ones(self.n_components) / self.n_components
    
    def gaussian_pdf(self, X, mean, cov):
        """Compute multivariate Gaussian PDF"""
        n_features = X.shape[1]
        
        # Add small value to diagonal for numerical stability
        cov = cov + torch.eye(n_features) * 1e-6
        
        # Compute probability
        diff = X - mean
        cov_inv = torch.inverse(cov)
        
        exponent = -0.5 * torch.sum(diff @ cov_inv * diff, dim=1)
        normalization = 1.0 / torch.sqrt((2 * np.pi) ** n_features * torch.det(cov))
        
        return normalization * torch.exp(exponent)
    
    def e_step(self, X):
        """
        E-step: Compute responsibilities (posterior probabilities).
        
        γᵢₖ = πₖ N(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ | μⱼ, Σⱼ)
        """
        n_samples = X.shape[0]
        responsibilities = torch.zeros(n_samples, self.n_components)
        
        # Compute weighted probabilities for each component
        for k in range(self.n_components):
            prob = self.gaussian_pdf(X, self.means[k], self.covs[k])
            responsibilities[:, k] = self.weights[k] * prob
        
        # Normalize to get posterior probabilities
        responsibilities = responsibilities / (responsibilities.sum(dim=1, keepdim=True) + 1e-10)
        
        return responsibilities
    
    def m_step(self, X, responsibilities):
        """
        M-step: Update parameters to maximize expected log-likelihood.
        """
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        N_k = responsibilities.sum(dim=0)
        
        # Update weights
        self.weights = N_k / n_samples
        
        # Update means
        for k in range(self.n_components):
            self.means[k] = (responsibilities[:, k:k+1] * X).sum(dim=0) / N_k[k]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covs[k] = (weighted_diff.T @ diff) / N_k[k]
            
            # Ensure positive definite
            self.covs[k] = self.covs[k] + torch.eye(n_features) * 1e-6
    
    def compute_log_likelihood(self, X):
        """Compute log-likelihood of data"""
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            # Mixture probability for this point
            prob_sum = 0.0
            for k in range(self.n_components):
                prob = self.gaussian_pdf(X[i:i+1], self.means[k], self.covs[k])
                prob_sum += self.weights[k] * prob
            
            log_likelihood += torch.log(prob_sum + 1e-10)
        
        return log_likelihood.item()
    
    def fit(self, X):
        """Fit GMM using EM algorithm"""
        self.initialize_parameters(X)
        
        print(f"   Running EM algorithm for {self.n_iterations} iterations...")
        
        for iteration in range(self.n_iterations):
            # E-step: compute responsibilities
            responsibilities = self.e_step(X)
            
            # M-step: update parameters
            self.m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_lik = self.compute_log_likelihood(X)
            self.history.append(log_lik)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration+1}/{self.n_iterations}, Log-Likelihood: {log_lik:.2f}")
        
        return self
    
    def predict(self, X):
        """Predict cluster assignments (hard clustering)"""
        responsibilities = self.e_step(X)
        return torch.argmax(responsibilities, dim=1)
    
    def predict_proba(self, X):
        """Predict cluster probabilities (soft clustering)"""
        return self.e_step(X)


def plot_gmm_results(X, labels, gmm, true_means):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # ================================================================
    # Plot 1: True Clustering (if labels available)
    # ================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    colors = ['red', 'blue', 'green']
    for k in range(gmm.n_components):
        mask = labels == k
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'True Cluster {k}')
    
    # Plot true means
    for k, mean in enumerate(true_means):
        ax1.scatter(mean[0], mean[1], c=colors[k], marker='*', s=500, 
                   edgecolors='black', linewidths=2, label=f'True μ_{k}')
    
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('True Clusters', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 2: Predicted Clustering (EM)
    # ================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    predicted_labels = gmm.predict(X)
    
    for k in range(gmm.n_components):
        mask = predicted_labels == k
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'Cluster {k}')
    
    # Plot estimated means and covariances
    for k in range(gmm.n_components):
        mean = gmm.means[k]
        cov = gmm.covs[k]
        
        # Plot mean
        ax2.scatter(mean[0], mean[1], c=colors[k], marker='X', s=500,
                   edgecolors='black', linewidths=2, label=f'Est μ_{k}')
        
        # Plot covariance ellipse
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item()))
        width, height = 2 * torch.sqrt(eigenvalues)
        
        ellipse = Ellipse(mean.numpy(), width.item(), height.item(), angle=angle,
                         facecolor='none', edgecolor=colors[k], linewidth=2, linestyle='--')
        ax2.add_patch(ellipse)
    
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title('EM Clustering Results', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 3: Soft Clustering (Responsibilities)
    # ================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    responsibilities = gmm.predict_proba(X)
    
    # Create RGB colors based on responsibilities
    rgb_colors = torch.zeros(len(X), 3)
    for k in range(min(3, gmm.n_components)):
        if k == 0:
            rgb_colors[:, 0] = responsibilities[:, k]  # Red
        elif k == 1:
            rgb_colors[:, 2] = responsibilities[:, k]  # Blue
        elif k == 2:
            rgb_colors[:, 1] = responsibilities[:, k]  # Green
    
    ax3.scatter(X[:, 0], X[:, 1], c=rgb_colors.numpy(), s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    for k in range(gmm.n_components):
        mean = gmm.means[k]
        ax3.scatter(mean[0], mean[1], c=colors[k], marker='X', s=500,
                   edgecolors='black', linewidths=2)
    
    ax3.set_xlabel('Feature 1', fontsize=12)
    ax3.set_ylabel('Feature 2', fontsize=12)
    ax3.set_title('Soft Clustering (Color = Responsibility)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 4: Log-Likelihood Convergence
    # ================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    ax4.plot(gmm.history, 'b-', linewidth=2)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Log-Likelihood', fontsize=12)
    ax4.set_title('EM Convergence', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 5: Component Weights
    # ================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    components = [f'Comp {k}' for k in range(gmm.n_components)]
    bars = ax5.bar(components, gmm.weights.numpy(), color=colors[:gmm.n_components], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax5.set_ylabel('Weight (π_k)', fontsize=12)
    ax5.set_title('Mixture Weights', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1.0])
    
    # ================================================================
    # Plot 6: Confusion Matrix
    # ================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Compute metrics
    ari = adjusted_rand_score(labels.numpy(), predicted_labels.numpy())
    nmi = normalized_mutual_info_score(labels.numpy(), predicted_labels.numpy())
    
    # Summary table
    table_data = [
        ['Metric', 'Value'],
        ['Components (K)', f'{gmm.n_components}'],
        ['Data points (N)', f'{len(X)}'],
        ['Final Log-Likelihood', f'{gmm.history[-1]:.2f}'],
        ['Adjusted Rand Index', f'{ari:.3f}'],
        ['Normalized Mutual Info', f'{nmi:.3f}'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('mixture_gaussians_em_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'mixture_gaussians_em_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("MIXTURE OF GAUSSIANS - EM Algorithm")
    print("=" * 80)
    
    # Generate data
    print("\n🎲 Generating data from Gaussian Mixture...")
    X, labels, true_means, true_covs, true_weights = generate_gmm_data(n_samples=300, n_components=3)
    
    print(f"   • Generated {len(X)} data points")
    print(f"   • Number of true components: {len(true_means)}")
    print(f"   • True mixture weights: {true_weights.numpy()}")
    
    # Fit GMM
    print("\n🔄 Fitting Gaussian Mixture Model...")
    print("-" * 80)
    
    gmm = GaussianMixture(n_components=3, n_iterations=50)
    gmm.fit(X)
    
    # Results
    print("\n📊 Results:")
    print("-" * 80)
    print(f"   Final log-likelihood: {gmm.history[-1]:.2f}")
    print(f"   Estimated mixture weights: {gmm.weights.numpy()}")
    print("\n   Estimated means:")
    for k in range(gmm.n_components):
        print(f"      Component {k}: {gmm.means[k].numpy()}")
    
    # Clustering evaluation
    predicted_labels = gmm.predict(X)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels.numpy(), predicted_labels.numpy())
    print(f"\n   Clustering quality (ARI): {ari:.3f}")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    plot_gmm_results(X, labels, gmm, true_means)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. EM algorithm handles latent (hidden) variables")
    print("   2. E-step: compute responsibilities (soft assignments)")
    print("   3. M-step: update parameters using weighted MLE")
    print("   4. Converges to local maximum (not necessarily global)")
    print("   5. Foundation for many ML algorithms (k-means, HMM, etc.)")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. MEDIUM: Automatic model selection
   - Try different numbers of components (K)
   - Use BIC or AIC for model selection
   - Plot BIC vs K

2. MEDIUM: Different covariance structures
   - Spherical: Σₖ = σₖ²I
   - Diagonal: Σₖ = diag(σₖ₁², ..., σₖₚ²)
   - Full: Σₖ (current implementation)
   - Compare performance and speed

3. CHALLENGING: Initialization strategies
   - Random initialization
   - K-means++ initialization
   - Multiple random restarts
   - Compare convergence

4. CHALLENGING: Bayesian GMM
   - Add Dirichlet prior on mixture weights
   - Add Gaussian-Wishart prior on means and covariances
   - Implement variational inference

5. CHALLENGING: Applications
   - Image segmentation using GMM
   - Anomaly detection (low probability points)
   - Density estimation and sampling
"""


if __name__ == "__main__":
    main()

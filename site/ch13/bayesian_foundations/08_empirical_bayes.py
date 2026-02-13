"""
Bayesian Inference - Module 8: Empirical Bayes
Level: Advanced
Topics: Empirical Bayes methods, hyperparameter estimation from data

Empirical Bayes estimates hyperparameters from the data itself, providing
a practical middle ground between fully Bayesian and frequentist approaches.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

"""
EMPIRICAL BAYES METHODOLOGY:

Standard Bayes: Prior parameters are fixed before seeing data
Empirical Bayes: Prior parameters are estimated from the data

Procedure:
1. Estimate hyperparameters from marginal distribution of data
2. Use these estimates as "prior" for individual inferences
3. Proceed with standard Bayesian inference

FAMOUS EXAMPLE: James-Stein Estimator
Shows that shrinkage estimators dominate MLE for 3+ parameters
"""

def baseball_empirical_bayes_demo():
    """
    Classic baseball batting average example of empirical Bayes.
    """
    print("="*70)
    print("EMPIRICAL BAYES: Baseball Batting Averages")
    print("="*70)
    
    # Simulate data: early season batting averages
    np.random.seed(42)
    n_players = 20
    true_abilities = np.random.beta(80, 220, n_players)  # True batting averages
    at_bats = np.random.randint(20, 100, n_players)
    hits = np.array([np.random.binomial(ab, ta) for ab, ta in zip(at_bats, true_abilities)])
    
    observed_avg = hits / at_bats
    
    print(f"\nNumber of players: {n_players}")
    print(f"At-bats range: {at_bats.min()}-{at_bats.max()}")
    
    # Empirical Bayes: Estimate Beta prior from observed averages
    # Method of moments
    mean_obs = np.mean(observed_avg)
    var_obs = np.var(observed_avg)
    
    # Beta distribution: mean = α/(α+β), var = αβ/[(α+β)²(α+β+1)]
    # Solve for α, β
    alpha_eb = mean_obs * (mean_obs * (1 - mean_obs) / var_obs - 1)
    beta_eb = (1 - mean_obs) * (mean_obs * (1 - mean_obs) / var_obs - 1)
    
    print(f"\nEmpirical Bayes prior: Beta({alpha_eb:.2f}, {beta_eb:.2f})")
    print(f"  Prior mean: {alpha_eb/(alpha_eb+beta_eb):.4f}")
    
    # Apply shrinkage
    eb_estimates = (hits + alpha_eb) / (at_bats + alpha_eb + beta_eb)
    
    # Compare with MLE
    mse_mle = np.mean((observed_avg - true_abilities)**2)
    mse_eb = np.mean((eb_estimates - true_abilities)**2)
    
    print(f"\nMean Squared Error:")
    print(f"  MLE:            {mse_mle:.6f}")
    print(f"  Empirical Bayes: {mse_eb:.6f}")
    print(f"  Improvement:     {(1 - mse_eb/mse_mle)*100:.1f}%")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(observed_avg, true_abilities, s=at_bats, alpha=0.6, label='MLE')
    plt.scatter(eb_estimates, true_abilities, s=at_bats, alpha=0.6, label='Empirical Bayes')
    plt.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5)
    plt.xlabel('Estimate', fontsize=12)
    plt.ylabel('True Ability', fontsize=12)
    plt.title('Estimates vs True Ability', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for i in range(n_players):
        plt.plot([observed_avg[i], eb_estimates[i]], [i, i], 'r-', linewidth=1.5, alpha=0.7)
        plt.plot(observed_avg[i], i, 'bo', markersize=6)
        plt.plot(eb_estimates[i], i, 'ro', markersize=6)
    plt.axvline(mean_obs, color='green', linestyle=':', linewidth=2, label='Grand mean')
    plt.xlabel('Batting Average', fontsize=12)
    plt.ylabel('Player', fontsize=12)
    plt.title('Shrinkage Toward Prior Mean', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('empirical_bayes_baseball.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 8: EMPIRICAL BAYES")
    print("="*70)
    
    baseball_empirical_bayes_demo()
    
    print("\n" + "="*70)
    print("MODULE 8 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Empirical Bayes estimates prior from data")
    print("2. Provides automatic shrinkage without full Bayesian machinery")
    print("3. Often outperforms MLE, especially with many parameters")
    print("4. Related to James-Stein estimator")
    print("\nNext: Module 9 - Bayesian Linear Regression")
    print("="*70)

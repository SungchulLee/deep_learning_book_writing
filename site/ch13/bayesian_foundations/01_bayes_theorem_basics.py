"""
Bayesian Inference - Module 1: Bayes' Theorem Basics
Level: Beginner
Topics: Fundamental Bayes' theorem, discrete examples, medical testing, coin flips

This module introduces the foundational concepts of Bayesian inference through
simple discrete examples. Students will learn to apply Bayes' theorem and 
understand the relationship between prior beliefs, likelihoods, and posterior
distributions.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# SECTION 1: BAYES' THEOREM - MATHEMATICAL FOUNDATION
# ============================================================================

"""
BAYES' THEOREM DERIVATION:

Starting from the definition of conditional probability:
P(A|B) = P(A ∩ B) / P(B)
P(B|A) = P(A ∩ B) / P(A)

From the first equation: P(A ∩ B) = P(A|B)P(B)
Substituting into the second: P(B|A) = P(A|B)P(B) / P(A)

In Bayesian terminology:
- A represents the hypothesis or parameter (θ)
- B represents the observed data (D)

Therefore, Bayes' theorem is:

P(θ|D) = P(D|θ) P(θ) / P(D)

Where:
- P(θ|D) is the POSTERIOR: probability of hypothesis given data
- P(D|θ) is the LIKELIHOOD: probability of data given hypothesis
- P(θ) is the PRIOR: initial probability of hypothesis
- P(D) is the EVIDENCE or MARGINAL LIKELIHOOD: total probability of data

The evidence can be computed using the law of total probability:
P(D) = Σ P(D|θᵢ)P(θᵢ) for all possible θᵢ
"""

def bayes_theorem_discrete(prior, likelihood, normalize=True):
    """
    Apply Bayes' theorem for discrete distributions.
    
    Parameters:
    -----------
    prior : array-like
        Prior probabilities for each hypothesis
    likelihood : array-like
        Likelihood of data under each hypothesis
    normalize : bool
        If True, normalize to ensure probabilities sum to 1
    
    Returns:
    --------
    posterior : numpy array
        Posterior probabilities for each hypothesis
    
    Mathematical formula:
    P(θ|D) = P(D|θ)P(θ) / Σ P(D|θᵢ)P(θᵢ)
    """
    # Convert to numpy arrays for numerical stability
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    
    # Compute unnormalized posterior (numerator of Bayes' theorem)
    unnormalized_posterior = likelihood * prior
    
    if normalize:
        # Compute evidence (marginal likelihood) - denominator of Bayes' theorem
        evidence = np.sum(unnormalized_posterior)
        
        # Avoid division by zero
        if evidence == 0:
            raise ValueError("Evidence is zero - cannot normalize posterior")
        
        # Compute normalized posterior
        posterior = unnormalized_posterior / evidence
    else:
        posterior = unnormalized_posterior
    
    return posterior

# ============================================================================
# EXAMPLE 1: MEDICAL TESTING - THE CLASSIC BAYESIAN PROBLEM
# ============================================================================

"""
MEDICAL TEST SCENARIO:

A disease affects 1% of the population (base rate).
A medical test for this disease has:
- Sensitivity (True Positive Rate): 95% - P(Test+|Disease+)
- Specificity (True Negative Rate): 90% - P(Test-|Disease-)

Question: If a person tests positive, what is the probability they have the disease?

This is a classic example demonstrating the importance of base rates and how
intuition can be misleading. Many people guess the answer is ~95%, but the
correct answer is much lower due to the low base rate.
"""

def medical_test_example():
    """
    Demonstrate Bayesian inference in medical testing.
    
    This example illustrates the counterintuitive result that even with a
    highly accurate test, a positive result doesn't necessarily mean high
    probability of disease when the base rate is low.
    """
    print("="*70)
    print("EXAMPLE 1: MEDICAL TESTING WITH BAYES' THEOREM")
    print("="*70)
    
    # Define the problem parameters
    base_rate = 0.01        # P(Disease) - prevalence in population
    sensitivity = 0.95      # P(Test+|Disease+) - true positive rate
    specificity = 0.90      # P(Test-|Disease-) - true negative rate
    
    # Derived quantities
    false_positive_rate = 1 - specificity  # P(Test+|Disease-)
    
    print(f"\nProblem Setup:")
    print(f"  Disease prevalence (base rate): {base_rate*100:.1f}%")
    print(f"  Test sensitivity: {sensitivity*100:.1f}%")
    print(f"  Test specificity: {specificity*100:.1f}%")
    print(f"  False positive rate: {false_positive_rate*100:.1f}%")
    
    # Define hypotheses: [Has Disease, No Disease]
    hypotheses = ['Has Disease', 'No Disease']
    
    # Prior probabilities
    prior = np.array([base_rate, 1 - base_rate])
    print(f"\nPrior Probabilities:")
    for hyp, p in zip(hypotheses, prior):
        print(f"  P({hyp}) = {p:.4f} ({p*100:.2f}%)")
    
    # Likelihood of positive test result under each hypothesis
    # P(Test+|Has Disease) = sensitivity
    # P(Test+|No Disease) = false_positive_rate
    likelihood = np.array([sensitivity, false_positive_rate])
    print(f"\nLikelihoods (given positive test):")
    for hyp, l in zip(hypotheses, likelihood):
        print(f"  P(Test+|{hyp}) = {l:.4f} ({l*100:.2f}%)")
    
    # Apply Bayes' theorem
    posterior = bayes_theorem_discrete(prior, likelihood)
    print(f"\nPosterior Probabilities (after positive test):")
    for hyp, p in zip(hypotheses, posterior):
        print(f"  P({hyp}|Test+) = {p:.4f} ({p*100:.2f}%)")
    
    # Key insight
    print(f"\n{'='*70}")
    print(f"KEY INSIGHT:")
    print(f"Even though the test is 95% sensitive, a positive test only gives")
    print(f"{posterior[0]*100:.1f}% probability of having the disease!")
    print(f"This is because the disease is rare (1% base rate).")
    print(f"{'='*70}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Prior distribution
    axes[0].bar(hypotheses, prior, color=['red', 'green'], alpha=0.7)
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Prior Distribution\n(Before Test)')
    axes[0].set_ylim([0, 1])
    for i, (hyp, p) in enumerate(zip(hypotheses, prior)):
        axes[0].text(i, p + 0.02, f'{p:.4f}', ha='center', fontsize=10)
    
    # Plot 2: Likelihood
    axes[1].bar(hypotheses, likelihood, color=['blue', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Likelihood\n(Positive Test | Hypothesis)')
    axes[1].set_ylim([0, 1])
    for i, (hyp, l) in enumerate(zip(hypotheses, likelihood)):
        axes[1].text(i, l + 0.02, f'{l:.4f}', ha='center', fontsize=10)
    
    # Plot 3: Posterior distribution
    axes[2].bar(hypotheses, posterior, color=['red', 'green'], alpha=0.7)
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Posterior Distribution\n(After Positive Test)')
    axes[2].set_ylim([0, 1])
    for i, (hyp, p) in enumerate(zip(hypotheses, posterior)):
        axes[2].text(i, p + 0.02, f'{p:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('medical_test_bayesian_inference.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return posterior

# ============================================================================
# EXAMPLE 2: COIN FLIP INFERENCE - DISCRETE PARAMETER SPACE
# ============================================================================

"""
COIN FLIP PROBLEM:

We have three coins:
- Coin 1: Fair coin (P(Heads) = 0.5)
- Coin 2: Biased toward heads (P(Heads) = 0.7)
- Coin 3: Biased toward tails (P(Heads) = 0.3)

We randomly select one coin (each equally likely) and flip it N times.
Question: Given the observed heads and tails, which coin was likely chosen?

This example demonstrates:
1. How data updates our beliefs
2. Sequential updating (flipping multiple times)
3. The effect of sample size on posterior certainty
"""

def coin_flip_inference(n_heads, n_tails, visualize=True):
    """
    Infer which of three coins was flipped based on observed data.
    
    Parameters:
    -----------
    n_heads : int
        Number of heads observed
    n_tails : int
        Number of tails observed
    visualize : bool
        Whether to create visualization
    
    Returns:
    --------
    posterior : numpy array
        Posterior probabilities for each coin
    """
    print("\n" + "="*70)
    print(f"EXAMPLE 2: COIN FLIP INFERENCE")
    print("="*70)
    
    # Define the three coins and their probabilities of heads
    coin_names = ['Coin 1 (Fair)', 'Coin 2 (Heads-biased)', 'Coin 3 (Tails-biased)']
    p_heads = np.array([0.5, 0.7, 0.3])  # Probability of heads for each coin
    
    # Prior: Each coin equally likely
    prior = np.array([1/3, 1/3, 1/3])
    print(f"\nObserved data: {n_heads} heads, {n_tails} tails")
    print(f"Total flips: {n_heads + n_tails}")
    
    print(f"\nPrior (before seeing data):")
    for name, p in zip(coin_names, prior):
        print(f"  P({name}) = {p:.4f}")
    
    # Compute likelihood for each coin using binomial distribution
    # P(D|θ) = P(n_heads heads in n_total flips | p_heads)
    # This follows the binomial distribution:
    # P(k successes in n trials) = C(n,k) * p^k * (1-p)^(n-k)
    
    n_total = n_heads + n_tails
    likelihood = stats.binom.pmf(n_heads, n_total, p_heads)
    
    print(f"\nLikelihood (probability of observed data given each coin):")
    for name, l in zip(coin_names, likelihood):
        print(f"  P(Data|{name}) = {l:.6f}")
    
    # Apply Bayes' theorem
    posterior = bayes_theorem_discrete(prior, likelihood)
    
    print(f"\nPosterior (after seeing data):")
    for name, p in zip(coin_names, posterior):
        print(f"  P({name}|Data) = {p:.4f} ({p*100:.1f}%)")
    
    # Identify most likely coin
    most_likely_idx = np.argmax(posterior)
    print(f"\nMost likely coin: {coin_names[most_likely_idx]}")
    print(f"Posterior probability: {posterior[most_likely_idx]*100:.1f}%")
    
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Prior
        axes[0].bar(range(3), prior, color=colors, alpha=0.7)
        axes[0].set_xticks(range(3))
        axes[0].set_xticklabels(coin_names, rotation=15, ha='right')
        axes[0].set_ylabel('Probability')
        axes[0].set_title('Prior Distribution')
        axes[0].set_ylim([0, 1])
        for i, p in enumerate(prior):
            axes[0].text(i, p + 0.02, f'{p:.3f}', ha='center')
        
        # Plot 2: Likelihood (normalized for visualization)
        likelihood_norm = likelihood / np.sum(likelihood)
        axes[1].bar(range(3), likelihood_norm, color=colors, alpha=0.7)
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(coin_names, rotation=15, ha='right')
        axes[1].set_ylabel('Relative Probability')
        axes[1].set_title(f'Likelihood\n({n_heads}H, {n_tails}T)')
        axes[1].set_ylim([0, 1])
        for i, l in enumerate(likelihood_norm):
            axes[1].text(i, l + 0.02, f'{l:.3f}', ha='center')
        
        # Plot 3: Posterior
        axes[2].bar(range(3), posterior, color=colors, alpha=0.7)
        axes[2].set_xticks(range(3))
        axes[2].set_xticklabels(coin_names, rotation=15, ha='right')
        axes[2].set_ylabel('Probability')
        axes[2].set_title('Posterior Distribution')
        axes[2].set_ylim([0, 1])
        for i, p in enumerate(posterior):
            axes[2].text(i, p + 0.02, f'{p:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'coin_inference_{n_heads}H_{n_tails}T.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return posterior

# ============================================================================
# EXAMPLE 3: SEQUENTIAL BAYESIAN UPDATING
# ============================================================================

"""
SEQUENTIAL UPDATING:

One of the key features of Bayesian inference is that we can update our beliefs
sequentially as new data arrives. The posterior from one update becomes the
prior for the next update.

This is expressed mathematically as:
P(θ|D₁,D₂) = P(D₂|θ)P(θ|D₁) / P(D₂|D₁)

This demonstrates that Bayesian inference is naturally suited for online learning
and streaming data scenarios.
"""

def sequential_coin_flips(flip_sequence, p_heads_coins=[0.5, 0.7, 0.3]):
    """
    Demonstrate sequential Bayesian updating with coin flips.
    
    Parameters:
    -----------
    flip_sequence : list of str
        Sequence of observed flips, e.g., ['H', 'H', 'T', 'H']
    p_heads_coins : list of float
        Probability of heads for each coin hypothesis
    
    Returns:
    --------
    posterior_history : list of numpy arrays
        Posterior distribution after each flip
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: SEQUENTIAL BAYESIAN UPDATING")
    print("="*70)
    
    coin_names = [f'Coin {i+1} (p={p:.1f})' for i, p in enumerate(p_heads_coins)]
    p_heads = np.array(p_heads_coins)
    
    # Initialize with uniform prior
    prior = np.ones(len(p_heads)) / len(p_heads)
    posterior_history = [prior.copy()]
    
    print(f"\nFlip sequence: {' -> '.join(flip_sequence)}")
    print(f"\nInitial prior (uniform):")
    for name, p in zip(coin_names, prior):
        print(f"  {name}: {p:.4f}")
    
    # Process each flip sequentially
    current_posterior = prior.copy()
    
    for flip_num, flip in enumerate(flip_sequence, 1):
        print(f"\n--- After Flip {flip_num}: {flip} ---")
        
        # Likelihood for this flip
        if flip == 'H':
            likelihood = p_heads  # P(Heads|each coin)
        else:  # flip == 'T'
            likelihood = 1 - p_heads  # P(Tails|each coin)
        
        # The previous posterior becomes the new prior
        prior = current_posterior.copy()
        
        # Apply Bayes' theorem
        current_posterior = bayes_theorem_discrete(prior, likelihood)
        posterior_history.append(current_posterior.copy())
        
        # Display results
        print(f"Likelihood of '{flip}':")
        for name, l in zip(coin_names, likelihood):
            print(f"  {name}: {l:.4f}")
        
        print(f"Updated posterior:")
        for name, p in zip(coin_names, current_posterior):
            print(f"  {name}: {p:.4f} ({p*100:.1f}%)")
    
    # Visualization of sequential updating
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_flips = len(flip_sequence)
    x_positions = np.arange(n_flips + 1)
    
    # Plot posterior probability for each coin over time
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for coin_idx, name in enumerate(coin_names):
        probabilities = [post[coin_idx] for post in posterior_history]
        ax.plot(x_positions, probabilities, marker='o', label=name, 
                color=colors[coin_idx], linewidth=2, markersize=8)
    
    # Add flip labels
    ax.set_xlabel('Flip Number', fontsize=12)
    ax.set_ylabel('Posterior Probability', fontsize=12)
    ax.set_title('Sequential Bayesian Updating: Coin Flip Inference', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Prior'] + [f'{i}:{f}' for i, f in enumerate(flip_sequence, 1)])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('sequential_updating.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return posterior_history

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 1: BAYES' THEOREM BASICS")
    print("="*70)
    print("\nThis module demonstrates fundamental Bayesian inference through")
    print("three classic examples with discrete parameter spaces.")
    
    # Example 1: Medical testing
    medical_posterior = medical_test_example()
    
    # Example 2: Single coin flip inference
    # Scenario: We flip a coin 10 times and observe 7 heads, 3 tails
    coin_posterior = coin_flip_inference(n_heads=7, n_tails=3)
    
    # Example 3: Sequential updating
    # Scenario: We observe a sequence of flips one by one
    flip_sequence = ['H', 'H', 'H', 'T', 'H', 'T', 'H', 'H']
    posterior_history = sequential_coin_flips(flip_sequence)
    
    print("\n" + "="*70)
    print("MODULE 1 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Bayes' theorem combines prior beliefs with data likelihood")
    print("2. Base rates (priors) matter - even accurate tests can be misleading")
    print("3. Bayesian inference naturally handles sequential data updates")
    print("4. The posterior becomes the prior for the next observation")
    print("\nNext: Module 2 - Continuous Bayesian Inference")
    print("="*70)

# ============================================================================
# EXERCISES FOR STUDENTS
# ============================================================================

"""
EXERCISE 1: Rare Disease
A disease affects 0.1% of the population. A test has 99% sensitivity and 
95% specificity. If someone tests positive, what's the probability they 
have the disease? Implement this using the functions above.

EXERCISE 2: Four Coins
Extend the coin flip example to four coins with p_heads = [0.2, 0.4, 0.6, 0.8].
Observe 15 flips with 11 heads. Which coin is most likely?

EXERCISE 3: Different Priors
In the medical test example, what if we have additional information that the
person has risk factors, increasing their prior probability to 5%? 
How does this change the posterior?

EXERCISE 4: Sensitivity Analysis
For the coin flip example, how many flips do we need to be 95% certain about
which coin we have? Test with different sequences.

EXERCISE 5: Real Data
Find a real dataset (e.g., spam classification, customer churn) and formulate
it as a Bayesian inference problem with discrete hypotheses.
"""

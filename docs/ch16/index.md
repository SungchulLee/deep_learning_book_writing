# Chapter 16: Bayesian Foundations

## Overview

This chapter develops the mathematical and philosophical foundations of Bayesian inference for machine learning and quantitative finance. Bayesian methods provide a coherent framework for updating beliefs in light of evidence, quantifying uncertainty, and making principled decisions under uncertainty.

The Bayesian paradigm interprets probability as a **degree of belief** rather than a long-run frequency. Parameters are random variables with distributions that encode our state of knowledge, and inference proceeds by conditioning on observed data through Bayes' theorem.

---

## Bayesian vs Frequentist Probability

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Nature of probability | Objective (property of world) | Subjective (state of knowledge) |
| Parameters | Fixed unknown constants | Random variables with distributions |
| Prior information | Not formally incorporated | Explicitly encoded in priors |
| Interpretation of $P(\theta \mid D)$ | Meaningless (parameters are fixed) | Posterior belief about $\theta$ |
| Inference foundation | Repeated sampling | Conditioning on evidence |

### Probability as Degree of Belief

Under the Bayesian view, probability quantifies an agent's **degree of belief** about a proposition. This interpretation extends probability to unrepeatable events — "The probability that it rained in London on January 1, 1800" has meaning as our uncertainty given historical records and climatological evidence.

### Coherence: The Dutch Book Argument

De Finetti (1931) showed that beliefs violating the probability axioms are **incoherent** — a bookie can construct bets guaranteeing a sure loss. Cox (1946) proved that any system of plausible reasoning satisfying basic desiderata must be isomorphic to the rules of probability theory.

---

## Chapter Structure

### 16.1 Core Theory

- **[Bayes' Theorem](bayesian_foundations/bayes_theorem.md)** — Derivation, components, sequential updating, and the base rate fallacy
- **[Prior, Likelihood, Posterior](bayesian_foundations/prior_likelihood_posterior.md)** — Rigorous treatment of each component, sufficient statistics, and the mechanics of updating
- **[Conjugate Priors](bayesian_foundations/conjugate_priors.md)** — Analytical posterior solutions: Beta-Binomial, Gamma-Poisson, Normal-Normal, Normal-Inverse-Gamma, and detailed derivations
- **[MAP Estimation](bayesian_foundations/map_estimation.md)** — Maximum a posteriori, comparison with MLE and posterior mean, and the MAP-regularization connection
- **[Credible Intervals](bayesian_foundations/credible_intervals.md)** — Equal-tailed and HPD intervals, and the distinction from frequentist confidence intervals

### 16.2 Bayesian Models

- **[Bayesian Linear Regression](bayesian_distributions/bayesian_linear_regression.md)** — Posterior distributions over weights, predictive uncertainty, and the conjugate Normal-Normal model
- **[Bayesian Logistic Regression](bayesian_distributions/bayesian_logistic_regression.md)** — Non-conjugate posterior inference with Laplace approximation
- **[Gaussian Processes](bayesian_distributions/gaussian_processes.md)** — Nonparametric Bayesian regression, kernel functions, and posterior prediction

### 16.3 Hierarchical Models

- **[Hierarchical Bayes](hierarchical/hierarchical_bayes.md)** — Multi-level models, partial pooling, shrinkage, and pooling strategies
- **[Multilevel Models](hierarchical/multilevel.md)** — Varying intercepts and slopes, group-level predictors
- **[Empirical Bayes](hierarchical/empirical_bayes.md)** — Data-driven hyperparameter estimation and James-Stein shrinkage

### 16.4 Model Comparison

- **[Bayesian Model Selection](model_comparison/selection.md)** — Model evidence (marginal likelihood) and Bayesian Occam's razor
- **[Bayes Factors](model_comparison/bayes_factors.md)** — Evidence ratios, interpretation scales, and Savage-Dickey density ratio
- **[Information Criteria](model_comparison/information_criteria.md)** — AIC, BIC, DIC, WAIC, and LOO-CV
- **[Hypothesis Testing](model_comparison/hypothesis_testing.md)** — Bayesian hypothesis testing and posterior odds

### 16.5 Finance Applications

- **[Bayesian Portfolio](finance/portfolio.md)** — Black-Litterman model and Bayesian portfolio optimization
- **[Parameter Uncertainty](finance/parameter_uncertainty.md)** — Estimation risk, shrinkage estimators, and predictive returns
- **[Regime Detection](finance/regime.md)** — Bayesian A/B testing, regime-switching models, and online signal tracking

---

## Prerequisites

- Probability distributions (Ch3), Maximum likelihood estimation (Ch5), Linear regression (Ch6)

## Key Connections

| Topic | Chapter | Connection |
|-------|---------|------------|
| Probabilistic Graphical Models | Ch17 | Bayesian networks as directed graphical models |
| Sampling and Inference | Ch18 | MCMC methods for posterior sampling |
| Approximate Inference | Ch19 | Variational inference and EM as posterior approximation |

---

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

---

# Appendix: Probability as Belief — Philosophical Foundations

The following provides a detailed treatment of the philosophical foundations of Bayesian probability, including formal coherence arguments and decision-theoretic justifications.

---

# Probability as Belief

The Bayesian approach to probability represents a fundamental shift in how we interpret and use probabilities. Rather than viewing probability as the long-run frequency of events, Bayesians interpret probability as a **degree of belief** or **state of knowledge**. This philosophical foundation has profound practical implications for machine learning, scientific inference, and decision-making under uncertainty.

---

## Two Philosophies of Probability

### The Frequentist Interpretation

Under the **frequentist** view, probability is defined as the limiting relative frequency of an event in an infinite sequence of identical, independent trials:

$$
P(A) = \lim_{n \to \infty} \frac{n_A}{n}
$$

where $n_A$ is the number of times event $A$ occurs in $n$ trials.

**Key characteristics**:

- Probabilities are objective properties of the world
- Parameters are fixed but unknown constants
- Probability statements apply only to repeatable experiments
- Inference proceeds through sampling distributions and hypothesis tests

**Conceptual limitation**: Consider the statement "The probability that it rained in London on January 1, 1800 is 0.3." Under strict frequentism, this statement is meaningless—we cannot repeat that specific day. Similarly, "The probability that dark matter exists" has no frequentist interpretation.

### The Bayesian Interpretation

Under the **Bayesian** view, probability quantifies an agent's **degree of belief** or **state of uncertainty** about a proposition:

$$
P(A) = \text{degree of belief in } A
$$

**Key characteristics**:

- Probabilities are subjective states of knowledge
- Parameters themselves have probability distributions
- Probability statements apply to any uncertain proposition
- Inference proceeds through conditioning on observed data

**Conceptual flexibility**: "The probability that it rained in London on January 1, 1800" now has meaning—it represents our uncertainty given historical records, climatology, and available evidence. "The probability that dark matter exists" similarly quantifies our state of knowledge.

### Comparison

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Nature of probability | Objective (property of world) | Subjective (state of knowledge) |
| Parameters | Fixed unknown constants | Random variables with distributions |
| Prior information | Not formally incorporated | Explicitly encoded in priors |
| Interpretation of $P(\theta \mid D)$ | Meaningless (parameters are fixed) | Posterior belief about $\theta$ |
| Inference foundation | Repeated sampling | Conditioning on evidence |
| Typical questions | "If $\theta = \theta_0$, what is $P(\text{data})$?" | "Given data, what is $P(\theta)$?" |

---

## Degrees of Belief: Formal Foundations

### Why Probabilities?

A natural question arises: if probability measures belief, why should beliefs follow the axioms of probability theory? The answer comes from **coherence** arguments—beliefs that violate probability axioms lead to inconsistencies that can be exploited.

### De Finetti's Dutch Book Argument

Bruno de Finetti (1931) showed that if an agent's beliefs violate the probability axioms, they can be forced into a **Dutch book**—a set of bets that guarantees a sure loss regardless of the outcome.

**Setup**: An agent assigns belief $b(A)$ to event $A$. They are willing to accept a bet that pays $\$S$ if $A$ occurs, at price $\$S \cdot b(A)$ (fair bet from their perspective).

**Theorem (Dutch Book)**: If an agent's beliefs $b(\cdot)$ violate any axiom of probability, a bookie can construct a set of bets such that the agent loses money regardless of which events occur.

**Example**: Suppose an agent believes $b(A) = 0.4$, $b(\bar{A}) = 0.5$ (violating $b(A) + b(\bar{A}) = 1$).

The bookie can:

- Sell the agent a bet on $A$ for $\$0.40$ (pays $\$1$ if $A$)
- Sell the agent a bet on $\bar{A}$ for $\$0.50$ (pays $\$1$ if $\bar{A}$)

Agent pays: $\$0.40 + \$0.50 = \$0.90$

Agent receives: $\$1$ (exactly one of $A$ or $\bar{A}$ occurs)

Agent's guaranteed loss: $\$0.90 - \$1 = -\$0.10$

**Conclusion**: Coherent beliefs must satisfy probability axioms. This provides a **normative** foundation for Bayesian probability—not a description of how people actually reason, but how they **should** reason to avoid guaranteed inconsistencies.

### Cox's Theorem

Richard Cox (1946) provided an alternative justification. He showed that any system of plausible reasoning satisfying certain desiderata must be isomorphic to probability theory.

**Cox's Desiderata**:

1. **Degrees of plausibility** are represented by real numbers
2. **Consistency**: If a conclusion can be reached multiple ways, all ways must give the same answer
3. **Common sense**: The system should agree with intuitive reasoning in simple cases
4. **Qualitative correspondence**: Greater plausibility should correspond to larger numbers

**Theorem (Cox, 1946)**: Any system satisfying these desiderata is equivalent to probability theory under a monotonic rescaling.

This is a remarkable result: probability theory is the **unique** consistent extension of Boolean logic to handle uncertainty. There are no alternatives that satisfy basic rationality requirements.

---

## Updating Beliefs: Bayes' Rule as Learning

### The Fundamental Update Equation

When we observe data $D$, our beliefs about hypothesis $H$ update according to Bayes' rule:

$$
P(H \mid D) = \frac{P(D \mid H) \, P(H)}{P(D)}
$$

In the belief interpretation:

- $P(H)$ = **prior belief** about $H$ before seeing $D$
- $P(D \mid H)$ = **likelihood** of observing $D$ given $H$ is true
- $P(H \mid D)$ = **posterior belief** about $H$ after seeing $D$
- $P(D)$ = **marginal likelihood** or **evidence** (normalizing constant)

### Belief Update as Information Processing

Bayes' rule describes how a rational agent should update beliefs when receiving new information. The posterior represents the optimal combination of:

1. **Prior knowledge**: What we believed before seeing the data
2. **Data evidence**: What the observations tell us

This update is:

- **Automatic**: No ad-hoc decisions about how to combine prior and data
- **Optimal**: Under certain loss functions, Bayesian updating minimizes expected error
- **Reversible**: Given the prior and posterior, we can reconstruct what data was observed

### Sequential Learning

A beautiful property of Bayesian updating is its **sequential consistency**. If we observe data $D_1$, then $D_2$, the result is the same whether we:

1. Update on $D_1$ to get posterior, then update on $D_2$
2. Update directly on $D = (D_1, D_2)$

**Proof**: 

$$
P(H \mid D_1, D_2) = \frac{P(D_2 \mid H, D_1) \, P(H \mid D_1)}{P(D_2 \mid D_1)}
$$

Using the posterior from $D_1$ as the new prior when updating on $D_2$. By the chain rule, this equals $P(H \mid D_1, D_2)$ computed directly.

This enables **online learning**: we can process data one observation at a time, maintaining a posterior that is always optimal given all data seen so far.

---

## Subjective vs Objective Bayes

### The Subjectivity Critique

A common criticism of Bayesian methods is their dependence on prior distributions. Different priors lead to different posteriors—doesn't this make Bayesian inference arbitrary?

### The Subjectivist Response (de Finetti, Savage)

**Subjective Bayesians** embrace the role of prior beliefs:

1. **Priors encode genuine information**: Our background knowledge, domain expertise, and previous experience are valuable and should inform inference
2. **Complete objectivity is impossible**: Even frequentist methods involve subjective choices (which test? what significance level?)
3. **Data overwhelms priors**: With sufficient data, the posterior concentrates around the truth regardless of (reasonable) prior choice
4. **Transparency**: At least the prior is explicit and can be debated

### The Objectivist Response (Jeffreys, Jaynes)

**Objective Bayesians** seek priors that minimize subjectivity:

1. **Reference priors**: Priors that maximize the expected information gain from data
2. **Jeffreys priors**: Priors that are invariant under reparameterization
3. **Maximum entropy priors**: Priors that assume nothing beyond stated constraints
4. **Empirical Bayes**: Estimate priors from the data itself

### Practical Resolution

In practice, the debate is often less important than it appears:

**With substantial data**: The likelihood dominates, and reasonable priors give nearly identical posteriors. The choice matters little.

**With limited data**: Prior choice genuinely affects conclusions. This is appropriate—with little data, conclusions *should* depend on prior knowledge. The Bayesian framework makes this dependence explicit and honest.

---

## Practical Implications for Machine Learning

### Parameters as Random Variables

In Bayesian machine learning, model parameters $\theta$ are treated as random variables with distributions, not fixed unknown constants.

**Frequentist neural network**: Parameters $\theta^*$ are point estimates

$$
\theta^* = \arg\max_\theta \log p(\mathcal{D} \mid \theta)
$$

**Bayesian neural network**: Parameters have a posterior distribution

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

### Uncertainty Quantification

The Bayesian approach naturally provides **uncertainty quantification**:

**Epistemic uncertainty** (model uncertainty): Captured by the posterior $p(\theta \mid \mathcal{D})$. With limited data, the posterior is spread out; with abundant data, it concentrates.

**Predictions include uncertainty**: Rather than a point prediction, we get a predictive distribution:

$$
p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta
$$

This integral averages predictions over all plausible parameter values, weighted by their posterior probability.

### Regularization as Prior Belief

Common regularization techniques have Bayesian interpretations:

| Regularization | Equivalent Prior | Interpretation |
|----------------|------------------|----------------|
| L2 (weight decay) | Gaussian prior $\theta \sim \mathcal{N}(0, \sigma^2 I)$ | Parameters likely near zero |
| L1 (Lasso) | Laplace prior $\theta \sim \text{Laplace}(0, b)$ | Parameters likely sparse |
| Dropout | Spike-and-slab prior | Random subset of parameters active |
| Early stopping | Implicit prior via initialization | Stay near initial values |

This reveals that seemingly "non-Bayesian" methods often have implicit Bayesian interpretations, making the prior assumptions visible.

---

## Python Implementation: Belief Updating

```python
"""
Bayesian Belief Updating: From Prior to Posterior

This module demonstrates core concepts of probability as belief
through interactive examples of belief updating via Bayes' rule.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from dataclasses import dataclass


@dataclass
class BeliefState:
    """Represents a discrete probability distribution over hypotheses."""
    hypotheses: np.ndarray  # Possible parameter values
    probabilities: np.ndarray  # Current beliefs P(hypothesis)
    
    def __post_init__(self):
        # Ensure probabilities sum to 1
        self.probabilities = self.probabilities / self.probabilities.sum()
    
    def expected_value(self) -> float:
        """Compute expected value under current beliefs."""
        return np.sum(self.hypotheses * self.probabilities)
    
    def variance(self) -> float:
        """Compute variance under current beliefs."""
        mean = self.expected_value()
        return np.sum((self.hypotheses - mean)**2 * self.probabilities)
    
    def entropy(self) -> float:
        """Compute entropy (uncertainty) of current beliefs."""
        p = self.probabilities[self.probabilities > 0]
        return -np.sum(p * np.log2(p))
    
    def credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """Compute central credible interval containing alpha probability mass."""
        cumsum = np.cumsum(self.probabilities)
        lower_idx = np.searchsorted(cumsum, (1 - alpha) / 2)
        upper_idx = np.searchsorted(cumsum, 1 - (1 - alpha) / 2)
        return self.hypotheses[lower_idx], self.hypotheses[upper_idx]


def bayesian_update(
    prior: BeliefState,
    likelihood_func: Callable[[np.ndarray, any], np.ndarray],
    observation: any
) -> BeliefState:
    """
    Update beliefs via Bayes' rule.
    
    Parameters
    ----------
    prior : BeliefState
        Current beliefs before observing data
    likelihood_func : callable
        Function(hypotheses, observation) -> P(observation | hypothesis)
    observation : any
        Observed data point
    
    Returns
    -------
    BeliefState
        Updated beliefs after conditioning on observation
    """
    # Compute likelihood for each hypothesis
    likelihoods = likelihood_func(prior.hypotheses, observation)
    
    # Apply Bayes' rule: posterior ∝ likelihood × prior
    unnormalized_posterior = likelihoods * prior.probabilities
    
    # Normalize
    posterior_probs = unnormalized_posterior / unnormalized_posterior.sum()
    
    return BeliefState(prior.hypotheses.copy(), posterior_probs)


def sequential_update(
    prior: BeliefState,
    likelihood_func: Callable,
    observations: List[any]
) -> List[BeliefState]:
    """
    Perform sequential Bayesian updates, returning belief history.
    
    Demonstrates that order of observations doesn't affect final posterior.
    """
    belief_history = [prior]
    current_belief = prior
    
    for obs in observations:
        current_belief = bayesian_update(current_belief, likelihood_func, obs)
        belief_history.append(current_belief)
    
    return belief_history


# =============================================================================
# Example: Coin Bias Estimation
# =============================================================================

def coin_likelihood(theta: np.ndarray, outcome: int) -> np.ndarray:
    """
    Likelihood of observing heads (1) or tails (0) given bias theta.
    
    P(heads | theta) = theta
    P(tails | theta) = 1 - theta
    """
    if outcome == 1:  # heads
        return theta
    else:  # tails
        return 1 - theta


def demonstrate_coin_inference():
    """
    Demonstrate belief updating for coin bias estimation.
    
    Shows how beliefs evolve as we observe coin flips.
    """
    # Hypothesis space: possible bias values
    theta_values = np.linspace(0.01, 0.99, 100)
    
    # Prior: uniform (maximum ignorance)
    uniform_prior = BeliefState(
        hypotheses=theta_values,
        probabilities=np.ones_like(theta_values)
    )
    
    # Generate "observed" data from a biased coin (true bias = 0.7)
    np.random.seed(42)
    true_bias = 0.7
    observations = (np.random.random(20) < true_bias).astype(int)
    
    # Sequential updates
    belief_history = sequential_update(uniform_prior, coin_likelihood, observations.tolist())
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    steps_to_show = [0, 1, 5, 10, 15, 20]
    
    for ax, step in zip(axes.flat, steps_to_show):
        belief = belief_history[step]
        ax.fill_between(belief.hypotheses, belief.probabilities, alpha=0.5, color='steelblue')
        ax.plot(belief.hypotheses, belief.probabilities, 'b-', linewidth=2)
        ax.axvline(true_bias, color='red', linestyle='--', label=f'True bias = {true_bias}')
        ax.axvline(belief.expected_value(), color='green', linestyle=':', 
                   label=f'E[θ] = {belief.expected_value():.3f}')
        
        ax.set_xlabel('θ (coin bias)')
        ax.set_ylabel('P(θ | data)')
        ax.set_title(f'After {step} observations')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    
    plt.suptitle('Bayesian Belief Updating: Coin Bias Estimation', fontsize=14)
    plt.tight_layout()
    plt.savefig('belief_updating_coin.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    final_belief = belief_history[-1]
    print("Coin Bias Estimation Summary")
    print("=" * 50)
    print(f"True bias: {true_bias}")
    print(f"Observations: {sum(observations)} heads in {len(observations)} flips")
    print(f"Posterior mean: {final_belief.expected_value():.4f}")
    print(f"Posterior std: {np.sqrt(final_belief.variance()):.4f}")
    print(f"95% credible interval: {final_belief.credible_interval()}")
    print(f"Entropy (bits): {final_belief.entropy():.2f}")


# =============================================================================
# Dutch Book Demonstration
# =============================================================================

@dataclass
class Bet:
    """A simple bet on an event."""
    event_name: str
    stake: float  # Amount bet
    fair_price: float  # Price paid (stake × believed probability)
    payoff_if_true: float  # Payoff if event occurs


def check_dutch_book(beliefs: dict, events_occurred: List[str]) -> dict:
    """
    Check if a set of beliefs allows a Dutch book.
    
    Parameters
    ----------
    beliefs : dict
        Mapping from event names to believed probabilities
    events_occurred : list
        Which events actually occurred
    
    Returns
    -------
    dict
        Analysis of betting outcomes
    """
    # Check probability axiom violations
    violations = []
    
    # For complementary events
    for event in beliefs:
        complement = f"not_{event}" if not event.startswith("not_") else event[4:]
        if complement in beliefs:
            if abs(beliefs[event] + beliefs[complement] - 1.0) > 1e-10:
                violations.append(f"P({event}) + P({complement}) = {beliefs[event] + beliefs[complement]} ≠ 1")
    
    # If violations exist, construct Dutch book
    if violations:
        print("Probability axiom violations detected:")
        for v in violations:
            print(f"  • {v}")
        
        # Demonstrate the Dutch book
        print("\nDutch Book Construction:")
        print("-" * 40)
        
        total_cost = sum(beliefs.values())
        print(f"Agent buys bets on all events for ${total_cost:.2f}")
        print(f"Exactly one event will occur, paying $1.00")
        print(f"Guaranteed loss: ${total_cost - 1:.2f}")
    
    return {
        'violations': violations,
        'is_coherent': len(violations) == 0
    }


def demonstrate_dutch_book():
    """Show how incoherent beliefs lead to guaranteed losses."""
    
    print("Dutch Book Demonstration")
    print("=" * 50)
    
    # Incoherent beliefs (probabilities sum to > 1)
    incoherent_beliefs = {
        'rain': 0.4,
        'not_rain': 0.7  # Should be 0.6!
    }
    
    print("\nIncoherent beliefs:")
    for event, prob in incoherent_beliefs.items():
        print(f"  P({event}) = {prob}")
    
    check_dutch_book(incoherent_beliefs, ['rain'])
    
    print("\n" + "=" * 50)
    
    # Coherent beliefs
    coherent_beliefs = {
        'rain': 0.4,
        'not_rain': 0.6
    }
    
    print("\nCoherent beliefs:")
    for event, prob in coherent_beliefs.items():
        print(f"  P({event}) = {prob}")
    
    result = check_dutch_book(coherent_beliefs, ['rain'])
    if result['is_coherent']:
        print("\n✓ Beliefs are coherent - no Dutch book possible")


# =============================================================================
# Comparing Bayesian and Frequentist Estimation
# =============================================================================

def compare_paradigms():
    """
    Compare Bayesian and frequentist approaches to parameter estimation.
    """
    np.random.seed(42)
    
    # True parameter
    true_mu = 5.0
    true_sigma = 2.0
    
    # Small sample
    n_samples = 10
    data = np.random.normal(true_mu, true_sigma, n_samples)
    
    print("Paradigm Comparison: Estimating Population Mean")
    print("=" * 60)
    print(f"True μ = {true_mu}, True σ = {true_sigma}")
    print(f"Sample: n = {n_samples}, x̄ = {data.mean():.3f}, s = {data.std():.3f}")
    print()
    
    # Frequentist approach
    print("FREQUENTIST APPROACH")
    print("-" * 40)
    sample_mean = data.mean()
    sample_se = data.std() / np.sqrt(n_samples)
    
    # 95% confidence interval
    from scipy import stats
    t_critical = stats.t.ppf(0.975, df=n_samples-1)
    ci_lower = sample_mean - t_critical * sample_se
    ci_upper = sample_mean + t_critical * sample_se
    
    print(f"Point estimate: μ̂ = {sample_mean:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print()
    print("Interpretation: If we repeated this experiment many times,")
    print("95% of such intervals would contain the true μ.")
    print("(NOT: 'There is 95% probability μ is in this interval')")
    print()
    
    # Bayesian approach
    print("BAYESIAN APPROACH")
    print("-" * 40)
    
    # Prior: N(0, 10) - weak prior centered at 0
    prior_mu = 0
    prior_sigma = 10
    
    # Posterior (conjugate update for known variance)
    # For simplicity, assume we know σ = 2
    known_sigma = true_sigma
    
    # Posterior precision = prior precision + data precision
    prior_precision = 1 / prior_sigma**2
    data_precision = n_samples / known_sigma**2
    posterior_precision = prior_precision + data_precision
    posterior_sigma = 1 / np.sqrt(posterior_precision)
    
    # Posterior mean = weighted average
    posterior_mu = (prior_precision * prior_mu + data_precision * sample_mean) / posterior_precision
    
    # 95% credible interval
    cred_lower = posterior_mu - 1.96 * posterior_sigma
    cred_upper = posterior_mu + 1.96 * posterior_sigma
    
    print(f"Prior: μ ~ N({prior_mu}, {prior_sigma}²)")
    print(f"Posterior: μ | data ~ N({posterior_mu:.3f}, {posterior_sigma:.3f}²)")
    print(f"Posterior mean: E[μ | data] = {posterior_mu:.3f}")
    print(f"95% Credible Interval: [{cred_lower:.3f}, {cred_upper:.3f}]")
    print()
    print("Interpretation: Given the data and prior,")
    print("there is 95% probability that μ lies in this interval.")
    print("(Exactly the statement people often want to make!)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEMONSTRATION: PROBABILITY AS BELIEF")
    print("=" * 60 + "\n")
    
    demonstrate_dutch_book()
    print("\n")
    demonstrate_coin_inference()
    print("\n")
    compare_paradigms()
```

---

## Key Takeaways

### Philosophical Foundation

1. **Bayesian probability = degree of belief**: Probability quantifies uncertainty about any proposition, not just repeatable experiments
2. **Coherence requires probability axioms**: Dutch book and Cox's theorem show that rational beliefs must follow probability theory
3. **Bayes' rule = optimal learning**: Updating beliefs via conditioning is the uniquely coherent way to incorporate new information

### Practical Implications

1. **Parameters have distributions**: In Bayesian ML, we maintain uncertainty over parameters rather than point estimates
2. **Priors encode knowledge**: Prior distributions formally incorporate background information
3. **Posteriors enable decisions**: Full posterior distributions support optimal decision-making under uncertainty

### Connection to Deep Learning

The Bayesian perspective motivates:

- **Bayesian Neural Networks**: Distributions over weights capture model uncertainty
- **Uncertainty quantification**: Essential for high-stakes applications
- **Regularization interpretation**: L2 regularization ↔ Gaussian prior
- **Active learning**: Acquiring data to maximally reduce uncertainty

---

## Summary

| Concept | Key Idea |
|---------|----------|
| **Probability as belief** | Probability quantifies subjective uncertainty, not objective frequency |
| **Dutch book argument** | Incoherent beliefs lead to guaranteed losses |
| **Cox's theorem** | Probability theory is the unique consistent logic of uncertainty |
| **Bayes' rule** | Optimal belief update: posterior ∝ likelihood × prior |
| **Sequential consistency** | Order of observations doesn't affect final posterior |
| **Subjectivity** | Prior choice is a feature, not a bug—makes assumptions explicit |

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Bayes' theorem mechanics | Ch13: Prior, Likelihood, Posterior | Mathematical machinery |
| Conjugate priors | Ch13: Conjugate Priors | Computational tractability |
| MCMC sampling | Ch13: MCMC | Computing posteriors when intractable |
| Bayesian neural networks | Ch13: BNN | Applying belief framework to deep learning |
| Variational inference | Ch13: VI | Approximate posterior inference |
| Model uncertainty | Ch32 | Epistemic vs aleatoric uncertainty |

### Key References

- de Finetti, B. (1931). Funzione caratteristica di un fenomeno aleatorio. *Atti della R. Academia Nazionale dei Lincei*.
- Cox, R. T. (1946). Probability, frequency, and reasonable expectation. *American Journal of Physics*, 14(1), 1-13.
- Savage, L. J. (1954). *The Foundations of Statistics*. John Wiley & Sons.
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

# Sources of Bias in ML Systems

## Overview

Bias in machine learning systems does not arise from a single cause. It enters through data collection, feature engineering, model training, and deployment—often compounding at each stage. Understanding the taxonomy of bias sources is the first step toward systematic mitigation.

## Taxonomy of Bias Sources

### Historical Bias

Training data reflects past human decisions that may have been discriminatory. Even when the data accurately represents the real world, the real world itself may be unfair.

$$P(\text{Historical Decision} \mid X, A=0) \neq P(\text{Historical Decision} \mid X, A=1)$$

where $A$ is the protected attribute. Historical bias is particularly insidious because the labels themselves are tainted—a model that perfectly learns the data distribution faithfully reproduces discrimination.

**Example in finance.** Historical mortgage approval data from the era of redlining encodes neighborhood-level racial discrimination. A model trained on these approvals learns that certain ZIP codes predict default, perpetuating geographic discrimination that correlates strongly with race.

```python
import numpy as np
import torch
from typing import Dict

def demonstrate_historical_bias():
    """
    Show how historical bias propagates through ML training.
    
    Simulates biased lending data where Group 1 faced a higher
    approval threshold historically, despite similar creditworthiness.
    """
    np.random.seed(42)
    n = 5000
    
    # True creditworthiness (unobserved) — identical distributions
    creditworthiness = np.random.normal(0.5, 0.15, n)
    group = np.random.randint(0, 2, n)
    
    # Historical decisions applied different thresholds
    threshold_0 = 0.4  # Group 0: lenient threshold
    threshold_1 = 0.6  # Group 1: strict threshold (discrimination)
    
    approved = np.where(
        group == 0,
        (creditworthiness > threshold_0).astype(int),
        (creditworthiness > threshold_1).astype(int),
    )
    
    # A model trained on this data inherits the bias
    for g in [0, 1]:
        mask = group == g
        rate = approved[mask].mean()
        avg_credit = creditworthiness[mask].mean()
        print(f"Group {g}: approval rate = {rate:.2%}, "
              f"avg creditworthiness = {avg_credit:.3f}")
    
    return {'creditworthiness': creditworthiness, 'group': group, 'approved': approved}

data = demonstrate_historical_bias()
```

### Representation Bias

Certain groups may be underrepresented in training data, leading to higher error rates for minority populations:

$$\frac{|\mathcal{D}_{A=0}|}{|\mathcal{D}|} \gg \frac{|\mathcal{D}_{A=1}|}{|\mathcal{D}|}$$

When a group constitutes only a small fraction of the training set, the model has fewer examples from which to learn group-specific patterns. This results in systematically worse predictions for underrepresented groups.

```python
def demonstrate_representation_bias():
    """
    Show how underrepresentation leads to worse model performance.
    """
    np.random.seed(42)
    
    # Group 0: 90% of data; Group 1: 10%
    n_group_0, n_group_1 = 4500, 500
    
    X_0 = np.random.randn(n_group_0, 5)
    X_1 = np.random.randn(n_group_1, 5) + 0.5  # Slightly shifted distribution
    
    # True relationship differs by group
    y_0 = (X_0[:, 0] + X_0[:, 1] > 0).astype(int)
    y_1 = (X_1[:, 0] - X_1[:, 2] > 0.5).astype(int)  # Different decision boundary
    
    print(f"Group 0 samples: {n_group_0} ({n_group_0/(n_group_0+n_group_1):.0%})")
    print(f"Group 1 samples: {n_group_1} ({n_group_1/(n_group_0+n_group_1):.0%})")
    print(f"\nA single model trained on pooled data will optimize primarily")
    print(f"for Group 0's decision boundary, harming Group 1 accuracy.")

demonstrate_representation_bias()
```

### Measurement Bias

Features may be measured differently or be less reliable for certain groups:

$$\text{Var}(X \mid A=0) \neq \text{Var}(X \mid A=1)$$

For example, credit scores may be noisier for individuals with thin credit files, who disproportionately belong to certain demographic groups. The model treats these noisy measurements as equally reliable, leading to biased predictions.

### Aggregation Bias

A single model may fail to capture distinct relationships across groups:

$$f^*(X \mid A=0) \neq f^*(X \mid A=1)$$

When the optimal decision function differs across groups but a single model is fit to pooled data, the model compromises between group-specific optima. The majority group typically benefits while the minority group suffers.

### Label Bias

Labels themselves may reflect human biases. In criminal justice, arrest records serve as proxies for criminal behavior, but arrest rates are influenced by policing patterns that vary across neighborhoods and demographics. In medicine, diagnostic labels reflect access to healthcare, which varies by socioeconomic status and geography.

### Feedback Loop Bias

Deployed models influence future data collection, creating self-reinforcing cycles:

$$\text{Model} \xrightarrow{\text{decisions}} \text{Outcomes} \xrightarrow{\text{data collection}} \text{Training Data} \xrightarrow{\text{training}} \text{Model}$$

A predictive policing model that directs more officers to certain neighborhoods generates more arrests in those neighborhoods, which in turn reinforces the model's prediction that those neighborhoods have more crime.

```python
def simulate_feedback_loop(n_rounds: int = 5):
    """
    Simulate how feedback loops amplify initial bias over time.
    """
    np.random.seed(42)
    
    # Initial crime rates are equal across two neighborhoods
    true_rate = {'A': 0.10, 'B': 0.10}
    
    # But initial policing allocation is biased
    patrol_allocation = {'A': 0.3, 'B': 0.7}
    
    print("Feedback Loop Simulation")
    print("-" * 55)
    print(f"{'Round':<8} {'Observed A':<15} {'Observed B':<15} {'Patrol A':<12} {'Patrol B'}")
    print("-" * 55)
    
    for round_num in range(n_rounds):
        # Observed crime = true crime × detection probability (∝ patrol)
        observed_A = true_rate['A'] * patrol_allocation['A']
        observed_B = true_rate['B'] * patrol_allocation['B']
        
        print(f"  {round_num+1:<6} {observed_A:<15.4f} {observed_B:<15.4f} "
              f"{patrol_allocation['A']:<12.2f} {patrol_allocation['B']:.2f}")
        
        # Update patrol based on observed rates (feedback)
        total_observed = observed_A + observed_B
        if total_observed > 0:
            patrol_allocation['A'] = observed_A / total_observed
            patrol_allocation['B'] = observed_B / total_observed

simulate_feedback_loop()
```

## Proxy Discrimination

Even when protected attributes are excluded from features, models can learn to discriminate through proxy variables—features that correlate with protected attributes. ZIP code correlates with race, name patterns correlate with ethnicity, and certain spending patterns correlate with gender.

Formally, if there exists a feature $X_j$ such that the mutual information $I(X_j; A)$ is high, then $X_j$ serves as a proxy for $A$. Removing $A$ from the feature set while retaining $X_j$ provides little protection against discrimination.

```python
def demonstrate_proxy_discrimination():
    """
    Show how proxy variables transmit bias even without
    direct access to protected attributes.
    """
    np.random.seed(42)
    n = 2000
    
    # Protected attribute (not used as feature)
    race = np.random.randint(0, 2, n)
    
    # ZIP code is correlated with race (proxy variable)
    # Group 0: more likely in ZIP range 10000-20000
    # Group 1: more likely in ZIP range 20000-30000
    zip_code = np.where(
        race == 0,
        np.random.randint(10000, 20000, n),
        np.random.randint(15000, 30000, n),
    )
    
    # Correlation between ZIP code and race
    correlation = np.corrcoef(zip_code, race)[0, 1]
    print(f"Correlation between ZIP code and race: {correlation:.3f}")
    print(f"A model using ZIP code as a feature effectively uses race as input.")

demonstrate_proxy_discrimination()
```

## Bias Across the ML Pipeline

Bias can enter at every stage:

| Stage | Bias Type | Example |
|-------|-----------|---------|
| Problem formulation | Framing bias | Predicting "recidivism" using re-arrest (not re-offense) |
| Data collection | Sampling bias | Training on data from a single geography |
| Feature engineering | Proxy bias | Using ZIP code as a feature |
| Labeling | Label bias | Using arrest records as ground truth for crime |
| Training | Aggregation bias | Single model for heterogeneous populations |
| Evaluation | Evaluation bias | Testing only on aggregate metrics, not per-group |
| Deployment | Feedback loop | Model predictions influence future training data |

## Key Takeaways

1. **Bias is systemic**: It enters at multiple stages of the ML pipeline, not just in the data
2. **Proxy discrimination** makes attribute removal ("fairness through unawareness") insufficient
3. **Feedback loops** can amplify small initial biases into large disparities over time
4. **Label bias** means even "ground truth" may be tainted by historical discrimination
5. **Awareness of bias sources** is prerequisite to effective mitigation

## Next Steps

- [Historical Context](history.md): Landmark cases and the evolution of fairness research
- [Demographic Parity](../definitions/demographic_parity.md): The first formal fairness definition

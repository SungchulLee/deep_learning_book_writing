# Introduction to Probabilistic Graphical Models

## The Curse of Dimensionality in Probability

When working with high-dimensional probability distributions, we face a fundamental challenge: the **exponential growth of parameters**. Consider a joint distribution over $n$ discrete random variables, each taking $k$ values. A full specification requires:

$$\text{Parameters} = k^n - 1$$

For 20 binary variables, this is over 1 million parameters. For 50 variables, it exceeds $10^{15}$—utterly intractable for storage, estimation, or inference.

## The Key Insight: Conditional Independence

The solution lies in recognizing that real-world variables rarely depend on all other variables. **Conditional independence** relationships allow us to factor complex distributions into products of simpler terms.

### Definition: Conditional Independence

Random variables $X$ and $Y$ are **conditionally independent** given $Z$, written $X \perp\!\!\!\perp Y \mid Z$, if:

$$P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)$$

Equivalently:
$$P(X \mid Y, Z) = P(X \mid Z)$$

This means that once we know $Z$, learning about $Y$ provides no additional information about $X$.

### Example: Medical Diagnosis

Consider three variables:
- $D$ = Disease (present/absent)
- $S_1$ = Symptom 1
- $S_2$ = Symptom 2

If both symptoms are caused by the disease but don't directly affect each other:

$$S_1 \perp\!\!\!\perp S_2 \mid D$$

Given knowledge of the disease status, the symptoms become independent. This is the **common cause** or **fork** structure.

## What Are Probabilistic Graphical Models?

A **Probabilistic Graphical Model (PGM)** is a pair $(G, P)$ where:

1. **$G$** is a graph structure encoding conditional independence relationships
2. **$P$** is a probability distribution that factors according to $G$

The graph provides:
- **Compact representation**: Only store local probability distributions
- **Independence structure**: Read off conditional independencies from the graph
- **Efficient inference**: Exploit structure for tractable computation
- **Interpretability**: Visualize relationships between variables

## The Two Major Families

### Directed Graphical Models (Bayesian Networks)

```
    Cloudy
    /    \
   v      v
 Rain    Sprinkler
    \    /
     v  v
   WetGrass
```

**Properties:**
- Directed Acyclic Graph (DAG)
- Edges represent causal or generative relationships
- Each node has a Conditional Probability Table (CPT): $P(X_i \mid \text{Parents}(X_i))$

**Factorization:**
$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

### Undirected Graphical Models (Markov Random Fields)

```
    A --- B
    |     |
    |     |
    C --- D
```

**Properties:**
- Undirected edges represent symmetric relationships
- Factors (potential functions) defined over cliques
- No notion of "parent" or "child"

**Factorization:**
$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

## Fundamental Operations on PGMs

### 1. Marginalization

Given joint distribution $P(X, Y)$, compute marginal $P(X)$:

$$P(X = x) = \sum_y P(X = x, Y = y)$$

### 2. Conditioning

Given joint $P(X, Y)$ and observation $Y = y$, compute posterior $P(X \mid Y = y)$:

$$P(X \mid Y = y) = \frac{P(X, Y = y)}{P(Y = y)} = \frac{P(X, Y = y)}{\sum_x P(X = x, Y = y)}$$

### 3. Inference Queries

Common queries on PGMs:

| Query Type | Description | Example |
|------------|-------------|---------|
| **Marginal** | $P(X)$ | What's the probability of disease? |
| **Conditional** | $P(X \mid E)$ | Probability of disease given symptoms? |
| **MAP** | $\arg\max_X P(X \mid E)$ | Most likely diagnosis given symptoms? |
| **MPE** | $\arg\max_X P(X, E)$ | Most probable complete explanation? |

## PyTorch Implementation: Probability Distribution

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class DiscreteDistribution:
    """
    Represents a discrete probability distribution over multiple variables.
    
    The distribution is stored as a multi-dimensional tensor where each
    dimension corresponds to a random variable.
    """
    
    def __init__(self, 
                 variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        """
        Initialize a discrete probability distribution.
        
        Args:
            variables: List of variable names in order
            cardinalities: Dict mapping variable names to number of values
            values: Tensor of probabilities (optional, defaults to uniform)
        """
        self.variables = variables
        self.cardinalities = cardinalities
        
        # Compute shape from cardinalities
        self.shape = tuple(cardinalities[var] for var in variables)
        
        if values is None:
            # Initialize with uniform distribution
            self.values = torch.ones(self.shape) / torch.prod(torch.tensor(self.shape)).float()
        else:
            self.values = values
            # Ensure normalization
            self.values = self.values / self.values.sum()
    
    def marginalize(self, keep_variables: List[str]) -> 'DiscreteDistribution':
        """
        Marginalize out variables not in keep_variables.
        
        P(X) = Σ_Y P(X, Y)
        
        Args:
            keep_variables: Variables to retain in the marginal
            
        Returns:
            New distribution over the specified variables
        """
        # Find axes to sum over
        sum_vars = [var for var in self.variables if var not in keep_variables]
        sum_axes = tuple(self.variables.index(var) for var in sum_vars)
        
        # Sum over those axes
        new_values = self.values.sum(dim=sum_axes)
        
        # Build new cardinalities
        new_cards = {var: self.cardinalities[var] for var in keep_variables}
        
        return DiscreteDistribution(keep_variables, new_cards, new_values)
    
    def condition(self, evidence: Dict[str, int]) -> 'DiscreteDistribution':
        """
        Condition on observed values.
        
        P(X | Y=y) = P(X, Y=y) / P(Y=y)
        
        Args:
            evidence: Dict mapping observed variables to their values
            
        Returns:
            Conditional distribution over non-observed variables
        """
        # Build indexing tuple
        indices = []
        remaining_vars = []
        
        for var in self.variables:
            if var in evidence:
                indices.append(evidence[var])
            else:
                indices.append(slice(None))
                remaining_vars.append(var)
        
        # Extract slice
        conditioned = self.values[tuple(indices)]
        
        # Normalize
        conditioned = conditioned / conditioned.sum()
        
        # Build new cardinalities
        new_cards = {var: self.cardinalities[var] for var in remaining_vars}
        
        return DiscreteDistribution(remaining_vars, new_cards, conditioned)
    
    def is_independent(self, 
                       var1: str, 
                       var2: str, 
                       tol: float = 1e-6) -> bool:
        """
        Test if two variables are marginally independent.
        
        X ⊥ Y iff P(X,Y) = P(X)P(Y)
        
        Args:
            var1, var2: Variable names to test
            tol: Numerical tolerance for comparison
            
        Returns:
            True if independent, False otherwise
        """
        # Get marginals
        p_var1 = self.marginalize([var1])
        p_var2 = self.marginalize([var2])
        
        # Get joint marginal over these two variables
        p_joint = self.marginalize([var1, var2])
        
        # Compute product of marginals
        # Need to reshape for broadcasting
        idx1 = p_joint.variables.index(var1)
        idx2 = p_joint.variables.index(var2)
        
        shape1 = [1, 1]
        shape1[idx1] = self.cardinalities[var1]
        shape2 = [1, 1]
        shape2[idx2] = self.cardinalities[var2]
        
        product = p_var1.values.view(*shape1) * p_var2.values.view(*shape2)
        
        # Compare with joint
        diff = torch.abs(p_joint.values - product)
        return diff.max().item() < tol
    
    def is_conditionally_independent(self,
                                     var1: str,
                                     var2: str,
                                     given: List[str],
                                     tol: float = 1e-6) -> bool:
        """
        Test if two variables are conditionally independent given others.
        
        X ⊥ Y | Z iff P(X,Y|Z=z) = P(X|Z=z)P(Y|Z=z) for all z
        
        Args:
            var1, var2: Variables to test
            given: Conditioning variables
            tol: Numerical tolerance
            
        Returns:
            True if conditionally independent, False otherwise
        """
        from itertools import product as cartesian_product
        
        # Test for each assignment to conditioning variables
        given_cards = [self.cardinalities[var] for var in given]
        
        for assignment in cartesian_product(*[range(c) for c in given_cards]):
            evidence = dict(zip(given, assignment))
            
            # Condition on this assignment
            conditioned = self.condition(evidence)
            
            # Test independence in the conditioned distribution
            if not conditioned.is_independent(var1, var2, tol):
                return False
        
        return True
    
    def entropy(self) -> torch.Tensor:
        """
        Compute Shannon entropy H(X) = -Σ P(x) log P(x).
        
        Returns:
            Entropy in nats
        """
        # Avoid log(0)
        probs = self.values.flatten()
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log(probs))
    
    def __repr__(self) -> str:
        return f"DiscreteDistribution({self.variables}, shape={self.shape})"


# Demonstration
if __name__ == "__main__":
    # Create a simple joint distribution P(X, Y, Z)
    dist = DiscreteDistribution(
        variables=['X', 'Y', 'Z'],
        cardinalities={'X': 2, 'Y': 2, 'Z': 2},
        values=torch.tensor([
            [[0.1, 0.05], [0.1, 0.15]],  # X=0
            [[0.15, 0.1], [0.2, 0.15]]   # X=1
        ])
    )
    
    print("Joint distribution P(X, Y, Z)")
    print(f"Shape: {dist.shape}")
    print(f"Entropy: {dist.entropy():.4f} nats")
    
    # Marginalize to get P(X)
    p_x = dist.marginalize(['X'])
    print(f"\nP(X): {p_x.values}")
    
    # Condition on Z=1 to get P(X, Y | Z=1)
    p_xy_given_z = dist.condition({'Z': 1})
    print(f"\nP(X, Y | Z=1):\n{p_xy_given_z.values}")
    
    # Test independence
    print(f"\nX ⊥ Y? {dist.is_independent('X', 'Y')}")
    print(f"X ⊥ Y | Z? {dist.is_conditionally_independent('X', 'Y', ['Z'])}")
```

## The Three Fundamental Structures

Understanding three basic structures is key to reading conditional independence from graphs:

### 1. Chain: $X \rightarrow Z \rightarrow Y$

```
X ──→ Z ──→ Y
```

- $X$ and $Y$ are **dependent** marginally
- $X$ and $Y$ are **independent** given $Z$
- Intuition: Information flows from $X$ to $Y$ through $Z$. Observing $Z$ blocks this flow.

$$X \perp\!\!\!\perp Y \mid Z$$

### 2. Fork (Common Cause): $X \leftarrow Z \rightarrow Y$

```
X ←── Z ──→ Y
```

- $X$ and $Y$ are **dependent** marginally (confounded by $Z$)
- $X$ and $Y$ are **independent** given $Z$
- Intuition: $Z$ is a common cause of both $X$ and $Y$. Once we know $Z$, they become independent.

$$X \perp\!\!\!\perp Y \mid Z$$

### 3. Collider (V-structure): $X \rightarrow Z \leftarrow Y$

```
X ──→ Z ←── Y
```

- $X$ and $Y$ are **independent** marginally
- $X$ and $Y$ are **dependent** given $Z$
- Intuition: $X$ and $Y$ are independent causes of $Z$. But if we observe $Z$, learning about one cause explains away the other.

$$X \perp\!\!\!\perp Y \text{ (marginal)}$$
$$X \not\perp\!\!\!\perp Y \mid Z \text{ (explaining away)}$$

## Example: The "Explaining Away" Effect

Consider:
- $X$ = Burglary occurred
- $Y$ = Earthquake occurred
- $Z$ = Alarm sounded

If the alarm sounds ($Z = 1$), and we learn there was an earthquake ($Y = 1$), this **decreases** our belief in a burglary ($X = 1$). The earthquake "explains away" the alarm, reducing the need for a burglary explanation.

```python
# Demonstrate explaining away
import torch

# Prior probabilities
p_burglary = 0.001
p_earthquake = 0.002

# P(Alarm | Burglary, Earthquake)
p_alarm_given_b_e = torch.tensor([
    [0.001, 0.29],   # Burglary=0: [Earthquake=0, Earthquake=1]
    [0.94, 0.95]     # Burglary=1: [Earthquake=0, Earthquake=1]
])

# Compute P(Alarm=1)
p_alarm = (
    p_alarm_given_b_e[0, 0] * (1-p_burglary) * (1-p_earthquake) +
    p_alarm_given_b_e[0, 1] * (1-p_burglary) * p_earthquake +
    p_alarm_given_b_e[1, 0] * p_burglary * (1-p_earthquake) +
    p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
)

# P(Burglary=1 | Alarm=1) using Bayes' rule
p_burglary_given_alarm = (
    p_alarm_given_b_e[1, 0] * p_burglary * (1-p_earthquake) +
    p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
) / p_alarm

# P(Burglary=1 | Alarm=1, Earthquake=1)
p_alarm_and_earthquake = (
    p_alarm_given_b_e[0, 1] * (1-p_burglary) * p_earthquake +
    p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
)
p_burglary_given_alarm_earthquake = (
    p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
) / p_alarm_and_earthquake

print(f"P(Burglary | Alarm) = {p_burglary_given_alarm:.4f}")
print(f"P(Burglary | Alarm, Earthquake) = {p_burglary_given_alarm_earthquake:.4f}")
print("\nExplaining away: Learning about earthquake decreased burglary probability!")
```

## Summary

| Concept | Description |
|---------|-------------|
| **PGM** | Graph + distribution factorization |
| **Conditional Independence** | $X \perp\!\!\!\perp Y \mid Z$ means knowing $Z$ makes $X$ and $Y$ independent |
| **Directed (Bayesian Networks)** | DAGs with CPTs, causal interpretation |
| **Undirected (MRFs)** | Symmetric dependencies, potential functions |
| **Key Operations** | Marginalization, conditioning, inference |

## Next Steps

In the following sections, we will:
1. Develop the theory of d-separation for reading independence from graphs
2. Build complete Bayesian networks with CPTs
3. Implement efficient inference algorithms
4. Learn parameters and structure from data

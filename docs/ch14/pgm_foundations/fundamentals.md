# PGM Fundamentals

## The Curse of Dimensionality in Probability

When working with high-dimensional probability distributions, we face a fundamental challenge: the **exponential growth of parameters**. Consider a joint distribution over $n$ discrete random variables, each taking $k$ values. A full specification requires:

$$\text{Parameters} = k^n - 1$$

For 20 binary variables, this exceeds one million parameters. For 50 variables, it exceeds $10^{15}$—utterly intractable for storage, estimation, or inference.

## The Key Insight: Conditional Independence

The solution lies in recognizing that real-world variables rarely depend on all other variables. **Conditional independence** relationships allow us to factor complex distributions into products of simpler terms.

### Definition: Conditional Independence

Random variables $X$ and $Y$ are **conditionally independent** given $Z$, written $X \perp\!\!\!\perp Y \mid Z$, if:

$$P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)$$

Equivalently:

$$P(X \mid Y, Z) = P(X \mid Z)$$

This means that once we know $Z$, learning about $Y$ provides no additional information about $X$.

### Example: Medical Diagnosis

Consider three variables: $D$ = Disease (present/absent), $S_1$ = Symptom 1, and $S_2$ = Symptom 2. If both symptoms are caused by the disease but don't directly affect each other:

$$S_1 \perp\!\!\!\perp S_2 \mid D$$

Given knowledge of the disease status, the symptoms become independent. This is the **common cause** (fork) structure.

## Formal Definition of a PGM

A **Probabilistic Graphical Model (PGM)** is a pair $(G, P)$ where:

1. **$G$** is a graph structure encoding conditional independence relationships
2. **$P$** is a probability distribution that factors according to $G$

The graph provides:

- **Compact representation**: Only local probability distributions need to be stored
- **Independence structure**: Conditional independencies can be read directly from the graph
- **Efficient inference**: Graph structure enables tractable computation
- **Interpretability**: Relationships between variables are visualized directly

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

A Bayesian network uses a directed acyclic graph (DAG) where edges represent causal or generative relationships. Each node has a Conditional Probability Table (CPT): $P(X_i \mid \text{Parents}(X_i))$.

**Factorization:**

$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

### Undirected Graphical Models (Markov Random Fields)

```
    A --- B
    |     |
    |     |
    C --- D
```

A Markov Random Field uses undirected edges representing symmetric relationships. Factors (potential functions) are defined over cliques, with no notion of "parent" or "child."

**Factorization:**

$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

## Fundamental Operations on PGMs

### Marginalization

Given joint distribution $P(X, Y)$, compute marginal $P(X)$:

$$P(X = x) = \sum_y P(X = x, Y = y)$$

### Conditioning

Given joint $P(X, Y)$ and observation $Y = y$, compute posterior $P(X \mid Y = y)$:

$$P(X \mid Y = y) = \frac{P(X, Y = y)}{P(Y = y)} = \frac{P(X, Y = y)}{\sum_x P(X = x, Y = y)}$$

### Inference Queries

Common queries on PGMs:

| Query Type | Description | Example |
|------------|-------------|---------|
| **Marginal** | $P(X)$ | What is the probability of disease? |
| **Conditional** | $P(X \mid E)$ | Probability of disease given symptoms? |
| **MAP** | $\arg\max_X P(X \mid E)$ | Most likely diagnosis given symptoms? |
| **MPE** | $\arg\max_X P(X, E)$ | Most probable complete explanation? |

## PyTorch Implementation: Discrete Distribution

The following class implements a discrete joint distribution with marginalization, conditioning, and independence testing:

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
        self.shape = tuple(cardinalities[var] for var in variables)
        
        if values is None:
            self.values = torch.ones(self.shape) / torch.prod(
                torch.tensor(self.shape)
            ).float()
        else:
            self.values = values
            self.values = self.values / self.values.sum()
    
    def marginalize(self, keep_variables: List[str]) -> 'DiscreteDistribution':
        """
        Marginalize out variables not in keep_variables.
        
        P(X) = sum_Y P(X, Y)
        """
        sum_vars = [var for var in self.variables if var not in keep_variables]
        sum_axes = tuple(self.variables.index(var) for var in sum_vars)
        new_values = self.values.sum(dim=sum_axes)
        new_cards = {var: self.cardinalities[var] for var in keep_variables}
        return DiscreteDistribution(keep_variables, new_cards, new_values)
    
    def condition(self, evidence: Dict[str, int]) -> 'DiscreteDistribution':
        """
        Condition on observed values.
        
        P(X | Y=y) = P(X, Y=y) / P(Y=y)
        """
        indices = []
        remaining_vars = []
        
        for var in self.variables:
            if var in evidence:
                indices.append(evidence[var])
            else:
                indices.append(slice(None))
                remaining_vars.append(var)
        
        conditioned = self.values[tuple(indices)]
        conditioned = conditioned / conditioned.sum()
        new_cards = {var: self.cardinalities[var] for var in remaining_vars}
        return DiscreteDistribution(remaining_vars, new_cards, conditioned)
    
    def is_independent(self, 
                       var1: str, 
                       var2: str, 
                       tol: float = 1e-6) -> bool:
        """
        Test if two variables are marginally independent.
        
        X _|_ Y iff P(X,Y) = P(X)P(Y)
        """
        p_var1 = self.marginalize([var1])
        p_var2 = self.marginalize([var2])
        p_joint = self.marginalize([var1, var2])
        
        idx1 = p_joint.variables.index(var1)
        idx2 = p_joint.variables.index(var2)
        
        shape1 = [1, 1]
        shape1[idx1] = self.cardinalities[var1]
        shape2 = [1, 1]
        shape2[idx2] = self.cardinalities[var2]
        
        product = p_var1.values.view(*shape1) * p_var2.values.view(*shape2)
        diff = torch.abs(p_joint.values - product)
        return diff.max().item() < tol
    
    def is_conditionally_independent(self,
                                     var1: str,
                                     var2: str,
                                     given: List[str],
                                     tol: float = 1e-6) -> bool:
        """
        Test if two variables are conditionally independent given others.
        
        X _|_ Y | Z iff P(X,Y|Z=z) = P(X|Z=z)P(Y|Z=z) for all z
        """
        from itertools import product as cartesian_product
        
        given_cards = [self.cardinalities[var] for var in given]
        
        for assignment in cartesian_product(*[range(c) for c in given_cards]):
            evidence = dict(zip(given, assignment))
            conditioned = self.condition(evidence)
            if not conditioned.is_independent(var1, var2, tol):
                return False
        return True
    
    def entropy(self) -> torch.Tensor:
        """Compute Shannon entropy H(X) = -sum P(x) log P(x) in nats."""
        probs = self.values.flatten()
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log(probs))
    
    def __repr__(self) -> str:
        return f"DiscreteDistribution({self.variables}, shape={self.shape})"


# --- Demonstration ---
if __name__ == "__main__":
    dist = DiscreteDistribution(
        variables=['X', 'Y', 'Z'],
        cardinalities={'X': 2, 'Y': 2, 'Z': 2},
        values=torch.tensor([
            [[0.1, 0.05], [0.1, 0.15]],   # X=0
            [[0.15, 0.1], [0.2, 0.15]]    # X=1
        ])
    )
    
    print(f"Joint distribution P(X, Y, Z), shape: {dist.shape}")
    print(f"Entropy: {dist.entropy():.4f} nats")
    
    p_x = dist.marginalize(['X'])
    print(f"\nP(X): {p_x.values}")
    
    p_xy_given_z = dist.condition({'Z': 1})
    print(f"\nP(X, Y | Z=1):\n{p_xy_given_z.values}")
    
    print(f"\nX _|_ Y? {dist.is_independent('X', 'Y')}")
    print(f"X _|_ Y | Z? {dist.is_conditionally_independent('X', 'Y', ['Z'])}")
```

## The Three Fundamental Structures

Understanding three basic graph structures is the key to reading conditional independence from directed graphs. These structures recur throughout PGM theory and practice.

### 1. Chain: $X \rightarrow Z \rightarrow Y$

```
X --> Z --> Y
```

$X$ and $Y$ are **dependent** marginally but **independent** given $Z$. Information flows from $X$ to $Y$ through $Z$; observing $Z$ blocks this flow.

$$X \perp\!\!\!\perp Y \mid Z$$

### 2. Fork (Common Cause): $X \leftarrow Z \rightarrow Y$

```
X <-- Z --> Y
```

$X$ and $Y$ are **dependent** marginally (confounded by $Z$) but **independent** given $Z$. Once we know the common cause $Z$, learning about one effect tells us nothing about the other.

$$X \perp\!\!\!\perp Y \mid Z$$

### 3. Collider (V-structure): $X \rightarrow Z \leftarrow Y$

```
X --> Z <-- Y
```

$X$ and $Y$ are **independent** marginally but **dependent** given $Z$. Observing the common effect $Z$ creates a dependency between its independent causes. This is the **explaining away** effect.

$$X \perp\!\!\!\perp Y \quad \text{(marginal)}, \qquad X \not\perp\!\!\!\perp Y \mid Z \quad \text{(explaining away)}$$

### The Explaining Away Effect

Consider $X$ = Burglary, $Y$ = Earthquake, and $Z$ = Alarm. If the alarm sounds ($Z = 1$) and we learn there was an earthquake ($Y = 1$), this **decreases** our belief in a burglary ($X = 1$). The earthquake "explains away" the alarm, reducing the need for a burglary explanation.

```python
import torch

# Prior probabilities
p_burglary = 0.001
p_earthquake = 0.002

# P(Alarm | Burglary, Earthquake)
p_alarm_given_b_e = torch.tensor([
    [0.001, 0.29],    # Burglary=0: [Earthquake=0, Earthquake=1]
    [0.94, 0.95]      # Burglary=1: [Earthquake=0, Earthquake=1]
])

# Compute P(Alarm=1)
p_alarm = (
    p_alarm_given_b_e[0, 0] * (1 - p_burglary) * (1 - p_earthquake)
    + p_alarm_given_b_e[0, 1] * (1 - p_burglary) * p_earthquake
    + p_alarm_given_b_e[1, 0] * p_burglary * (1 - p_earthquake)
    + p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
)

# P(Burglary=1 | Alarm=1) via Bayes' rule
p_burglary_given_alarm = (
    p_alarm_given_b_e[1, 0] * p_burglary * (1 - p_earthquake)
    + p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
) / p_alarm

# P(Burglary=1 | Alarm=1, Earthquake=1)
p_alarm_and_earthquake = (
    p_alarm_given_b_e[0, 1] * (1 - p_burglary) * p_earthquake
    + p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
)
p_burglary_given_alarm_earthquake = (
    p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
) / p_alarm_and_earthquake

print(f"P(Burglary | Alarm)                = {p_burglary_given_alarm:.4f}")
print(f"P(Burglary | Alarm, Earthquake)    = {p_burglary_given_alarm_earthquake:.4f}")
print("\nExplaining away: learning about the earthquake decreased burglary probability.")
```

## Summary

| Structure | Marginal | Conditioned on Middle | Rule |
|-----------|----------|----------------------|------|
| Chain: $A \to B \to C$ | Dependent | Independent | Blocking |
| Fork: $A \leftarrow B \to C$ | Dependent | Independent | Blocking |
| Collider: $A \to B \leftarrow C$ | Independent | Dependent | Explaining away |

| Concept | Description |
|---------|-------------|
| **PGM** | Graph + distribution factorization |
| **Conditional Independence** | $X \perp\!\!\!\perp Y \mid Z$: knowing $Z$ makes $X$ and $Y$ independent |
| **Directed (Bayesian Network)** | DAGs with CPTs, causal interpretation |
| **Undirected (MRF)** | Symmetric dependencies, potential functions |
| **Key Operations** | Marginalization, conditioning, inference |

## Quantitative Finance Application: Factor Models as PGMs

Classical factor models in quantitative finance are naturally expressed as PGMs. Consider a single-factor model where a market factor $F$ drives $n$ asset returns $R_1, \ldots, R_n$:

$$R_i = \alpha_i + \beta_i F + \epsilon_i, \qquad \epsilon_i \perp\!\!\!\perp \epsilon_j \mid F$$

This is exactly the **fork** structure: $R_i \leftarrow F \rightarrow R_j$. Asset returns are marginally correlated (through the common market factor) but conditionally independent given the factor. Multi-factor models extend this to multiple common causes, with the graph encoding which factors drive which assets. Recognizing this structure enables efficient covariance estimation and risk decomposition.

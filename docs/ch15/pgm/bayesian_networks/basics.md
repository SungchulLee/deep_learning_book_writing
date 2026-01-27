# Bayesian Networks: Structure and Semantics

## What is a Bayesian Network?

A **Bayesian Network** (also called a Belief Network or Directed Graphical Model) is a probabilistic graphical model that represents a joint probability distribution using:

1. **A Directed Acyclic Graph (DAG)** $G = (V, E)$
   - Nodes $V$ represent random variables
   - Directed edges $E$ represent direct probabilistic dependencies

2. **Conditional Probability Distributions (CPDs)**
   - For each node $X_i$, a distribution $P(X_i \mid \text{Parents}(X_i))$
   - Stored as Conditional Probability Tables (CPTs) for discrete variables

## The Factorization Property

The key property of Bayesian Networks is that the joint distribution factors as:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

where $\text{Pa}(X_i)$ denotes the parents of $X_i$ in the DAG.

### Why This Matters

Consider $n$ binary variables:
- **Full joint table**: $2^n - 1$ parameters
- **Bayesian Network** with max $k$ parents: At most $n \cdot 2^k$ parameters

For $n=20$ and $k=3$: 1,048,575 vs 160 parameters!

## Conditional Probability Tables (CPTs)

A CPT specifies $P(X \mid \text{Parents}(X))$ for all combinations of parent values.

### Example: Weather Network

```
    Cloudy
    /    \
   ↓      ↓
 Rain    Sprinkler
    \    /
     ↓  ↓
   WetGrass
```

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class CPT:
    """
    Conditional Probability Table for a discrete random variable.
    
    Stores P(Variable | Parents) as a multi-dimensional tensor where:
    - First dimensions correspond to parent variables (in order)
    - Last dimension corresponds to the variable itself
    """
    
    def __init__(self,
                 variable: str,
                 parents: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        """
        Initialize a CPT.
        
        Args:
            variable: Name of the variable
            parents: List of parent variable names (in order)
            cardinalities: Dict mapping all variable names to their cardinalities
            values: Optional tensor of conditional probabilities
                   Shape: (card_parent1, card_parent2, ..., card_variable)
        """
        self.variable = variable
        self.parents = parents
        self.cardinalities = cardinalities
        
        # Compute shape: parent cardinalities + variable cardinality
        self.shape = tuple(cardinalities[p] for p in parents) + (cardinalities[variable],)
        
        if values is None:
            # Initialize with uniform distribution
            self.values = torch.ones(self.shape) / cardinalities[variable]
        else:
            self.values = values.float()
            self._validate_normalization()
    
    def _validate_normalization(self, tol: float = 1e-5):
        """Ensure each conditional distribution sums to 1."""
        # Sum over the last axis (the variable)
        sums = self.values.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=tol):
            print(f"Warning: CPT for {self.variable} not normalized. Normalizing...")
            self.values = self.values / sums.unsqueeze(-1)
    
    def get_probability(self, 
                       variable_value: int,
                       parent_values: Dict[str, int]) -> float:
        """
        Get P(Variable=value | Parents=parent_values).
        
        Args:
            variable_value: Value of the variable
            parent_values: Dict mapping parent names to their values
            
        Returns:
            Conditional probability
        """
        # Build index tuple
        index = tuple(parent_values[p] for p in self.parents) + (variable_value,)
        return self.values[index].item()
    
    def sample(self, parent_values: Dict[str, int]) -> int:
        """
        Sample from P(Variable | Parents=parent_values).
        
        Args:
            parent_values: Dict mapping parent names to their values
            
        Returns:
            Sampled value for the variable
        """
        # Get the conditional distribution
        index = tuple(parent_values[p] for p in self.parents)
        probs = self.values[index]
        
        # Sample from categorical distribution
        return torch.multinomial(probs, 1).item()
    
    def to_factor(self) -> 'Factor':
        """Convert CPT to a Factor for inference."""
        variables = self.parents + [self.variable]
        return Factor(variables, self.cardinalities, self.values)
    
    def __repr__(self) -> str:
        parent_str = ', '.join(self.parents) if self.parents else '∅'
        return f"CPT(P({self.variable} | {parent_str}), shape={self.shape})"


class BayesianNetwork:
    """
    A Bayesian Network: DAG structure with CPTs.
    
    Represents: P(X1, ..., Xn) = ∏ P(Xi | Parents(Xi))
    """
    
    def __init__(self):
        """Initialize an empty Bayesian Network."""
        self.variables: List[str] = []
        self.cardinalities: Dict[str, int] = {}
        self.parents: Dict[str, List[str]] = {}
        self.cpts: Dict[str, CPT] = {}
    
    def add_variable(self, name: str, cardinality: int, parents: List[str] = None):
        """
        Add a variable to the network.
        
        Args:
            name: Variable name
            cardinality: Number of values the variable can take
            parents: List of parent variable names
        """
        if parents is None:
            parents = []
        
        # Validate parents exist
        for parent in parents:
            if parent not in self.variables:
                raise ValueError(f"Parent {parent} not in network. Add variables in topological order.")
        
        self.variables.append(name)
        self.cardinalities[name] = cardinality
        self.parents[name] = parents
    
    def set_cpt(self, variable: str, values: torch.Tensor):
        """
        Set the CPT for a variable.
        
        Args:
            variable: Variable name
            values: CPT values tensor
        """
        if variable not in self.variables:
            raise ValueError(f"Variable {variable} not in network")
        
        parents = self.parents[variable]
        self.cpts[variable] = CPT(variable, parents, self.cardinalities, values)
    
    def get_children(self, variable: str) -> List[str]:
        """Get children of a variable."""
        return [v for v in self.variables if variable in self.parents[v]]
    
    def topological_order(self) -> List[str]:
        """Get variables in topological order."""
        # Variables are already added in topological order by construction
        return self.variables.copy()
    
    def joint_probability(self, assignment: Dict[str, int]) -> float:
        """
        Compute P(X1=x1, ..., Xn=xn) using the chain rule.
        
        P(X) = ∏ P(Xi | Parents(Xi))
        
        Args:
            assignment: Complete assignment to all variables
            
        Returns:
            Joint probability
        """
        prob = 1.0
        for var in self.variables:
            parents = self.parents[var]
            parent_values = {p: assignment[p] for p in parents}
            var_value = assignment[var]
            
            prob *= self.cpts[var].get_probability(var_value, parent_values)
        
        return prob
    
    def forward_sample(self) -> Dict[str, int]:
        """
        Generate a sample using forward (ancestral) sampling.
        
        Samples variables in topological order, conditioning on sampled parents.
        
        Returns:
            Dict mapping each variable to its sampled value
        """
        sample = {}
        
        for var in self.topological_order():
            parents = self.parents[var]
            parent_values = {p: sample[p] for p in parents}
            
            sample[var] = self.cpts[var].sample(parent_values)
        
        return sample
    
    def generate_samples(self, n_samples: int) -> List[Dict[str, int]]:
        """Generate multiple samples."""
        return [self.forward_sample() for _ in range(n_samples)]
    
    def __repr__(self) -> str:
        edges = []
        for var in self.variables:
            for parent in self.parents[var]:
                edges.append(f"{parent}→{var}")
        return f"BayesianNetwork({len(self.variables)} vars, edges: {', '.join(edges)})"


def build_weather_network() -> BayesianNetwork:
    """
    Build the classic weather/sprinkler network.
    
    Structure:
        Cloudy
        /    \
       ↓      ↓
     Rain    Sprinkler
        \    /
         ↓  ↓
       WetGrass
    """
    bn = BayesianNetwork()
    
    # Add variables in topological order
    bn.add_variable('Cloudy', 2, parents=[])
    bn.add_variable('Sprinkler', 2, parents=['Cloudy'])
    bn.add_variable('Rain', 2, parents=['Cloudy'])
    bn.add_variable('WetGrass', 2, parents=['Sprinkler', 'Rain'])
    
    # Set CPTs
    
    # P(Cloudy)
    bn.set_cpt('Cloudy', torch.tensor([0.5, 0.5]))
    
    # P(Sprinkler | Cloudy)
    # Less likely to use sprinkler when cloudy
    bn.set_cpt('Sprinkler', torch.tensor([
        [0.5, 0.5],   # P(Sprinkler | Cloudy=0)
        [0.9, 0.1]    # P(Sprinkler | Cloudy=1)
    ]))
    
    # P(Rain | Cloudy)
    # More likely to rain when cloudy
    bn.set_cpt('Rain', torch.tensor([
        [0.8, 0.2],   # P(Rain | Cloudy=0)
        [0.2, 0.8]    # P(Rain | Cloudy=1)
    ]))
    
    # P(WetGrass | Sprinkler, Rain)
    bn.set_cpt('WetGrass', torch.tensor([
        [[1.0, 0.0],    # Sprinkler=0, Rain=0: Grass dry
         [0.1, 0.9]],   # Sprinkler=0, Rain=1: Grass wet (rain)
        [[0.1, 0.9],    # Sprinkler=1, Rain=0: Grass wet (sprinkler)
         [0.01, 0.99]]  # Sprinkler=1, Rain=1: Grass very wet
    ]))
    
    return bn


def demonstrate_bayesian_network():
    """Demonstrate Bayesian Network operations."""
    
    print("Building Weather Bayesian Network")
    print("=" * 60)
    
    bn = build_weather_network()
    print(f"\nNetwork: {bn}")
    
    # Show CPTs
    print("\nConditional Probability Tables:")
    print("-" * 60)
    for var in bn.variables:
        print(f"\n{bn.cpts[var]}")
        print(f"Values:\n{bn.cpts[var].values}")
    
    # Compute some joint probabilities
    print("\n\nJoint Probability Examples:")
    print("-" * 60)
    
    scenarios = [
        {'Cloudy': 0, 'Sprinkler': 1, 'Rain': 0, 'WetGrass': 1},
        {'Cloudy': 1, 'Sprinkler': 0, 'Rain': 1, 'WetGrass': 1},
        {'Cloudy': 1, 'Sprinkler': 1, 'Rain': 1, 'WetGrass': 1},
    ]
    
    for scenario in scenarios:
        prob = bn.joint_probability(scenario)
        desc = ", ".join(f"{k}={v}" for k, v in scenario.items())
        print(f"\nP({desc}) = {prob:.6f}")
        
        # Show factorization
        print("  = P(Cloudy) × P(Sprinkler|Cloudy) × P(Rain|Cloudy) × P(WetGrass|Sprinkler,Rain)")
        
        factors = []
        for var in bn.variables:
            parents = bn.parents[var]
            parent_vals = {p: scenario[p] for p in parents}
            p = bn.cpts[var].get_probability(scenario[var], parent_vals)
            if parents:
                parent_str = ','.join(f'{p}={parent_vals[p]}' for p in parents)
                factors.append(f"P({var}={scenario[var]}|{parent_str})={p:.3f}")
            else:
                factors.append(f"P({var}={scenario[var]})={p:.3f}")
        print(f"  = {' × '.join(factors)}")
    
    # Forward sampling
    print("\n\nForward Sampling:")
    print("-" * 60)
    
    samples = bn.generate_samples(1000)
    
    # Compute empirical probabilities
    print("\nEmpirical vs True Probabilities:")
    
    # P(Cloudy=1)
    empirical_cloudy = sum(s['Cloudy'] for s in samples) / len(samples)
    print(f"P(Cloudy=1): True=0.500, Empirical={empirical_cloudy:.3f}")
    
    # P(Rain=1 | Cloudy=1)
    cloudy_samples = [s for s in samples if s['Cloudy'] == 1]
    empirical_rain_given_cloudy = sum(s['Rain'] for s in cloudy_samples) / len(cloudy_samples)
    print(f"P(Rain=1|Cloudy=1): True=0.800, Empirical={empirical_rain_given_cloudy:.3f}")
    
    # P(WetGrass=1)
    empirical_wet = sum(s['WetGrass'] for s in samples) / len(samples)
    # True probability computed by summing over all configurations
    true_wet = sum(bn.joint_probability(dict(zip(bn.variables, vals))) 
                   for vals in [(c, s, r, 1) for c in [0,1] for s in [0,1] for r in [0,1]])
    print(f"P(WetGrass=1): True={true_wet:.3f}, Empirical={empirical_wet:.3f}")

demonstrate_bayesian_network()
```

## Classic Bayesian Networks

### The Alarm Network

A famous diagnostic network modeling a home security system:

```
  Burglary    Earthquake
      \          /
       ↓        ↓
         Alarm
        /      \
       ↓        ↓
  JohnCalls  MaryCalls
```

**Key properties:**
- Burglary and Earthquake are independent causes of Alarm
- JohnCalls and MaryCalls are independent given Alarm
- Burglary and Earthquake become dependent when Alarm is observed (explaining away)

### The Student Network

Models factors affecting a student's performance:

```
  Difficulty    Intelligence
      \            /
       ↓          ↓
         Grade
           |
           ↓
        Letter
```

**Variables:**
- Difficulty: Easy (0) or Hard (1)
- Intelligence: Low (0) or High (1)
- Grade: A (0), B (1), or C (2)
- Letter: Weak (0) or Strong (1)

### The Asia Network

A simplified medical diagnosis network:

```
  VisitAsia    Smoking
      |           |
      ↓           ↓
    TB        LungCancer    Bronchitis
      \          /              |
       ↓        ↓               |
        TBorCancer              |
           |                    |
           ↓                    ↓
          Xray              Dyspnea
```

## Parameter Counting

The number of independent parameters in a CPT for variable $X_i$ with parents $\text{Pa}(X_i)$:

$$\text{Parameters}(X_i) = (|\text{Val}(X_i)| - 1) \times \prod_{j \in \text{Pa}(X_i)} |\text{Val}(X_j)|$$

The $-1$ accounts for the normalization constraint (probabilities must sum to 1).

### Example

For the Weather network:
- Cloudy: $(2-1) \times 1 = 1$ parameter
- Sprinkler: $(2-1) \times 2 = 2$ parameters
- Rain: $(2-1) \times 2 = 2$ parameters
- WetGrass: $(2-1) \times 2 \times 2 = 4$ parameters

**Total: 9 parameters** (vs $2^4 - 1 = 15$ for full joint table)

## Querying Bayesian Networks

### Types of Queries

1. **Prior Query**: $P(X)$ without any evidence
2. **Posterior Query**: $P(X \mid E=e)$ given evidence
3. **MAP Query**: $\arg\max_X P(X \mid E=e)$ most probable state
4. **MPE Query**: Most probable complete assignment

### Inference by Enumeration

The simplest (but slowest) inference method:

$$P(X \mid E=e) = \frac{P(X, E=e)}{P(E=e)} = \frac{\sum_Y P(X, E=e, Y)}{\sum_{X,Y} P(X, E=e, Y)}$$

where $Y$ are the hidden (non-query, non-evidence) variables.

```python
class InferenceByEnumeration:
    """
    Exact inference by summing over all configurations.
    
    Time complexity: O(d^n) where d is max cardinality, n is number of variables.
    Practical only for small networks!
    """
    
    def __init__(self, bn: BayesianNetwork):
        self.bn = bn
    
    def query(self, 
              query_vars: List[str],
              evidence: Dict[str, int] = None) -> torch.Tensor:
        """
        Compute P(query_vars | evidence) by enumeration.
        
        Args:
            query_vars: Variables to compute distribution over
            evidence: Observed variable values
            
        Returns:
            Tensor of shape (card_q1, card_q2, ...) with probabilities
        """
        if evidence is None:
            evidence = {}
        
        # Identify hidden variables
        hidden_vars = [v for v in self.bn.variables 
                       if v not in query_vars and v not in evidence]
        
        # Shape of result
        result_shape = tuple(self.bn.cardinalities[v] for v in query_vars)
        result = torch.zeros(result_shape)
        
        # Enumerate all configurations
        from itertools import product
        
        query_cards = [self.bn.cardinalities[v] for v in query_vars]
        hidden_cards = [self.bn.cardinalities[v] for v in hidden_vars]
        
        for query_vals in product(*[range(c) for c in query_cards]):
            query_assignment = dict(zip(query_vars, query_vals))
            
            # Sum over hidden variables
            prob_sum = 0.0
            for hidden_vals in product(*[range(c) for c in hidden_cards]):
                hidden_assignment = dict(zip(hidden_vars, hidden_vals))
                
                # Complete assignment
                full_assignment = {**query_assignment, **hidden_assignment, **evidence}
                
                # Compute joint probability
                prob_sum += self.bn.joint_probability(full_assignment)
            
            result[query_vals] = prob_sum
        
        # Normalize to get conditional probability
        result = result / result.sum()
        
        return result


def demonstrate_inference():
    """Demonstrate inference in a Bayesian Network."""
    
    print("\n\nBayesian Network Inference")
    print("=" * 60)
    
    bn = build_weather_network()
    inference = InferenceByEnumeration(bn)
    
    # Query 1: P(Rain)
    print("\nQuery 1: P(Rain)")
    p_rain = inference.query(['Rain'])
    print(f"P(Rain=0) = {p_rain[0]:.4f}")
    print(f"P(Rain=1) = {p_rain[1]:.4f}")
    
    # Query 2: P(Rain | Cloudy=1)
    print("\nQuery 2: P(Rain | Cloudy=1)")
    p_rain_given_cloudy = inference.query(['Rain'], {'Cloudy': 1})
    print(f"P(Rain=0 | Cloudy=1) = {p_rain_given_cloudy[0]:.4f}")
    print(f"P(Rain=1 | Cloudy=1) = {p_rain_given_cloudy[1]:.4f}")
    
    # Query 3: P(Rain | WetGrass=1) - diagnostic inference
    print("\nQuery 3: P(Rain | WetGrass=1) - Diagnostic Reasoning")
    p_rain_given_wet = inference.query(['Rain'], {'WetGrass': 1})
    print(f"P(Rain=0 | WetGrass=1) = {p_rain_given_wet[0]:.4f}")
    print(f"P(Rain=1 | WetGrass=1) = {p_rain_given_wet[1]:.4f}")
    
    # Query 4: P(Rain, Sprinkler | WetGrass=1)
    print("\nQuery 4: P(Rain, Sprinkler | WetGrass=1)")
    p_joint_given_wet = inference.query(['Rain', 'Sprinkler'], {'WetGrass': 1})
    print("            Sprinkler=0  Sprinkler=1")
    print(f"Rain=0      {p_joint_given_wet[0,0]:.4f}        {p_joint_given_wet[0,1]:.4f}")
    print(f"Rain=1      {p_joint_given_wet[1,0]:.4f}        {p_joint_given_wet[1,1]:.4f}")
    
    # Query 5: P(Cloudy | WetGrass=1) - explaining away
    print("\nQuery 5: P(Cloudy | WetGrass=1)")
    p_cloudy_given_wet = inference.query(['Cloudy'], {'WetGrass': 1})
    print(f"P(Cloudy=0 | WetGrass=1) = {p_cloudy_given_wet[0]:.4f}")
    print(f"P(Cloudy=1 | WetGrass=1) = {p_cloudy_given_wet[1]:.4f}")
    
    # Compare with knowing sprinkler was on (explaining away)
    print("\nQuery 6: P(Cloudy | WetGrass=1, Sprinkler=1) - Explaining Away")
    p_cloudy_given_wet_sprinkler = inference.query(['Cloudy'], {'WetGrass': 1, 'Sprinkler': 1})
    print(f"P(Cloudy=0 | WetGrass=1, Sprinkler=1) = {p_cloudy_given_wet_sprinkler[0]:.4f}")
    print(f"P(Cloudy=1 | WetGrass=1, Sprinkler=1) = {p_cloudy_given_wet_sprinkler[1]:.4f}")
    print("\nNote: Knowing sprinkler was on explains the wet grass,")
    print("reducing our belief that it was cloudy (which would have caused rain).")

demonstrate_inference()
```

## Summary

| Concept | Description |
|---------|-------------|
| **Bayesian Network** | DAG + CPTs representing a joint distribution |
| **CPT** | $P(X \mid \text{Parents}(X))$ for each node |
| **Factorization** | $P(X_1,\ldots,X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))$ |
| **Forward Sampling** | Sample in topological order, conditioning on parents |
| **Inference** | Compute $P(\text{Query} \mid \text{Evidence})$ |

## Key Advantages of Bayesian Networks

1. **Compact representation**: Exponential reduction in parameters
2. **Interpretable structure**: Visualize dependencies between variables
3. **Modular specification**: Define each variable's local distribution independently
4. **Efficient inference**: Exploit structure for tractable computation
5. **Causal reasoning**: Can sometimes interpret edges as causal relationships

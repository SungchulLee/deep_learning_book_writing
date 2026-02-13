# Bayesian Networks

## What Is a Bayesian Network?

A **Bayesian Network** (also called a Belief Network or Directed Graphical Model) is a probabilistic graphical model that represents a joint probability distribution using:

1. **A Directed Acyclic Graph (DAG)** $G = (V, E)$ where nodes $V$ represent random variables and directed edges $E$ represent direct probabilistic dependencies.

2. **Conditional Probability Distributions (CPDs)** — for each node $X_i$, a distribution $P(X_i \mid \text{Parents}(X_i))$, stored as Conditional Probability Tables (CPTs) for discrete variables.

## The Factorization Property

The joint distribution factors as:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

where $\text{Pa}(X_i)$ denotes the parents of $X_i$ in the DAG.

### Parameter Savings

Consider $n$ binary variables. A full joint table requires $2^n - 1$ parameters, while a Bayesian network with at most $k$ parents per node requires at most $n \cdot 2^k$ parameters. For $n = 20$ and $k = 3$, this is 1,048,575 versus 160 parameters.

## Conditional Probability Tables

A CPT specifies $P(X \mid \text{Parents}(X))$ for all combinations of parent values. The number of independent parameters in a CPT for variable $X_i$ with parents $\text{Pa}(X_i)$ is:

$$\text{Parameters}(X_i) = (|\text{Val}(X_i)| - 1) \times \prod_{j \in \text{Pa}(X_i)} |\text{Val}(X_j)|$$

The $-1$ accounts for the normalization constraint.

## PyTorch Implementation

### CPT Class

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class CPT:
    """
    Conditional Probability Table for a discrete random variable.
    
    Stores P(Variable | Parents) as a multi-dimensional tensor where
    the first dimensions correspond to parent variables (in order) and
    the last dimension corresponds to the variable itself.
    """
    
    def __init__(self,
                 variable: str,
                 parents: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        self.variable = variable
        self.parents = parents
        self.cardinalities = cardinalities
        self.shape = tuple(cardinalities[p] for p in parents) + (cardinalities[variable],)
        
        if values is None:
            self.values = torch.ones(self.shape) / cardinalities[variable]
        else:
            self.values = values.float()
            self._validate_normalization()
    
    def _validate_normalization(self, tol: float = 1e-5):
        """Ensure each conditional distribution sums to 1."""
        sums = self.values.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=tol):
            print(f"Warning: CPT for {self.variable} not normalized. Normalizing...")
            self.values = self.values / sums.unsqueeze(-1)
    
    def get_probability(self, variable_value: int,
                       parent_values: Dict[str, int]) -> float:
        """Get P(Variable=value | Parents=parent_values)."""
        index = tuple(parent_values[p] for p in self.parents) + (variable_value,)
        return self.values[index].item()
    
    def sample(self, parent_values: Dict[str, int]) -> int:
        """Sample from P(Variable | Parents=parent_values)."""
        index = tuple(parent_values[p] for p in self.parents)
        probs = self.values[index]
        return torch.multinomial(probs, 1).item()
    
    def __repr__(self) -> str:
        parent_str = ', '.join(self.parents) if self.parents else 'empty'
        return f"CPT(P({self.variable} | {parent_str}), shape={self.shape})"
```

### Bayesian Network Class

```python
class BayesianNetwork:
    """
    A Bayesian Network: DAG structure with CPTs.
    
    Represents: P(X1, ..., Xn) = prod_i P(Xi | Parents(Xi))
    """
    
    def __init__(self):
        self.variables: List[str] = []
        self.cardinalities: Dict[str, int] = {}
        self.parents: Dict[str, List[str]] = {}
        self.cpts: Dict[str, CPT] = {}
    
    def add_variable(self, name: str, cardinality: int, 
                     parents: List[str] = None):
        """Add a variable. Variables must be added in topological order."""
        if parents is None:
            parents = []
        for parent in parents:
            if parent not in self.variables:
                raise ValueError(
                    f"Parent {parent} not in network. "
                    f"Add variables in topological order."
                )
        self.variables.append(name)
        self.cardinalities[name] = cardinality
        self.parents[name] = parents
    
    def set_cpt(self, variable: str, values: torch.Tensor):
        """Set the CPT for a variable."""
        parents = self.parents[variable]
        self.cpts[variable] = CPT(
            variable, parents, self.cardinalities, values
        )
    
    def get_children(self, variable: str) -> List[str]:
        return [v for v in self.variables if variable in self.parents[v]]
    
    def topological_order(self) -> List[str]:
        return self.variables.copy()
    
    def joint_probability(self, assignment: Dict[str, int]) -> float:
        """
        Compute P(X1=x1, ..., Xn=xn) using the chain rule.
        """
        prob = 1.0
        for var in self.variables:
            parents = self.parents[var]
            parent_values = {p: assignment[p] for p in parents}
            prob *= self.cpts[var].get_probability(assignment[var], parent_values)
        return prob
    
    def forward_sample(self) -> Dict[str, int]:
        """
        Generate a sample using ancestral sampling.
        
        Samples variables in topological order, conditioning on
        already-sampled parents.
        """
        sample = {}
        for var in self.topological_order():
            parent_values = {p: sample[p] for p in self.parents[var]}
            sample[var] = self.cpts[var].sample(parent_values)
        return sample
    
    def generate_samples(self, n_samples: int) -> List[Dict[str, int]]:
        return [self.forward_sample() for _ in range(n_samples)]
    
    def __repr__(self) -> str:
        edges = []
        for var in self.variables:
            for parent in self.parents[var]:
                edges.append(f"{parent}->{var}")
        return f"BayesianNetwork({len(self.variables)} vars, edges: {', '.join(edges)})"
```

## Example: The Weather Network

```
    Cloudy
    /    \
   v      v
 Rain    Sprinkler
    \    /
     v  v
   WetGrass
```

```python
def build_weather_network() -> BayesianNetwork:
    """Build the classic weather/sprinkler network."""
    bn = BayesianNetwork()
    
    bn.add_variable('Cloudy', 2, parents=[])
    bn.add_variable('Sprinkler', 2, parents=['Cloudy'])
    bn.add_variable('Rain', 2, parents=['Cloudy'])
    bn.add_variable('WetGrass', 2, parents=['Sprinkler', 'Rain'])
    
    # P(Cloudy)
    bn.set_cpt('Cloudy', torch.tensor([0.5, 0.5]))
    
    # P(Sprinkler | Cloudy): less likely when cloudy
    bn.set_cpt('Sprinkler', torch.tensor([
        [0.5, 0.5],    # Cloudy=0
        [0.9, 0.1]     # Cloudy=1
    ]))
    
    # P(Rain | Cloudy): more likely when cloudy
    bn.set_cpt('Rain', torch.tensor([
        [0.8, 0.2],    # Cloudy=0
        [0.2, 0.8]     # Cloudy=1
    ]))
    
    # P(WetGrass | Sprinkler, Rain)
    bn.set_cpt('WetGrass', torch.tensor([
        [[1.0, 0.0],     # Sprinkler=0, Rain=0
         [0.1, 0.9]],    # Sprinkler=0, Rain=1
        [[0.1, 0.9],     # Sprinkler=1, Rain=0
         [0.01, 0.99]]   # Sprinkler=1, Rain=1
    ]))
    
    return bn


# Parameter counting for the Weather network:
# Cloudy:    (2-1) * 1     = 1 parameter
# Sprinkler: (2-1) * 2     = 2 parameters
# Rain:      (2-1) * 2     = 2 parameters
# WetGrass:  (2-1) * 2 * 2 = 4 parameters
# Total: 9 parameters  (vs 2^4 - 1 = 15 for full joint)
```

## Classic Bayesian Networks

### The Alarm Network

```
  Burglary    Earthquake
      \          /
       v        v
         Alarm
        /      \
       v        v
  JohnCalls  MaryCalls
```

Burglary and Earthquake are independent causes of Alarm. JohnCalls and MaryCalls are independent given Alarm. Burglary and Earthquake become dependent when Alarm is observed (explaining away).

### The Student Network

```
  Difficulty    Intelligence
      \            /
       v          v
         Grade
           |
           v
        Letter
```

Variables: Difficulty (Easy/Hard), Intelligence (Low/High), Grade (A/B/C), Letter (Weak/Strong). This network illustrates how a ternary variable (Grade) with two binary parents leads to a CPT with $2 \times 2 \times 3 = 12$ entries but only $(3-1) \times 4 = 8$ free parameters.

## Querying Bayesian Networks

### Types of Queries

1. **Prior Query**: $P(X)$ without any evidence
2. **Posterior Query**: $P(X \mid E = e)$ given evidence
3. **MAP Query**: $\arg\max_X P(X \mid E = e)$ — most probable state
4. **MPE Query**: Most probable complete assignment

### Inference by Enumeration

The simplest exact inference method computes:

$$P(X \mid E = e) = \frac{\sum_Y P(X, E = e, Y)}{\sum_{X,Y} P(X, E = e, Y)}$$

where $Y$ are the hidden (non-query, non-evidence) variables.

```python
class InferenceByEnumeration:
    """
    Exact inference by summing over all configurations.
    
    Time complexity: O(d^n) — practical only for small networks.
    """
    
    def __init__(self, bn: BayesianNetwork):
        self.bn = bn
    
    def query(self, query_vars: List[str],
              evidence: Dict[str, int] = None) -> torch.Tensor:
        """Compute P(query_vars | evidence) by enumeration."""
        if evidence is None:
            evidence = {}
        
        from itertools import product
        
        hidden_vars = [
            v for v in self.bn.variables
            if v not in query_vars and v not in evidence
        ]
        
        result_shape = tuple(self.bn.cardinalities[v] for v in query_vars)
        result = torch.zeros(result_shape)
        
        query_cards = [self.bn.cardinalities[v] for v in query_vars]
        hidden_cards = [self.bn.cardinalities[v] for v in hidden_vars]
        
        for query_vals in product(*[range(c) for c in query_cards]):
            query_assignment = dict(zip(query_vars, query_vals))
            prob_sum = 0.0
            
            for hidden_vals in product(*[range(c) for c in hidden_cards]):
                hidden_assignment = dict(zip(hidden_vars, hidden_vals))
                full_assignment = {
                    **query_assignment, **hidden_assignment, **evidence
                }
                prob_sum += self.bn.joint_probability(full_assignment)
            
            result[query_vals] = prob_sum
        
        result = result / result.sum()
        return result
```

### Demonstration: Inference Queries

```python
def demonstrate_inference():
    """Demonstrate inference in the Weather network."""
    bn = build_weather_network()
    inference = InferenceByEnumeration(bn)
    
    # P(Rain)
    p_rain = inference.query(['Rain'])
    print(f"P(Rain=1) = {p_rain[1]:.4f}")
    
    # P(Rain | Cloudy=1)
    p_rain_cloudy = inference.query(['Rain'], {'Cloudy': 1})
    print(f"P(Rain=1 | Cloudy=1) = {p_rain_cloudy[1]:.4f}")
    
    # P(Rain | WetGrass=1) — diagnostic reasoning
    p_rain_wet = inference.query(['Rain'], {'WetGrass': 1})
    print(f"P(Rain=1 | WetGrass=1) = {p_rain_wet[1]:.4f}")
    
    # Explaining away: P(Cloudy | WetGrass=1) vs P(Cloudy | WetGrass=1, Sprinkler=1)
    p_cloudy_wet = inference.query(['Cloudy'], {'WetGrass': 1})
    p_cloudy_wet_spr = inference.query(['Cloudy'], {'WetGrass': 1, 'Sprinkler': 1})
    print(f"\nP(Cloudy=1 | WetGrass=1) = {p_cloudy_wet[1]:.4f}")
    print(f"P(Cloudy=1 | WetGrass=1, Sprinkler=1) = {p_cloudy_wet_spr[1]:.4f}")
    print("Sprinkler explains the wet grass, reducing belief in Cloudy (and thus Rain).")


demonstrate_inference()
```

## Application: Medical Diagnosis

Medical diagnosis is one of the most successful applications of Bayesian networks. The domain is well-suited because uncertainty is inherent, diseases cause symptoms (natural causal direction), expert knowledge can specify local conditional probabilities, and the reasoning is interpretable.

### The Noisy-OR Model

A powerful simplification for networks where multiple diseases can independently cause a symptom:

$$P(\text{Symptom} = 0 \mid D_1, \ldots, D_n) = (1 - \lambda_0) \prod_{i: D_i = 1} (1 - \lambda_i)$$

This reduces parameters from $O(2^n)$ to $O(n)$ while maintaining the intuitive interpretation that each active disease independently "tries" to cause the symptom.

### Respiratory Diagnosis Network

```python
from itertools import product as cartesian_product


class MedicalBayesianNetwork:
    """A Bayesian Network specialized for medical diagnosis."""
    
    def __init__(self):
        self.risk_factors: List[str] = []
        self.diseases: List[str] = []
        self.symptoms: List[str] = []
        self.cardinalities: Dict[str, int] = {}
        self.cpts: Dict[str, Tuple[List[str], torch.Tensor]] = {}
    
    def add_risk_factor(self, name: str, cardinality: int = 2):
        self.risk_factors.append(name)
        self.cardinalities[name] = cardinality
    
    def add_disease(self, name: str,
                    risk_factor_parents: List[str] = None,
                    base_probability: float = 0.01):
        self.diseases.append(name)
        self.cardinalities[name] = 2
        if risk_factor_parents:
            parents = risk_factor_parents
            shape = tuple(self.cardinalities[p] for p in parents) + (2,)
            cpt = torch.ones(shape) * base_probability
            cpt[..., 0] = 1 - cpt[..., 1]
        else:
            parents = []
            cpt = torch.tensor([1 - base_probability, base_probability])
        self.cpts[name] = (parents, cpt)
    
    def add_symptom(self, name: str, disease_parents: List[str],
                    sensitivity: Dict[str, float] = None,
                    specificity: float = 0.95):
        """Add a symptom node using the Noisy-OR model."""
        self.symptoms.append(name)
        self.cardinalities[name] = 2
        parents = disease_parents
        shape = (2,) * len(disease_parents) + (2,)
        cpt = torch.zeros(shape)
        
        if sensitivity is None:
            sensitivity = {d: 0.8 for d in disease_parents}
        
        for idx in cartesian_product(*[range(2) for _ in disease_parents]):
            disease_states = dict(zip(disease_parents, idx))
            prob_no_symptom = specificity
            for d, state in disease_states.items():
                if state == 1:
                    prob_no_symptom *= (1 - sensitivity[d])
            cpt[idx + (0,)] = prob_no_symptom
            cpt[idx + (1,)] = 1 - prob_no_symptom
        
        self.cpts[name] = (parents, cpt)
    
    def set_cpt(self, variable: str, parents: List[str], 
                values: torch.Tensor):
        self.cpts[variable] = (parents, values)
    
    def compute_joint_probability(self, assignment: Dict[str, int]) -> float:
        prob = 1.0
        for var in self.risk_factors + self.diseases + self.symptoms:
            parents, cpt = self.cpts.get(var, ([], torch.tensor([0.5, 0.5])))
            if parents:
                index = tuple(assignment[p] for p in parents) + (assignment[var],)
            else:
                index = (assignment[var],)
            prob *= cpt[index].item()
        return prob
    
    def differential_diagnosis(self, evidence: Dict[str, int],
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank diseases by posterior probability given evidence."""
        results = {}
        for disease in self.diseases:
            probs = torch.zeros(2)
            hidden_vars = (
                [d for d in self.diseases if d != disease and d not in evidence]
                + [s for s in self.symptoms if s not in evidence]
                + [r for r in self.risk_factors if r not in evidence]
            )
            hidden_cards = [self.cardinalities[v] for v in hidden_vars]
            
            for disease_val in [0, 1]:
                total = 0.0
                for hidden_vals in cartesian_product(
                    *[range(c) for c in hidden_cards]
                ):
                    assignment = dict(evidence)
                    assignment[disease] = disease_val
                    assignment.update(dict(zip(hidden_vars, hidden_vals)))
                    total += self.compute_joint_probability(assignment)
                probs[disease_val] = total
            
            probs = probs / probs.sum()
            results[disease] = probs[1].item()
        
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def demonstrate_medical_diagnosis():
    """Demonstrate medical diagnosis with a Bayesian Network."""
    bn = MedicalBayesianNetwork()
    
    # Risk factors
    bn.add_risk_factor('Smoking', 2)
    bn.add_risk_factor('Age', 3)   # 0=Young, 1=Middle, 2=Old
    bn.cpts['Smoking'] = ([], torch.tensor([0.7, 0.3]))
    bn.cpts['Age'] = ([], torch.tensor([0.3, 0.4, 0.3]))
    
    # Diseases
    bn.add_disease('Cold', base_probability=0.1)
    bn.add_disease('Flu', base_probability=0.05)
    bn.add_disease('Pneumonia', risk_factor_parents=['Age'])
    bn.add_disease('LungCancer', risk_factor_parents=['Smoking', 'Age'])
    bn.add_disease('TB', base_probability=0.001)
    
    bn.set_cpt('Pneumonia', ['Age'], torch.tensor([
        [0.99, 0.01], [0.98, 0.02], [0.95, 0.05]
    ]))
    bn.set_cpt('LungCancer', ['Smoking', 'Age'], torch.tensor([
        [[0.9999, 0.0001], [0.999, 0.001], [0.995, 0.005]],
        [[0.999, 0.001], [0.99, 0.01], [0.95, 0.05]]
    ]))
    
    # Symptoms (Noisy-OR)
    bn.add_symptom('Fever',
                   disease_parents=['Cold', 'Flu', 'Pneumonia', 'TB'],
                   sensitivity={'Cold': 0.3, 'Flu': 0.9, 'Pneumonia': 0.8, 'TB': 0.7})
    bn.add_symptom('Cough',
                   disease_parents=['Cold', 'Flu', 'Pneumonia', 'LungCancer', 'TB'],
                   sensitivity={'Cold': 0.7, 'Flu': 0.8, 'Pneumonia': 0.9,
                               'LungCancer': 0.6, 'TB': 0.8})
    bn.add_symptom('Fatigue',
                   disease_parents=['Flu', 'Pneumonia', 'LungCancer', 'TB'],
                   sensitivity={'Flu': 0.8, 'Pneumonia': 0.7,
                               'LungCancer': 0.8, 'TB': 0.9})
    bn.add_symptom('WeightLoss',
                   disease_parents=['LungCancer', 'TB'],
                   sensitivity={'LungCancer': 0.6, 'TB': 0.7})
    
    # Case 1: Patient with fever and cough
    print("Case 1: Fever + Cough")
    for disease, prob in bn.differential_diagnosis({'Fever': 1, 'Cough': 1}):
        print(f"  {disease}: {prob*100:.2f}%")
    
    # Case 2: Elderly smoker with concerning symptoms
    print("\nCase 2: Age=Old, Smoker, Cough + Fatigue + WeightLoss")
    evidence = {'Age': 2, 'Smoking': 1, 'Cough': 1, 'Fatigue': 1, 'WeightLoss': 1}
    for disease, prob in bn.differential_diagnosis(evidence):
        print(f"  {disease}: {prob*100:.2f}%")


demonstrate_medical_diagnosis()
```

### Real-World Medical Bayesian Networks

| System | Domain | Scale |
|--------|--------|-------|
| QMR-DT | General medicine | ~600 diseases, ~4,000 symptoms |
| PATHFINDER | Lymph-node pathology | ~140 variables |
| HEPAR II | Liver disorders | Expert-constructed |

## Summary

| Concept | Description |
|---------|-------------|
| **Bayesian Network** | DAG + CPTs representing a joint distribution |
| **CPT** | $P(X \mid \text{Parents}(X))$ for each node |
| **Factorization** | $P(X_1,\ldots,X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))$ |
| **Forward Sampling** | Sample in topological order, conditioning on parents |
| **Noisy-OR** | Compact CPT parametrization, $O(n)$ instead of $O(2^n)$ |
| **Inference** | Compute $P(\text{Query} \mid \text{Evidence})$ |

## Quantitative Finance Application

Bayesian networks are used in credit risk modeling where the causal structure mirrors the medical diagnosis pattern: macroeconomic factors (risk factors) influence sector health (diseases), which in turn drives observable indicators like credit spreads and default rates (symptoms). The Noisy-OR model naturally extends to scenarios where multiple risk factors can independently trigger a credit event, and the explaining away effect captures the intuition that observing a specific cause (e.g., sector-wide downturn) reduces the posterior probability of alternative explanations (e.g., firm-specific fraud).

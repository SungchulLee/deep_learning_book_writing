# Structure Learning for Bayesian Networks

## The Structure Learning Problem

Given a dataset $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$ of $N$ independent samples, **structure learning** aims to discover the DAG structure $G$ that best explains the data.

This is fundamentally different from parameter learning:
- **Parameter learning**: Given structure $G$, learn CPT parameters
- **Structure learning**: Discover the graph $G$ itself

## Why Structure Learning is Hard

### Combinatorial Explosion

The number of possible DAGs over $n$ nodes grows super-exponentially:

| Nodes | DAGs |
|-------|------|
| 3 | 25 |
| 5 | 29,281 |
| 10 | $4.2 \times 10^{18}$ |
| 20 | $2.3 \times 10^{72}$ |

### NP-Hardness

Finding the optimal structure (for most scoring functions) is NP-hard. We must use:
- Heuristic search methods
- Constraint-based methods
- Hybrid approaches

## Two Main Approaches

### 1. Constraint-Based Methods

**Idea**: Test conditional independence relationships in data, use them to constrain the structure.

**Key algorithm**: PC (Peter-Clark) algorithm

**Advantages**:
- Theoretically motivated
- Can identify Markov equivalence class
- Polynomial in number of independence tests

**Disadvantages**:
- Sensitive to errors in independence tests
- Requires many tests for dense graphs

### 2. Score-Based Methods

**Idea**: Define a scoring function that measures how well a structure fits the data, then search for high-scoring structures.

**Key algorithm**: Hill climbing with BIC score

**Advantages**:
- More robust to individual test errors
- Can incorporate prior knowledge
- Flexible scoring functions

**Disadvantages**:
- May get stuck in local optima
- Computationally expensive for large networks

## Constraint-Based: The PC Algorithm

### Algorithm Overview

1. **Start** with complete undirected graph
2. **Skeleton learning**: Remove edges based on conditional independence
3. **V-structure identification**: Orient colliders ($X \rightarrow Z \leftarrow Y$)
4. **Edge propagation**: Orient remaining edges to avoid new v-structures or cycles

### Independence Testing

For discrete variables, we typically use the **chi-squared test**:

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

where $O_{ij}$ are observed counts and $E_{ij}$ are expected counts under independence.

```python
import torch
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
import networkx as nx

class IndependenceTest:
    """Statistical tests for conditional independence."""
    
    @staticmethod
    def chi_squared_test(data: pd.DataFrame,
                         var1: str,
                         var2: str,
                         given: List[str] = None,
                         alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Test conditional independence using chi-squared test.
        
        H0: var1 ⊥ var2 | given (independent)
        H1: var1 ⊥̸ var2 | given (dependent)
        
        Args:
            data: DataFrame with observations
            var1, var2: Variables to test
            given: Conditioning variables (empty for marginal independence)
            alpha: Significance level
            
        Returns:
            (is_independent, p_value): Tuple of test result and p-value
        """
        if given is None or len(given) == 0:
            # Marginal independence test
            contingency = pd.crosstab(data[var1], data[var2])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            return p_value > alpha, p_value
        
        # Conditional independence: test for each value of conditioning vars
        p_values = []
        
        for _, group in data.groupby(given):
            if len(group) < 5:  # Skip small groups
                continue
            
            try:
                contingency = pd.crosstab(group[var1], group[var2])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                p_values.append(p_value)
            except:
                continue
        
        if not p_values:
            return True, 1.0  # Not enough data, assume independent
        
        # Conservative: use minimum p-value
        min_p = min(p_values)
        return min_p > alpha, min_p


class PCAlgorithm:
    """
    PC (Peter-Clark) algorithm for structure learning.
    
    Returns a CPDAG (Completed Partially Directed Acyclic Graph)
    representing the Markov equivalence class.
    """
    
    def __init__(self, alpha: float = 0.05, max_cond_size: int = None):
        """
        Initialize PC algorithm.
        
        Args:
            alpha: Significance level for independence tests
            max_cond_size: Maximum conditioning set size (None for unlimited)
        """
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.tester = IndependenceTest()
        self.sep_sets = {}  # Store separating sets
    
    def learn_skeleton(self, data: pd.DataFrame) -> nx.Graph:
        """
        Learn the skeleton (undirected edges) of the network.
        
        Phase 1 of PC algorithm: Remove edges based on conditional independence.
        
        Args:
            data: DataFrame with observations
            
        Returns:
            Undirected graph (skeleton)
        """
        variables = list(data.columns)
        n = len(variables)
        
        # Start with complete graph
        skeleton = nx.Graph()
        skeleton.add_nodes_from(variables)
        for i in range(n):
            for j in range(i+1, n):
                skeleton.add_edge(variables[i], variables[j])
        
        print(f"PC Algorithm: Learning skeleton from {len(data)} samples")
        print(f"Variables: {variables}")
        print(f"Starting with complete graph ({skeleton.number_of_edges()} edges)")
        
        # Test conditional independence with increasing conditioning set size
        cond_size = 0
        max_size = self.max_cond_size or (n - 2)
        
        while cond_size <= max_size:
            print(f"\nTesting with conditioning set size {cond_size}...")
            
            edges_removed = 0
            edges_to_test = list(skeleton.edges())
            
            for (var1, var2) in edges_to_test:
                if not skeleton.has_edge(var1, var2):
                    continue  # Already removed
                
                # Get potential conditioning sets (neighbors)
                neighbors = (set(skeleton.neighbors(var1)) | 
                           set(skeleton.neighbors(var2))) - {var1, var2}
                
                if len(neighbors) < cond_size:
                    continue
                
                # Test all conditioning sets of current size
                found_independence = False
                for cond_set in combinations(neighbors, cond_size):
                    cond_list = list(cond_set)
                    
                    is_indep, p_val = self.tester.chi_squared_test(
                        data, var1, var2, cond_list, self.alpha
                    )
                    
                    if is_indep:
                        # Remove edge and store separating set
                        skeleton.remove_edge(var1, var2)
                        self.sep_sets[(var1, var2)] = set(cond_list)
                        self.sep_sets[(var2, var1)] = set(cond_list)
                        edges_removed += 1
                        
                        print(f"  Removed {var1}--{var2} | {{{', '.join(cond_list)}}} "
                              f"(p={p_val:.4f})")
                        found_independence = True
                        break
                
                if found_independence:
                    continue
            
            if edges_removed == 0 and cond_size > 0:
                break  # No more edges to remove
            
            cond_size += 1
        
        print(f"\nSkeleton has {skeleton.number_of_edges()} edges")
        return skeleton
    
    def orient_v_structures(self, skeleton: nx.Graph) -> nx.DiGraph:
        """
        Orient v-structures (colliders).
        
        Phase 2: For each triple X--Z--Y where X and Y are not adjacent,
        if Z is not in the separating set of X and Y, orient as X→Z←Y.
        
        Args:
            skeleton: Undirected skeleton graph
            
        Returns:
            Partially directed graph (PDAG)
        """
        # Create mixed graph (some edges directed, some not)
        pdag = nx.DiGraph()
        pdag.add_nodes_from(skeleton.nodes())
        
        # Initially add all edges as undirected (both directions)
        for u, v in skeleton.edges():
            pdag.add_edge(u, v)
            pdag.add_edge(v, u)
        
        print("\nOrienting v-structures...")
        
        # Find and orient v-structures
        for z in skeleton.nodes():
            neighbors = list(skeleton.neighbors(z))
            
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    # Check if X and Y are not adjacent
                    if skeleton.has_edge(x, y):
                        continue
                    
                    # Check if Z is in separating set
                    sep_set = self.sep_sets.get((x, y), set())
                    
                    if z not in sep_set:
                        # Orient as X → Z ← Y
                        # Remove the reverse edges
                        if pdag.has_edge(z, x):
                            pdag.remove_edge(z, x)
                        if pdag.has_edge(z, y):
                            pdag.remove_edge(z, y)
                        
                        print(f"  V-structure: {x} → {z} ← {y}")
        
        return pdag
    
    def propagate_orientations(self, pdag: nx.DiGraph) -> nx.DiGraph:
        """
        Propagate edge orientations using Meek's rules.
        
        Phase 3: Apply rules to orient undirected edges while avoiding
        new v-structures or cycles.
        
        Args:
            pdag: Partially directed graph
            
        Returns:
            Completed PDAG (CPDAG)
        """
        print("\nPropagating orientations...")
        
        changed = True
        while changed:
            changed = False
            
            # Find undirected edges (edges in both directions)
            for u, v in list(pdag.edges()):
                if pdag.has_edge(v, u):  # Undirected
                    # Rule 1: If X→Y and Y--Z and X not adjacent to Z, orient Y→Z
                    # Rule 2: If X→Z→Y and X--Y, orient X→Y
                    # Rule 3: If X--Y and X--Z1, X--Z2, Z1→Y, Z2→Y, Z1 not adj Z2, orient X→Y
                    
                    # Apply Rule 1
                    for w in pdag.predecessors(u):
                        if not pdag.has_edge(u, w):  # w→u (directed)
                            if not pdag.has_edge(w, v) and not pdag.has_edge(v, w):
                                # w not adjacent to v, orient u→v
                                if pdag.has_edge(v, u):
                                    pdag.remove_edge(v, u)
                                    print(f"  Rule 1: {u} → {v}")
                                    changed = True
        
        return pdag
    
    def learn_structure(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Learn Bayesian Network structure from data.
        
        Args:
            data: DataFrame with observations
            
        Returns:
            CPDAG representing the Markov equivalence class
        """
        # Phase 1: Learn skeleton
        skeleton = self.learn_skeleton(data)
        
        # Phase 2: Orient v-structures
        pdag = self.orient_v_structures(skeleton)
        
        # Phase 3: Propagate orientations
        cpdag = self.propagate_orientations(pdag)
        
        return cpdag
```

## Score-Based: Hill Climbing with BIC

### The BIC Score

The **Bayesian Information Criterion (BIC)** balances model fit against complexity:

$$\text{BIC}(G, \mathcal{D}) = \log P(\mathcal{D} \mid G, \hat{\theta}_{\text{MLE}}) - \frac{d}{2} \log N$$

where:
- First term: Log-likelihood under MLE parameters
- Second term: Penalty for model complexity
- $d$: Number of free parameters
- $N$: Number of samples

### Decomposability

BIC is **decomposable**: it can be written as a sum of local scores:

$$\text{BIC}(G, \mathcal{D}) = \sum_{i=1}^{n} \text{BIC}(X_i \mid \text{Pa}_G(X_i))$$

This allows efficient local search—we only need to recompute scores for affected nodes.

### Hill Climbing Algorithm

1. Start with initial graph (empty or random)
2. Evaluate all single-edge operations: add, delete, reverse
3. Apply the operation that most improves the score
4. Repeat until no improvement possible

```python
class ScoreBasedLearning:
    """
    Score-based structure learning using hill climbing.
    
    Uses BIC score for model selection.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with data.
        
        Args:
            data: DataFrame with observations
        """
        self.data = data
        self.variables = list(data.columns)
        self.n_samples = len(data)
        
        # Cache for local scores
        self._score_cache = {}
    
    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Compute local BIC score for a variable given its parents.
        
        BIC_local(X | Pa) = log L(X | Pa) - (d/2) log N
        
        Args:
            variable: The child variable
            parents: List of parent variables
            
        Returns:
            Local BIC score
        """
        cache_key = (variable, tuple(sorted(parents)))
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        
        var_card = self.data[variable].nunique()
        
        if not parents:
            # No parents: just count frequencies
            counts = self.data[variable].value_counts().values
            log_lik = np.sum(counts * np.log(counts / self.n_samples + 1e-10))
            num_params = var_card - 1
        else:
            # Has parents: count for each parent configuration
            log_lik = 0.0
            num_params = 0
            
            for _, group in self.data.groupby(parents):
                counts = group[variable].value_counts().values
                n_local = counts.sum()
                
                if n_local > 0:
                    probs = counts / n_local
                    log_lik += np.sum(counts * np.log(probs + 1e-10))
                
                num_params += var_card - 1
        
        # BIC score
        score = log_lik - (num_params / 2) * np.log(self.n_samples)
        
        self._score_cache[cache_key] = score
        return score
    
    def total_score(self, graph: nx.DiGraph) -> float:
        """Compute total BIC score for a graph."""
        score = 0.0
        for var in self.variables:
            parents = list(graph.predecessors(var))
            score += self.local_score(var, parents)
        return score
    
    def hill_climbing(self, 
                      max_parents: int = 3,
                      max_iterations: int = 100,
                      verbose: bool = True) -> nx.DiGraph:
        """
        Learn structure using hill climbing search.
        
        Args:
            max_parents: Maximum number of parents per node
            max_iterations: Maximum search iterations
            verbose: Whether to print progress
            
        Returns:
            Learned DAG
        """
        # Start with empty graph
        graph = nx.DiGraph()
        graph.add_nodes_from(self.variables)
        
        current_score = self.total_score(graph)
        
        if verbose:
            print(f"\nHill Climbing Structure Learning")
            print(f"Samples: {self.n_samples}, Variables: {len(self.variables)}")
            print(f"Max parents: {max_parents}")
            print(f"Initial score: {current_score:.2f}")
            print("-" * 60)
        
        for iteration in range(max_iterations):
            best_delta = 0
            best_operation = None
            best_graph = None
            
            # Try all possible operations
            
            # 1. Add edge
            for u in self.variables:
                for v in self.variables:
                    if u == v or graph.has_edge(u, v):
                        continue
                    
                    # Check max parents constraint
                    if len(list(graph.predecessors(v))) >= max_parents:
                        continue
                    
                    # Try adding edge
                    new_graph = graph.copy()
                    new_graph.add_edge(u, v)
                    
                    # Check acyclicity
                    if not nx.is_directed_acyclic_graph(new_graph):
                        continue
                    
                    # Compute score change (only affected node)
                    old_score = self.local_score(v, list(graph.predecessors(v)))
                    new_score = self.local_score(v, list(new_graph.predecessors(v)))
                    delta = new_score - old_score
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_operation = ('add', u, v)
                        best_graph = new_graph
            
            # 2. Delete edge
            for u, v in graph.edges():
                new_graph = graph.copy()
                new_graph.remove_edge(u, v)
                
                old_score = self.local_score(v, list(graph.predecessors(v)))
                new_score = self.local_score(v, list(new_graph.predecessors(v)))
                delta = new_score - old_score
                
                if delta > best_delta:
                    best_delta = delta
                    best_operation = ('delete', u, v)
                    best_graph = new_graph
            
            # 3. Reverse edge
            for u, v in graph.edges():
                if len(list(graph.predecessors(u))) >= max_parents:
                    continue
                
                new_graph = graph.copy()
                new_graph.remove_edge(u, v)
                new_graph.add_edge(v, u)
                
                if not nx.is_directed_acyclic_graph(new_graph):
                    continue
                
                # Both nodes affected
                old_score_u = self.local_score(u, list(graph.predecessors(u)))
                old_score_v = self.local_score(v, list(graph.predecessors(v)))
                new_score_u = self.local_score(u, list(new_graph.predecessors(u)))
                new_score_v = self.local_score(v, list(new_graph.predecessors(v)))
                
                delta = (new_score_u + new_score_v) - (old_score_u + old_score_v)
                
                if delta > best_delta:
                    best_delta = delta
                    best_operation = ('reverse', u, v)
                    best_graph = new_graph
            
            # Apply best operation
            if best_operation is None:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
            
            graph = best_graph
            current_score += best_delta
            
            if verbose:
                op, u, v = best_operation
                print(f"Iter {iteration+1}: {op} {u}→{v}, Δ={best_delta:.2f}, "
                      f"Score={current_score:.2f}")
        
        if verbose:
            print(f"\nFinal graph: {graph.number_of_edges()} edges")
            print(f"Final score: {current_score:.2f}")
        
        return graph
```

## Demonstration: Learning from Synthetic Data

```python
def demonstrate_structure_learning():
    """Demonstrate structure learning on synthetic data."""
    
    print("=" * 70)
    print("Structure Learning Demonstration")
    print("=" * 70)
    
    # Generate synthetic data from known structure
    np.random.seed(42)
    n_samples = 1000
    
    # True structure: A → B → C, A → C
    #     A
    #    / \
    #   ↓   ↓
    #   B → C
    
    # Sample A
    A = np.random.binomial(1, 0.5, n_samples)
    
    # Sample B | A
    p_B = 0.3 + 0.4 * A  # P(B=1|A)
    B = np.random.binomial(1, p_B)
    
    # Sample C | A, B
    p_C = 0.2 + 0.3 * A + 0.3 * B
    p_C = np.clip(p_C, 0, 1)
    C = np.random.binomial(1, p_C)
    
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    
    print(f"\nTrue structure: A → B → C, A → C")
    print(f"Generated {n_samples} samples")
    
    # Score-based learning
    print("\n" + "=" * 70)
    print("Score-Based Learning (Hill Climbing with BIC)")
    print("=" * 70)
    
    learner = ScoreBasedLearning(data)
    learned_graph = learner.hill_climbing(max_parents=2, verbose=True)
    
    print(f"\nLearned edges: {list(learned_graph.edges())}")
    
    # Constraint-based learning
    print("\n" + "=" * 70)
    print("Constraint-Based Learning (PC Algorithm)")
    print("=" * 70)
    
    pc = PCAlgorithm(alpha=0.05)
    cpdag = pc.learn_structure(data)
    
    print(f"\nLearned CPDAG edges: {list(cpdag.edges())}")

# Run demonstration
demonstrate_structure_learning()
```

## Challenges in Structure Learning

### 1. Sample Complexity

Structure learning requires many samples:
- Independence tests need sufficient data
- Rare configurations may be unobserved
- Rule of thumb: $N \geq 5 \cdot 2^k$ for $k$ parent configurations

### 2. Markov Equivalence

Multiple DAGs can encode the same conditional independencies. We can only identify the **equivalence class**, not the unique DAG.

**Equivalence**: DAGs $G_1$ and $G_2$ are equivalent iff they have:
- Same skeleton (undirected edges)
- Same v-structures (colliders)

### 3. Hidden Confounders

Unobserved common causes can create spurious dependencies, leading to incorrect edges.

### 4. Local Optima

Score-based methods may get stuck in local optima. Solutions:
- Random restarts
- Simulated annealing
- Genetic algorithms
- MCMC over structures

## Summary

| Approach | Algorithm | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| Constraint-based | PC | Theoretically sound | Sensitive to errors |
| Score-based | Hill Climbing | Robust | Local optima |
| Hybrid | MMHC | Best of both | Complex |

## Key Takeaways

1. **Structure learning is hard**: NP-hard, super-exponential search space
2. **Two paradigms**: Constraint-based (test independence) vs score-based (optimize)
3. **Markov equivalence**: Can only identify equivalence class
4. **Practical considerations**: Need enough data, handle missing values, incorporate domain knowledge
5. **Validation**: Use held-out data, cross-validation, or expert review

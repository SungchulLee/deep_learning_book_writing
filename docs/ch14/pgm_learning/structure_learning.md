# Structure Learning

## The Structure Learning Problem

Given a dataset $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$ of $N$ independent samples, **structure learning** aims to discover the DAG structure $G$ that best explains the data. This is fundamentally harder than parameter learning: instead of filling in numerical values for a given graph, we must search over the space of possible graphs.

## Why Structure Learning Is Hard

### Combinatorial Explosion

The number of possible DAGs over $n$ nodes grows super-exponentially:

| Nodes | DAGs |
|-------|------|
| 3 | 25 |
| 5 | 29,281 |
| 10 | $4.2 \times 10^{18}$ |
| 20 | $2.3 \times 10^{72}$ |

Finding the optimal structure (for most scoring functions) is NP-hard. Practical methods use heuristic search, constraint-based testing, or hybrid approaches.

## Two Main Approaches

### Constraint-Based Methods

**Idea**: Test conditional independence relationships in data, use them to constrain the graph structure.

**Key algorithm**: PC (Peter-Clark) algorithm

**Strengths**: Theoretically motivated, can identify Markov equivalence classes, polynomial in the number of independence tests.

**Weaknesses**: Sensitive to errors in individual independence tests, requires many tests for dense graphs.

### Score-Based Methods

**Idea**: Define a scoring function that measures how well a structure fits the data, then search for high-scoring structures.

**Key algorithm**: Hill climbing with BIC score

**Strengths**: More robust to individual test errors, can incorporate prior knowledge, flexible scoring functions.

**Weaknesses**: May get stuck in local optima, computationally expensive for large networks.

## Constraint-Based: The PC Algorithm

### Algorithm Overview

1. **Start** with complete undirected graph
2. **Skeleton learning**: Remove edges based on conditional independence tests with increasing conditioning set size
3. **V-structure identification**: Orient colliders $X \to Z \leftarrow Y$ where $Z$ is not in the separating set of $X$ and $Y$
4. **Edge propagation**: Orient remaining edges using Meek's rules to avoid new v-structures or cycles

### Independence Testing

For discrete variables, the **chi-squared test** is standard:

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

where $O_{ij}$ are observed counts and $E_{ij}$ are expected counts under independence.

### Implementation

```python
import torch
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import Dict, List, Set, Tuple
from itertools import combinations
import networkx as nx


class IndependenceTest:
    """Statistical tests for conditional independence."""
    
    @staticmethod
    def chi_squared_test(data: pd.DataFrame, var1: str, var2: str,
                         given: List[str] = None,
                         alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Test conditional independence using chi-squared test.
        
        H0: var1 _|_ var2 | given (independent)
        H1: var1 not _|_ var2 | given (dependent)
        
        Returns:
            (is_independent, p_value)
        """
        if given is None or len(given) == 0:
            contingency = pd.crosstab(data[var1], data[var2])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            return p_value > alpha, p_value
        
        # Conditional: test for each value of conditioning variables
        p_values = []
        for _, group in data.groupby(given):
            if len(group) < 5:
                continue
            try:
                contingency = pd.crosstab(group[var1], group[var2])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                p_values.append(p_value)
            except Exception:
                continue
        
        if not p_values:
            return True, 1.0
        
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
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.tester = IndependenceTest()
        self.sep_sets: Dict[Tuple[str, str], Set[str]] = {}
    
    def learn_skeleton(self, data: pd.DataFrame) -> nx.Graph:
        """
        Phase 1: Learn the skeleton by removing edges
        based on conditional independence tests.
        """
        variables = list(data.columns)
        n = len(variables)
        
        # Start with complete graph
        skeleton = nx.Graph()
        skeleton.add_nodes_from(variables)
        for i in range(n):
            for j in range(i + 1, n):
                skeleton.add_edge(variables[i], variables[j])
        
        print(f"PC: Skeleton learning from {len(data)} samples, "
              f"{len(variables)} variables")
        print(f"Starting with {skeleton.number_of_edges()} edges")
        
        cond_size = 0
        max_size = self.max_cond_size or (n - 2)
        
        while cond_size <= max_size:
            edges_removed = 0
            edges_to_test = list(skeleton.edges())
            
            for (var1, var2) in edges_to_test:
                if not skeleton.has_edge(var1, var2):
                    continue
                
                neighbors = (set(skeleton.neighbors(var1))
                           | set(skeleton.neighbors(var2))) - {var1, var2}
                
                if len(neighbors) < cond_size:
                    continue
                
                for cond_set in combinations(neighbors, cond_size):
                    cond_list = list(cond_set)
                    is_indep, p_val = self.tester.chi_squared_test(
                        data, var1, var2, cond_list, self.alpha
                    )
                    
                    if is_indep:
                        skeleton.remove_edge(var1, var2)
                        self.sep_sets[(var1, var2)] = set(cond_list)
                        self.sep_sets[(var2, var1)] = set(cond_list)
                        edges_removed += 1
                        print(f"  Removed {var1}--{var2} | "
                              f"{{{', '.join(cond_list)}}} (p={p_val:.4f})")
                        break
            
            if edges_removed == 0 and cond_size > 0:
                break
            cond_size += 1
        
        print(f"Skeleton: {skeleton.number_of_edges()} edges")
        return skeleton
    
    def orient_v_structures(self, skeleton: nx.Graph) -> nx.DiGraph:
        """
        Phase 2: Orient v-structures (colliders).
        
        For each triple X--Z--Y where X and Y are not adjacent,
        if Z is not in the separating set of X and Y, orient as X->Z<-Y.
        """
        pdag = nx.DiGraph()
        pdag.add_nodes_from(skeleton.nodes())
        
        # Initially all edges are undirected (both directions)
        for u, v in skeleton.edges():
            pdag.add_edge(u, v)
            pdag.add_edge(v, u)
        
        for z in skeleton.nodes():
            neighbors = list(skeleton.neighbors(z))
            for i, x in enumerate(neighbors):
                for y in neighbors[i + 1:]:
                    if skeleton.has_edge(x, y):
                        continue
                    
                    sep_set = self.sep_sets.get((x, y), set())
                    if z not in sep_set:
                        # Orient as X -> Z <- Y
                        if pdag.has_edge(z, x):
                            pdag.remove_edge(z, x)
                        if pdag.has_edge(z, y):
                            pdag.remove_edge(z, y)
                        print(f"  V-structure: {x} -> {z} <- {y}")
        
        return pdag
    
    def propagate_orientations(self, pdag: nx.DiGraph) -> nx.DiGraph:
        """
        Phase 3: Apply Meek's rules to orient remaining edges
        while avoiding new v-structures or cycles.
        """
        changed = True
        while changed:
            changed = False
            for u, v in list(pdag.edges()):
                if pdag.has_edge(v, u):  # Undirected edge
                    # Rule 1: If W->U and U--V and W not adj to V, orient U->V
                    for w in pdag.predecessors(u):
                        if not pdag.has_edge(u, w):  # W->U is directed
                            if not pdag.has_edge(w, v) and not pdag.has_edge(v, w):
                                if pdag.has_edge(v, u):
                                    pdag.remove_edge(v, u)
                                    changed = True
        return pdag
    
    def learn_structure(self, data: pd.DataFrame) -> nx.DiGraph:
        """Learn Bayesian Network structure from data."""
        skeleton = self.learn_skeleton(data)
        pdag = self.orient_v_structures(skeleton)
        cpdag = self.propagate_orientations(pdag)
        return cpdag
```

## Score-Based: Hill Climbing with BIC

### The BIC Score

The **Bayesian Information Criterion (BIC)** balances model fit against complexity:

$$\text{BIC}(G, \mathcal{D}) = \log P(\mathcal{D} \mid G, \hat{\theta}_{\text{MLE}}) - \frac{d}{2} \log N$$

where the first term is the log-likelihood under MLE parameters, $d$ is the number of free parameters, and $N$ is the sample size. The penalty term prevents overfitting by penalizing complex structures.

### Decomposability

BIC is **decomposable** — it can be written as a sum of local scores:

$$\text{BIC}(G) = \sum_{i=1}^{n} \text{BIC}(X_i \mid \text{Pa}_G(X_i))$$

This allows efficient local search: when evaluating edge additions, deletions, or reversals, only the affected variables' local scores need recomputation.

### Hill Climbing Algorithm

1. Start with an initial graph (typically empty)
2. Evaluate all single-edge operations: add, delete, reverse
3. Apply the operation that most improves the total score
4. Repeat until no improvement is possible

```python
class ScoreBasedLearning:
    """Score-based structure learning via hill climbing with BIC."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.variables = list(data.columns)
        self.n_samples = len(data)
        self._score_cache: Dict[Tuple, float] = {}
    
    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Compute local BIC score: log L(X | Pa) - (d/2) log N.
        """
        cache_key = (variable, tuple(sorted(parents)))
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        
        var_card = self.data[variable].nunique()
        
        if not parents:
            counts = self.data[variable].value_counts().values
            log_lik = np.sum(counts * np.log(counts / self.n_samples + 1e-10))
            num_params = var_card - 1
        else:
            log_lik = 0.0
            num_params = 0
            for _, group in self.data.groupby(parents):
                counts = group[variable].value_counts().values
                n_local = counts.sum()
                if n_local > 0:
                    probs = counts / n_local
                    log_lik += np.sum(counts * np.log(probs + 1e-10))
                num_params += var_card - 1
        
        score = log_lik - (num_params / 2) * np.log(self.n_samples)
        self._score_cache[cache_key] = score
        return score
    
    def total_score(self, graph: nx.DiGraph) -> float:
        return sum(
            self.local_score(var, list(graph.predecessors(var)))
            for var in self.variables
        )
    
    def hill_climbing(self, max_parents: int = 3,
                      max_iterations: int = 100,
                      verbose: bool = True) -> nx.DiGraph:
        """Learn structure via greedy hill climbing."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.variables)
        current_score = self.total_score(graph)
        
        if verbose:
            print(f"Hill Climbing: {self.n_samples} samples, "
                  f"{len(self.variables)} variables")
            print(f"Initial score: {current_score:.2f}")
        
        for iteration in range(max_iterations):
            best_delta = 0
            best_op = None
            best_graph = None
            
            # Try adding edges
            for u in self.variables:
                for v in self.variables:
                    if u == v or graph.has_edge(u, v):
                        continue
                    if len(list(graph.predecessors(v))) >= max_parents:
                        continue
                    
                    new_graph = graph.copy()
                    new_graph.add_edge(u, v)
                    if not nx.is_directed_acyclic_graph(new_graph):
                        continue
                    
                    old_s = self.local_score(v, list(graph.predecessors(v)))
                    new_s = self.local_score(v, list(new_graph.predecessors(v)))
                    delta = new_s - old_s
                    
                    if delta > best_delta:
                        best_delta, best_op, best_graph = delta, ('add', u, v), new_graph
            
            # Try deleting edges
            for u, v in graph.edges():
                new_graph = graph.copy()
                new_graph.remove_edge(u, v)
                
                old_s = self.local_score(v, list(graph.predecessors(v)))
                new_s = self.local_score(v, list(new_graph.predecessors(v)))
                delta = new_s - old_s
                
                if delta > best_delta:
                    best_delta, best_op, best_graph = delta, ('del', u, v), new_graph
            
            # Try reversing edges
            for u, v in graph.edges():
                if len(list(graph.predecessors(u))) >= max_parents:
                    continue
                new_graph = graph.copy()
                new_graph.remove_edge(u, v)
                new_graph.add_edge(v, u)
                if not nx.is_directed_acyclic_graph(new_graph):
                    continue
                
                old_su = self.local_score(u, list(graph.predecessors(u)))
                old_sv = self.local_score(v, list(graph.predecessors(v)))
                new_su = self.local_score(u, list(new_graph.predecessors(u)))
                new_sv = self.local_score(v, list(new_graph.predecessors(v)))
                delta = (new_su + new_sv) - (old_su + old_sv)
                
                if delta > best_delta:
                    best_delta, best_op, best_graph = delta, ('rev', u, v), new_graph
            
            if best_op is None:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            graph = best_graph
            current_score += best_delta
            
            if verbose:
                op, u, v = best_op
                print(f"  Iter {iteration + 1}: {op} {u}->{v}, "
                      f"delta={best_delta:.2f}, score={current_score:.2f}")
        
        return graph
```

## Demonstration

```python
def demonstrate_structure_learning():
    """Demonstrate both constraint-based and score-based learning."""
    np.random.seed(42)
    n_samples = 1000
    
    # True structure: A -> B -> C, A -> C
    A = np.random.binomial(1, 0.5, n_samples)
    p_B = 0.3 + 0.4 * A
    B = np.random.binomial(1, p_B)
    p_C = np.clip(0.2 + 0.3 * A + 0.3 * B, 0, 1)
    C = np.random.binomial(1, p_C)
    
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    print(f"True structure: A -> B -> C, A -> C")
    print(f"Samples: {n_samples}\n")
    
    # Score-based
    print("=" * 60)
    print("Score-Based Learning (Hill Climbing + BIC)")
    print("=" * 60)
    learner = ScoreBasedLearning(data)
    learned = learner.hill_climbing(max_parents=2)
    print(f"Learned edges: {list(learned.edges())}")
    
    # Constraint-based
    print("\n" + "=" * 60)
    print("Constraint-Based Learning (PC Algorithm)")
    print("=" * 60)
    pc = PCAlgorithm(alpha=0.05)
    cpdag = pc.learn_structure(data)
    print(f"Learned CPDAG edges: {list(cpdag.edges())}")


demonstrate_structure_learning()
```

## Challenges in Structure Learning

### Markov Equivalence

Multiple DAGs can encode the same conditional independencies. Structure learning can only identify the **Markov equivalence class**, not the unique DAG.

**Equivalence criterion**: Two DAGs are Markov equivalent if and only if they have the same skeleton and the same v-structures.

For example, $A \to B \to C$ and $A \leftarrow B \leftarrow C$ are equivalent (same skeleton, no v-structures), but $A \to B \leftarrow C$ is in a different class (it has a v-structure at $B$).

### Sample Complexity

Structure learning requires sufficient data. Independence tests need enough observations per conditioning configuration, and rare configurations may be unobserved. A rule of thumb: $N \geq 5 \cdot 2^k$ for $k$ parent configurations.

### Hidden Confounders

Unobserved common causes can create spurious dependencies, leading to incorrect edges. Causal discovery methods like FCI (Fast Causal Inference) extend the PC algorithm to handle latent confounders.

### Local Optima

Score-based methods may converge to local optima. Remedies include random restarts, simulated annealing, tabu search, and MCMC over graph structures.

## Summary

| Approach | Algorithm | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| Constraint-based | PC | Theoretically sound, identifies equivalence class | Sensitive to test errors |
| Score-based | Hill Climbing + BIC | Robust, flexible | Local optima |
| Hybrid | MMHC | Best of both | Complex implementation |

## Key Takeaways

1. **Structure learning is NP-hard**: The search space grows super-exponentially.
2. **Two paradigms**: Constraint-based (test independence) vs score-based (optimize a score).
3. **Markov equivalence**: We can only identify the equivalence class, not the unique DAG.
4. **BIC is decomposable**: Enables efficient local search via edge operations.
5. **Domain knowledge helps**: Constraining the search space with expert knowledge dramatically improves results.

## Quantitative Finance Application

Structure learning from financial time series enables data-driven discovery of lead-lag relationships and common drivers across assets. Applying the PC algorithm to daily returns of sector ETFs can reveal which sectors lead or lag others, with v-structures identifying pairs that are marginally dependent but conditionally independent given a third sector. Score-based learning with the BIC score can discover the sparse dependency structure among macro-financial variables (GDP growth, inflation, interest rates, credit spreads), providing an empirical alternative to hand-specified VAR models. The learned structure can then be used for scenario analysis: given an interest rate shock, the graph propagates its effect to dependent variables according to the discovered causal ordering.

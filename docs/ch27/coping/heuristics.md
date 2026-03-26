# Heuristics for NP-Hard Problems

When provable approximation guarantees are unavailable or the approximation ratio is too loose, **heuristic methods** offer practical alternatives. These algorithms sacrifice worst-case guarantees for empirical performance, often finding near-optimal solutions on real-world instances. While heuristics lack formal approximation ratios, they are the workhorse of combinatorial optimization in practice.

## Local Search

**Local search** starts with a feasible solution and iteratively improves it by exploring **neighbors** --- solutions reachable by small modifications.

### Framework

1. Start with an initial feasible solution $s_0$.
2. Define a **neighborhood** $N(s)$ for each solution $s$.
3. While $N(s)$ contains an improving neighbor:
    - Move to the best (or any) improving neighbor.
4. Return $s$ (a **local optimum**).

### Neighborhood Design

The choice of neighborhood determines the algorithm's quality:

| Problem | Neighborhood | Move |
|---------|-------------|------|
| TSP | 2-opt | Swap two edges in the tour |
| TSP | 3-opt | Replace three edges |
| Graph Partition | Swap | Exchange one vertex between parts |
| SAT | Flip | Flip one variable |

Larger neighborhoods explore more solutions per step but cost more to search.

### Limitations

Local search can get stuck at **local optima** far from the global optimum. The solution landscape may have many local optima separated by deep valleys.

## Simulated Annealing

**Simulated annealing (SA)** escapes local optima by accepting worse solutions with a probability that decreases over time, mimicking the physical process of metal cooling.

### Algorithm

1. Start with solution $s$ and initial temperature $T_0$.
2. At each step, generate a random neighbor $s'$ from $N(s)$.
3. If $s'$ is better, accept it.
4. If $s'$ is worse, accept it with probability:

$$
\Pr[\text{accept}] = \exp\left(-\frac{\Delta f}{T}\right)
$$

where $\Delta f = f(s') - f(s) > 0$ is the cost increase.

5. Decrease $T$ according to a **cooling schedule** (e.g., $T_{k+1} = \alpha T_k$ with $\alpha \approx 0.95$).
6. Stop when $T$ drops below a threshold or time runs out.

### Convergence

Under a sufficiently slow cooling schedule ($T_k = c / \log k$), SA converges to the global optimum with probability 1. However, this theoretical schedule is impractically slow --- it requires exponential time.

In practice, geometric cooling ($T_{k+1} = \alpha T_k$) with $\alpha \in [0.9, 0.99]$ works well.

## Genetic Algorithms

**Genetic algorithms (GAs)** maintain a **population** of solutions and evolve them through selection, crossover, and mutation, inspired by biological evolution.

### Components

1. **Encoding:** Represent each solution as a chromosome (e.g., a bitstring for subset problems, a permutation for TSP).
2. **Fitness function:** Evaluate solution quality.
3. **Selection:** Choose parents proportional to fitness (tournament, roulette wheel).
4. **Crossover:** Combine two parents to produce offspring (e.g., one-point crossover, order crossover for permutations).
5. **Mutation:** Randomly modify offspring with small probability (e.g., bit flip, swap two positions).
6. **Replacement:** Form the next generation from offspring and (possibly) surviving parents.

### Advantages and Limitations

- **Advantages:** Explores diverse regions of the search space simultaneously; works well when the fitness landscape has multiple basins.
- **Limitations:** Many hyperparameters to tune (population size, mutation rate, crossover type); no convergence guarantees; often slow compared to tailored heuristics.

## Tabu Search

**Tabu search** enhances local search by maintaining a **tabu list** of recently visited solutions (or moves), preventing the algorithm from cycling back.

### Key Ideas

1. At each step, move to the best neighbor even if it worsens the objective.
2. Maintain a list of recent moves (length $\ell$); these moves are **tabu** (forbidden).
3. **Aspiration criterion:** Override the tabu status if a move leads to a solution better than the best known.

The tabu list prevents cycling and forces exploration beyond local optima.

## Comparison of Heuristic Methods

| Method | Escapes Local Optima | Memory | Parameters | Best For |
|--------|---------------------|--------|-----------|---------|
| Local Search | No | $O(1)$ | Neighborhood choice | Quick baseline |
| Simulated Annealing | Yes (probabilistic) | $O(1)$ | Temperature schedule | Continuous landscapes |
| Genetic Algorithm | Yes (population) | $O(\text{pop})$ | Many | Diverse landscapes |
| Tabu Search | Yes (memory) | $O(\ell)$ | Tabu tenure $\ell$ | Discrete optimization |

## When to Use Heuristics

!!! warning "No Guarantees"
    Heuristics provide no worst-case approximation guarantees. Use them when:

    - The problem has no known constant-factor approximation.
    - Instance-specific performance matters more than worst-case bounds.
    - The search space has exploitable structure (smoothness, decomposability).
    - Exact or approximation algorithms are too slow for the required instance size.

??? example "Example: 2-Opt for TSP"
    **Instance:** 5 cities with an initial tour $A \to B \to C \to D \to E \to A$ of cost 25.

    **2-opt neighborhood:** Remove two edges and reconnect the tour. For example, remove $(B,C)$ and $(D,E)$, reconnect as $A \to B \to D \to C \to E \to A$.

    **Iteration 1:** The reconnection reduces cost to 22. Accept.

    **Iteration 2:** Try all 2-opt swaps on the new tour. The best swap reduces cost to 20. Accept.

    **Iteration 3:** No improving 2-opt swap exists. Return tour of cost 20.

    The 2-opt local optimum may not be globally optimal, but it is typically within a few percent of OPT on real instances.

## Reference

- Aarts, E., & Lenstra, J. K. (2003). *Local Search in Combinatorial Optimization*. Princeton University Press.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
- Talbi, E.-G. (2009). *Metaheuristics: From Design to Implementation*. Wiley.

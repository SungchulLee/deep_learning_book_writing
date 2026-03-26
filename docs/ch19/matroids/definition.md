# Matroids

Greedy algorithms work optimally for some problems (minimum spanning tree, interval scheduling) but fail for others (traveling salesman, general knapsack). What structural property distinguishes these cases? A **matroid** is an abstract combinatorial structure that captures exactly when a greedy algorithm is guaranteed to find an optimal solution. Whenever a problem has matroid structure, the simple strategy of always taking the best available element produces the global optimum.

## Formal Definition

A **matroid** is a pair $M = (S, \mathcal{I})$ where $S$ is a finite ground set and $\mathcal{I} \subseteq 2^S$ is a family of subsets (called **independent sets**) satisfying three axioms:

**Axiom 1 (Non-emptiness).** $\emptyset \in \mathcal{I}$.

**Axiom 2 (Hereditary property).** If $B \in \mathcal{I}$ and $A \subseteq B$, then $A \in \mathcal{I}$. Every subset of an independent set is independent.

**Axiom 3 (Exchange property).** If $A, B \in \mathcal{I}$ and $|A| < |B|$, then there exists $x \in B \setminus A$ such that $A \cup \{x\} \in \mathcal{I}$.

The exchange property is the crucial axiom. It guarantees that all **maximal** independent sets (called **bases**) have the same cardinality, just as all bases of a vector space have the same dimension.

## Terminology

- **Independent set.** A member of $\mathcal{I}$.
- **Dependent set.** A subset of $S$ not in $\mathcal{I}$.
- **Circuit.** A minimal dependent set (removing any element makes it independent).
- **Base.** A maximal independent set.
- **Rank.** The rank of a set $A \subseteq S$ is the size of the largest independent subset of $A$: $r(A) = \max\{|B| : B \subseteq A,\; B \in \mathcal{I}\}$.

## Examples

### Uniform Matroid

$U_{k,n} = (S, \mathcal{I})$ where $|S| = n$ and $\mathcal{I} = \{A \subseteq S : |A| \le k\}$. Every subset of size at most $k$ is independent. The bases are all subsets of size exactly $k$.

### Linear (Vector) Matroid

Let $S$ be a set of vectors in $\mathbb{R}^d$. Define $\mathcal{I}$ as the collection of linearly independent subsets of $S$. The exchange property follows from the Steinitz exchange lemma in linear algebra.

### Graphic Matroid

Given a graph $G = (V, E)$, let $S = E$ and $\mathcal{I} = \{F \subseteq E : F \text{ is acyclic}\}$. The independent sets are forests, the bases are spanning trees, and the circuits are simple cycles. This matroid underlies the correctness of Kruskal's algorithm.

### Partition Matroid

Let $S = S_1 \cup S_2 \cup \cdots \cup S_k$ be a partition. Given bounds $b_1, \dots, b_k$, define $\mathcal{I} = \{A \subseteq S : |A \cap S_i| \le b_i \text{ for all } i\}$.

## Key Properties

!!! note "All Bases Have Equal Size"
    In any matroid, all bases have the same cardinality. This follows directly from the exchange property: if bases $B_1$ and $B_2$ had different sizes, the smaller one could be extended, contradicting maximality.

!!! note "Matroid Duality"
    Given matroid $M = (S, \mathcal{I})$, the **dual matroid** $M^* = (S, \mathcal{I}^*)$ where $B^*$ is a base of $M^*$ if and only if $S \setminus B^*$ is a base of $M$. The dual of a graphic matroid is called a **cographic matroid**.

## Verification

```python
"""
Matroid axiom verification.

Checks whether a given family of sets satisfies the three matroid
axioms: non-emptiness, hereditary property, and exchange property.
"""

from itertools import combinations

# === Matroid Checker ===

def is_matroid(ground_set: set, independent: list[frozenset]) -> bool:
    """Check if (ground_set, independent) forms a matroid.

    Args:
        ground_set: The finite ground set S.
        independent: List of independent sets (as frozensets).

    Returns:
        True if the three matroid axioms are satisfied.
    """
    ind_set = set(independent)

    # Axiom 1: Non-emptiness
    if frozenset() not in ind_set:
        print("Fails Axiom 1: empty set not independent")
        return False

    # Axiom 2: Hereditary property
    for s in independent:
        for size in range(len(s)):
            for subset in combinations(s, size):
                if frozenset(subset) not in ind_set:
                    print(f"Fails Axiom 2: {set(subset)} not independent")
                    return False

    # Axiom 3: Exchange property
    for a in independent:
        for b in independent:
            if len(a) < len(b):
                found = False
                for x in b - a:
                    if frozenset(a | {x}) in ind_set:
                        found = True
                        break
                if not found:
                    print(f"Fails Axiom 3: {set(a)}, {set(b)}")
                    return False

    return True


# === Demonstration ===

if __name__ == "__main__":
    # Uniform matroid U_{2,3}
    S = {1, 2, 3}
    I = [frozenset(), frozenset({1}), frozenset({2}), frozenset({3}),
         frozenset({1,2}), frozenset({1,3}), frozenset({2,3})]
    print(f"U(2,3) is matroid: {is_matroid(S, I)}")

    # NOT a matroid: {1,2} and {3,4} independent but not {1,3}
    S2 = {1, 2, 3, 4}
    I2 = [frozenset(), frozenset({1}), frozenset({2}), frozenset({3}),
          frozenset({4}), frozenset({1,2}), frozenset({3,4})]
    print(f"Non-matroid check: {is_matroid(S2, I2)}")
```

**Output:**

```
U(2,3) is matroid: True
Fails Axiom 3: {1}, {3, 4}
Non-matroid check: False
```

The uniform matroid $U_{2,3}$ satisfies all three axioms. The second example fails the exchange property: $\{1\}$ and $\{3,4\}$ are independent with $|\{1\}| < |\{3,4\}|$, but neither $\{1,3\}$ nor $\{1,4\}$ is independent.

## Reference

- Whitney, H. (1935). On the abstract properties of linear dependence. *American Journal of Mathematics*, 57(3), 509--533.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 16: Greedy Algorithms.
- Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.

# Expected Search Time

The central promise of hash tables is $O(1)$ expected-time lookup. This section makes that promise precise by deriving the expected number of elements examined during a search operation in a hash table with chaining, under the simple uniform hashing assumption (SUHA). The analysis distinguishes between **unsuccessful search** (the key is not in the table) and **successful search** (the key is present), as these cases have different costs.

## Setup and Assumptions

Consider a hash table with $m$ slots and $n$ stored keys, using separate chaining for collision resolution. Each slot $j$ contains a linked list (chain) of all keys $k$ with $h(k) = j$.

**Simple Uniform Hashing Assumption (SUHA).** Each key is equally likely to hash to any of the $m$ slots, independently of all other keys:

$$
\Pr[h(k) = j] = \frac{1}{m} \quad \text{for all } k,\ j \in \{0, 1, \ldots, m-1\}
$$

Under SUHA, the expected length of each chain is the load factor:

$$
\alpha = \frac{n}{m}
$$

## Unsuccessful Search

An unsuccessful search for a key $k \notin T$ (where $T$ is the set of stored keys) computes $h(k)$ and traverses the entire chain at slot $h(k)$, since $k$ is not found until the end of the chain.

**Theorem.** Under SUHA, the expected time of an unsuccessful search is:

$$
\Theta(1 + \alpha)
$$

*Proof.* The cost consists of two parts:

1. Computing $h(k)$: $O(1)$ time.
2. Traversing the chain at slot $h(k)$: the expected chain length is $\alpha$.

The expected number of elements examined is exactly $\alpha$, since under SUHA each of the $n$ keys lands in slot $h(k)$ with probability $1/m$, and by linearity of expectation:

$$
\mathbb{E}[\text{chain length at slot } h(k)] = \sum_{i=1}^{n} \Pr[h(k_i) = h(k)] = n \cdot \frac{1}{m} = \alpha
$$

Adding the $O(1)$ cost of computing the hash gives $\Theta(1 + \alpha)$. The $1 +$ term ensures the bound is $\Theta(1)$ when $\alpha = 0$ (empty table). $\square$

## Successful Search

A successful search for a key $k \in T$ computes $h(k)$ and traverses the chain at slot $h(k)$ until $k$ is found. On average, $k$ is not at the end of the chain, so fewer elements are examined than in an unsuccessful search.

**Theorem.** Under SUHA, the expected time of a successful search is:

$$
\Theta\!\left(1 + \frac{\alpha}{2}\right)
$$

More precisely, the expected number of elements examined (including the target) is:

$$
1 + \frac{\alpha}{2} - \frac{1}{2m} = 1 + \frac{n-1}{2m}
$$

*Proof.* Assume keys are inserted in order $k_1, k_2, \ldots, k_n$ and each new key is appended to the end of its chain. When searching for $k_i$ (the $i$-th key inserted), we must traverse all keys in the same chain that were inserted **after** $k_i$, plus $k_i$ itself.

For a key $k_j$ inserted after $k_i$ (i.e., $j > i$), the probability that $k_j$ is in the same chain as $k_i$ is $1/m$ under SUHA. Define the indicator variable:

$$
X_{ij} = \mathbf{1}[h(k_j) = h(k_i)]
$$

The expected number of elements examined when searching for $k_i$ is:

$$
1 + \sum_{j=i+1}^{n} \mathbb{E}[X_{ij}] = 1 + \frac{n - i}{m}
$$

Averaging over all $n$ keys (each equally likely to be searched):

$$
\frac{1}{n} \sum_{i=1}^{n} \left(1 + \frac{n - i}{m}\right) = 1 + \frac{1}{nm} \sum_{i=1}^{n} (n - i)
$$

The summation evaluates to:

$$
\sum_{i=1}^{n} (n - i) = \sum_{j=0}^{n-1} j = \frac{n(n-1)}{2}
$$

Substituting:

$$
1 + \frac{1}{nm} \cdot \frac{n(n-1)}{2} = 1 + \frac{n-1}{2m} = 1 + \frac{\alpha}{2} - \frac{1}{2m}
$$

For large $n$, this is $\Theta(1 + \alpha/2)$. $\square$

## Interpretation

The factor of $1/2$ difference between successful and unsuccessful search is intuitive: an unsuccessful search must examine every element in the chain (average length $\alpha$), while a successful search stops, on average, halfway through the chain (average cost $\alpha/2$).

When the load factor $\alpha$ is bounded by a constant (e.g., $\alpha \leq 0.75$), both search types run in $O(1)$ expected time:

$$
\Theta(1 + \alpha) = \Theta(1 + 0.75) = \Theta(1)
$$

This is the fundamental result that justifies the $O(1)$ expected-time claim for hash tables.

## Expected Time for Open Addressing

For hash tables using open addressing (no chaining), the expected probe counts under uniform hashing are:

**Unsuccessful search:**

$$
\mathbb{E}[\text{probes}] \leq \frac{1}{1 - \alpha}
$$

**Successful search:**

$$
\mathbb{E}[\text{probes}] \leq \frac{1}{\alpha} \ln \frac{1}{1 - \alpha}
$$

These bounds assume the **uniform hashing assumption** (each probe sequence is an independent random permutation of the slots), which is stronger than SUHA. Open addressing degrades more sharply as $\alpha \to 1$ because every probe inspects a slot that is occupied with probability $\alpha$.

??? example "Expected Probes at Various Load Factors"

    | Load factor $\alpha$ | Unsuccessful (chaining) | Successful (chaining) | Unsuccessful (open addressing) | Successful (open addressing) |
    |---|---|---|---|---|
    | 0.50 | 1.50 | 1.25 | 2.00 | 1.39 |
    | 0.75 | 1.75 | 1.38 | 4.00 | 1.85 |
    | 0.90 | 1.90 | 1.45 | 10.00 | 2.56 |
    | 0.99 | 1.99 | 1.50 | 100.00 | 4.65 |

    Chaining degrades linearly with $\alpha$, while open addressing degrades dramatically as $\alpha$ approaches 1. This is why open addressing implementations typically maintain $\alpha \leq 0.75$.

## Conditional Expectation View

The expected search time can also be understood through conditional expectation. Let $L_j$ denote the length of the chain at slot $j$. Then:

$$
\mathbb{E}[\text{cost of unsuccessful search}] = \sum_{j=0}^{m-1} \Pr[h(k) = j] \cdot \mathbb{E}[L_j] = \frac{1}{m} \sum_{j=0}^{m-1} \mathbb{E}[L_j] = \frac{1}{m} \cdot n = \alpha
$$

This decomposition makes explicit that the $O(1)$ guarantee depends on both SUHA (which ensures $\Pr[h(k) = j] = 1/m$) and the uniform distribution of chain lengths (which ensures no slot has a disproportionately long chain).

## Summary

Under the simple uniform hashing assumption, the expected search time in a hash table with chaining is $\Theta(1 + \alpha)$ for unsuccessful search and $\Theta(1 + \alpha/2)$ for successful search, where $\alpha = n/m$ is the load factor. When $\alpha$ is bounded by a constant, both operations take $O(1)$ expected time. Open addressing achieves similar bounds but degrades more sharply as $\alpha$ approaches 1. These results form the theoretical foundation for the practical performance of hash tables.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)

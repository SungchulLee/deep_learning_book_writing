# Decision Tree Model

How fast can sorting possibly be? To answer this question, we need a framework for reasoning about **all possible** comparison-based sorting algorithms simultaneously, not just the ones we know. The **decision tree model** provides exactly this framework. It abstracts any comparison-based sorting algorithm as a binary tree where each internal node represents a comparison and each leaf represents a final permutation of the input. By analyzing the structure of these trees, we can derive lower bounds that apply to every comparison-based sorting algorithm, known or unknown.

## What Is a Decision Tree

A **decision tree** for sorting $n$ elements is a full binary tree that models the execution of a comparison-based sorting algorithm on all possible inputs of size $n$. The tree has the following structure:

- **Internal nodes.** Each internal node is labeled with a comparison $a_i \leq a_j$ for some indices $i$ and $j$. The left child corresponds to the outcome "yes" ($a_i \leq a_j$), and the right child corresponds to "no" ($a_i > a_j$).

- **Leaves.** Each leaf is labeled with a permutation $\pi$ of $\{1, 2, \ldots, n\}$, indicating the final rearrangement that produces the sorted output.

- **Root-to-leaf paths.** Each path from the root to a leaf represents the sequence of comparisons made by the algorithm on a particular input. The length of this path is the number of comparisons performed.

For a given input, the algorithm starts at the root, evaluates the comparison at each internal node, follows the corresponding branch, and arrives at a leaf that specifies the correct output permutation.

## Example: Sorting Three Elements

Consider sorting $\langle a_1, a_2, a_3 \rangle$ using comparisons. The decision tree for insertion sort on three elements looks like:

```
                    a1 ≤ a2?
                   /        \
                Yes          No
               /              \
          a2 ≤ a3?          a1 ≤ a3?
         /       \          /       \
       Yes       No       Yes       No
       /          \        /          \
   (1,2,3)    a1 ≤ a3?  (2,1,3)   a2 ≤ a3?
               /    \              /      \
             Yes    No           Yes      No
             /        \          /          \
         (1,3,2)  (3,1,2)   (2,3,1)    (3,2,1)
```

This tree has $3! = 6$ leaves, one for each permutation of three elements. The height of the tree is $3$, meaning the worst case requires $3$ comparisons. Some inputs require only $2$ comparisons (e.g., the already-sorted input follows the leftmost path of length $2$).

## Key Properties

### Every Permutation Must Appear

A correct sorting algorithm must handle every possible input ordering. Since there are $n!$ permutations of $n$ distinct elements, the decision tree must have **at least** $n!$ leaves. If any permutation were missing, there would exist an input for which the algorithm produces the wrong output.

### Height Equals Worst-Case Comparisons

The **height** $h$ of the decision tree equals the maximum number of comparisons the algorithm makes on any input. This is the algorithm's worst-case comparison count.

### Leaves Bound the Height

A binary tree of height $h$ has at most $2^h$ leaves. Since the decision tree must have at least $n!$ leaves:

$$
2^h \geq n!
$$

Taking logarithms:

$$
h \geq \log_2(n!)
$$

This inequality is the foundation of the $\Omega(n \log n)$ lower bound, which is proved in detail on the [Proof](proof.md) page.

## Decision Trees for Specific Algorithms

Different sorting algorithms produce different decision trees for the same $n$, but all must satisfy the $n!$ leaf requirement.

### Insertion Sort

Insertion sort's decision tree is a left-skewed tree: when the input is already sorted, the algorithm follows the leftmost path (only $n - 1$ comparisons). When the input is reverse sorted, it follows the longest path ($n(n-1)/2$ comparisons). The tree has height $\Theta(n^2)$ — far above the $\Omega(n \log n)$ lower bound.

### Merge Sort

Merge sort's decision tree is more balanced. Its height is $\Theta(n \log n)$, which matches the lower bound up to constant factors. This means merge sort is **asymptotically optimal** in the comparison model.

### Optimal Sorting Networks

For small $n$, the minimum-height decision tree can be found by exhaustive search. The minimum number of comparisons needed to sort $n$ elements is known for small values:

| $n$ | Minimum comparisons |
|-----|-------------------|
| 2 | 1 |
| 3 | 3 |
| 4 | 5 |
| 5 | 7 |
| 6 | 10 |

For large $n$, the exact minimum is unknown, but it lies between $\lceil \log_2(n!) \rceil$ and the number of comparisons used by the best known algorithms.

## Assumptions of the Model

The decision tree model makes several assumptions that are important to state explicitly:

1. **Comparison-based.** The algorithm can only learn about the relative order of elements through pairwise comparisons. It cannot examine the bits of the keys, compute hash functions, or use arithmetic on the keys.

2. **Deterministic.** Each comparison has a fixed outcome for a given input. Randomized algorithms can be modeled by considering the expected depth of a random root-to-leaf path.

3. **Distinct elements.** The standard lower bound assumes all $n$ elements are distinct. With duplicates, the number of distinct permutations is less than $n!$, so the lower bound is weaker.

!!! warning "Model Limitations"
    The decision tree model does not account for non-comparison operations. Algorithms like counting sort and radix sort bypass the $\Omega(n \log n)$ bound by exploiting the internal structure of keys (e.g., treating them as integers in a fixed range). These algorithms do not fit the decision tree framework because they use operations other than comparisons.

## Connection to Information Theory

The decision tree lower bound has an elegant information-theoretic interpretation. Before sorting, the algorithm has no information about which of the $n!$ permutations is the correct one. Each comparison is a yes/no question that provides at most $1$ bit of information. To distinguish among $n!$ possibilities, the algorithm needs at least $\log_2(n!)$ bits, which requires at least $\log_2(n!)$ comparisons.

This connection to information theory explains why the lower bound is so robust: it does not depend on the specific algorithm, only on the number of possible outputs and the information gained per comparison.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Section 8.1.
- Knuth, D. E. (1997). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley. Section 5.3.

# Randomized Priorities

A [treap](structure.md) assigns each node both a **key** (obeying BST order) and a **priority** (obeying min-heap or max-heap order).  When priorities are chosen uniformly at random, the resulting tree has the same distribution as a **random binary search tree** — a BST built by inserting the keys in a uniformly random permutation.  This connection gives treaps expected $O(\log n)$ height without any explicit balancing rotations, and it is the random priorities that make this possible.

## Why Randomization Works

A BST built by inserting $n$ keys in a random order has expected height $O(\log n)$.  The challenge is that real-world insertion orders are rarely random — sorted or nearly-sorted input produces a degenerate linear tree.  A treap sidesteps this problem by assigning each key a random priority at insertion time.  The tree structure is then determined by the priority ordering, which is random regardless of the key insertion order.

!!! note "Priority determines structure"
    For a fixed set of $n$ key–priority pairs with distinct priorities, there is exactly one treap satisfying both BST order on keys and heap order on priorities.  Since the priorities are random, the resulting treap is equivalent to a random BST, and all structural properties of random BSTs carry over.

## Expected Height

The expected height of a random BST (and hence a treap) with $n$ nodes is:

$$
E[h] = O(\log n)
$$

More precisely, the expected depth of any node $x$ with rank $i$ (i.e., the $i$-th smallest key) is:

$$
E[\text{depth}(x)] = H_i + H_{n - i + 1} - 1
$$

where $H_k = \sum_{j=1}^{k} 1/j$ is the $k$-th harmonic number.  Since $H_k = O(\log k)$, the maximum expected depth over all nodes is at most $2 \ln n + O(1) \approx 1.39 \log_2 n$.

## Proof Sketch: Connection to Random BSTs

Consider $n$ key–priority pairs $(k_1, p_1), \ldots, (k_n, p_n)$ where the priorities are drawn independently from a continuous distribution (ensuring distinct priorities with probability 1).

The node with the **smallest priority** (for a min-heap treap) becomes the root.  Its key partitions the remaining keys into left and right subtrees, exactly as in a random BST where the root is chosen uniformly at random.  The argument applies recursively to each subtree.

Formally, the probability that key $k_j$ becomes the root equals $1/n$, matching the random BST model.  By induction on tree size, the treap and random BST have identical distributions. $\square$

## Priority Distribution Choices

Any continuous probability distribution works:

| Distribution | Common choice? | Notes |
|-------------|----------------|-------|
| Uniform on $[0, 1]$ | Yes | Simple and effective |
| Uniform integers on $[1, M]$ | Yes, for large $M$ | Small probability of ties |
| Exponential | Sometimes | Memoryless property useful in analysis |

!!! warning "Distinct priorities"
    If two nodes receive the same priority, the treap structure is not unique.  Using a sufficiently large integer range (e.g., 64-bit random integers) makes collisions negligibly unlikely.  Alternatively, break ties by key value.

## Expected Operation Costs

Since the treap has the same distribution as a random BST, all expected costs follow from random BST analysis:

| Operation | Expected time |
|-----------|---------------|
| Search | $O(\log n)$ |
| Insert | $O(\log n)$ |
| Delete | $O(\log n)$ |
| [Split / Join](split_merge.md) | $O(\log n)$ |

These are **expected** bounds over the random priority choices, not amortized bounds.  Each individual operation has $O(\log n)$ expected cost regardless of the history of previous operations.

## Comparison with Deterministic Balancing

| Property | Treap | AVL / Red-black |
|----------|-------|-----------------|
| Balance guarantee | Expected $O(\log n)$ | Worst-case $O(\log n)$ |
| Implementation complexity | Simple | Moderate to complex |
| Rotations per insert | Expected $O(1)$ | Worst-case $O(\log n)$ |
| Deterministic? | No (randomized) | Yes |

The expected number of rotations per insertion in a treap is less than 2, making treaps very efficient in practice.  The trade-off is the lack of worst-case guarantees — an adversary who can observe the random priorities could force $O(n)$ operations.  In practice, this is addressed by keeping priorities secret or using cryptographic randomness.

## Reference

- Aragon, C. R., & Seidel, R. (1989). Randomized search trees. *30th IEEE Symposium on Foundations of Computer Science*, 540–545.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Problem 13-4. MIT Press.

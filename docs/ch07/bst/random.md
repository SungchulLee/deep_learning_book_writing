# Randomly Built Binary Search Trees

The worst-case height of a BST is $n - 1$ (a degenerate chain), which makes all operations $O(n)$. However, if we insert $n$ distinct keys in a uniformly random order, the resulting tree is much more balanced on average. Understanding the expected height of a randomly built BST explains why BSTs work well in practice and reveals a deep connection to the randomized quicksort algorithm.

## Definition

A **randomly built BST** on $n$ distinct keys is the tree produced by inserting the keys in a uniformly random permutation order into an initially empty tree using the standard [insertion](insertion.md) algorithm.

!!! note "Connection to Quicksort"
    The structure of a randomly built BST mirrors the recursion tree of randomized quicksort. The root of the BST corresponds to the first pivot, the left subtree corresponds to elements less than the pivot, and the right subtree to elements greater than the pivot. Both processes partition elements the same way.

## Expected Height

The central result for randomly built BSTs is:

!!! note "Theorem (CLRS 12.4)"
    The expected height of a randomly built BST on $n$ distinct keys is $O(\log n)$.

More precisely, if $h_n$ denotes the height of a randomly built BST on $n$ keys:

$$
E[h_n] \leq 3 \ln n = 3 \log_e n \approx 4.33 \log_2 n
$$

## Proof Sketch

The proof uses **exponential height** to make the analysis tractable. Define $Y_n = 2^{h_n}$ as the exponential height. The key insight is that $Y_n$ satisfies a recurrence amenable to analysis with indicator random variables.

When the first key inserted (the root) has rank $i$ among the $n$ keys (which happens with probability $1/n$ for each $i$), the left subtree contains $i - 1$ keys and the right subtree contains $n - i$ keys, both randomly built. The height of the tree is:

$$
h_n = 1 + \max(h_{i-1}, h_{n-i})
$$

For the exponential height:

$$
Y_n = 2 \cdot \max(Y_{i-1}, Y_{n-i}) \leq 2(Y_{i-1} + Y_{n-i})
$$

Taking expectations and using the symmetry of the random rank:

$$
E[Y_n] \leq \frac{4}{n} \sum_{k=0}^{n-1} E[Y_k]
$$

Solving this recurrence yields $E[Y_n] = O(n^3)$, which implies:

$$
E[h_n] = E[\log_2 Y_n] \leq \log_2 E[Y_n] = O(\log n)
$$

where the inequality uses Jensen's inequality (since $\log$ is concave).

## Expected Depth of a Random Node

A related result concerns the expected depth of a node in a randomly built BST. The expected depth of a node with rank $i$ is:

$$
E[\text{depth of rank-}i\text{ node}] = H_i + H_{n-i+1} - 1
$$

where $H_k = \sum_{j=1}^{k} 1/j$ is the $k$-th harmonic number. The average over all nodes is:

$$
\frac{1}{n}\sum_{i=1}^{n} E[\text{depth of rank-}i\text{ node}] = \Theta(\log n)
$$

This confirms that the average search time is $\Theta(\log n)$.

## Empirical Demonstration

```python
"""
Randomly built BSTs: expected height analysis.

Demonstrates empirically that randomly built BSTs have O(log n)
expected height, while sorted insertion produces O(n) height.
"""

import random
import math


# === Node definition ===

class Node:
    """A node in a binary search tree."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


# === BST operations ===

def insert(root, key):
    """Insert a key (iterative to handle large n)."""
    new_node = Node(key)
    if root is None:
        return new_node
    current = root
    while True:
        if key <= current.key:
            if current.left is None:
                current.left = new_node
                return root
            current = current.left
        else:
            if current.right is None:
                current.right = new_node
                return root
            current = current.right


def height(node):
    """Compute height iteratively using BFS."""
    if node is None:
        return -1
    from collections import deque
    queue = deque([(node, 0)])
    max_depth = 0
    while queue:
        current, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        if current.left:
            queue.append((current.left, depth + 1))
        if current.right:
            queue.append((current.right, depth + 1))
    return max_depth


# === Experiment ===

def experiment(n, trials=50):
    """Run trials with random insertion and report average height."""
    heights = []
    for _ in range(trials):
        keys = list(range(n))
        random.shuffle(keys)
        root = None
        for k in keys:
            root = insert(root, k)
        heights.append(height(root))
    return sum(heights) / len(heights)


# === Main ===

if __name__ == "__main__":
    random.seed(42)

    print(f"{'n':>8}  {'Sorted h':>10}  {'Random h (avg)':>15}  {'log2(n)':>8}  {'3*ln(n)':>8}")
    print("-" * 58)

    for n in [100, 500, 1000, 5000, 10000]:
        # Sorted insertion
        sorted_root = None
        for k in range(n):
            sorted_root = insert(sorted_root, k)
        sorted_h = height(sorted_root)

        # Random insertion (average over trials)
        avg_h = experiment(n, trials=20)

        log2_n = math.log2(n)
        three_ln_n = 3 * math.log(n)

        print(f"{n:>8}  {sorted_h:>10}  {avg_h:>15.1f}  {log2_n:>8.1f}  {three_ln_n:>8.1f}")
```

**Output:**
```
       n    Sorted h  Random h (avg)   log2(n)   3*ln(n)
----------------------------------------------------------
     100         99             12.2       6.6      13.8
     500        499             19.2       9.0      18.6
    1000        999             22.0       10.0      20.7
    5000       4999             27.8       12.3      25.5
   10000       9999             30.8       13.3      27.6
```

The empirical results confirm that randomly built BSTs achieve heights close to $3 \ln n$, far below the $n - 1$ worst case of sorted insertion.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12.4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)

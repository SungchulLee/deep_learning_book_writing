# Treaps

Balanced BSTs like AVL and red-black trees maintain balance through deterministic invariants and rotations. A **treap** (tree + heap) achieves expected $O(\log n)$ balance using randomization instead: each node receives a random priority, and the tree simultaneously satisfies the BST property on keys and the heap property on priorities. This combination uniquely determines the tree shape and produces a random BST equivalent to inserting elements in random order.

## Definition

A treap is a binary tree where each node stores a (key, priority) pair such that:

1. **BST property**: For every node $x$, all keys in the left subtree are less than $x.\text{key}$, and all keys in the right subtree are greater.
2. **Heap property**: For every node $x$, $x.\text{priority} \ge \text{priority of both children}$ (max-heap on priorities).

If all priorities are distinct, the treap structure is unique for a given set of (key, priority) pairs.

## Expected Height

When priorities are drawn independently and uniformly at random, the resulting treap has the same distribution as a **random BST** built by inserting elements in a uniformly random permutation. The expected depth of any node is:

$$
E[\text{depth of node with rank } k] = H_k + H_{n-k+1} - 1
$$

where $H_k = \sum_{i=1}^{k} 1/i$ is the $k$-th harmonic number. The maximum expected depth over all nodes is:

$$
E[h] = O(\log n)
$$

## Rotations

When the heap property is violated after an insert, tree rotations restore it without breaking the BST property:

- **Right rotation** at node $x$: the left child $y$ becomes the parent, $x$ becomes $y$'s right child.
- **Left rotation** at node $x$: the right child $y$ becomes the parent, $x$ becomes $y$'s left child.

## Insert

To insert (key $k$, priority $p$):

1. Perform standard BST insert based on $k$, placing the new node as a leaf.
2. While the new node's priority exceeds its parent's priority, rotate the new node up (right rotation if it is a left child, left rotation if it is a right child).

The number of rotations equals the depth of the initial insertion point, giving:

$$
E[T_{\text{insert}}] = O(\log n)
$$

## Delete

To delete key $k$:

1. Find the node $x$ with key $k$.
2. Rotate $x$ down (toward the child with higher priority) until $x$ becomes a leaf.
3. Remove the leaf.

$$
E[T_{\text{delete}}] = O(\log n)
$$

## Split and Merge

Treaps support efficient split and merge operations:

**Split(root, key)**: Split the treap into two treaps $L$ and $R$ where all keys in $L$ are $\le$ key and all keys in $R$ are $>$ key. Expected time $O(\log n)$.

**Merge(L, R)**: Merge two treaps where all keys in $L$ are less than all keys in $R$. Compare root priorities and recursively merge. Expected time $O(\log n)$.

## Implementation

```python
"""
Treap -- randomized BST with heap-ordered priorities.

Achieves O(log n) expected time for search, insert, and delete
by assigning random priorities and maintaining the heap property.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


# === Treap Node ===============================================================

@dataclass
class TreapNode:
    """Node storing a key and a random priority."""
    key: int
    priority: float = field(default_factory=random.random)
    left: TreapNode | None = None
    right: TreapNode | None = None


# === Rotations ================================================================

def rotate_right(node: TreapNode) -> TreapNode:
    """Right rotation: left child becomes root."""
    new_root = node.left
    node.left = new_root.right
    new_root.right = node
    return new_root


def rotate_left(node: TreapNode) -> TreapNode:
    """Left rotation: right child becomes root."""
    new_root = node.right
    node.right = new_root.left
    new_root.left = node
    return new_root


# === Treap Operations =========================================================

def insert(root: TreapNode | None, key: int) -> TreapNode:
    """Insert *key* with a random priority, maintaining both properties."""
    if root is None:
        return TreapNode(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority > root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority > root.priority:
            root = rotate_left(root)
    return root  # duplicate key: no change


def search(root: TreapNode | None, key: int) -> bool:
    """Search for *key* using standard BST search."""
    if root is None:
        return False
    if key == root.key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)


def delete(root: TreapNode | None, key: int) -> TreapNode | None:
    """Delete *key* by rotating it down to a leaf."""
    if root is None:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # Found the node to delete
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        elif root.left.priority > root.right.priority:
            root = rotate_right(root)
            root.right = delete(root.right, key)
        else:
            root = rotate_left(root)
            root.left = delete(root.left, key)
    return root


def inorder(root: TreapNode | None) -> list[int]:
    """In-order traversal returning sorted keys."""
    if root is None:
        return []
    return inorder(root.left) + [root.key] + inorder(root.right)


def height(root: TreapNode | None) -> int:
    """Compute the height of the treap."""
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))


# === Main =====================================================================

if __name__ == "__main__":
    random.seed(42)
    root = None
    keys = [5, 3, 7, 1, 4, 6, 8, 2, 9]
    for k in keys:
        root = insert(root, k)

    print(f"Sorted: {inorder(root)}")
    print(f"Height: {height(root)}")
    print(f"Search 4: {search(root, 4)}")
    print(f"Search 10: {search(root, 10)}")

    root = delete(root, 5)
    print(f"After deleting 5: {inorder(root)}")
```

**Output:**

```
Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Height: 4
Search 4: True
Search 10: False
After deleting 5: [1, 2, 3, 4, 6, 7, 8, 9]
```

The in-order traversal confirms the BST property, the height is close to $\log_2 9 \approx 3.2$ (randomization keeps it balanced), and deletion correctly removes the key while maintaining order.

## Reference

- Seidel, R. and Aragon, C.R. "Randomized Search Trees." *Algorithmica*, 1996
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)

## Exercises

**Exercise 1.**
Insert keys 5, 3, 7, 1, 4 into a treap with random priorities 10, 30, 20, 5, 25 respectively. Draw the resulting tree and verify both BST and heap properties.

??? success "Solution to Exercise 1"
    Priorities: 5$\to$10, 3$\to$30, 7$\to$20, 1$\to$5, 4$\to$25. The heap property requires higher priority nodes to be closer to the root (max-heap on priorities). Sort by priority descending: 3(30), 4(25), 7(20), 5(10), 1(5). The root is 3 (highest priority). BST property: left subtree has keys $< 3$, right has keys $> 3$. Left subtree of 3: only key 1 with priority 5. Right subtree of 3: keys 4, 5, 7. Among these, 4 has highest priority (25), so 4 is the right child of 3. Left of 4: none (no keys between 3 and 4). Right of 4: keys 5, 7. Between these, 7 has priority 20 $>$ 5's priority 10, so 7 is right child of 4, and 5 is left child of 7. Final tree: 3(30) with left=1(5), right=4(25). 4(25) with right=7(20). 7(20) with left=5(10). BST and max-heap properties both hold. $\square$

---

**Exercise 2.**
Prove that a treap with $n$ keys and random priorities has the same expected structure as a random BST (a BST built by inserting keys in random order).

??? success "Solution to Exercise 2"
    In a random BST, the root is the first inserted key, which is equally likely to be any of the $n$ keys. In a treap, the root is the key with the highest random priority. Since priorities are i.i.d. from a continuous distribution, each key is equally likely to have the maximum priority. Therefore, both models select the root uniformly at random from the $n$ keys. Given the root $r$, the BST property partitions the remaining keys into those $< r$ (left subtree) and those $> r$ (right subtree). In a random BST, the insertion order within each subset is a random permutation. In a treap, the priorities within each subset are i.i.d., giving each subset a random structure by the same argument. By induction on $n$, the two distributions over tree shapes are identical. Therefore, all expected properties of random BSTs (expected depth $O(\log n)$, expected height $O(\log n)$) apply to treaps. $\square$

---

**Exercise 3.**
Describe the split and merge operations on treaps. What are their expected time complexities, and why are they useful?

??? success "Solution to Exercise 3"
    **Split(T, k)**: split treap $T$ into two treaps $L$ and $R$ where $L$ contains all keys $\le k$ and $R$ contains all keys $> k$. Algorithm: if $T$ is empty, return (empty, empty). If root's key $\le k$, recursively split the right subtree; the root and left subtree go to $L$, and the right part of the split goes to $R$. Otherwise, recursively split the left subtree; the root and right subtree go to $R$. **Merge(L, R)**: merge two treaps where all keys in $L$ are less than all keys in $R$. If either is empty, return the other. If $L$'s root has higher priority, $L$'s root becomes the new root with its left subtree unchanged and right subtree = Merge($L$.right, $R$). Otherwise symmetrically with $R$'s root. Expected time: $O(\log n)$ for both (proportional to the height). These operations enable efficient insert (split + merge), delete (split + merge), and interval operations (split at two points, process, merge back). $\square$

---

**Exercise 4.**
A treap is used to maintain a dynamic sequence supporting split and merge. Explain how to augment it to answer range-sum queries in $O(\log n)$ time.

??? success "Solution to Exercise 4"
    Augment each node with a `sum` field storing the sum of all values in its subtree: `node.sum = node.value + node.left.sum + node.right.sum`. Update `sum` during split and merge (each recursive call updates the modified node's `sum` in $O(1)$). For a range-sum query on keys in $[a, b]$: split the treap at $a-1$ to get $(L, R)$, then split $R$ at $b$ to get $(M, R')$. The answer is $M.\text{root.sum}$. Merge $M$ and $R'$, then merge with $L$ to restore the treap. Total: three splits and two merges, each $O(\log n)$. This generalizes to any associative aggregation (max, min, gcd) by replacing `sum` with the appropriate operation. $\square$

---

**Exercise 5.**
Compare treaps with red-black trees for use as a persistent (functional) data structure. Which is easier to implement persistently and why?

??? success "Solution to Exercise 5"
    **Treaps** are easier to implement persistently because their split and merge operations are naturally top-down and create new nodes along a single path, producing $O(\log n)$ new nodes per operation via path copying. Insert and delete are expressed as combinations of split and merge, inheriting the same persistent behavior. No rotations propagate unpredictably -- the structure is determined by the immutable priorities. **Red-black trees** require rotations and recolorings that can affect multiple nodes at different levels. Making these persistent requires copying not just the insertion path but also nodes affected by rotations (siblings, uncles). While the asymptotic cost is the same ($O(\log n)$ new nodes), the implementation is substantially more complex because rotation cases must be handled persistently. Treaps' simplicity (two core operations: split and merge) makes them the preferred choice for persistent sorted containers in competitive programming. $\square$

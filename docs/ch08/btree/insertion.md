# B-Tree Insertion and Splitting

Inserting a key into a [B-tree](definition.md) always adds the key to a **leaf node**.  If the leaf is not full (it has fewer than $2t - 1$ keys), the insertion simply places the key in sorted position.  The interesting case arises when the leaf is already full — the algorithm must **split** the node before (or during) insertion to maintain the B-tree properties.  A well-designed insertion procedure splits full nodes proactively on the way down, guaranteeing that the target leaf always has room for the new key.

## Splitting a Full Node

Splitting is the fundamental structural operation in B-tree insertion.  When a node $y$ is full (contains $2t - 1$ keys), it is divided into two nodes around its **median key** $y.key_t$:

1. The first $t - 1$ keys remain in $y$ (the left half).
2. The last $t - 1$ keys move to a new node $z$ (the right half).
3. The median key $y.key_t$ is **pushed up** into $y$'s parent.

After the split, the parent gains one additional key and one additional child pointer.

$$
\underbrace{k_1, \ldots, k_{t-1}}_{\text{stay in } y}, \quad \underbrace{k_t}_{\text{pushed up}}, \quad \underbrace{k_{t+1}, \ldots, k_{2t-1}}_{\text{move to } z}
$$

??? example "Splitting a full node with t = 3"
    A full node with $2t - 1 = 5$ keys:

    ```
    Parent: [..., P, ...]
                  |
    Full child: [A, B, C, D, E]
    ```

    After splitting on the median C:

    ```
    Parent: [..., P, C, ...]
                  |     |
              [A, B]  [D, E]
    ```

    The median C moves up to the parent, and the full child splits into two nodes with 2 keys each.

## Proactive Splitting (Single-Pass Insertion)

The CLRS insertion algorithm uses a **single downward pass** from root to leaf.  As the algorithm descends, it splits any full node it encounters — even if the split is not immediately necessary.  This ensures that when a child needs to receive a pushed-up median key, the parent is guaranteed to have room.

The invariant is:

$$
\text{Every node on the path from root to leaf is non-full when first visited}
$$

**Special case: splitting the root.**  If the root itself is full, a new empty root is created, the old root becomes its only child, and then the old root is split.  This is the only operation that increases the height of a B-tree.

## Insertion Algorithm

Given a B-tree with minimum degree $t$ and a key $k$ to insert:

1. If the root is full, create a new root, make the old root its child, and split the old root.
2. Starting at the (now non-full) root, descend toward the leaf:
      - At each internal node, find the child $c_i$ that should contain $k$.
      - If $c_i$ is full, split it before descending.
3. Insert $k$ into the leaf in sorted position.

Because every node encountered on the path is non-full by the time we process it, the insertion never needs to backtrack.

## Pseudocode

```
B-TREE-INSERT(T, k):
    r = T.root
    if r.n == 2t - 1:                  # root is full
        s = new node                   # s becomes the new root
        T.root = s
        s.leaf = false
        s.n = 0
        s.c[1] = r
        B-TREE-SPLIT-CHILD(s, 1)       # split the old root
        B-TREE-INSERT-NONFULL(s, k)
    else:
        B-TREE-INSERT-NONFULL(r, k)

B-TREE-INSERT-NONFULL(x, k):
    i = x.n
    if x.leaf:
        # Shift keys right and insert k
        while i >= 1 and k < x.key[i]:
            x.key[i+1] = x.key[i]
            i = i - 1
        x.key[i+1] = k
        x.n = x.n + 1
    else:
        # Find the child to descend into
        while i >= 1 and k < x.key[i]:
            i = i - 1
        i = i + 1
        if x.c[i].n == 2t - 1:        # child is full
            B-TREE-SPLIT-CHILD(x, i)
            if k > x.key[i]:
                i = i + 1
        B-TREE-INSERT-NONFULL(x.c[i], k)
```

## Complexity

| Operation | Time | Disk accesses |
|-----------|------|---------------|
| Insert | $O(t \log_t n)$ | $O(\log_t n)$ |
| Split | $O(t)$ | $O(1)$ |

Each level of the tree requires at most one split ($O(t)$ work to copy keys and update pointers) and the tree has $O(\log_t n)$ levels.  The number of disk writes is at most $O(\log_t n)$ because at most one split occurs per level, and each split writes 3 nodes (the two halves and the parent).

!!! tip "Amortized split cost"
    Although the worst case involves a split at every level, this cannot happen frequently.  A node can only be split after $t - 1$ insertions fill it.  Over a sequence of $n$ insertions, the total number of splits is at most $n - 1$, giving an amortized cost of $O(1)$ splits per insertion.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 18.2. MIT Press.
- Bayer, R., & McCreight, E. (1972). Organization and maintenance of large ordered indexes. *Acta Informatica*, 1(3), 173–189.

# Ukkonen's Algorithm

Constructing a suffix tree by inserting all $n+1$ suffixes independently takes $O(n^2)$ time. Ukkonen (1995) devised an **online** algorithm that builds the suffix tree in $O(n)$ time by processing the string one character at a time, from left to right. The algorithm maintains an **implicit suffix tree** for the prefix seen so far and extends it as each new character arrives. Three key optimizations -- suffix links, the active point trick, and the rule-3 skip -- reduce the total work from $O(n^2)$ to $O(n)$. This section presents the algorithm in full detail.

## Implicit Suffix Trees

For a prefix $T[0..i]$ that does not end with the sentinel $\$$, some suffixes may be prefixes of other suffixes. In this case, the suffix tree for $T[0..i]$ is **implicit**: some suffixes end in the middle of an edge rather than at a leaf. Ukkonen's algorithm builds these implicit trees incrementally and the final tree (after appending $\$$) becomes the proper suffix tree.

The **implicit suffix tree** for $T[0..i]$ contains all suffixes of $T[0..i]$, but without the guarantee that each suffix ends at a leaf. It can be obtained from the explicit suffix tree of $T[0..i]\$$ by removing the sentinel, then removing any edges with empty labels, and then removing any internal nodes with only one child.

## Phases and Extensions

The algorithm processes the string in $n + 1$ **phases**, one for each character. In phase $i + 1$ (processing character $T[i]$), the algorithm must ensure that every suffix of $T[0..i]$ is present in the tree.

Each phase consists of up to $i + 1$ **extensions**, one for each suffix $T[j..i]$ where $0 \leq j \leq i$. In extension $j$ of phase $i + 1$, the algorithm ensures that suffix $T[j..i]$ is in the tree.

## The Three Extension Rules

When extending the tree to include suffix $T[j..i]$ (which was previously $T[j..i-1]$, now extended by character $T[i]$), one of three rules applies:

**Rule 1 (Leaf extension)**: If the path for $T[j..i-1]$ ends at a leaf, extend the leaf edge label to include $T[i]$. This happens automatically if edge labels are stored as open-ended intervals $(l, \infty)$, which is the **global end** trick.

**Rule 2 (Branch)**: If the path for $T[j..i-1]$ ends at a non-leaf and no edge continues with character $T[i]$, create a new leaf edge labeled $T[i]$ (and a new internal node if the path ends in the middle of an edge).

**Rule 3 (Do nothing)**: If the path for $T[j..i-1]$ ends at a non-leaf and an edge continues with $T[i]$, the suffix $T[j..i]$ is already implicitly present. Do nothing.

!!! note "Rule 3 and early termination"
    Once Rule 3 applies in extension $j$, it will also apply in all subsequent extensions $j+1, j+2, \ldots, i$ of the same phase. This is because if $T[j..i]$ is already in the tree, then so is every shorter suffix $T[j'..i]$ for $j' > j$. The algorithm can therefore stop the current phase immediately when Rule 3 fires.

## Suffix Links

A **suffix link** connects an internal node with path label $xw$ (where $x$ is a single character and $w$ is a possibly empty string) to the internal node with path label $w$.

$$
\text{suffixLink}(v) = u \quad \text{where } \text{path}(v) = x \cdot \text{path}(u)
$$

Suffix links are crucial for efficiency: after processing extension $j$ (for suffix $T[j..i-1]$), the algorithm follows the suffix link to jump to the node for $T[j+1..i-1]$, avoiding the need to walk from the root for each extension.

!!! tip "Suffix link existence"
    Every internal node in a suffix tree has a suffix link (this can be proven by induction on the construction process). During construction, whenever Rule 2 creates a new internal node, its suffix link is set in the next extension when the corresponding shorter suffix is processed.

## The Active Point

The algorithm maintains an **active point** $(v, e, \ell)$ consisting of:

- **Active node** $v$: the current internal node
- **Active edge** $e$: the first character of the edge being traversed (or null if at a node)
- **Active length** $\ell$: how far along the active edge the algorithm has progressed

The active point tracks where in the tree the next extension should begin, avoiding redundant traversals.

### Active Point Update Rules

After each extension:

- If **Rule 3** fires (character already present): increment the active length by 1. If this walks past the end of the current edge, follow that edge and reset the active point at the new node.
- If **Rule 2** fires at the root: decrement the active length by 1 and update the active edge to $T[j+1]$.
- If **Rule 2** fires at a non-root node: follow the suffix link from the active node (or go to the root if no suffix link exists).

## The Global End Trick

Edge labels to leaves are stored with an open right boundary: $(l, e)$ where $e$ is a global variable that increments with each phase. This means Rule 1 (leaf extensions) happens implicitly for all existing leaves when the global end is incremented, requiring no explicit work.

This reduces the number of explicit extension operations per phase to only those that create new nodes (Rule 2) or find that the suffix is already present (Rule 3).

## Complexity Analysis

**Time complexity**: The total number of Rule 2 extensions across all phases is at most $2n$, because each Rule 2 extension creates a new leaf, and there are at most $n + 1$ leaves. The Rule 3 extensions take $O(1)$ each (due to early termination). Suffix link traversals contribute $O(n)$ total time by an amortized argument. Therefore:

$$
T(n) = O(n)
$$

**Space complexity**: The suffix tree has $O(n)$ nodes and edges, so:

$$
S(n) = O(n)
$$

## Algorithm Summary

```
UKKONEN(T[0..n]):
    Create root node
    global_end = -1
    active_node = root
    active_edge = null
    active_length = 0
    remaining = 0

    for i = 0 to n:
        global_end = i          // implicit Rule 1 for all leaves
        remaining += 1
        last_new_node = null

        while remaining > 0:
            if active_length == 0:
                active_edge = T[i]

            if no edge from active_node starts with active_edge:
                // Rule 2: create new leaf from active_node
                create leaf edge from active_node labeled T[i..global_end]
                if last_new_node != null:
                    last_new_node.suffix_link = active_node
                    last_new_node = null
            else:
                edge = edge from active_node starting with active_edge
                if active_length >= edge.length:
                    // walk down the edge
                    active_node = edge.target
                    active_length -= edge.length
                    active_edge = T[i - active_length] (if active_length > 0)
                    continue

                if T[edge.start + active_length] == T[i]:
                    // Rule 3: character already in tree
                    active_length += 1
                    if last_new_node != null:
                        last_new_node.suffix_link = active_node
                    break

                // Rule 2: split edge and create new leaf
                split edge, creating new internal node
                create leaf edge from new node
                if last_new_node != null:
                    last_new_node.suffix_link = new_node
                last_new_node = new_node

            remaining -= 1
            if active_node == root and active_length > 0:
                active_length -= 1
                active_edge = T[i - remaining + 1]
            elif active_node.suffix_link exists:
                active_node = active_node.suffix_link
            else:
                active_node = root

    return tree
```

## Comparison with Other Construction Algorithms

| Algorithm | Year | Time | Online | Approach |
|-----------|------|------|--------|----------|
| Weiner | 1973 | $O(n)$ | No | Right-to-left, extension links |
| McCreight | 1976 | $O(n)$ | No | Right-to-left, suffix links |
| Ukkonen | 1995 | $O(n)$ | Yes | Left-to-right, implicit trees |

Ukkonen's algorithm is the most commonly taught because its online nature is intuitive: it extends the tree one character at a time. All three achieve the same $O(n)$ complexity.

## Reference

- Ukkonen, E. (1995). *On-line construction of suffix trees*. Algorithmica, 14(3), 249-260.
- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press, Chapter 6.

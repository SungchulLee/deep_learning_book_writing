# Path Copying

When a node in a linked structure changes, only that node and its ancestors need updating -- the rest of the structure can be shared verbatim. **Path copying** exploits this observation by duplicating only the nodes on the root-to-modification path, linking them to the unchanged subtrees of the previous version. This yields a simple, general technique for making any tree-based data structure persistent.

## Key Observation

In a rooted tree, every node has a unique path from the root. Modifying a node at depth $d$ affects only $d + 1$ nodes (the node itself plus its $d$ ancestors). All other nodes are unreachable from the modification point without going through an ancestor, so they can be shared.

$$
\text{Nodes copied per update} = O(d)
$$

For a balanced tree of $n$ nodes, $d = O(\log n)$, giving logarithmic overhead per operation.

## Algorithm

Given a tree rooted at $r_v$ (version $v$) and a modification at node $x$:

1. **Copy the root**: create $r_{v+1}$ with the same key/data as $r_v$.
2. **Walk toward** $x$: at each level, copy the node on the path to $x$ and point the copied parent to the new child. Point all other children to the originals from version $v$.
3. **Apply the modification** at the copied version of $x$.
4. Store $r_{v+1}$ as the root of version $v+1$.

The result is two trees, $r_v$ and $r_{v+1}$, sharing all nodes except those on the copied path.

## Complexity Analysis

For a tree with branching factor $b$ and height $h$:

| Operation | Time | Extra Space |
|---|---|---|
| Modify (insert/delete) | $O(h)$ | $O(h)$ |
| Search | $O(h)$ | $O(1)$ |

For a balanced BST ($h = O(\log n)$):

$$
T_{\text{modify}} = O(\log n), \quad S_{\text{modify}} = O(\log n)
$$

After $M$ total modifications, the total space is:

$$
S_{\text{total}} = O(n + M \log n)
$$

where $n$ is the initial tree size.

!!! tip "Path copying vs fat nodes"
    Path copying has faster reads ($O(\log n)$ standard BST search) but uses more space per update ($O(\log n)$ vs $O(1)$ amortized for fat nodes). It is the preferred technique when read performance matters and the version tree is queried frequently.

## Linked Lists

Path copying also works for singly linked lists. Modifying the $k$-th node requires copying all $k$ nodes from the head to that position, since each predecessor's "next" pointer must change:

$$
T_{\text{modify at position } k} = O(k), \quad S_{\text{modify}} = O(k)
$$

For modifications at the head ($k = 1$), the cost is $O(1)$, which is why cons lists (prepend-only linked lists) are naturally persistent.

## Full Persistence

Path copying directly supports **full persistence** -- modifying any version, not just the latest. Modifying version $v$ at node $x$ copies the root-to-$x$ path in version $v$'s tree, creating a new version that branches from $v$. The version history forms a tree (or DAG) rather than a linear sequence.

## Implementation

```python
"""
Path Copying -- persistent linked list and BST.

Demonstrates path copying on two data structures: a singly linked
list (copy the prefix) and a BST (copy the root-to-node path).
"""

from __future__ import annotations
from dataclasses import dataclass


# === Persistent Linked List ===================================================

@dataclass(frozen=True)
class ListNode:
    """Immutable linked list node."""
    value: int
    next: ListNode | None = None


def list_set(head: ListNode | None, index: int, value: int) -> ListNode | None:
    """Return a new list with position *index* changed to *value*."""
    if head is None:
        raise IndexError("index out of range")
    if index == 0:
        return ListNode(value, head.next)  # copy this node, share the tail
    return ListNode(head.value, list_set(head.next, index - 1, value))


def to_list(head: ListNode | None) -> list[int]:
    """Convert linked list to Python list."""
    result = []
    while head is not None:
        result.append(head.value)
        head = head.next
    return result


# === Persistent BST ===========================================================

@dataclass(frozen=True)
class TreeNode:
    """Immutable BST node."""
    key: int
    left: TreeNode | None = None
    right: TreeNode | None = None


def tree_insert(root: TreeNode | None, key: int) -> TreeNode:
    """Return a new tree with *key* inserted via path copying."""
    if root is None:
        return TreeNode(key)
    if key < root.key:
        return TreeNode(root.key, tree_insert(root.left, key), root.right)
    elif key > root.key:
        return TreeNode(root.key, root.left, tree_insert(root.right, key))
    return root  # duplicate


def tree_inorder(root: TreeNode | None) -> list[int]:
    """In-order traversal."""
    if root is None:
        return []
    return tree_inorder(root.left) + [root.key] + tree_inorder(root.right)


# === Main =====================================================================

if __name__ == "__main__":
    # --- Persistent linked list ---
    v0 = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    v1 = list_set(v0, 2, 99)  # change index 2 from 3 to 99

    print("Linked list path copying:")
    print(f"  v0: {to_list(v0)}")
    print(f"  v1: {to_list(v1)}")
    # Tail sharing check
    print(f"  v0 tail is v1 tail? {v0.next.next.next is v1.next.next.next}")

    # --- Persistent BST ---
    print("\nBST path copying:")
    trees = [None]
    for k in [5, 3, 7, 1, 6]:
        trees.append(tree_insert(trees[-1], k))

    for i, t in enumerate(trees):
        print(f"  v{i}: {tree_inorder(t)}")

    # Sharing: inserting 6 (goes right->left) does not copy left subtree
    print(f"  v5.left is v4.left? {trees[5].left is trees[4].left}")
```

**Output:**

```
Linked list path copying:
  v0: [1, 2, 3, 4]
  v1: [1, 2, 99, 4]
  v0 tail is v1 tail? True

BST path copying:
  v0: []
  v1: [5]
  v2: [3, 5]
  v3: [3, 5, 7]
  v4: [1, 3, 5, 7]
  v5: [1, 3, 5, 6, 7]
  v5.left is v4.left? True
```

The tail-sharing check for the linked list confirms that modifying index 2 copies only nodes at indices 0, 1, and 2, while sharing the tail at index 3. The BST sharing check shows that inserting 6 (which goes into the right subtree) leaves the left subtree physically shared between versions.

## Reference

- Driscoll, J.R., Sarnak, N., Sleator, D.D., and Tarjan, R.E. "Making Data Structures Persistent." *JCSS*, 1989
- Okasaki, C. *Purely Functional Data Structures.* Cambridge University Press, 1998

## Exercises

**Exercise 1.**
Draw the state of a persistent BST (using path copying) after inserting keys 5, 3, 7, 2 sequentially. Show which nodes are shared between versions.

??? success "Solution to Exercise 1"
    Version 0: empty. Version 1: single node [5]. Version 2: insert 3 -- copy root to get [5'], set 5'.left = new node [3]. Version 1's root [5] is unchanged. Version 3: insert 7 -- copy root to get [5''], set 5''.left = [3] (shared from v2), 5''.right = new node [7]. Version 2's root [5'] is unchanged, with its left child [3] shared by both v2 and v3. Version 4: insert 2 -- copy root to [5'''], copy [3] to [3']. Set 5'''.left = [3'], 3'.left = new node [2], 5'''.right = [7] (shared). Shared nodes: [7] is shared by v3 and v4; original [3] is still referenced by v2; [5] by v1. Each version has its own root pointer. Total new nodes per version: 1, 2, 2, 3 (matching path lengths). $\square$

---

**Exercise 2.**
Prove that path copying on a balanced BST with $n$ nodes creates $O(\log n)$ new nodes per update and uses $O(n + m \log n)$ total space after $m$ updates.

??? success "Solution to Exercise 2"
    An update (insert or delete) in a balanced BST modifies nodes along a root-to-leaf path of length $O(\log n)$. Path copying duplicates each node on this path, creating $O(\log n)$ new nodes per update. Unchanged subtrees are shared via pointers, requiring no additional copies. The initial tree has $O(n)$ nodes. Each of the $m$ updates adds $O(\log n)$ nodes. Total: $O(n) + m \cdot O(\log n) = O(n + m \log n)$. This bound is tight: in the worst case, each update modifies a distinct root-to-leaf path of length $\Theta(\log n)$, and no nodes from different updates are shared (though in practice, updates to nearby keys share subtrees, using less space). $\square$

---

**Exercise 3.**
Path copying requires storing a root pointer per version. If there are $m$ versions, what is the overhead of the version table, and how can it be made more efficient for range version queries?

??? success "Solution to Exercise 3"
    The version table stores $m$ root pointers, one per version. If each pointer is 8 bytes, the overhead is $8m$ bytes -- negligible compared to the $O(n + m \log n)$ node storage. For range version queries (e.g., "find the first version where key $k$ exists"), a linear scan of the version table costs $O(m)$. To improve this: (1) store the version table as a sorted array and binary search for the relevant version in $O(\log m)$; (2) build a persistent segment tree over the version indices, enabling range queries in $O(\log m)$; (3) for specific queries like "when was key $k$ first inserted," augment each version's root with metadata and use fractional cascading across versions for $O(\log n + \log m)$ queries. $\square$

---

**Exercise 4.**
Explain why path copying does not work efficiently for data structures with back-pointers (e.g., doubly-linked lists). What alternative persistence technique handles this case?

??? success "Solution to Exercise 4"
    Path copying requires that modifying a node only necessitates updating its ancestors (nodes that point to it). In a tree, each node has exactly one parent, so copying a node requires updating only the parent's pointer -- propagating up the path to the root. In a doubly-linked list, each node has both a `next` and `prev` pointer. Copying a node requires updating both its predecessor's `next` and its successor's `prev`. Updating the predecessor triggers copying it, which requires updating its predecessor, cascading through the entire list. The result is $O(n)$ copies per modification -- no better than full copying. The fat-node technique handles this case: instead of copying nodes, modifications are stored as timestamped logs within each node, achieving $O(1)$ amortized space per modification regardless of pointer structure. $\square$

---

**Exercise 5.**
Design a persistent stack using path copying. What are the time and space complexities for push, pop, and top operations? How many versions can be maintained simultaneously?

??? success "Solution to Exercise 5"
    A stack is a singly-linked list with a top pointer. Push: create a new node pointing to the current top. The new top is the new node; the old version's top pointer is unchanged. No copying needed -- this is structural sharing, equivalent to path copying on a list of depth 1. Pop: the new version's top points to `top.next`. Again, no copying. Top: return `top.value`. All operations are $O(1)$ time and $O(1)$ space (push creates one node; pop and top create zero nodes). The number of simultaneous versions is unlimited -- each version is just a pointer to a node in the shared linked list. With $m$ push operations across all versions, total space is $O(m)$. This is the simplest example of a persistent data structure: a cons-list is inherently persistent because it has no back-pointers and no mutations. $\square$

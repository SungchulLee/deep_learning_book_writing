# Left-Leaning Red-Black Trees

Standard [red-black trees](properties.md) allow red links on either side of a node, leading to many cases during [insertion](insert_fixup.md) and [deletion](delete_fixup.md).  Sedgewick's **left-leaning red-black (LLRB) tree** adds a single additional invariant — red links lean left — that cuts the number of cases roughly in half.  The resulting code is dramatically simpler while maintaining the same $O(\log n)$ worst-case guarantees.

## The Left-Leaning Invariant

An LLRB tree is a red-black tree with one extra rule:

- **No node has a red right child without also having a red left child.**

Equivalently, if a node has exactly one red child, that child must be the left child.  This eliminates the symmetric cases that complicate standard red-black tree operations.

## Correspondence with 2-3 Trees

An LLRB tree is a binary representation of a **2-3 tree**.  Each type of 2-3 tree node maps to a specific LLRB configuration:

| 2-3 tree node | LLRB representation |
|---------------|---------------------|
| 2-node (1 key, 2 children) | Black node with two black children |
| 3-node (2 keys, 3 children) | Black node with a red left child |

Because a 3-node always has its second key as a red left child of the first key, the structure is deterministic — there is exactly one LLRB tree for each 2-3 tree.

!!! note "Why left-leaning simplifies code"
    Standard red-black trees represent 2-3-4 trees, where a 4-node can have two red children.  LLRB trees represent only 2-3 trees, eliminating 4-nodes entirely.  Combined with the left-leaning constraint, each operation has fewer cases to handle.

## Key Operations

### Rotation and Color Flip

Three local transformations maintain the LLRB invariants:

**Left rotation:** converts a right-leaning red link to a left-leaning one.

**Right rotation:** temporarily creates a right-leaning red link (used during insertion to fix consecutive left-leaning red links).

**Color flip:** when both children are red (a temporary 4-node), flip all three colors — the parent becomes red and both children become black.  This splits the 4-node and pushes the middle key up.

### Insertion

LLRB insertion follows a simple recursive pattern:

1. Insert the new key as a **red** node at the leaf level (standard BST insertion).
2. On the way back up the recursion:
      - If the right child is red and the left child is black, **left-rotate**.
      - If the left child is red and the left child's left child is also red, **right-rotate**.
      - If both children are red, **color-flip**.

These three checks, applied in order after each recursive call, restore all LLRB invariants.

```python
"""Left-leaning red-black tree insertion."""

from __future__ import annotations


# === Constants ===

RED = True
BLACK = False


# === Node Definition ===

class Node:
    """LLRB tree node with a color bit."""

    def __init__(self, key: int, color: bool = RED):
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None
        self.color = color


# === Helper Functions ===

def is_red(node: Node | None) -> bool:
    """Return True if the node exists and is red."""
    return node is not None and node.color == RED


def rotate_left(h: Node) -> Node:
    """Rotate a right-leaning red link to lean left."""
    x = h.right
    h.right = x.left
    x.left = h
    x.color = h.color
    h.color = RED
    return x


def rotate_right(h: Node) -> Node:
    """Rotate a left-leaning red link to lean right (temporary)."""
    x = h.left
    h.left = x.right
    x.right = h
    x.color = h.color
    h.color = RED
    return x


def flip_colors(h: Node) -> None:
    """Split a temporary 4-node by flipping colors."""
    h.color = RED
    h.left.color = BLACK
    h.right.color = BLACK


# === Insertion ===

def insert(node: Node | None, key: int) -> Node:
    """Insert a key into the LLRB subtree rooted at *node*."""
    if node is None:
        return Node(key, RED)

    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    # Duplicate keys are ignored

    # Fix-up on the way back up
    if is_red(node.right) and not is_red(node.left):
        node = rotate_left(node)
    if is_red(node.left) and is_red(node.left.left):
        node = rotate_right(node)
    if is_red(node.left) and is_red(node.right):
        flip_colors(node)

    return node


# === Demonstration ===

if __name__ == "__main__":
    root: Node | None = None
    for key in [7, 3, 18, 10, 22, 8, 11, 26]:
        root = insert(root, key)
        root.color = BLACK  # root is always black

    def inorder(node: Node | None) -> list[int]:
        """Collect keys in sorted order."""
        if node is None:
            return []
        return inorder(node.left) + [node.key] + inorder(node.right)

    print(f"Inorder: {inorder(root)}")
    # [3, 7, 8, 10, 11, 18, 22, 26]
```

## Complexity

| Operation | Time |
|-----------|------|
| Search | $O(\log n)$ |
| Insert | $O(\log n)$ |
| Delete | $O(\log n)$ |

The height of an LLRB tree is at most $2 \log_2(n + 1)$, the same bound as a standard red-black tree.  The constant factors in practice are comparable, but the code is significantly shorter.

## LLRB vs Standard Red-Black Trees

| Aspect | Standard RB | LLRB |
|--------|-------------|------|
| Underlying tree | 2-3-4 tree | 2-3 tree |
| Red link direction | Either side | Left only |
| Insertion cases | 3 + 3 symmetric | 3 (no symmetric cases) |
| Deletion complexity | 4 + 4 symmetric cases | Fewer cases |
| Implementation size | ~100 lines | ~40 lines |

## Reference

- Sedgewick, R. (2008). Left-leaning red-black trees. *Dagstuhl Workshop on Data Structures*.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.), Section 3.3. Addison-Wesley.

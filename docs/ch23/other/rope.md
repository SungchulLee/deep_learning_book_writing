# Rope Data Structure

Arrays and linked lists both have $O(n)$ worst-case performance for string concatenation or insertion in the middle. A **rope** is a binary tree where each leaf holds a short string fragment and each internal node stores the total length of its left subtree. This structure supports concatenation, splitting, and insertion in $O(\log n)$ time, making it ideal for text editors and other applications that perform frequent edits on large strings.

## Structure

A rope is a binary tree with the following properties:

- **Leaf nodes** store short string fragments (typically up to some threshold length).
- **Internal nodes** store a `weight` field equal to the total length of all leaves in the left subtree.
- The full string is obtained by an in-order traversal of the leaves.

For the string "Hello_World":

```
        [5]
       /    \
    "Hello"  [1]
            /   \
          "_"  "World"
```

The root's weight is 5 (length of "Hello"), and the right child's weight is 1 (length of "_").

## Operations

### Index — $O(\log n)$

To find character at position $i$:

1. If $i < \text{weight}$, recurse into the left subtree.
2. Otherwise, recurse into the right subtree with $i \leftarrow i - \text{weight}$.
3. At a leaf, return the character at position $i$ within the fragment.

### Concatenation — $O(1)$ or $O(\log n)$

Create a new root with the two ropes as left and right children. The weight is the total length of the left rope. If balancing is required, this takes $O(\log n)$.

### Split — $O(\log n)$

Split a rope at position $i$ into two ropes (characters $0 \ldots i-1$ and $i \ldots n-1$):

1. Navigate to position $i$, splitting nodes along the path.
2. Reassemble the left and right portions.

### Insert — $O(\log n)$

Insert a string at position $i$: split at $i$, concatenate the left part with the new string, then concatenate with the right part.

### Delete — $O(\log n)$

Delete characters from position $i$ to $j$: split at $i$ and at $j$, then concatenate the outer parts.

## Complexity Comparison

| Operation | Array | Rope |
|---|---|---|
| Index | $O(1)$ | $O(\log n)$ |
| Concatenate | $O(n)$ | $O(\log n)$ |
| Split | $O(n)$ | $O(\log n)$ |
| Insert | $O(n)$ | $O(\log n)$ |
| Delete | $O(n)$ | $O(\log n)$ |

## Python Implementation

```python
"""
Rope Data Structure — Binary Tree of String Fragments.

Supports efficient concatenation, splitting, index access,
and insertion for large mutable strings.
"""


# === Rope Node ===

class RopeNode:
    """A node in the rope binary tree."""

    def __init__(
        self, text: str = "",
        left: "RopeNode | None" = None,
        right: "RopeNode | None" = None,
    ):
        if left is None and right is None:
            # Leaf node
            self.text = text
            self.weight = len(text)
            self.left = None
            self.right = None
        else:
            # Internal node
            self.text = ""
            self.left = left
            self.right = right
            self.weight = left.total_length() if left else 0

    def total_length(self) -> int:
        """Return the total string length represented by this subtree."""
        if self.left is None and self.right is None:
            return len(self.text)
        length = self.weight
        if self.right:
            length += self.right.total_length()
        return length

    def index(self, i: int) -> str:
        """Return the character at position i."""
        if self.left is None and self.right is None:
            return self.text[i]
        if i < self.weight:
            return self.left.index(i) if self.left else ""
        return self.right.index(i - self.weight) if self.right else ""

    def to_string(self) -> str:
        """Collect the full string by in-order traversal."""
        if self.left is None and self.right is None:
            return self.text
        result = ""
        if self.left:
            result += self.left.to_string()
        if self.right:
            result += self.right.to_string()
        return result


# === Rope Operations ===

def concatenate(left: RopeNode | None, right: RopeNode | None) -> RopeNode:
    """Concatenate two ropes into a new rope."""
    if left is None:
        return right
    if right is None:
        return left
    return RopeNode(left=left, right=right)


def split(node: RopeNode, i: int) -> tuple[RopeNode | None, RopeNode | None]:
    """Split a rope at position i into (left, right)."""
    if node.left is None and node.right is None:
        # Leaf node
        if i <= 0:
            return None, node
        if i >= len(node.text):
            return node, None
        return RopeNode(node.text[:i]), RopeNode(node.text[i:])

    if i < node.weight:
        left_split, right_split = split(node.left, i) if node.left else (None, None)
        return left_split, concatenate(right_split, node.right)
    elif i > node.weight:
        left_split, right_split = (
            split(node.right, i - node.weight) if node.right else (None, None)
        )
        return concatenate(node.left, left_split), right_split
    else:
        return node.left, node.right


def insert(node: RopeNode, i: int, text: str) -> RopeNode:
    """Insert text at position i."""
    left, right = split(node, i)
    new_leaf = RopeNode(text)
    return concatenate(concatenate(left, new_leaf), right)


def delete(node: RopeNode, i: int, j: int) -> RopeNode:
    """Delete characters from position i to j (exclusive)."""
    left, temp = split(node, i)
    _, right = split(temp, j - i)
    return concatenate(left, right)


# === Main ===

if __name__ == "__main__":
    # Build a rope from fragments
    rope = concatenate(RopeNode("Hello"), RopeNode("_World"))
    print(f"Rope: '{rope.to_string()}'")
    print(f"Length: {rope.total_length()}")
    print(f"Index 6: '{rope.index(6)}'")

    # Insert
    rope = insert(rope, 5, " Beautiful")
    print(f"After insert: '{rope.to_string()}'")

    # Delete
    rope = delete(rope, 5, 15)
    print(f"After delete: '{rope.to_string()}'")
    # Output:
    # Rope: 'Hello_World'
    # Length: 11
    # Index 6: 'W'
    # After insert: 'Hello Beautiful_World'
    # After delete: 'Hello_World'
```

## Reference

- Boehm, H., Atkinson, R., & Plass, M. (1995). Ropes: An alternative to strings. *Software: Practice and Experience*, 25(12), 1315-1330.

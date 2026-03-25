# Radix Trees

In a standard trie, every character of every key occupies its own node. When long keys share few common prefixes, this creates chains of single-child nodes that waste both memory and traversal time. A **radix tree** (also called a **compressed trie** or **Patricia trie**) eliminates this redundancy by merging each chain of single-child nodes into a single edge labeled with the entire substring. The result is a tree whose internal node count never exceeds the number of stored keys, regardless of key length.

## From Tries to Radix Trees

Consider inserting the keys `"romane"`, `"romanus"`, `"romulus"`, `"rubens"`, and `"ruber"` into a standard trie. The path from the root through `r → o → m → a → n` consists of nodes that each have exactly one child — five nodes to represent what a single edge label `"roman"` could capture. A radix tree compresses exactly these single-child chains.

!!! note "Compression Rule"
    An internal node in a radix tree has at least two children (except possibly the root). Whenever a node would have exactly one child, that node is merged with its child, and their edge labels are concatenated.

## Formal Definition

A radix tree over an alphabet $\Sigma$ stores a set $S$ of strings. Each edge carries a label $\ell \in \Sigma^+$ (a non-empty string). For any internal node $v$ (other than a single-key root), $v$ has at least two children. The key associated with any node is the concatenation of edge labels on the path from the root to that node. A node is marked as a **terminal** if the corresponding key belongs to $S$.

**Space complexity.** A radix tree storing $n$ keys has at most $n$ terminal nodes and at most $n - 1$ internal nodes (excluding the root), giving $O(n)$ nodes total — independent of key length.

## Node Structure

Each node in a radix tree stores:

- A dictionary mapping the **first character** of each outgoing edge label to a `(label, child)` pair.
- A boolean flag indicating whether the node represents a complete key.

```python
"""
Radix tree (compressed trie) implementation.

Demonstrates insertion, search, prefix collection, and deletion
on a space-efficient compressed trie structure.
"""


# === Node Definition ===

class RadixNode:
    """A single node in the radix tree."""

    def __init__(self):
        self.children = {}   # first_char -> (label, child_node)
        self.is_terminal = False


# === Radix Tree ===

class RadixTree:
    """Compressed trie that merges single-child chains into edge labels."""

    def __init__(self):
        self.root = RadixNode()

    # --- Insertion ---

    def insert(self, key: str) -> None:
        """Insert a key into the radix tree.

        Walk down the tree following matching edge labels. When a
        mismatch occurs partway through an edge label, split that
        edge at the mismatch point and attach new branches.
        """
        node = self.root
        i = 0  # position in key

        while i < len(key):
            ch = key[i]
            if ch not in node.children:
                # No matching edge — create a new one for the remainder
                node.children[ch] = (key[i:], RadixNode())
                node.children[ch][1].is_terminal = True
                return

            label, child = node.children[ch]
            # Find how much of the edge label matches the key
            j = 0
            while j < len(label) and i + j < len(key) and label[j] == key[i + j]:
                j += 1

            if j == len(label):
                # Full edge match — continue from the child
                node = child
                i += j
            else:
                # Partial match — split the edge at position j
                split_node = RadixNode()
                # Edge from current node to split node: label[:j]
                # Edge from split node to original child: label[j:]
                split_node.children[label[j]] = (label[j:], child)
                node.children[ch] = (label[:j], split_node)

                if i + j < len(key):
                    # Remaining key goes as a new edge from split node
                    remainder = key[i + j:]
                    new_node = RadixNode()
                    new_node.is_terminal = True
                    split_node.children[remainder[0]] = (remainder, new_node)
                else:
                    # Key ends exactly at the split point
                    split_node.is_terminal = True
                return

        # Key exhausted exactly at an existing node
        node.is_terminal = True

    # --- Search ---

    def search(self, key: str) -> bool:
        """Return True if key is stored in the radix tree."""
        node = self.root
        i = 0

        while i < len(key):
            ch = key[i]
            if ch not in node.children:
                return False

            label, child = node.children[ch]
            if not key[i:i + len(label)] == label:
                return False
            i += len(label)
            node = child

        return node.is_terminal

    # --- Prefix Search ---

    def starts_with(self, prefix: str) -> bool:
        """Return True if any stored key begins with the given prefix."""
        node = self.root
        i = 0

        while i < len(prefix):
            ch = prefix[i]
            if ch not in node.children:
                return False

            label, child = node.children[ch]
            remaining = len(prefix) - i
            if remaining <= len(label):
                return prefix[i:] == label[:remaining]
            if not prefix[i:i + len(label)] == label:
                return False
            i += len(label)
            node = child

        return True

    # --- Collect All Keys ---

    def _collect(self, node: RadixNode, prefix: str, results: list):
        """Recursively collect all keys under a node."""
        if node.is_terminal:
            results.append(prefix)
        for ch in sorted(node.children):
            label, child = node.children[ch]
            self._collect(child, prefix + label, results)

    def all_keys(self) -> list:
        """Return all keys in sorted order."""
        results = []
        self._collect(self.root, "", results)
        return results

    # --- Delete ---

    def delete(self, key: str) -> bool:
        """Delete a key from the radix tree. Returns True if found."""
        return self._delete(self.root, key, 0)

    def _delete(self, node: RadixNode, key: str, depth: int) -> bool:
        if depth == len(key):
            if not node.is_terminal:
                return False
            node.is_terminal = False
            return True

        ch = key[depth]
        if ch not in node.children:
            return False

        label, child = node.children[ch]
        if not key[depth:depth + len(label)] == label:
            return False

        found = self._delete(child, key, depth + len(label))
        if not found:
            return False

        # Clean up: remove childless non-terminal nodes
        if not child.is_terminal and not child.children:
            del node.children[ch]
        # Merge: if child has exactly one child, compress the edge
        elif not child.is_terminal and len(child.children) == 1:
            only_ch = next(iter(child.children))
            only_label, only_grandchild = child.children[only_ch]
            node.children[ch] = (label + only_label, only_grandchild)

        return True


# === Demonstration ===

if __name__ == "__main__":
    tree = RadixTree()

    words = ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon", "ruler"]
    for w in words:
        tree.insert(w)

    print("All keys:", tree.all_keys())
    print("Search 'romane':", tree.search("romane"))
    print("Search 'roman':", tree.search("roman"))
    print("Starts with 'rom':", tree.starts_with("rom"))

    tree.delete("romane")
    print("After deleting 'romane':", tree.all_keys())
```

**Output:**
```
All keys: ['romane', 'romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'ruler']
Search 'romane': True
Search 'roman': False
Starts with 'rom': True
After deleting 'romane': ['romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'ruler']
```

## How Insertion Works

Insertion follows the key character by character, matching against edge labels:

1. **No matching edge.** Create a new edge from the current node with the remaining key as its label.
2. **Full edge match.** Consume the entire edge label and continue from the child node with the remaining key.
3. **Partial edge match at position $j$.** Split the edge into two parts at position $j$: the matched prefix becomes the label of a new intermediate node, and the unmatched suffix becomes a child edge. The remaining key becomes another child edge of the intermediate node.

Each insertion touches at most $O(m)$ nodes, where $m$ is the key length.

## How Search Works

Search mirrors insertion but never modifies the tree. At each node, it checks whether the outgoing edge label matches the corresponding portion of the query key:

- If the edge label matches fully, advance to the child and continue.
- If the edge label does not match, the key is absent.
- If the key is exhausted at a node, check the terminal flag.

Search runs in $O(m)$ time, where $m$ is the length of the query key.

## Complexity Analysis

Let $n$ denote the number of stored keys and $m$ the length of the key being operated on.

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Insert    | $O(m)$         | $O(m)$ new edges |
| Search    | $O(m)$         | $O(1)$          |
| Delete    | $O(m)$         | $O(1)$          |
| Space     | —              | $O(n \cdot \bar{m})$ total |

Here $\bar{m}$ denotes the average key length. The $O(n)$ node bound makes radix trees significantly more space-efficient than standard tries when keys are long.

## Radix Trees vs Standard Tries

| Property | Standard Trie | Radix Tree |
|----------|--------------|------------|
| Nodes | $O(n \cdot m_{\max})$ | $O(n)$ |
| Single-child chains | Present | Eliminated |
| Edge labels | Single characters | Substrings |
| Implementation complexity | Simple | Moderate (edge splitting) |
| Cache behavior | Poor (many pointer hops) | Better (fewer nodes) |

The radix tree's advantage grows as key length increases relative to the number of keys. For short keys over small alphabets, the simpler standard trie may be preferable.

## Applications

Radix trees appear in many practical systems:

- **IP routing tables.** Longest-prefix matching on binary representations of IP addresses.
- **In-memory databases.** Adaptive Radix Trees (ART) power index structures in modern databases like HyPer.
- **Linux kernel.** The kernel uses radix trees for page cache lookups and other internal mappings.
- **Autocomplete and spell checking.** Prefix-based lookups benefit from the compressed structure.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 12.
- Morrison, D. R. (1968). PATRICIA — Practical Algorithm to Retrieve Information Coded in Alphanumeric. *Journal of the ACM*, 15(4), 514-534.

# Huffman Coding

When transmitting or storing text, using a fixed-length code (like ASCII, with 8 bits per character) wastes space if some characters appear far more often than others. **Huffman coding** assigns shorter bit strings to frequent characters and longer ones to rare characters, producing the optimal **prefix-free** variable-length code. The algorithm is a beautiful application of the greedy paradigm: it builds the code tree from the bottom up by repeatedly merging the two least-frequent symbols.

## Prefix-Free Codes

A **prefix-free code** (sometimes called a "prefix code") is a binary code in which no codeword is a prefix of another. This property ensures that a concatenated bit string can be decoded unambiguously from left to right, without delimiters.

For example, $\{0, 10, 110, 111\}$ is prefix-free, while $\{0, 01, 10, 1\}$ is not (since $0$ is a prefix of $01$, and $1$ is a prefix of $10$).

Prefix-free codes correspond one-to-one with binary trees: each leaf represents a character, and the path from root to leaf (left = 0, right = 1) gives the codeword. The codeword length $d_i$ equals the depth of character $i$ in the tree.

## Optimization Objective

Given $n$ characters with frequencies $f_1, f_2, \ldots, f_n$, the **cost** of a prefix-free code is the expected number of bits per character:

$$
B(T) = \sum_{i=1}^{n} f_i \cdot d_i
$$

where $d_i$ is the depth of character $i$ in the code tree $T$. Huffman coding constructs the tree $T^*$ that minimizes $B(T)$.

## Algorithm

!!! note "Huffman's Algorithm"
    1. Create a leaf node for each character with its frequency. Insert all nodes into a min-priority queue $Q$ keyed by frequency.
    2. While $|Q| > 1$:
        - Extract the two nodes $x$ and $y$ with the smallest frequencies.
        - Create a new internal node $z$ with $f_z = f_x + f_y$, with $x$ as its left child and $y$ as its right child.
        - Insert $z$ into $Q$.
    3. The remaining node in $Q$ is the root of the Huffman tree.

The algorithm is greedy because at each step it makes the locally optimal choice: merge the two least-frequent nodes. This ensures that the least-frequent characters end up deepest in the tree (longest codewords), while the most frequent characters remain near the root (shortest codewords).

## Worked Example

**Characters and frequencies:**

| Character | a  | b  | c  | d  | e   | f  |
|-----------|----|----|----|----|-----|----|
| Frequency | 5  | 9  | 12 | 13 | 16  | 45 |

**Step-by-step construction:**

1. **Merge** $a$ (5) and $b$ (9) $\to$ internal node (14).
2. **Merge** $c$ (12) and $d$ (13) $\to$ internal node (25).
3. **Merge** (14) and $e$ (16) $\to$ internal node (30).
4. **Merge** (25) and (30) $\to$ internal node (55).
5. **Merge** $f$ (45) and (55) $\to$ root (100).

**Resulting codes:**

| Character | Frequency | Code | Depth |
|-----------|-----------|------|-------|
| f         | 45        | 0    | 1     |
| c         | 12        | 100  | 3     |
| d         | 13        | 101  | 3     |
| a         | 5         | 1100 | 4     |
| b         | 9         | 1101 | 4     |
| e         | 16        | 111  | 3     |

**Cost:**

$$
B(T) = 45 \cdot 1 + 12 \cdot 3 + 13 \cdot 3 + 5 \cdot 4 + 9 \cdot 4 + 16 \cdot 3 = 224
$$

A fixed 3-bit code would cost $100 \times 3 = 300$ bits. Huffman coding saves 25.3%.

## Python Implementation

```python
"""
Huffman coding: build an optimal prefix-free binary code.

Uses a greedy strategy of repeatedly merging the two lowest-frequency
symbols to construct a binary tree that minimizes expected code length.
"""

import heapq
from collections import Counter


# === Huffman Tree Node ===

class HuffmanNode:
    """A node in the Huffman tree."""

    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


# === Build Huffman Tree ===

def build_huffman_tree(frequencies):
    """Build a Huffman tree from character frequencies.

    Args:
        frequencies: dict mapping characters to their frequencies

    Returns:
        Root node of the Huffman tree
    """
    heap = [HuffmanNode(char=c, freq=f) for c, f in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right,
        )
        heapq.heappush(heap, merged)

    return heap[0]


# === Extract Codes ===

def extract_codes(node, prefix="", codes=None):
    """Extract binary codes from a Huffman tree.

    Args:
        node: current node in the tree
        prefix: binary string built so far
        codes: dict accumulating character -> code mappings

    Returns:
        Dict mapping characters to their Huffman codes
    """
    if codes is None:
        codes = {}

    if node.char is not None:
        codes[node.char] = prefix if prefix else "0"
    else:
        if node.left:
            extract_codes(node.left, prefix + "0", codes)
        if node.right:
            extract_codes(node.right, prefix + "1", codes)

    return codes


# === Compute Cost ===

def huffman_cost(codes, frequencies):
    """Compute the total weighted path length of a Huffman code.

    Args:
        codes: dict mapping characters to binary code strings
        frequencies: dict mapping characters to frequencies

    Returns:
        Total cost sum(f_i * d_i)
    """
    return sum(frequencies[c] * len(code) for c, code in codes.items())


if __name__ == "__main__":
    frequencies = {"a": 5, "b": 9, "c": 12, "d": 13, "e": 16, "f": 45}

    tree = build_huffman_tree(frequencies)
    codes = extract_codes(tree)

    print("Huffman Codes:")
    print(f"{'Char':>5} {'Freq':>5} {'Code':>6} {'Depth':>6}")
    print("-" * 24)
    for char in sorted(codes, key=lambda c: len(codes[c])):
        print(f"{char:>5} {frequencies[char]:>5} {codes[char]:>6} {len(codes[char]):>6}")

    cost = huffman_cost(codes, frequencies)
    total_freq = sum(frequencies.values())
    fixed_cost = total_freq * 3  # 6 chars need at least 3 bits

    print(f"\nTotal cost: {cost}")
    print(f"Fixed 3-bit cost: {fixed_cost}")
    print(f"Savings: {(1 - cost / fixed_cost) * 100:.1f}%")
```

**Output:**
```
Huffman Codes:
 Char  Freq   Code  Depth
------------------------
    f    45      0      1
    c    12    100      3
    d    13    101      3
    e    16    111      3
    a     5   1100      4
    b     9   1101      4

Total cost: 224
Fixed 3-bit cost: 300
Savings: 25.3%
```

## Complexity Analysis

Let $n$ be the number of distinct characters.

- **Building the initial heap:** $O(n)$.
- **Main loop:** $n - 1$ iterations, each extracting two elements and inserting one: $O(n \log n)$.
- **Extracting codes:** $O(n)$ (one traversal of all leaves).
- **Total:** $O(n \log n)$.

**Space:** $O(n)$ for the tree and codes.

## Properties of Huffman Codes

1. **Optimality.** Among all prefix-free codes, Huffman coding minimizes $\sum f_i d_i$ (proved on the Huffman Optimality Proof page).
2. **Full binary tree.** Every internal node has exactly two children. If a node had only one child, the child's codeword could be shortened.
3. **Lowest-frequency characters are siblings at maximum depth.** This follows directly from the greedy construction.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16.3. MIT Press.
- Huffman, D. A. (1952). A method for the construction of minimum-redundancy codes. *Proceedings of the IRE*, 40(9), 1098--1101.

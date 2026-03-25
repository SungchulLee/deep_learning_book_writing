# Huffman Coding

Fixed-length codes like ASCII assign the same number of bits to every symbol, regardless of how often each symbol appears.  When some symbols occur far more frequently than others, this wastes bits.  Huffman coding exploits frequency imbalance by assigning shorter codes to common symbols and longer codes to rare ones, producing a **prefix-free** code that is provably optimal among all prefix-free codes.  This page presents the algorithm, proves its optimality, and walks through a complete example.

## Prefix-Free Codes

A **prefix-free code** assigns a binary codeword to each symbol such that no codeword is a prefix of another.  This property guarantees unambiguous decoding without delimiters: the decoder reads bits left to right, and the moment a valid codeword is recognized, it emits the corresponding symbol and starts the next codeword.

Prefix-free codes correspond naturally to binary trees.  Each leaf represents a symbol, and the path from root to leaf defines the codeword (left = 0, right = 1).  The depth of a leaf equals its codeword length.

## Optimal Code Objective

Given an alphabet $\Sigma = \{a_1, a_2, \dots, a_m\}$ with frequencies $f_1, f_2, \dots, f_m$, the cost of a prefix-free code $C$ is the expected codeword length

$$
B(C) = \sum_{i=1}^{m} f_i \cdot d_i
$$

where $d_i$ is the depth (codeword length) of symbol $a_i$ in the code tree.  Huffman's algorithm finds a code $C^*$ that minimizes $B(C)$ over all prefix-free codes.

## Algorithm

Huffman's greedy strategy builds the optimal tree bottom-up:

1. Create a leaf node for each symbol with its frequency.
2. Insert all nodes into a min-priority queue keyed by frequency.
3. While the queue contains more than one node:
    - Extract the two nodes $x$ and $y$ with the smallest frequencies.
    - Create a new internal node $z$ with $f_z = f_x + f_y$, setting $x$ and $y$ as its children.
    - Insert $z$ back into the queue.
4. The remaining node is the root of the Huffman tree.

## Complexity

| Step | Time |
|------|------|
| Build initial heap | $O(m)$ |
| $m - 1$ extract-min + insert | $O(m \log m)$ |
| **Total** | $O(m \log m)$ |

where $m = |\Sigma|$ is the alphabet size.  Note that the complexity depends on the alphabet size, not the input length.

## Worked Example

Consider five symbols with frequencies:

| Symbol | A | B | C | D | E |
|--------|---|---|---|---|---|
| Frequency | 45 | 13 | 12 | 16 | 9 |

**Step-by-step construction:**

1. Initial queue: E(9), C(12), B(13), D(16), A(45)
2. Merge E(9) + C(12) = EC(21).  Queue: B(13), D(16), EC(21), A(45)
3. Merge B(13) + D(16) = BD(29).  Queue: EC(21), BD(29), A(45)
4. Merge EC(21) + BD(29) = ECBD(50).  Queue: A(45), ECBD(50)
5. Merge A(45) + ECBD(50) = root(95)

**Resulting codes:**

| Symbol | Frequency | Codeword | Bits |
|--------|-----------|----------|------|
| A      | 45        | 0        | 1    |
| B      | 13        | 110      | 3    |
| C      | 12        | 101      | 3    |
| D      | 16        | 111      | 3    |
| E      | 9         | 100      | 3    |

**Weighted path length:**

$$
B = 45 \cdot 1 + 13 \cdot 3 + 12 \cdot 3 + 16 \cdot 3 + 9 \cdot 3 = 45 + 150 = 195
$$

A fixed 3-bit code would cost $95 \times 3 = 285$ bits, so Huffman saves about 32%.

## Optimality

??? note "Proof sketch of optimality"
    The proof proceeds by two key lemmas:

    **Lemma 1 (Greedy choice):** An optimal tree exists in which the two least-frequent symbols are siblings at the maximum depth.

    *Proof:* Take any optimal tree $T^*$.  If the two least-frequent symbols $x, y$ are not deepest siblings, swap them with the current deepest siblings.  Since $f_x$ and $f_y$ are smallest, this swap does not increase the cost.

    **Lemma 2 (Optimal substructure):** Let $z$ be the internal node formed by merging $x$ and $y$ with $f_z = f_x + f_y$.  Then $T$ is optimal for the original alphabet if and only if $T'$ (with $z$ replacing the subtree $\{x, y\}$) is optimal for the reduced alphabet.

    *Proof:* The cost satisfies $B(T) = B(T') + f_x + f_y$, so minimizing $B(T')$ also minimizes $B(T)$.

    Together, these lemmas establish that the greedy merging strategy produces an optimal prefix-free code.

## Implementation

```python
"""
Huffman Coding -- build an optimal prefix-free code from symbol frequencies.

Demonstrates the greedy tree construction and code extraction using a
min-heap priority queue.
"""

import heapq
from collections import Counter

# === Tree Node ===============================================================

class HuffmanNode:
    """A node in the Huffman tree."""

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


# === Tree Construction =======================================================

def build_huffman_tree(frequencies: dict[str, int]) -> HuffmanNode:
    """Build a Huffman tree from a frequency dictionary."""
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]


# === Code Extraction ==========================================================

def extract_codes(node: HuffmanNode, prefix: str = "") -> dict[str, str]:
    """Extract binary codes by traversing the Huffman tree."""
    if node.symbol is not None:
        return {node.symbol: prefix or "0"}
    codes = {}
    codes.update(extract_codes(node.left, prefix + "0"))
    codes.update(extract_codes(node.right, prefix + "1"))
    return codes


# === Main =====================================================================

if __name__ == "__main__":
    frequencies = {"A": 45, "B": 13, "C": 12, "D": 16, "E": 9}
    print("Frequencies:", frequencies)

    tree = build_huffman_tree(frequencies)
    codes = extract_codes(tree)

    print("\nHuffman Codes:")
    for symbol in sorted(codes):
        print(f"  {symbol}: {codes[symbol]}")

    total_bits = sum(frequencies[s] * len(codes[s]) for s in frequencies)
    fixed_bits = sum(frequencies.values()) * 3
    print(f"\nTotal bits (Huffman) : {total_bits}")
    print(f"Total bits (fixed-3) : {fixed_bits}")
    print(f"Savings              : {(1 - total_bits / fixed_bits) * 100:.1f}%")
```

**Output:**
```
Frequencies: {'A': 45, 'B': 13, 'C': 12, 'D': 16, 'E': 9}

Huffman Codes:
  A: 0
  B: 110
  C: 101
  D: 111
  E: 100

Total bits (Huffman) : 195
Total bits (fixed-3) : 285
Savings              : 31.6%
```

## Relationship to Entropy

The Shannon source coding theorem states that no lossless code can achieve an expected length below the entropy

$$
H = -\sum_{i=1}^{m} p_i \log_2 p_i
$$

where $p_i = f_i / \sum_j f_j$.  Huffman coding satisfies

$$
H \leq B(C^*) < H + 1
$$

so the optimal Huffman code is within one bit per symbol of the entropy bound.  For large alphabets or when symbol probabilities are far from powers of $1/2$, arithmetic coding can approach $H$ more closely.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)

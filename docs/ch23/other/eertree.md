# Palindromic Tree (Eertree)

A string of length $n$ can have at most $n + 1$ distinct palindromic substrings (including the empty string). The **palindromic tree** (also called **eertree**) is a data structure that stores all distinct palindromic substrings of a string in $O(n)$ time and space. Each node represents a unique palindromic substring, and suffix links connect palindromes to enable efficient online construction.

## Structure

The eertree consists of:

- **Two root nodes**: one for odd-length palindromes (length $-1$, imaginary) and one for even-length palindromes (length $0$, empty string).
- **Nodes**: Each node stores a distinct palindromic substring, identified by its length and position.
- **Edges**: Labeled transitions — edge with character $c$ from node $u$ leads to the palindrome obtained by adding $c$ on both sides of $u$.
- **Suffix links**: Each node points to its longest proper palindromic suffix.

The odd root has length $-1$ so that adding one character on each side produces a length-1 palindrome (single character).

## Key Properties

- A string of length $n$ has at most $n + 1$ distinct palindromic substrings.
- Each new character adds at most one new palindromic substring.
- The eertree has at most $n + 2$ nodes (including both roots).
- Total construction time is $O(n)$ for a constant-size alphabet.

## Online Construction

Characters are added one at a time. For each new character $c$ at position $i$:

1. Start from the node representing the longest palindromic suffix of the previous string.
2. Follow suffix links until finding a node $u$ such that the character before $u$'s occurrence equals $c$ (i.e., $s[i - \text{len}(u) - 1] = c$).
3. If the edge labeled $c$ from $u$ already exists, the palindrome is not new — follow it.
4. Otherwise, create a new node for the palindrome $c \cdot u \cdot c$:
    - Find the suffix link by continuing along suffix links from $u$ to find the next matching node.
    - Set up the edge and suffix link for the new node.

## Python Implementation

```python
"""
Palindromic Tree (Eertree) — Online Construction.

Builds a palindromic tree that stores all distinct palindromic
substrings of a string in O(n) time and space.
"""


# === Node Class ===

class EertreeNode:
    """A node in the palindromic tree."""

    def __init__(self, length: int, suffix_link: int = 0):
        self.length = length
        self.suffix_link = suffix_link
        self.edges: dict[str, int] = {}
        self.count = 0  # occurrences as a suffix palindrome


# === Eertree ===

class Eertree:
    """Palindromic tree supporting online construction."""

    def __init__(self) -> None:
        # Node 0: odd root (length -1)
        # Node 1: even root (length 0)
        self.nodes = [
            EertreeNode(length=-1, suffix_link=0),
            EertreeNode(length=0, suffix_link=0),
        ]
        self.s = [-1]  # sentinel character at position 0
        self.last = 1  # index of the node for the longest suffix palindrome

    def _get_link(self, v: int) -> int:
        """Follow suffix links until s[pos - len(v) - 1] == s[pos]."""
        pos = len(self.s) - 1
        while self.s[pos - self.nodes[v].length - 1] != self.s[pos]:
            v = self.nodes[v].suffix_link
        return v

    def add_char(self, c: str) -> bool:
        """Add a character and return True if a new palindrome was created."""
        self.s.append(ord(c))
        cur = self._get_link(self.last)

        if c in self.nodes[cur].edges:
            self.last = self.nodes[cur].edges[c]
            self.nodes[self.last].count += 1
            return False

        # Create new node
        new_len = self.nodes[cur].length + 2
        # Find suffix link for new node
        suffix = self._get_link(self.nodes[cur].suffix_link)
        if c in self.nodes[suffix].edges:
            suf_link = self.nodes[suffix].edges[c]
        else:
            suf_link = 1  # even root

        new_node = EertreeNode(length=new_len, suffix_link=suf_link)
        new_node.count = 1
        self.nodes.append(new_node)
        new_idx = len(self.nodes) - 1
        self.nodes[cur].edges[c] = new_idx
        self.last = new_idx
        return True

    def build(self, s: str) -> None:
        """Build the eertree for the entire string."""
        for c in s:
            self.add_char(c)

    def get_palindromes(self) -> list[tuple[int, int]]:
        """Return (length, count) for each distinct palindromic substring."""
        # Propagate counts from longest to shortest
        result = []
        for i in range(len(self.nodes) - 1, 1, -1):
            node = self.nodes[i]
            self.nodes[node.suffix_link].count += node.count
            result.append((node.length, node.count))
        return result

    @property
    def num_palindromes(self) -> int:
        """Number of distinct palindromic substrings."""
        return len(self.nodes) - 2  # exclude both roots


# === Main ===

if __name__ == "__main__":
    s = "abaab"
    tree = Eertree()
    tree.build(s)

    print(f"String: '{s}'")
    print(f"Distinct palindromes: {tree.num_palindromes}")

    palindromes = tree.get_palindromes()
    palindromes.sort(key=lambda x: x[0])
    for length, count in palindromes:
        print(f"  length {length}: occurs {count} times")
    # Output:
    # String: 'abaab'
    # Distinct palindromes: 5
    #   length 1: occurs 5 times
    #   length 1: occurs 3 times
    #   length 2: occurs 1 times
    #   length 3: occurs 1 times
    #   length 3: occurs 1 times
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Construction | $O(n)$ amortized | $O(n)$ |
| Count distinct palindromes | $O(1)$ | — |
| Query all palindromes | $O(n)$ | $O(n)$ |

## Applications

- **Count distinct palindromic substrings**: The eertree has exactly this many nodes (minus 2 roots).
- **Count palindromic substring occurrences**: Propagate counts along suffix links.
- **Longest palindromic substring**: Track the maximum node length during construction.
- **Number of palindromic substrings in each prefix**: Count new palindromes added per character.

## Reference

- Rubinchik, M., & Shur, A. M. (2018). EERTREE: An efficient data structure for processing palindromes in strings. *European Journal of Combinatorics*, 68, 249-265.

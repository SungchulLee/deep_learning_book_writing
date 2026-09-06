# Autocomplete with Tries

Autocomplete is one of the most visible applications of the trie data structure. When a user types a prefix into a search box, the system must rapidly retrieve all stored strings (or the top-ranked ones) that begin with that prefix. A trie handles this naturally: navigate to the node corresponding to the prefix in $O(p)$ time, then collect all descendants. This section explains the algorithm, analyzes its complexity, and provides a complete implementation.

## Algorithm

Autocomplete via trie proceeds in three steps:

1. **Navigate to the prefix node**: Starting from the root, follow the path for each character of the prefix. If the path breaks, return an empty result.
2. **Collect all completions**: From the prefix node, perform a DFS (or BFS) to enumerate every path that leads to a word-endpoint node.
3. **Rank results** (optional): Sort completions by frequency, recency, or relevance score stored at each endpoint node.

## Implementation

```python
"""Autocomplete system using a trie data structure.

Stores words with associated frequency counts and returns
completions ranked by frequency.
"""


# === Trie Node and Autocomplete Trie ===
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False
        self.freq = 0


class AutocompleteTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, freq=1):
        """Insert a word with an associated frequency."""
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True
        node.freq += freq

    def autocomplete(self, prefix, limit=5):
        """Return up to `limit` completions for the given prefix, sorted by frequency."""
        node = self.root
        for c in prefix:
            if c not in node.children:
                return []
            node = node.children[c]
        results = []
        self._collect(node, prefix, results)
        results.sort(key=lambda x: -x[1])
        return [word for word, freq in results[:limit]]

    def _collect(self, node, path, results):
        if node.end:
            results.append((path, node.freq))
        for c, child in node.children.items():
            self._collect(child, path + c, results)


# === Main ===
if __name__ == "__main__":
    trie = AutocompleteTrie()
    for word, freq in [("apple", 50), ("app", 30), ("application", 20),
                       ("bat", 10), ("ball", 15), ("banana", 25)]:
        trie.insert(word, freq)

    print("Prefix 'app':", trie.autocomplete("app"))
    print("Prefix 'ba':", trie.autocomplete("ba"))
    print("Prefix 'xyz':", trie.autocomplete("xyz"))
```

**Output:**
```
Prefix 'app': ['apple', 'app', 'application']
Prefix 'ba': ['banana', 'ball', 'bat']
Prefix 'xyz': []
```

## Complexity

| Phase | Time |
|:---|:---:|
| Navigate to prefix | $O(p)$ |
| Collect all completions | $O(k)$ |
| Sort by frequency | $O(k \log k)$ |
| **Total** | $O(p + k \log k)$ |

Here $p$ is the prefix length and $k$ is the total size of all matching words. In practice, $k$ is bounded by the vocabulary size, and the sort can be avoided entirely by maintaining a priority queue or precomputed top-$k$ list at each node.

## References

[Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)

## Exercises

**Exercise 1.**
A trie stores the words "cat", "car", "card", "care", "dog", "do". Draw the trie and trace the autocomplete query for prefix "car".

??? success "Solution to Exercise 1"
    Trie structure: root -> c -> a -> t($), r($) -> d($), e($); root -> d -> o($) -> g($). (where $ marks end of word). For prefix "car": navigate root -> c -> a -> r. From the "r" node, collect all descendants with end-of-word markers: "car" (r is marked), "card" (r -> d, marked), "care" (r -> e, marked). Result: ["car", "card", "care"]. The navigation takes $O(|prefix|) = O(3)$ steps, and collecting descendants takes $O(k)$ where $k$ is the number of results (here 3). $\square$

---

**Exercise 2.**
Analyze the time and space complexity of autocomplete using a trie with $n$ total characters across all stored strings. How does it compare to binary search on a sorted array?

??? success "Solution to Exercise 2"
    **Trie**: space $O(n \times |\Sigma|)$ where $|\Sigma|$ is the alphabet size (each node has up to $|\Sigma|$ child pointers). With hash maps at each node: $O(n)$ space. Autocomplete query for prefix $p$ with $k$ results: $O(|p| + k)$ time ($|p|$ to navigate, $k$ to collect results via DFS). **Sorted array + binary search**: space $O(n)$. Autocomplete: binary search for the first string $\ge p$ in $O(|p| \log m)$ where $m$ is the number of strings, then scan forward collecting $k$ results in $O(k \cdot |p|)$ for comparisons. Total: $O(|p| \log m + k \cdot |p|)$. The trie is faster for autocomplete by a factor of $\log m$ in the navigation phase and avoids repeated string comparisons. The sorted array uses less space and supports efficient rank queries. $\square$

---

**Exercise 3.**
Design an autocomplete system that returns the top-$k$ results by popularity. What augmentation does the trie need?

??? success "Solution to Exercise 3"
    Augment each trie node with a priority queue (min-heap) of the top-$k$ completions reachable from that node, along with their popularity scores. During trie construction: for each word inserted, propagate its score up to all ancestor nodes, maintaining only the top $k$ at each node. Autocomplete query: navigate to the prefix node in $O(|p|)$ and return the stored top-$k$ list in $O(k)$. Space overhead: $O(n \cdot k)$ where $n$ is the number of trie nodes. Update cost when a word's popularity changes: propagate from the word's leaf up to the root, updating each ancestor's top-$k$ heap in $O(k)$ per node, for $O(|w| \cdot k)$ total per update. This precomputation makes queries extremely fast at the cost of higher update cost and memory. $\square$

---

**Exercise 4.**
Explain how a ternary search tree (TST) reduces the space overhead of a trie while maintaining efficient prefix lookups.

??? success "Solution to Exercise 4"
    A standard trie node has $|\Sigma|$ child pointers (e.g., 26 for lowercase English), most of which are null in sparse tries. A ternary search tree replaces the $|\Sigma|$-way branch with a binary search tree at each level: each node stores one character and three pointers (less-than, equal, greater-than). Lookup at a node: if the query character equals the node's character, follow the "equal" pointer (advance to next character in the query). If less, follow "less-than" (same position in query, different character). If greater, follow "greater-than." Space: each stored character uses one node with 3 pointers (vs. one node with $|\Sigma|$ pointers in a trie). For sparse tries, the TST uses significantly less memory. Prefix lookup: $O(|p| + \log |\Sigma|)$ per character (the BST at each level costs $O(\log |\Sigma|)$). For $|\Sigma| = 26$: $\log 26 \approx 5$ extra comparisons per character, a modest overhead. $\square$

---

**Exercise 5.**
A search engine processes 10,000 autocomplete queries per second, each with a prefix of average length 5. The dictionary has 10 million words. Estimate the memory and compute requirements.

??? success "Solution to Exercise 5"
    **Memory**: with 10 million words of average length 8 characters, total characters = $8 \times 10^7$. Trie nodes: roughly $5 \times 10^7$ (many words share prefixes). Each node: 1 byte (character) + 8 bytes (child pointer or hash map entry) + 1 byte (end flag) $\approx 10$ bytes. Total: $5 \times 10^8$ bytes $= 500$ MB. With top-10 precomputed results per node (8 bytes each): additional $5 \times 10^7 \times 80 = 4$ GB. A compressed trie (DAFSA/DAWG) can reduce the base structure to $\sim$100 MB. **Compute**: each query navigates 5 nodes + returns top-10 results $\approx 15$ memory accesses. At $\sim$10 ns per L3 cache hit: $150$ ns per query. For 10,000 qps: $1.5$ ms total CPU time per second -- trivially handled by one core. The bottleneck is memory bandwidth, not computation. $\square$

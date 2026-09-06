# Trie Complexity Analysis

Understanding the time and space complexity of trie operations is essential for deciding when a trie is the right data structure for a problem. This page provides a systematic analysis of each operation, compares tries with alternative data structures, and identifies the scenarios where tries offer a clear advantage.

## Operation Complexities

Let $m$ denote the length of the query string, $n$ the number of stored strings, $L$ the total length of all stored strings, and $|\Sigma|$ the alphabet size.

| Operation | Time | Notes |
|:---|:---:|:---|
| Insert | $O(m)$ | Create at most $m$ new nodes |
| Search (exact) | $O(m)$ | Follow at most $m$ edges |
| Delete | $O(m)$ | May require cleanup of empty nodes |
| Prefix search | $O(p + k)$ | $p$ = prefix length, $k$ = output size |
| Longest prefix match | $O(m)$ | Used in IP routing |

All operations are **independent of $n$** -- a critical advantage when the dataset is large but individual strings are short.

## Space Complexity

The space depends on the node representation:

| Representation | Space per node | Total space |
|:---|:---:|:---|
| Array of size $\lvert\Sigma\rvert$ | $O(\lvert\Sigma\rvert)$ | $O(L \cdot \lvert\Sigma\rvert)$ |
| Hash map | $O(\text{avg children})$ | $O(L)$ on average |
| Compressed (Patricia) | $O(1)$ amortized | $O(n)$ nodes, $O(L)$ for labels |

For small alphabets (e.g., DNA with $|\Sigma| = 4$), the array representation is practical. For large alphabets (e.g., Unicode), hash maps or compressed tries are preferred.

## Comparison with Other Data Structures

| Data Structure | Exact Lookup | Prefix Search | Sorted Order | Space |
|:---|:---:|:---:|:---:|:---|
| Trie | $O(m)$ | $O(p + k)$ | Yes (via DFS) | $O(L \cdot \lvert\Sigma\rvert)$ or $O(L)$ |
| Hash table | $O(m)$ avg | $O(n \cdot m)$ | No | $O(L)$ |
| Balanced BST | $O(m \log n)$ | $O(m \log n + k)$ | Yes | $O(L)$ |
| Sorted array | $O(m \log n)$ | $O(m \log n + k)$ | Yes | $O(L)$ |

The trie's distinguishing advantage is **prefix search in $O(p + k)$** -- no other standard data structure matches this without additional indexing.

!!! tip "When to Choose a Trie"
    Tries are the best choice when:

    - **Prefix-based queries** are frequent (autocomplete, spell-check, IP routing)
    - **Strings share long common prefixes**, making the trie compact
    - **Worst-case guarantees** matter: trie operations have no adversarial inputs (unlike hash tables)

    Tries are a poor choice when the alphabet is very large, strings share few prefixes, and only exact lookup is needed -- a hash table will be simpler and faster.

## References

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)

## Exercises

**Exercise 1.**
Analyze the time and space complexity of inserting a string of length $m$ into a trie with an alphabet of size $|\Sigma|$.

??? success "Solution to Exercise 1"
    Time: $O(m)$ -- traverse or create one node per character. At each step, check if the child for the current character exists ($O(1)$ with an array of size $|\Sigma|$ or $O(1)$ expected with a hash map). Create the child if absent. Total: $m$ steps, each $O(1)$, giving $O(m)$. Space for the new string: at most $m$ new nodes (if no prefix is shared with existing strings). Each node has $|\Sigma|$ child pointers with array representation, or variable children with hash maps. Worst-case space per node: $O(|\Sigma|)$ with arrays, $O(1)$ average with hash maps. Total trie space for $n$ strings of average length $L$: $O(n \cdot L \cdot |\Sigma|)$ with arrays, $O(n \cdot L)$ with hash maps (but with higher constant factors). $\square$

---

**Exercise 2.**
Compare a trie with a hash set for the operation "check if any stored string has prefix $p$." Which is more efficient?

??? success "Solution to Exercise 2"
    **Trie**: navigate from root following the characters of $p$. If we successfully traverse all $|p|$ characters (all intermediate nodes exist), the answer is yes (at least one string passes through this node). Time: $O(|p|)$. **Hash set**: stores complete strings, not prefixes. To check if any string has prefix $p$, we must iterate over all stored strings and check each one: $O(n \cdot |p|)$ where $n$ is the number of strings. Alternatively, precompute all prefixes of all strings and store them in the hash set: $O(1)$ lookup per query, but $O(n \cdot L)$ preprocessing space where $L$ is the average string length. The trie is clearly superior for prefix queries: $O(|p|)$ with no extra space beyond the trie itself. This is the trie's defining advantage. $\square$

---

**Exercise 3.**
A compressed trie (Patricia trie) reduces space by collapsing chains of single-child nodes. Analyze the space savings for a dictionary of $n$ English words.

??? success "Solution to Exercise 3"
    An uncompressed trie for $n$ words of average length $L$ has up to $nL$ nodes. Many nodes on shared prefixes have only one child (e.g., the suffix of a unique word). A Patricia trie collapses such chains: each edge stores a substring rather than a single character. The resulting trie has at most $2n - 1$ nodes (each internal node has $\ge 2$ children; with $n$ leaves, the number of internal nodes is at most $n - 1$). Space: $O(n)$ nodes plus $O(nL)$ total edge label length (which can be represented by start/end pointers into the original strings, requiring $O(n)$ space). For $n = 100{,}000$ English words: uncompressed trie might have $\sim 500{,}000$ nodes; Patricia trie has $\sim 200{,}000$ nodes -- a 60% reduction. $\square$

---

**Exercise 4.**
Describe how a trie supports autocomplete (finding all strings with a given prefix) and analyze the time complexity.

??? success "Solution to Exercise 4"
    Navigate from the root following the prefix characters: $O(|p|)$ to reach the prefix node. From there, perform a DFS/BFS to collect all descendant leaf nodes, each representing a stored string. Time: $O(|p| + k \cdot L_{\text{avg}})$ where $k$ is the number of matching strings and $L_{\text{avg}}$ is the average length of the suffix beyond the prefix. If only the top-$k$ results by popularity are needed, augment each node with a precomputed list of top-$k$ descendants. Autocomplete query: $O(|p| + k)$ -- navigate to the prefix node and read the top-$k$ list. This is the approach used by search engine autocomplete boxes. $\square$

---

**Exercise 5.**
Compare a trie with a sorted array + binary search for dictionary operations (insert, search, prefix search). Under what conditions does each win?

??? success "Solution to Exercise 5"
    | Operation | Trie | Sorted Array + Binary Search |
    |---|---|---|
    | Insert | $O(m)$ | $O(n \cdot m)$ (shift + compare) |
    | Search | $O(m)$ | $O(m \log n)$ |
    | Prefix search (all $k$ matches) | $O(m + k \cdot L)$ | $O(m \log n + k \cdot L)$ |
    | Space | $O(S \cdot |\Sigma|)$ or $O(S)$ | $O(S)$ compact |

    where $m$ = query length, $n$ = number of strings, $S$ = total string characters, $L$ = avg string length. Trie wins when: (1) prefix queries are common; (2) the dictionary is dynamic (frequent inserts/deletes); (3) $|\Sigma|$ is small (26 for lowercase English). Sorted array wins when: (1) the dictionary is static (built once, queried many times); (2) memory is tight (arrays are more compact); (3) $|\Sigma|$ is large (Unicode), making trie nodes expensive. For most autocomplete and spell-check applications, tries are preferred. $\square$

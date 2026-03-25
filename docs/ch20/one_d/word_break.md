# Word Break

The word break problem asks whether a given string can be segmented into a sequence of words from a dictionary.  This is a natural application of one-dimensional dynamic programming where the state represents a position in the string and the transition checks all possible last words ending at that position.  The problem arises in natural language processing (tokenization), search query parsing, and domain name analysis.

## Problem Statement

Given a string $s$ of length $n$ and a dictionary $D$ (a set of valid words), determine whether $s$ can be partitioned into one or more words, each of which belongs to $D$.

**Example:** With $s = \texttt{"leetcode"}$ and $D = \{\texttt{"leet"}, \texttt{"code"}\}$, the answer is true because $s$ can be split as $\texttt{"leet"} + \texttt{"code"}$.

**Example:** With $s = \texttt{"catsandog"}$ and $D = \{\texttt{"cats"}, \texttt{"dog"}, \texttt{"sand"}, \texttt{"and"}, \texttt{"cat"}\}$, the answer is false.

## Recurrence

Let $dp[i]$ be true if the prefix $s[0..i-1]$ (the first $i$ characters) can be segmented into dictionary words.  For each position $i$, check all possible last words $s[j..i-1]$ for $0 \le j < i$:

$$
dp[i] = \bigvee_{\substack{0 \le j < i \\ s[j..i-1] \in D}} dp[j]
$$

with base case $dp[0] = \text{true}$ (the empty prefix is trivially segmentable).

In words: position $i$ is reachable if there exists some earlier position $j$ that is reachable, and the substring from $j$ to $i$ is a valid dictionary word.

## Tabulation

```python
"""
Word break: determine if a string can be segmented into dictionary words.
"""


# ===================================================================
# Approach 1: Tabulation (bottom-up)
# ===================================================================
def word_break(s: str, word_dict: list[str]) -> bool:
    """Check if s can be segmented. Time: O(n^2 * L), Space: O(n)."""
    n = len(s)
    words = set(word_dict)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break

    return dp[n]
```

The outer loop runs $n$ times.  The inner loop runs up to $n$ times, and each substring check takes $O(L)$ where $L$ is the maximum word length.  Total time is $O(n^2 L)$, which can be improved by limiting $j$ to check only word-length windows.

## Optimized with Maximum Word Length

Since dictionary words have bounded length, the inner loop can be restricted:

```python
# ===================================================================
# Approach 2: Optimized with max word length
# ===================================================================
def word_break_optimized(s: str, word_dict: list[str]) -> bool:
    """Optimized version limiting inner loop range. Time: O(n * L), Space: O(n)."""
    n = len(s)
    words = set(word_dict)
    max_len = max(len(w) for w in words) if words else 0
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break

    return dp[n]
```

By checking only substrings up to the maximum word length $L$, the inner loop runs at most $L$ times per position, giving $O(nL)$ time.

## Reconstructing the Segmentation

To find the actual word segmentation, track which split points lead to valid segmentations:

```python
# ===================================================================
# Approach 3: With reconstruction
# ===================================================================
def word_break_segment(s: str, word_dict: list[str]) -> list[str] | None:
    """Return one valid segmentation, or None if impossible."""
    n = len(s)
    words = set(word_dict)
    dp = [False] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                parent[i] = j
                break

    if not dp[n]:
        return None

    # Backtrack to recover the words
    result = []
    idx = n
    while idx > 0:
        result.append(s[parent[idx]:idx])
        idx = parent[idx]

    return list(reversed(result))
```

## Complexity

| Approach | Time | Space |
|----------|------|-------|
| Basic tabulation | $O(n^2 L)$ | $O(n)$ |
| Optimized | $O(nL)$ | $O(n + \|D\|)$ |
| With reconstruction | $O(n^2 L)$ | $O(n)$ |

Here $n$ is the string length, $L$ is the maximum word length, and $|D|$ is the dictionary size.

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_cases = [
        ("leetcode", ["leet", "code"]),
        ("applepenapple", ["apple", "pen"]),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"]),
    ]
    for s, dictionary in test_cases:
        result = word_break(s, dictionary)
        segmentation = word_break_segment(s, dictionary)
        print(f"s='{s}' -> {result}, segmentation={segmentation}")
```

**Output:**
```
s='leetcode' -> True, segmentation=['leet', 'code']
s='applepenapple' -> True, segmentation=['apple', 'pen', 'apple']
s='catsandog' -> False, segmentation=None
```

!!! note "All segmentations variant"
    A harder variant asks for **all possible** segmentations.  This requires exploring all valid splits using backtracking with memoization, and the output can be exponential in size.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.

# Aho-Corasick

The Aho-Corasick algorithm is a multi-pattern string matching algorithm that finds all occurrences of a set of patterns $\{P_1, P_2, \ldots, P_k\}$ simultaneously in a text $T$ of length $n$. Let $m = \sum |P_i|$ be the total length of all patterns and $z$ be the number of matches. The algorithm runs in $O(n + m + z)$ time.

## Overview

The algorithm works in two phases:

1. **Preprocessing:** Build a trie (keyword tree) from all patterns, then augment it with failure links and output links (dictionary links) to create a finite automaton.
2. **Search:** Feed the text through the automaton character by character. At each state, report all patterns that end at that position.

$$
\text{Total time} = \underbrace{O(m)}_{\text{build trie + links}} + \underbrace{O(n)}_{\text{scan text}} + \underbrace{O(z)}_{\text{report matches}} = O(n + m + z)
$$

## Implementation

```python
from collections import deque

class AhoCorasick:
    def __init__(self):
        self.goto = [{}]       # goto function (trie edges)
        self.fail = [0]        # failure links
        self.output = [[]]     # output function (pattern indices)

    def add_pattern(self, pattern: str, index: int):
        """Insert a pattern into the trie."""
        state = 0
        for ch in pattern:
            if ch not in self.goto[state]:
                self.goto[state][ch] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            state = self.goto[state][ch]
        self.output[state].append(index)

    def build(self):
        """Compute failure links and propagate output links via BFS."""
        queue = deque()
        for ch, s in self.goto[0].items():
            queue.append(s)
            self.fail[s] = 0
        while queue:
            r = queue.popleft()
            for ch, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while state != 0 and ch not in self.goto[state]:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(ch, 0)
                if self.fail[s] == s:
                    self.fail[s] = 0
                self.output[s] = self.output[s] + self.output[self.fail[s]]

    def search(self, text: str, patterns: list[str]) -> list[tuple[int, str]]:
        """Search text for all pattern occurrences."""
        results = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for pid in self.output[state]:
                results.append((i - len(patterns[pid]) + 1, patterns[pid]))
        return results


# Example
patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick()
for i, p in enumerate(patterns):
    ac.add_pattern(p, i)
ac.build()

text = "ahishers"
matches = ac.search(text, patterns)
for pos, pat in sorted(matches):
    print(f"  Pattern '{pat}' found at position {pos}")
# Output:
#   Pattern 'his' found at position 1
#   Pattern 'he' found at position 4
#   Pattern 'she' found at position 3
#   Pattern 'hers' found at position 4
```

## Complexity Analysis

| Phase | Time | Space |
|-------|------|-------|
| Build trie | $O(m)$ | $O(m \cdot |\Sigma|)$ |
| Build failure links | $O(m)$ | included above |
| Search | $O(n + z)$ | $O(1)$ extra |
| **Total** | $O(n + m + z)$ | $O(m \cdot |\Sigma|)$ |


# Reference

[Aho, Corasick - Efficient String Matching: An Aid to Bibliographic Search (1975)](https://doi.org/10.1145/360825.360855)

[Aho-Corasick Algorithm - CP-Algorithms](https://cp-algorithms.com/string/aho_corasick.html)

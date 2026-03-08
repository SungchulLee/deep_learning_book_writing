# Failure Links

Failure links (also called suffix links) are the key augmentation that transforms the Aho-Corasick trie into a powerful automaton. They are the multi-pattern generalization of the KMP failure function.

## Definition

For a state $s$ in the trie representing the string $w$, the failure link $f(s)$ points to the state representing the longest proper suffix of $w$ that is also a prefix of some pattern in the trie.

$$

f(s) = \text{state for the longest proper suffix of } \text{str}(s) \text{ that exists in the trie}

$$

If no proper suffix of $w$ exists in the trie, then $f(s)$ points to the root.

## Construction via BFS

Failure links are computed using a BFS traversal of the trie. States at depth 1 always have their failure link pointing to the root. For deeper states, we follow the parent's failure link chain until we find a state with a matching transition.

```python
from collections import deque

def build_failure_links(goto, fail):
    """
    Build failure links for the Aho-Corasick automaton.

    goto: list of dicts, goto[state][char] = next_state
    fail: list of ints, fail[state] will be filled in
    """
    queue = deque()

    # Depth-1 states: failure link -> root (0)
    for ch, s in goto[0].items():
        fail[s] = 0
        queue.append(s)

    # BFS for deeper states
    while queue:
        r = queue.popleft()
        for ch, s in goto[r].items():
            queue.append(s)
            state = fail[r]
            while state != 0 and ch not in goto[state]:
                state = fail[state]
            fail[s] = goto[state].get(ch, 0)
            if fail[s] == s:
                fail[s] = 0

    return fail

# Example: trie for patterns ["ACC", "ATC", "CAT"]
goto = [
    {'A': 1, 'C': 7},  # root (0)
    {'C': 2, 'T': 4},  # state 1: "A"
    {'C': 3},           # state 2: "AC"
    {},                 # state 3: "ACC" (match)
    {'C': 5},           # state 4: "AT"
    {},                 # state 5: "ATC" (match)
    {},                 # (unused)
    {'A': 8},           # state 7: "C"
    {'T': 9},           # state 8: "CA"
    {},                 # state 9: "CAT" (match)
]
fail = [0] * len(goto)
build_failure_links(goto, fail)
print("Failure links:", fail)
# fail[2] -> 7 (suffix "C" of "AC" matches state for "C")
# fail[8] -> 1 (suffix "A" of "CA" matches state for "A")
```

## Why Failure Links Work

When processing a text character and the current trie state has no transition for it, following the failure link is equivalent to considering the next longest suffix that could still match. This is analogous to how the KMP failure function avoids re-examining characters. The chain of failure links from any state $s$ back to the root represents all suffixes of $\text{str}(s)$ that are prefixes of some pattern.

## Complexity

- **Construction:** $O(m)$ where $m$ is the total length of all patterns. Each state is enqueued once in the BFS, and the failure link chain traversal is amortized $O(1)$ per state.
- **Space:** $O(1)$ per state for the failure link pointer.

# Reference

[Aho, Corasick - Efficient String Matching (1975)](https://doi.org/10.1145/360825.360855)

[Aho-Corasick - CP-Algorithms](https://cp-algorithms.com/string/aho_corasick.html)

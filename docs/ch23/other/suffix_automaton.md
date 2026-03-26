# Suffix Automaton

A **suffix automaton** (also called DAWG — Directed Acyclic Word Graph) is the smallest deterministic finite automaton (DFA) that accepts exactly all suffixes of a given string. It can be built in $O(n)$ time and space, and it enables substring queries, counting distinct substrings, and finding the longest common substring — all efficiently.

## Key Properties

For a string $s$ of length $n$:

- The suffix automaton has at most $2n - 1$ states and at most $3n - 4$ transitions.
- It accepts a string $w$ if and only if $w$ is a substring of $s$.
- Each state represents an **equivalence class** of substrings that occur at the same set of ending positions in $s$.

## Endpos Sets and Equivalence Classes

Define $\text{endpos}(w)$ as the set of ending positions where substring $w$ occurs in $s$. Two substrings $u$ and $v$ belong to the same equivalence class if $\text{endpos}(u) = \text{endpos}(v)$.

Each state in the suffix automaton corresponds to one equivalence class. The **suffix link** of a state points to the state representing the longest proper suffix that belongs to a different equivalence class.

## Online Construction

The suffix automaton is built incrementally by adding one character at a time. When adding character $c$ to the automaton for $s[0 \ldots i-1]$:

1. Create a new state `cur` for the extended string.
2. Starting from the state representing the previous full string, follow suffix links and add transitions labeled $c$ to `cur`.
3. If a transition labeled $c$ already exists from some ancestor, handle the suffix link of `cur` by potentially cloning the target state.

## Python Implementation

```python
"""
Suffix Automaton (DAWG) — Online Construction in O(n).

Builds the smallest DFA that accepts all suffixes of a string.
Supports substring checking, counting distinct substrings,
and finding the longest common substring.
"""


# === State Class ===

class State:
    """A state in the suffix automaton."""

    def __init__(self) -> None:
        self.length = 0       # length of the longest string in this class
        self.link = -1        # suffix link
        self.transitions: dict[str, int] = {}
        self.count = 0        # number of times endpos set changes


# === Suffix Automaton ===

class SuffixAutomaton:
    """Online suffix automaton construction."""

    def __init__(self) -> None:
        init_state = State()
        init_state.length = 0
        init_state.link = -1
        self.states = [init_state]
        self.last = 0  # index of the state for the current full string

    def extend(self, c: str) -> None:
        """Add a character to the suffix automaton."""
        cur = len(self.states)
        new_state = State()
        new_state.length = self.states[self.last].length + 1
        new_state.count = 1
        self.states.append(new_state)

        p = self.last
        while p != -1 and c not in self.states[p].transitions:
            self.states[p].transitions[c] = cur
            p = self.states[p].link

        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].transitions[c]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                # Clone state q
                clone = len(self.states)
                cloned = State()
                cloned.length = self.states[p].length + 1
                cloned.link = self.states[q].link
                cloned.transitions = dict(self.states[q].transitions)
                self.states.append(cloned)

                while p != -1 and self.states[p].transitions.get(c) == q:
                    self.states[p].transitions[c] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[cur].link = clone

        self.last = cur

    def build(self, s: str) -> None:
        """Build the suffix automaton for string s."""
        for c in s:
            self.extend(c)

    def contains(self, pattern: str) -> bool:
        """Check if pattern is a substring of the original string."""
        cur = 0
        for c in pattern:
            if c not in self.states[cur].transitions:
                return False
            cur = self.states[cur].transitions[c]
        return True

    def count_distinct_substrings(self) -> int:
        """Count the number of distinct non-empty substrings."""
        total = 0
        for i in range(1, len(self.states)):
            state = self.states[i]
            link_len = self.states[state.link].length if state.link >= 0 else 0
            total += state.length - link_len
        return total


# === Longest Common Substring ===

def longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring using a suffix automaton."""
    sa = SuffixAutomaton()
    sa.build(s1)

    cur = 0
    cur_len = 0
    best_len = 0
    best_end = 0

    for i, c in enumerate(s2):
        while cur != 0 and c not in sa.states[cur].transitions:
            cur = sa.states[cur].link
            cur_len = sa.states[cur].length

        if c in sa.states[cur].transitions:
            cur = sa.states[cur].transitions[c]
            cur_len += 1
        else:
            cur = 0
            cur_len = 0

        if cur_len > best_len:
            best_len = cur_len
            best_end = i

    return s2[best_end - best_len + 1:best_end + 1]


# === Main ===

if __name__ == "__main__":
    s = "abcbc"
    sa = SuffixAutomaton()
    sa.build(s)

    print(f"String: '{s}'")
    print(f"States: {len(sa.states)}")
    print(f"Distinct substrings: {sa.count_distinct_substrings()}")

    for pattern in ["abc", "bcb", "cb", "xyz"]:
        print(f"  Contains '{pattern}': {sa.contains(pattern)}")

    lcs = longest_common_substring("abcdef", "zbcdf")
    print(f"\nLCS of 'abcdef' and 'zbcdf': '{lcs}'")
    # Output:
    # String: 'abcbc'
    # States: 8
    # Distinct substrings: 12
    #   Contains 'abc': True
    #   Contains 'bcb': True
    #   Contains 'cb': True
    #   Contains 'xyz': False
    #
    # LCS of 'abcdef' and 'zbcdf': 'bcd'
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Construction | $O(n)$ | $O(n)$ states, $O(n |\Sigma|)$ transitions |
| Substring check | $O(|w|)$ | — |
| Count distinct substrings | $O(n)$ | — |
| Longest common substring | $O(n + m)$ | $O(n)$ |

## Reference

- Blumer, A., Blumer, J., Haussler, D., McConnell, R., & Ehrenfeucht, A. (1987). Complete inverted files for efficient text retrieval and analysis. *Journal of the ACM*, 34(3), 578-595.

# Dictionary Links

Dictionary links (also called output links or dictionary suffix links) are an optimization in the Aho-Corasick automaton that enables efficient enumeration of all matching patterns at each text position. They form a secondary chain alongside failure links.

## Motivation

When processing a text position, the current state $s$ may represent a string whose suffixes match multiple patterns. The output for state $s$ must include not only the pattern ending at $s$ itself (if any), but also all patterns reachable via the failure link chain. Naively following the entire failure link chain at every text position would be too slow.

## Definition

The dictionary link $\text{dict}(s)$ points to the nearest ancestor of $s$ on the failure link chain that is an output state (i.e., a state where some pattern ends):

$$
\text{dict}(s) = \begin{cases}
f(s) & \text{if } f(s) \text{ is an output state}\\
\text{dict}(f(s)) & \text{otherwise}
\end{cases}
$$

## Construction

Dictionary links are computed during the same BFS that builds failure links:

```python
from collections import deque

def build_with_dict_links(goto, fail, output):
    """
    Build failure links and dictionary links.

    output[s] = list of pattern indices that end at state s.
    dict_link[s] = nearest output state via failure chain.
    """
    num_states = len(goto)
    dict_link = [0] * num_states
    queue = deque()

    for ch, s in goto[0].items():
        fail[s] = 0
        dict_link[s] = 0
        queue.append(s)

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
            if output[fail[s]]:
                dict_link[s] = fail[s]
            else:
                dict_link[s] = dict_link[fail[s]]

    return dict_link


def collect_outputs(state, output, dict_link):
    """Collect all matching patterns at a given state."""
    results = list(output[state])
    s = dict_link[state]
    while s != 0:
        results.extend(output[s])
        s = dict_link[s]
    return results
```

## Complexity Analysis

Without dictionary links, reporting all $z$ matches could take $O(n \cdot m)$ in the worst case, because at each text position we might traverse the entire failure link chain. With dictionary links, the chain of output states is compressed:

- **Construction:** $O(m)$ as part of the BFS.
- **Match reporting:** $O(1)$ per reported match, since each dictionary link hop yields at least one output.
- **Total search time:** $O(n + z)$, achieving optimal output-sensitive complexity.

Dictionary links are essential for the theoretical $O(n + m + z)$ guarantee of Aho-Corasick.


# Reference

[Aho, Corasick - Efficient String Matching (1975)](https://doi.org/10.1145/360825.360855)

[Aho-Corasick Algorithm - Stanford CS166](https://web.stanford.edu/class/cs166/)

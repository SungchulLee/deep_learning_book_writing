# DFA Minimization

DFA minimization reduces a deterministic finite automaton to its smallest equivalent DFA. Two DFAs are equivalent if they accept the same language. The minimized DFA is unique (up to state renaming) for any given regular language.

## Hopcroft's Algorithm

The most efficient DFA minimization algorithm is Hopcroft's algorithm (1971), which runs in $O(n \log n)$ time where $n$ is the number of DFA states. It works by iteratively refining a partition of states.

### Key Idea

Two states $p$ and $q$ are **distinguishable** if there exists a string $w$ such that exactly one of $\delta^*(p, w)$ and $\delta^*(q, w)$ is an accepting state. The minimized DFA merges all indistinguishable states.

### Algorithm (Moore's Method)

The simpler Moore's method partitions states iteratively:

1. Start with two groups: accepting states $F$ and non-accepting states $Q \setminus F$.
2. For each group, split it if two states in the group transition to different groups on some input character.
3. Repeat until no group can be split.

$$
\text{States } p, q \text{ are equivalent} \iff \forall a \in \Sigma: \delta(p,a) \text{ and } \delta(q,a) \text{ are in the same group}
$$

```python
def minimize_dfa(states, alphabet, transitions, start, accept_states):
    """Minimize a DFA using partition refinement."""
    # Remove unreachable states
    reachable = set()
    queue = [start]
    reachable.add(start)
    while queue:
        s = queue.pop()
        for a in alphabet:
            t = transitions.get((s, a))
            if t is not None and t not in reachable:
                reachable.add(t)
                queue.append(t)
    states = states & reachable
    accept_states = accept_states & reachable

    # Initial partition
    non_accept = states - accept_states
    partition = []
    if accept_states:
        partition.append(frozenset(accept_states))
    if non_accept:
        partition.append(frozenset(non_accept))

    def state_to_group(s, part):
        for i, group in enumerate(part):
            if s in group:
                return i
        return -1

    changed = True
    while changed:
        changed = False
        new_partition = []
        for group in partition:
            splits = {}
            for s in group:
                signature = tuple(
                    state_to_group(transitions.get((s, a)), partition)
                    if transitions.get((s, a)) is not None else -1
                    for a in sorted(alphabet)
                )
                splits.setdefault(signature, set()).add(s)
            if len(splits) > 1:
                changed = True
            for sub in splits.values():
                new_partition.append(frozenset(sub))
        partition = new_partition

    # Build minimized DFA
    group_rep = {}
    for group in partition:
        rep = min(group)
        for s in group:
            group_rep[s] = rep

    new_states = {group_rep[s] for s in states}
    new_trans = {}
    for (s, a), t in transitions.items():
        if s in states and t in states:
            new_trans[(group_rep[s], a)] = group_rep[t]
    new_start = group_rep[start]
    new_accept = {group_rep[s] for s in accept_states}

    return new_states, new_trans, new_start, new_accept


# Example
states = {0, 1, 2, 3, 4}
alphabet = {'a', 'b'}
transitions = {
    (0, 'a'): 1, (0, 'b'): 2,
    (1, 'a'): 1, (1, 'b'): 3,
    (2, 'a'): 1, (2, 'b'): 2,
    (3, 'a'): 1, (3, 'b'): 4,
    (4, 'a'): 1, (4, 'b'): 2,
}
start = 0
accept = {4}

new_states, new_trans, new_start, new_accept = minimize_dfa(
    states, alphabet, transitions, start, accept
)
print(f"Original states: {len(states)}")
print(f"Minimized states: {len(new_states)}")
# States 0 and 2 are equivalent (both non-accepting, same behavior)
```

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Moore's method | $O(n^2 \cdot |\Sigma|)$ | $O(n)$ |
| Hopcroft's algorithm | $O(n \log n \cdot |\Sigma|)$ | $O(n)$ |

## Myhill-Nerode Theorem

The theoretical foundation for DFA minimization is the **Myhill-Nerode theorem**, which states that the number of states in the minimum DFA for a language $L$ equals the number of equivalence classes of the right-invariant equivalence relation $\equiv_L$ defined by:

$$
x \equiv_L y \iff \forall z \in \Sigma^*: (xz \in L \Leftrightarrow yz \in L)
$$

This guarantees that the minimized DFA is unique.


# Reference

[Hopcroft - An n log n Algorithm for Minimizing States in a Finite Automaton (1971)](https://doi.org/10.1016/B978-0-12-417750-5.50022-1)

[Introduction to Automata Theory - Hopcroft, Motwani, Ullman, Chapter 4](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-automata-theory-languages-and-computation/P200000003517)

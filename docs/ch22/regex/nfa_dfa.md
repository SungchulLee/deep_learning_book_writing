# NFA to DFA

The subset construction (also called the powerset construction) converts a nondeterministic finite automaton (NFA) into an equivalent deterministic finite automaton (DFA). The resulting DFA has at most $2^n$ states for an NFA with $n$ states, though in practice it is usually much smaller.

## Algorithm: Subset Construction

Each DFA state corresponds to a set of NFA states. The DFA start state is the epsilon closure of the NFA start state. For each DFA state $S$ and input character $a$, the DFA transition goes to the epsilon closure of $\bigcup_{s \in S} \delta(s, a)$.

$$

\text{DFA state } D = \{q_1, q_2, \ldots, q_k\} \subseteq Q_{\text{NFA}}

$$

$$

\delta_{\text{DFA}}(D, a) = \varepsilon\text{-closure}\!\left(\bigcup_{q \in D} \delta_{\text{NFA}}(q, a)\right)

$$

A DFA state $D$ is accepting if $D \cap F_{\text{NFA}} \neq \emptyset$.

```python
from collections import deque

def epsilon_closure(states, nfa_epsilon):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in nfa_epsilon.get(s, []):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)

def subset_construction(nfa_start, nfa_accept_states, nfa_transitions, nfa_epsilon, alphabet):
    """Convert NFA to DFA using subset construction."""
    start = epsilon_closure({nfa_start}, nfa_epsilon)
    dfa_states = {start}
    dfa_trans = {}
    dfa_accept = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current & nfa_accept_states:
            dfa_accept.add(current)
        for a in alphabet:
            next_nfa = set()
            for s in current:
                next_nfa.update(nfa_transitions.get((s, a), set()))
            next_state = epsilon_closure(next_nfa, nfa_epsilon)
            if next_state:
                dfa_trans[(current, a)] = next_state
                if next_state not in dfa_states:
                    dfa_states.add(next_state)
                    queue.append(next_state)

    return start, dfa_accept, dfa_trans

# Example NFA for regex (a|b)*abb
nfa_transitions = {
    (2, 'a'): {3}, (4, 'b'): {5},
    (7, 'a'): {8}, (8, 'b'): {9}, (9, 'b'): {10},
}
nfa_epsilon = {0: [1, 7], 1: [2, 4], 3: [6], 5: [6], 6: [1, 7]}
alphabet = {'a', 'b'}

start, accept, trans = subset_construction(0, {10}, nfa_transitions, nfa_epsilon, alphabet)
print(f"DFA start: {sorted(start)}")
print(f"Number of DFA states: {len({start} | {v for v in trans.values()})}")
```

## Complexity Analysis

- **Time:** $O(2^n \cdot |\Sigma|)$ in the worst case, where $n$ is the number of NFA states.
- **Space:** $O(2^n)$ DFA states in the worst case.
- **In practice:** The number of reachable DFA states is usually much smaller than $2^n$.

## Worst-Case Example

The language $L_k = \{w \in \{a,b\}^* : \text{the } k\text{-th symbol from the end is } a\}$ requires an NFA with $O(k)$ states but a DFA with $2^k$ states. This demonstrates that the exponential blowup is tight.

# Reference

[Introduction to Automata Theory - Hopcroft, Motwani, Ullman, Chapter 2](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-automata-theory-languages-and-computation/P200000003517)

[Subset Construction - Wikipedia](https://en.wikipedia.org/wiki/Powerset_construction)

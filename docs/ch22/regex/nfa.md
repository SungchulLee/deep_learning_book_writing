# NFA Construction

A nondeterministic finite automaton (NFA) is a foundational model for regular expression matching. An NFA consists of a finite set of states, an alphabet, a transition function that can map a state and symbol to multiple states (including epsilon/empty transitions), a start state, and a set of accepting states.

## Formal Definition

An NFA is a 5-tuple $(Q, \Sigma, \delta, q_0, F)$ where:

$$

\begin{aligned}
Q &: \text{finite set of states}\\
\Sigma &: \text{input alphabet}\\
\delta &: Q \times (\Sigma \cup \{\varepsilon\}) \to \mathcal{P}(Q) \quad \text{(transition function)}\\
q_0 &\in Q : \text{start state}\\
F &\subseteq Q : \text{set of accepting states}
\end{aligned}

$$

An NFA accepts a string $w$ if there exists at least one path from $q_0$ to some state in $F$ that spells out $w$ (with arbitrary epsilon transitions).

## From Regular Expression to NFA

Every regular expression $r$ can be converted to an equivalent NFA using a compositional (inductive) approach. The three basic operations are:

1. **Single character $a$:** Create two states $s_0 \xrightarrow{a} s_1$.
2. **Concatenation $r_1 r_2$:** Connect the accept state of $N(r_1)$ to the start state of $N(r_2)$ via an $\varepsilon$-transition.
3. **Union $r_1 | r_2$:** Create a new start state with $\varepsilon$-transitions to both NFAs, and both accept states connect via $\varepsilon$ to a new accept state.
4. **Kleene star $r^*$:** Add $\varepsilon$-transitions for repetition and bypassing.

```python
class State:
    _id_counter = 0
    def __init__(self, is_accept=False):
        self.id = State._id_counter
        State._id_counter += 1
        self.is_accept = is_accept
        self.transitions = {}  # char -> list of states
        self.epsilon = []      # epsilon transitions

    def add_transition(self, char, state):
        self.transitions.setdefault(char, []).append(state)

    def add_epsilon(self, state):
        self.epsilon.append(state)

class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def char_nfa(c):
    start = State()
    accept = State(is_accept=True)
    start.add_transition(c, accept)
    return NFA(start, accept)

def concat_nfa(n1, n2):
    n1.accept.is_accept = False
    n1.accept.add_epsilon(n2.start)
    return NFA(n1.start, n2.accept)

def union_nfa(n1, n2):
    start = State()
    accept = State(is_accept=True)
    start.add_epsilon(n1.start)
    start.add_epsilon(n2.start)
    n1.accept.is_accept = False
    n1.accept.add_epsilon(accept)
    n2.accept.is_accept = False
    n2.accept.add_epsilon(accept)
    return NFA(start, accept)

def star_nfa(n1):
    start = State()
    accept = State(is_accept=True)
    start.add_epsilon(n1.start)
    start.add_epsilon(accept)
    n1.accept.is_accept = False
    n1.accept.add_epsilon(n1.start)
    n1.accept.add_epsilon(accept)
    return NFA(start, accept)
```

## Simulating an NFA

To check if an NFA accepts a string, we track the set of current states (including epsilon closures):

```python
def epsilon_closure(states):
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in s.epsilon:
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return closure

def nfa_accepts(nfa, text):
    current = epsilon_closure({nfa.start})
    for ch in text:
        next_states = set()
        for state in current:
            for target in state.transitions.get(ch, []):
                next_states.add(target)
        current = epsilon_closure(next_states)
    return any(s.is_accept for s in current)
```

## Complexity

- **Construction:** $O(r)$ states and transitions for a regular expression of length $r$.
- **Simulation:** $O(n \cdot r)$ per string of length $n$, since each step may visit $O(r)$ states.

# Reference

[Introduction to Automata Theory, Languages, and Computation - Hopcroft, Motwani, Ullman](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-automata-theory-languages-and-computation/P200000003517)

[Regular Expression Matching Can Be Simple And Fast - Russ Cox](https://swtch.com/~rsc/regexp/regexp1.html)

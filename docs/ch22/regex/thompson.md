# Thompson's Construction

Thompson's construction is a systematic algorithm for converting any regular expression into an equivalent nondeterministic finite automaton (NFA). Proposed by Ken Thompson in 1968, it produces an NFA with at most $2r$ states for a regular expression of length $r$, where each state has at most two outgoing transitions.

## Algorithm

The construction is recursive, mirroring the structure of the regular expression. Every sub-NFA has exactly one start state and one accepting state.

### Base Cases

**Empty string $\varepsilon$:** $q_0 \xrightarrow{\varepsilon} q_f$

**Single character $a$:** $q_0 \xrightarrow{a} q_f$

### Inductive Cases

**Concatenation $r_1 r_2$:** Merge the accept state of $N(r_1)$ with the start state of $N(r_2)$.

**Union $r_1 | r_2$:** New start state with $\varepsilon$-transitions to both sub-NFAs; both accept states connect via $\varepsilon$ to a new accept state.

**Kleene star $r_1^*$:** New start state and accept state, with $\varepsilon$-transitions allowing zero or more repetitions.

## Implementation

```python
class State:
    _counter = 0
    def __init__(self):
        self.id = State._counter
        State._counter += 1
        self.char_trans = {}
        self.epsilon = []
        self.is_accept = False

class Fragment:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def thompson(postfix):
    """Build NFA from postfix regex. Operators: . (concat), | (union), * (star)"""
    stack = []
    for ch in postfix:
        if ch == '.':
            f2 = stack.pop()
            f1 = stack.pop()
            f1.accept.epsilon.append(f2.start)
            f1.accept.is_accept = False
            stack.append(Fragment(f1.start, f2.accept))
        elif ch == '|':
            f2 = stack.pop()
            f1 = stack.pop()
            start = State()
            accept = State()
            accept.is_accept = True
            start.epsilon = [f1.start, f2.start]
            f1.accept.epsilon.append(accept)
            f1.accept.is_accept = False
            f2.accept.epsilon.append(accept)
            f2.accept.is_accept = False
            stack.append(Fragment(start, accept))
        elif ch == '*':
            f = stack.pop()
            start = State()
            accept = State()
            accept.is_accept = True
            start.epsilon = [f.start, accept]
            f.accept.epsilon = [f.start, accept]
            f.accept.is_accept = False
            stack.append(Fragment(start, accept))
        else:
            start = State()
            accept = State()
            accept.is_accept = True
            start.char_trans[ch] = accept
            stack.append(Fragment(start, accept))
    return stack.pop()


def infix_to_postfix(regex):
    """Convert infix regex to postfix with explicit concat operator."""
    output = []
    for i, ch in enumerate(regex):
        output.append(ch)
        if ch not in ('(', '|') and i + 1 < len(regex):
            next_ch = regex[i + 1]
            if next_ch not in (')', '|', '*'):
                output.append('.')
    expr = ''.join(output)

    precedence = {'*': 3, '.': 2, '|': 1}
    result = []
    ops = []
    for ch in expr:
        if ch == '(':
            ops.append(ch)
        elif ch == ')':
            while ops and ops[-1] != '(':
                result.append(ops.pop())
            ops.pop()
        elif ch in precedence:
            while (ops and ops[-1] != '(' and
                   ops[-1] in precedence and
                   precedence[ops[-1]] >= precedence[ch]):
                result.append(ops.pop())
            ops.append(ch)
        else:
            result.append(ch)
    while ops:
        result.append(ops.pop())
    return ''.join(result)


# Example: regex "a(b|c)*d"
postfix = infix_to_postfix("a(b|c)*d")
print(f"Postfix: {postfix}")
# Output: Postfix: abc|*.d.

nfa = thompson(postfix)
print(f"Start state: {nfa.start.id}, Accept state: {nfa.accept.id}")
```

## Properties

1. **At most $2r$ states** for a regex of length $r$ (each operator creates at most 2 new states).
2. **Each state has at most two outgoing transitions** (either one character transition, or up to two epsilon transitions).
3. **The accept state has no outgoing transitions.**
4. **The construction is linear:** $O(r)$ time and space.


# Reference

[Thompson - Regular Expression Search Algorithm (1968)](https://doi.org/10.1145/363347.363387)

[Regular Expression Matching Can Be Simple And Fast - Russ Cox](https://swtch.com/~rsc/regexp/regexp1.html)

# Balanced Parentheses

Compilers, text editors, and configuration parsers all need to verify that opening brackets are correctly matched with closing brackets. A string like `({[]})` is balanced because every opener has a corresponding closer in the right order, while `({[})` is not because the square bracket closes before the curly brace. The stack is the natural data structure for this problem: its LIFO property ensures that the most recent unmatched opener is always on top, ready to be matched against the next closer. This page describes the balanced parentheses algorithm, proves its correctness, and extends it to handle multiple bracket types.

## The Algorithm

Given a string containing bracket characters, determine whether all brackets are properly nested and matched.

1. Initialize an empty stack
2. For each character in the string:
    - If it is an **opening bracket** (`(`, `[`, `{`), push it onto the stack
    - If it is a **closing bracket** (`)`, `]`, `}`):
        - If the stack is empty, return **unbalanced** (no matching opener)
        - Pop the top of the stack and check that it matches the closing bracket
        - If it does not match, return **unbalanced**
3. After processing all characters, if the stack is empty, the string is **balanced**; otherwise it is **unbalanced** (unmatched openers remain)

The algorithm runs in $O(n)$ time and $O(n)$ space, where $n$ is the length of the string. Each character is processed exactly once, and each character is pushed and popped at most once.

??? example "Trace: `({[]})`"
    | Step | Char | Action | Stack |
    |------|------|--------|-------|
    | 1 | `(` | Push | `(` |
    | 2 | `{` | Push | `( {` |
    | 3 | `[` | Push | `( { [` |
    | 4 | `]` | Pop `[`, matches | `( {` |
    | 5 | `}` | Pop `{`, matches | `(` |
    | 6 | `)` | Pop `(`, matches | (empty) |

    Stack is empty at the end --- **balanced**.

??? example "Trace: `({[)}`"
    | Step | Char | Action | Stack |
    |------|------|--------|-------|
    | 1 | `(` | Push | `(` |
    | 2 | `{` | Push | `( {` |
    | 3 | `[` | Push | `( { [` |
    | 4 | `)` | Pop `[`, does NOT match `)` | **FAIL** |

    Mismatch detected --- **unbalanced**.

## Implementation

```python
"""
Balanced parentheses — check whether brackets are properly matched.

Uses a stack to match each closing bracket with the most recent
unmatched opening bracket, supporting (, [, and { bracket types.
"""


# === Balanced Parentheses Checker =============================================

MATCHING = {")": "(", "]": "[", "}": "{"}
OPENERS = set(MATCHING.values())
CLOSERS = set(MATCHING.keys())


def is_balanced(s):
    """Check whether the brackets in string s are balanced.

    Time:  O(n) — single pass through the string.
    Space: O(n) — stack can hold up to n/2 openers.
    """
    stack = []
    for i, ch in enumerate(s):
        if ch in OPENERS:
            stack.append(ch)
        elif ch in CLOSERS:
            if not stack:
                return False, f"Unmatched closer '{ch}' at position {i}"
            top = stack.pop()
            if top != MATCHING[ch]:
                return False, f"Mismatch: '{top}' at stack vs '{ch}' at position {i}"
    if stack:
        return False, f"Unmatched opener(s): {stack}"
    return True, "Balanced"


# === Demonstration ============================================================

if __name__ == "__main__":
    test_cases = [
        "({[]})",
        "(())[{}]",
        "((()))",
        "",
        "({[)}",
        "(((",
        "())",
        "}{",
        "a * (b + c) - [d / {e + f}]",
    ]

    print(f"{'Expression':<35s} {'Balanced?':<12s} {'Detail'}")
    print("-" * 75)
    for expr in test_cases:
        balanced, detail = is_balanced(expr)
        display = repr(expr) if expr else "''"
        print(f"{display:<35s} {str(balanced):<12s} {detail}")
```

**Output:**
```
Expression                          Balanced?    Detail
---------------------------------------------------------------------------
'({[]})'                            True         Balanced
'(())[{}]'                          True         Balanced
'((()))'                            True         Balanced
''                                  True         Balanced
'({[)}'                             False        Mismatch: '[' at stack vs ')' at position 3
'((('                               False        Unmatched opener(s): ['(', '(', '(']
'())'                               False        Unmatched closer ')' at position 2
'}{'                                False        Unmatched closer '}' at position 0
'a * (b + c) - [d / {e + f}]'       True         Balanced
```

The algorithm correctly identifies balanced and unbalanced strings, providing specific error messages for mismatches and unmatched brackets.

## Correctness

The algorithm is correct because the stack maintains the following invariant:

!!! info "Stack Invariant"
    At any point during the scan, the stack contains exactly the unmatched opening brackets, in the order they were encountered. The top of the stack is the most recent unmatched opener.

**Why this guarantees correct matching**: bracket nesting is inherently LIFO. If we encounter `( [ {`, the `{` must be closed before the `[`, and the `[` before the `(`. The stack enforces exactly this order by always matching against the top element.

**Completeness**: the algorithm detects all three types of imbalance:

1. **Extra closer** --- the stack is empty when a closer arrives
2. **Mismatched pair** --- the top of the stack does not match the closer
3. **Extra opener** --- the stack is non-empty after all characters are processed

## Complexity

$$
T(n) = O(n) \quad \text{and} \quad S(n) = O(n)
$$

The time bound is tight because every character must be examined. The space bound is tight for strings like `(((...(` with $n/2$ or more opening brackets.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.

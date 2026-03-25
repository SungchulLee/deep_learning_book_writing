# Infix to Postfix

Humans write arithmetic in **infix notation** --- the operator sits between its operands, as in `3 + 4 * 2`. This notation is convenient to read but difficult to evaluate programmatically because it requires knowing operator precedence and handling parentheses. **Postfix notation** (Reverse Polish Notation, RPN) places operators after their operands: `3 4 2 * +`. This representation is unambiguous --- no parentheses are ever needed --- and can be evaluated in a single left-to-right pass with a stack. The **Shunting Yard algorithm**, invented by Edsger Dijkstra, converts infix expressions to postfix using a stack to temporarily hold operators. This page describes the algorithm in detail and traces through several examples.

## Three Notation Systems

The same arithmetic expression can be written in three notations:

| Notation | Format | Example (for $3 + 4 \times 2$) |
|----------|--------|-------------------------------|
| **Infix** | Operator between operands | `3 + 4 * 2` |
| **Postfix** (RPN) | Operator after operands | `3 4 2 * +` |
| **Prefix** (Polish) | Operator before operands | `+ 3 * 4 2` |

Postfix and prefix notations are parenthesis-free: the order of operators encodes precedence unambiguously. Compilers typically convert infix source code to postfix (or a tree representation) before generating machine instructions.

## The Shunting Yard Algorithm

The algorithm uses an **output queue** and an **operator stack**. It processes tokens from left to right:

1. **Number** --- send directly to the output queue
2. **Operator** $o_1$ --- while there is an operator $o_2$ on top of the stack with greater or equal precedence (for left-associative operators), pop $o_2$ to the output queue. Then push $o_1$ onto the stack.
3. **Left parenthesis** `(` --- push onto the stack
4. **Right parenthesis** `)` --- pop operators from the stack to the output queue until a left parenthesis is encountered, then discard the left parenthesis

After all tokens are processed, pop any remaining operators from the stack to the output queue.

The algorithm runs in $O(n)$ time and $O(n)$ space, where $n$ is the number of tokens. Each token is pushed and popped at most once.

??? example "Worked Example: `3 + 4 * 2`"
    | Step | Token | Action | Output Queue | Operator Stack |
    |------|-------|--------|--------------|----------------|
    | 1 | `3` | Output | `3` | |
    | 2 | `+` | Push (stack empty) | `3` | `+` |
    | 3 | `4` | Output | `3 4` | `+` |
    | 4 | `*` | Push (`*` > `+`) | `3 4` | `+ *` |
    | 5 | `2` | Output | `3 4 2` | `+ *` |
    | 6 | End | Pop all | `3 4 2 * +` | |

    Result: `3 4 2 * +`

??? example "Worked Example: `(3 + 4) * 2`"
    | Step | Token | Action | Output Queue | Operator Stack |
    |------|-------|--------|--------------|----------------|
    | 1 | `(` | Push | | `(` |
    | 2 | `3` | Output | `3` | `(` |
    | 3 | `+` | Push | `3` | `( +` |
    | 4 | `4` | Output | `3 4` | `( +` |
    | 5 | `)` | Pop until `(` | `3 4 +` | |
    | 6 | `*` | Push | `3 4 +` | `*` |
    | 7 | `2` | Output | `3 4 + 2` | `*` |
    | 8 | End | Pop all | `3 4 + 2 *` | |

    Result: `3 4 + 2 *`

## Implementation

```python
"""
Infix to postfix conversion — Dijkstra's Shunting Yard algorithm.

Converts infix arithmetic expressions to postfix (RPN) notation
using a stack to manage operator precedence and parentheses.
"""


# === Shunting Yard Algorithm ==================================================

PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def tokenize(expression):
    """Split an expression string into a list of tokens."""
    tokens = []
    i = 0
    while i < len(expression):
        if expression[i].isspace():
            i += 1
        elif expression[i] in "()+-*/":
            tokens.append(expression[i])
            i += 1
        else:
            j = i
            while j < len(expression) and (expression[j].isdigit() or expression[j] == "."):
                j += 1
            tokens.append(expression[i:j])
            i = j
    return tokens


def infix_to_postfix(expression):
    """Convert an infix expression to postfix (RPN) notation.

    Uses Dijkstra's Shunting Yard algorithm.
    Assumes left-to-right associativity for all operators.

    Time:  O(n) where n is the number of tokens.
    Space: O(n) for the operator stack and output queue.
    """
    tokens = tokenize(expression)
    output = []
    operator_stack = []

    for token in tokens:
        if token in PRECEDENCE:
            # Pop operators with >= precedence (left-associative)
            while (operator_stack and
                   operator_stack[-1] != "(" and
                   operator_stack[-1] in PRECEDENCE and
                   PRECEDENCE[operator_stack[-1]] >= PRECEDENCE[token]):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            operator_stack.pop()  # discard the "("
        else:
            output.append(token)  # operand

    # Pop remaining operators
    while operator_stack:
        output.append(operator_stack.pop())

    return " ".join(output)


# === Postfix Evaluator (for verification) =====================================

def evaluate_postfix(postfix_str):
    """Evaluate a postfix expression to verify conversion correctness."""
    operators = {
        "+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "/": lambda a, b: a / b,
    }
    stack = []
    for token in postfix_str.split():
        if token in operators:
            b, a = stack.pop(), stack.pop()
            stack.append(operators[token](a, b))
        else:
            stack.append(float(token))
    return stack[0]


# === Demonstration ============================================================

if __name__ == "__main__":
    test_cases = [
        "3 + 4 * 2",
        "(3 + 4) * 2",
        "5 + (1 + 2) * 4 - 3",
        "2 * 3 + 4",
        "2 + 3 * 4 + 5",
        "(2 + 3) * (4 + 5)",
    ]

    print(f"{'Infix':<30s} {'Postfix':<25s} {'Value':>8s}")
    print("-" * 65)
    for expr in test_cases:
        postfix = infix_to_postfix(expr)
        value = evaluate_postfix(postfix)
        print(f"{expr:<30s} {postfix:<25s} {value:>8.1f}")
```

**Output:**
```
Infix                          Postfix                     Value
-----------------------------------------------------------------
3 + 4 * 2                      3 4 2 * +                     11.0
(3 + 4) * 2                    3 4 + 2 *                     14.0
5 + (1 + 2) * 4 - 3            5 1 2 + 4 * + 3 -            14.0
2 * 3 + 4                      2 3 * 4 +                     10.0
2 + 3 * 4 + 5                  2 3 4 * + 5 +                 19.0
(2 + 3) * (4 + 5)              2 3 + 4 5 + *                 45.0
```

Each infix expression is correctly converted to its postfix equivalent. The postfix evaluator confirms that the converted expression produces the same value as the original.

## Handling Right-Associative Operators

The algorithm above assumes left-to-right associativity: operators of equal precedence are evaluated left to right. For right-associative operators like exponentiation (`^`), the comparison changes from "greater or equal" to "strictly greater":

- **Left-associative**: pop while precedence of stack top $\geq$ current operator
- **Right-associative**: pop while precedence of stack top $>$ current operator

This ensures that `2 ^ 3 ^ 4` converts to `2 3 4 ^ ^` (evaluated as $2^{(3^4)}$) rather than `2 3 ^ 4 ^` (which would give $(2^3)^4$).

## Complexity Analysis

$$
T(n) = O(n) \quad \text{and} \quad S(n) = O(n)
$$

Each of the $n$ tokens is processed exactly once. Each token enters the operator stack at most once and leaves at most once, so the total number of stack operations is at most $2n$. The output queue grows to at most $n$ entries.

## Reference

- Dijkstra, E. W. (1961). "Algol 60 translation." *ALGOL Bulletin*, Supplement 10.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.

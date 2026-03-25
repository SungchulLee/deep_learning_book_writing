# Expression Evaluation

Arithmetic expressions like `3 + 4 * 2` require careful handling of operator precedence and parentheses. In **infix notation** --- the notation humans use --- multiplication binds tighter than addition, and parentheses override the default precedence. Evaluating such expressions directly is cumbersome because the evaluator must look ahead to determine whether a pending operator should be applied now or later. **Postfix notation** (also called Reverse Polish Notation, RPN) eliminates this problem entirely: operands appear before their operators, and no parentheses are needed. A stack makes postfix evaluation straightforward --- scan left to right, push operands, and apply operators as they appear. This page explains both the postfix evaluation algorithm and a direct infix evaluator that uses two stacks.

## Postfix Evaluation Algorithm

Given a postfix expression as a sequence of tokens (operands and operators), the evaluation algorithm works as follows:

1. Initialize an empty stack
2. For each token from left to right:
    - If the token is a **number**, push it onto the stack
    - If the token is an **operator** ($+$, $-$, $\times$, $\div$), pop two operands, apply the operator, and push the result
3. After all tokens are processed, the stack contains exactly one element: the result

The algorithm runs in $O(n)$ time and $O(n)$ space, where $n$ is the number of tokens.

??? example "Worked Example: Postfix Evaluation"
    Evaluate the postfix expression `3 4 2 * +` (equivalent to infix `3 + 4 * 2`):

    | Step | Token | Action | Stack |
    |------|-------|--------|-------|
    | 1 | `3` | Push 3 | `[3]` |
    | 2 | `4` | Push 4 | `[3, 4]` |
    | 3 | `2` | Push 2 | `[3, 4, 2]` |
    | 4 | `*` | Pop 2 and 4, push $4 \times 2 = 8$ | `[3, 8]` |
    | 5 | `+` | Pop 8 and 3, push $3 + 8 = 11$ | `[11]` |

    Result: **11**

## Two-Stack Infix Evaluation

Dijkstra's two-stack algorithm evaluates fully parenthesized infix expressions directly. It uses one stack for operands and one for operators:

1. **Left parenthesis** `(` --- ignore (or push to operator stack for tracking)
2. **Number** --- push onto the operand stack
3. **Operator** --- push onto the operator stack
4. **Right parenthesis** `)` --- pop an operator and two operands, apply the operator, push the result

For expressions that are not fully parenthesized, the algorithm is extended with precedence rules: before pushing an operator, pop and apply any operators on the stack that have equal or higher precedence.

```python
"""
Expression evaluation — postfix and infix evaluation using stacks.

Demonstrates the postfix (RPN) evaluation algorithm and a two-stack
infix evaluator that handles operator precedence.
"""


# === Postfix Evaluator ========================================================

def evaluate_postfix(expression):
    """Evaluate a postfix (RPN) expression.

    Args:
        expression: space-separated string of numbers and operators.

    Returns:
        The numeric result.

    Time:  O(n) where n is the number of tokens.
    Space: O(n) for the operand stack.
    """
    operators = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }
    stack = []
    tokens = expression.split()

    print(f"  Evaluating postfix: {expression}")
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            result = operators[token](a, b)
            stack.append(result)
            print(f"    {token}: {a} {token} {b} = {result}  →  stack = {stack}")
        else:
            stack.append(float(token))
            print(f"    push {token}  →  stack = {stack}")

    return stack[0]


# === Infix Evaluator with Precedence ==========================================

PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def evaluate_infix(expression):
    """Evaluate an infix expression with operator precedence and parentheses.

    Uses two stacks: one for operands, one for operators.
    Handles +, -, *, / and parentheses.

    Time:  O(n) where n is the number of tokens.
    Space: O(n) for the two stacks.
    """
    def apply_operator(ops, vals):
        op = ops.pop()
        b = vals.pop()
        a = vals.pop()
        operators = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
                     "*": lambda a, b: a * b, "/": lambda a, b: a / b}
        vals.append(operators[op](a, b))

    def tokenize(expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
            elif expr[i] in "()+-*/":
                tokens.append(expr[i])
                i += 1
            else:
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                    j += 1
                tokens.append(expr[i:j])
                i = j
        return tokens

    tokens = tokenize(expression)
    vals = []
    ops = []

    for token in tokens:
        if token == "(":
            ops.append(token)
        elif token == ")":
            while ops[-1] != "(":
                apply_operator(ops, vals)
            ops.pop()  # remove the "("
        elif token in PRECEDENCE:
            while (ops and ops[-1] != "(" and
                   ops[-1] in PRECEDENCE and
                   PRECEDENCE[ops[-1]] >= PRECEDENCE[token]):
                apply_operator(ops, vals)
            ops.append(token)
        else:
            vals.append(float(token))

    while ops:
        apply_operator(ops, vals)

    return vals[0]


# === Demonstration ============================================================

if __name__ == "__main__":
    # Postfix evaluation
    result1 = evaluate_postfix("3 4 2 * +")
    print(f"  Result: {result1}")
    print()

    result2 = evaluate_postfix("5 1 2 + 4 * + 3 -")
    print(f"  Result: {result2}")
    print()

    # Infix evaluation
    expressions = [
        "3 + 4 * 2",
        "(3 + 4) * 2",
        "5 + (1 + 2) * 4 - 3",
        "10 / 2 + 3 * 4",
    ]
    print("  Infix evaluation:")
    for expr in expressions:
        result = evaluate_infix(expr)
        print(f"    {expr:30s} = {result}")
```

**Output:**
```
  Evaluating postfix: 3 4 2 * +
    push 3  →  stack = [3.0]
    push 4  →  stack = [3.0, 4.0]
    push 2  →  stack = [3.0, 4.0, 2.0]
    *: 4.0 * 2.0 = 8.0  →  stack = [3.0, 8.0]
    +: 3.0 + 8.0 = 11.0  →  stack = [11.0]
  Result: 11.0

  Evaluating postfix: 5 1 2 + 4 * + 3 -
    push 5  →  stack = [5.0]
    push 1  →  stack = [5.0, 1.0]
    push 2  →  stack = [5.0, 1.0, 2.0]
    +: 1.0 + 2.0 = 3.0  →  stack = [5.0, 3.0]
    push 4  →  stack = [5.0, 3.0, 4.0]
    *: 3.0 * 4.0 = 12.0  →  stack = [5.0, 12.0]
    +: 5.0 + 12.0 = 17.0  →  stack = [17.0]
    push 3  →  stack = [17.0, 3.0]
    -: 17.0 - 3.0 = 14.0  →  stack = [14.0]
  Result: 14.0

  Infix evaluation:
    3 + 4 * 2                      = 11.0
    (3 + 4) * 2                    = 14.0
    5 + (1 + 2) * 4 - 3            = 14.0
    10 / 2 + 3 * 4                 = 17.0
```

The postfix evaluator traces each step, showing how the stack grows and shrinks. The infix evaluator produces the same results as standard arithmetic, correctly respecting operator precedence and parentheses.

## Correctness

The postfix evaluation algorithm is correct because postfix notation encodes the evaluation order explicitly. Each operator immediately follows its operands, so when the algorithm encounters an operator, the top two stack elements are guaranteed to be its operands (assuming the input expression is valid).

For the infix evaluator, correctness follows from the precedence-based deferral rule: an operator is applied only after all higher-precedence operators to its right have been applied. This produces the same result as the standard mathematical interpretation.

!!! warning "Invalid Expressions"
    Both algorithms assume valid input. A postfix expression with too few operands for an operator will cause an underflow error. An infix expression with mismatched parentheses will produce incorrect results or raise exceptions. Production implementations should include input validation.

## Complexity Summary

| Algorithm | Time | Space |
|---|---|---|
| Postfix evaluation | $O(n)$ | $O(n)$ |
| Two-stack infix evaluation | $O(n)$ | $O(n)$ |

Both algorithms make a single pass through the token sequence. Each token is pushed and popped at most once, giving linear time.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
- Dijkstra, E. W. (1961). "Algol 60 translation." *ALGOL Bulletin*, Supplement 10.

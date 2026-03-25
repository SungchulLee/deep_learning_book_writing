# Function Call Simulation

Every time a program calls a function, the runtime pushes a **stack frame** containing the return address, local variables, and parameters. When the function returns, its frame is popped and execution resumes at the saved return address. This mechanism --- the **call stack** --- is the reason recursive algorithms work at all: each recursive invocation gets its own frame with its own copy of local state. Understanding this process is essential for converting recursive algorithms to iterative ones and for diagnosing stack overflow errors. This page explains call stack mechanics in detail and demonstrates how to simulate them with an explicit stack data structure.

## Stack Frame Anatomy

Each stack frame contains three pieces of information:

1. **Return address** --- where to resume execution after the function completes
2. **Parameters** --- the arguments passed to the function
3. **Local variables** --- any variables declared inside the function body

When function `A` calls function `B`, the runtime:

1. Pushes a new frame for `B` (with `B`'s parameters and locals) onto the call stack
2. Transfers control to `B`'s first instruction
3. When `B` returns, pops `B`'s frame
4. Resumes `A` at the instruction following the call

The LIFO ordering of the stack ensures that the most recently called function finishes first, which matches the nesting structure of function calls.

## Call Stack Depth and Space

For a chain of $n$ nested function calls, the call stack grows to depth $n$, consuming $O(n)$ memory. Most language runtimes impose a maximum stack depth (e.g., Python defaults to 1000). A recursive function with recursion depth $d$ requires $O(d)$ stack space. If $d$ exceeds the limit, a **stack overflow** occurs.

$$
\text{Space}(n) = n \times (\text{frame size}) = O(n)
$$

This is why deeply recursive algorithms sometimes need to be converted to iterative versions using an explicit stack.

## Simulating Recursion with an Explicit Stack

Any recursive algorithm can be converted to an iterative one by replacing the implicit call stack with a user-managed stack. Each "frame" on the explicit stack stores whatever state the recursive call would have needed. The following example demonstrates this with factorial computation.

```python
"""
Function call simulation — modeling the call stack with an explicit stack.

Demonstrates how recursive function calls correspond to push/pop operations
on a stack, and how to convert recursion to iteration using this insight.
"""


# === Stack Frame Representation ===============================================

class Frame:
    """Represents a single stack frame in the simulated call stack."""

    def __init__(self, func_name, params, return_addr=None):
        self.func_name = func_name
        self.params = params
        self.return_addr = return_addr
        self.local_vars = {}

    def __repr__(self):
        return f"Frame({self.func_name}, params={self.params}, locals={self.local_vars})"


# === Recursive Factorial (uses implicit call stack) ===========================

def factorial_recursive(n):
    """Compute n! recursively. Each call creates an implicit stack frame."""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)


# === Iterative Factorial via Explicit Stack ===================================

def factorial_stack_simulation(n):
    """Compute n! by simulating the call stack explicitly.

    Each 'call' pushes a frame; each 'return' pops a frame and
    passes the result to the caller.
    """
    # Phase 1: Simulate the recursive descent (push frames)
    call_stack = []
    print("  === Pushing frames (recursive descent) ===")
    for i in range(n, 0, -1):
        frame = Frame("factorial", {"n": i})
        call_stack.append(frame)
        print(f"    PUSH {frame}")

    # Phase 2: Simulate the returns (pop frames and accumulate result)
    print("  === Popping frames (returning results) ===")
    result = 1  # base case: factorial(1) = 1
    while call_stack:
        frame = call_stack.pop()
        current_n = frame.params["n"]
        result *= current_n
        print(f"    POP  {frame.func_name}(n={current_n}) → result so far = {result}")

    return result


# === Simulating Mutual Recursion ==============================================

def simulate_even_odd(n):
    """Simulate mutually recursive is_even/is_odd using an explicit stack.

    is_even(n) = is_odd(n-1), is_odd(n) = is_even(n-1)
    is_even(0) = True, is_odd(0) = False
    """
    call_stack = [Frame("is_even", {"n": n})]
    print(f"  Checking if {n} is even via mutual recursion simulation:")

    while True:
        frame = call_stack[-1]
        fn = frame.func_name
        current_n = frame.params["n"]

        if current_n == 0:
            # Base case
            result = (fn == "is_even")
            call_stack.pop()
            print(f"    BASE  {fn}(0) = {result}")
            break

        # Simulate the mutual call
        next_fn = "is_odd" if fn == "is_even" else "is_even"
        new_frame = Frame(next_fn, {"n": current_n - 1})
        call_stack.append(new_frame)
        print(f"    CALL  {fn}({current_n}) → {next_fn}({current_n - 1})")

    # Unwind remaining frames
    while call_stack:
        frame = call_stack.pop()
        print(f"    RETURN {frame.func_name}({frame.params['n']}) = {result}")

    return result


# === Demonstration ============================================================

if __name__ == "__main__":
    # Recursive factorial
    n = 5
    print(f"Recursive factorial({n}) = {factorial_recursive(n)}")
    print()

    # Stack-simulated factorial
    print(f"Stack-simulated factorial({n}):")
    result = factorial_stack_simulation(n)
    print(f"  Result: {result}")
    print()

    # Mutual recursion simulation
    for test_n in [4, 7]:
        answer = simulate_even_odd(test_n)
        print(f"  is_even({test_n}) = {answer}")
        print()
```

**Output:**
```
Recursive factorial(5) = 120

Stack-simulated factorial(5):
  === Pushing frames (recursive descent) ===
    PUSH Frame(factorial, params={'n': 5}, locals={})
    PUSH Frame(factorial, params={'n': 4}, locals={})
    PUSH Frame(factorial, params={'n': 3}, locals={})
    PUSH Frame(factorial, params={'n': 2}, locals={})
    PUSH Frame(factorial, params={'n': 1}, locals={})
  === Popping frames (returning results) ===
    POP  factorial(n=1) → result so far = 1
    POP  factorial(n=2) → result so far = 2
    POP  factorial(n=3) → result so far = 6
    POP  factorial(n=4) → result so far = 24
    POP  factorial(n=5) → result so far = 120
  Result: 120

  Checking if 4 is even via mutual recursion simulation:
    CALL  is_even(4) → is_odd(3)
    CALL  is_odd(3) → is_even(2)
    CALL  is_even(2) → is_odd(1)
    CALL  is_odd(1) → is_even(0)
    BASE  is_even(0) = True
    RETURN is_odd(1) = True
    RETURN is_even(2) = True
    RETURN is_odd(3) = True
    RETURN is_even(4) = True
  is_even(4) = True

  Checking if 7 is even via mutual recursion simulation:
    CALL  is_even(7) → is_odd(6)
    CALL  is_odd(6) → is_even(5)
    CALL  is_even(5) → is_odd(4)
    CALL  is_odd(4) → is_even(3)
    CALL  is_even(3) → is_odd(2)
    CALL  is_odd(2) → is_even(1)
    CALL  is_even(1) → is_odd(0)
    BASE  is_odd(0) = False
    RETURN is_even(1) = False
    RETURN is_odd(2) = False
    RETURN is_even(3) = False
    RETURN is_odd(4) = False
    RETURN is_even(5) = False
    RETURN is_odd(6) = False
    RETURN is_even(7) = False
  is_even(7) = False

```

The simulation makes visible what normally happens behind the scenes. Each recursive call corresponds to a `PUSH` operation, and each return corresponds to a `POP`. The stack depth at any moment equals the current recursion depth.

## When to Use Explicit Stacks

Converting recursion to iteration via an explicit stack is useful in three situations:

1. **Avoiding stack overflow** --- languages with limited call stack size (Python, Java) can overflow on deep recursion. An explicit stack uses heap memory, which is typically much larger.
2. **Performance** --- eliminating function call overhead (parameter passing, frame allocation) can provide a constant-factor speedup.
3. **State inspection** --- an explicit stack can be inspected, serialized, or modified during execution, which is impossible with the implicit call stack.

!!! tip "General Conversion Pattern"
    To convert any recursive function to an iterative one: (1) identify what state each recursive call needs, (2) define a frame structure holding that state, (3) replace each recursive call with a push, and (4) replace each return with a pop.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.

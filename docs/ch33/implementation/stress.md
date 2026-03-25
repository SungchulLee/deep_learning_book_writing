# Stress Testing

When sample test cases pass but hidden cases fail, the most effective debugging strategy is stress testing: automatically generating random inputs, running both a brute-force and an optimized solution, and reporting the first discrepancy. A single stress test script can find bugs that hours of manual inspection would miss.

## The Stress Testing Framework

A stress test has three components:

1. **Generator**: produces random valid inputs.
2. **Brute-force solution**: a simple, obviously correct but slow implementation.
3. **Optimized solution**: the candidate solution to verify.

The test loop generates an input, feeds it to both solutions, and compares outputs. The first mismatch reveals a counterexample.

```python
"""
Stress testing framework for competitive programming.

Compares a brute-force solution against an optimized solution
on randomly generated inputs to find counterexamples.
"""

import random

# ===================================================================
# Brute-Force Solution
# ===================================================================

def brute_force(arr):
    """Find maximum subarray sum using O(n^2) brute force."""
    n = len(arr)
    if n == 0:
        return 0
    best = arr[0]
    for i in range(n):
        current = 0
        for j in range(i, n):
            current += arr[j]
            best = max(best, current)
    return best

# ===================================================================
# Optimized Solution (Kadane's Algorithm)
# ===================================================================

def optimized(arr):
    """Find maximum subarray sum using Kadane's algorithm, O(n)."""
    if not arr:
        return 0
    max_ending = arr[0]
    max_so_far = arr[0]
    for i in range(1, len(arr)):
        max_ending = max(arr[i], max_ending + arr[i])
        max_so_far = max(max_so_far, max_ending)
    return max_so_far

# ===================================================================
# Random Input Generator
# ===================================================================

def generate_input(max_n=20, max_val=100):
    """Generate a random array of integers."""
    n = random.randint(1, max_n)
    arr = [random.randint(-max_val, max_val) for _ in range(n)]
    return arr

# ===================================================================
# Stress Test Loop
# ===================================================================

def stress_test(num_tests=10000):
    """Run stress test comparing brute force and optimized solutions."""
    for test in range(1, num_tests + 1):
        arr = generate_input()
        expected = brute_force(arr)
        actual = optimized(arr)

        if expected != actual:
            print(f"MISMATCH on test {test}!")
            print(f"Input: {arr}")
            print(f"Brute force: {expected}")
            print(f"Optimized:   {actual}")
            return False

        if test % 1000 == 0:
            print(f"Passed {test} tests...")

    print(f"All {num_tests} tests passed.")
    return True

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    random.seed(42)
    stress_test(5000)
```

**Output:**
```
Passed 1000 tests...
Passed 2000 tests...
Passed 3000 tests...
Passed 4000 tests...
Passed 5000 tests...
All 5000 tests passed.
```

## Designing Good Random Generators

The generator determines the quality of the stress test. A generator that only produces "nice" inputs will miss the bugs triggered by adversarial cases.

### Key Principles

**Cover the constraint space.** Generate inputs at all scales:

- Small inputs ($n = 1, 2, 3$) catch off-by-one errors and base case bugs.
- Medium inputs ($n = 10$--$20$) are large enough to exercise algorithm logic while staying small enough for brute-force comparison.
- Vary value ranges: include negative values, zeros, maximum values, and mixed positive/negative.

**Use structured random generation.** For graph problems, random generators should produce:

- Trees (connected, $n - 1$ edges).
- Dense graphs (many edges).
- Disconnected graphs (multiple components).
- Specific structures (paths, stars, complete bipartite graphs) with some probability.

### Generator Patterns

| Problem type | Generator strategy |
|---|---|
| Array problems | Random arrays of varying lengths and value ranges |
| Graph problems | Random trees, random graphs with controlled density |
| String problems | Random strings over controlled alphabets |
| Geometry problems | Random points with controlled coordinate ranges |
| Multi-test problems | Random number of test cases with varying sizes |

## When to Use External Processes

For compiled solutions (C++), the stress test invokes external executables:

```python
"""
External process stress testing.

Runs compiled C++ solutions and compares their outputs.
"""

import subprocess
import random

# ===================================================================
# Generator and Runner
# ===================================================================

def generate_and_test(max_n=15, num_tests=1000):
    """Generate inputs and compare two external solutions."""
    for test in range(1, num_tests + 1):
        n = random.randint(1, max_n)
        arr = [random.randint(-100, 100) for _ in range(n)]
        input_data = f"{n}\n{' '.join(map(str, arr))}\n"

        try:
            brute_out = subprocess.run(
                ["./brute"], input=input_data,
                capture_output=True, text=True, timeout=5
            ).stdout.strip()

            fast_out = subprocess.run(
                ["./fast"], input=input_data,
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"Test {test}: TIMEOUT")
            print(f"Input: {input_data}")
            continue

        if brute_out != fast_out:
            print(f"MISMATCH on test {test}!")
            print(f"Input:\n{input_data}")
            print(f"Brute: {brute_out}")
            print(f"Fast:  {fast_out}")
            return

    print(f"All {num_tests} tests passed.")

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    generate_and_test()
```

## Debugging with Counterexamples

When the stress test finds a mismatch:

1. **Save the failing input** to a file for reproducibility.
2. **Minimize the input** by removing elements that do not affect the failure. Often, the minimal failing case is just 2--5 elements.
3. **Trace the optimized solution** on the minimal input step by step.
4. **Fix the bug** and re-run the full stress test to confirm.

## Stress Testing Best Practices

| Practice | Rationale |
|---|---|
| Keep brute force obviously correct | A bug in the brute force wastes the entire effort |
| Use a fixed random seed for reproducibility | `random.seed(42)` ensures the same test sequence |
| Start with small inputs | Small counterexamples are easier to debug |
| Run at least 10,000 tests | Many bugs trigger only on specific structures |
| Time both solutions | Catches TLE issues in the optimized solution |
| Test on edge cases separately | Stress tests may never generate $n = 0$ or $n = 1$ |

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)

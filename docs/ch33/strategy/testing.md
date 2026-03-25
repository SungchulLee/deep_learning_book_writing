# Testing and Debugging

Even experienced competitive programmers rarely submit correct solutions on the first attempt. A systematic testing and debugging workflow catches errors before submission, reducing penalty time and wrong-answer frustration. This section presents a layered testing strategy -- from manual tracing through automated stress testing -- along with debugging techniques for quickly isolating faults.

## The Testing Pyramid

Testing proceeds from cheap and fast checks to expensive and thorough ones.

```
        ┌──────────┐
        │  Stress  │   Automated random testing
       ─┤  Testing ├─
      ┌─┴──────────┴─┐
      │  Edge Cases  │   Boundary and degenerate inputs
     ─┤              ├─
    ┌─┴──────────────┴─┐
    │  Sample Cases    │   Problem's provided examples
   ─┤                  ├─
  ┌─┴──────────────────┴─┐
  │  Manual Tracing       │   Dry-run on paper
  └───────────────────────┘
```

### Level 1 -- Manual Tracing

Before running any code, trace your algorithm on the first sample input by hand. Walk through each variable, each loop iteration, and each recursive call. This catches logical errors in the algorithm design before implementation details obscure them.

### Level 2 -- Sample Cases

Run your code on all provided sample inputs. Verify not just correctness but exact output format -- spacing, newlines, and precision.

!!! warning "Samples Are Necessary but Not Sufficient"
    Problem setters design samples to illustrate the problem, not to test edge cases. Passing all samples provides no guarantee of correctness.

### Level 3 -- Edge Cases

Construct the boundary inputs identified in the edge case analysis (see the [Edge Cases](edge_cases.md) section). Focus on:

- Minimum valid input size.
- Maximum valid input size (test for TLE and MLE).
- Degenerate structures (sorted arrays, star graphs, single-character strings).
- Values at integer boundaries ($0$, $-1$, $2^{31} - 1$, $10^{18}$).

### Level 4 -- Stress Testing

Stress testing automates the search for counterexamples by comparing an optimized solution against a known-correct brute-force solution on random inputs.

```python
"""
Stress testing framework for competitive programming.

Generates random inputs, runs both a brute-force and an optimized
solution, and reports the first input where outputs differ.
"""

import random
import subprocess

# ===================================================================
# Configuration
# ===================================================================

MAX_N = 20       # Keep small so brute force finishes quickly
NUM_TESTS = 1000

# ===================================================================
# Random Input Generator
# ===================================================================

def generate_input():
    """Generate a random test case as a string."""
    n = random.randint(1, MAX_N)
    arr = [random.randint(-100, 100) for _ in range(n)]
    return f"{n}\n{' '.join(map(str, arr))}\n"

# ===================================================================
# Solution Runners
# ===================================================================

def run_solution(executable, input_data):
    """Run a compiled solution and return its stdout."""
    result = subprocess.run(
        [executable],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=5
    )
    return result.stdout.strip()

# ===================================================================
# Stress Test Loop
# ===================================================================

if __name__ == "__main__":
    for test_num in range(1, NUM_TESTS + 1):
        inp = generate_input()
        out_brute = run_solution("./brute", inp)
        out_fast = run_solution("./fast", inp)

        if out_brute != out_fast:
            print(f"MISMATCH on test {test_num}!")
            print(f"Input:\n{inp}")
            print(f"Brute: {out_brute}")
            print(f"Fast:  {out_fast}")
            break
        else:
            print(f"Test {test_num}: OK")
    else:
        print(f"All {NUM_TESTS} tests passed.")
```

## Debugging Techniques

When a test fails, the following techniques help isolate the bug.

### Print Debugging

Insert targeted print statements at decision points -- not everywhere. Focus on:

- **Loop invariants**: Print the state at the start of each loop iteration.
- **Recursive calls**: Print arguments and return values.
- **Conditional branches**: Print which branch was taken and why.

!!! tip "Use stderr for Debug Output"
    Print debug output to `stderr` so it does not interfere with the judge's output comparison. In C++: `cerr << "x=" << x << endl;`. In Python: `print(x, file=sys.stderr)`.

### Binary Search on Bugs

If the input that causes failure is large, use a binary search strategy:

1. Remove the second half of the input. Does the bug persist?
2. If yes, the bug is in the first half. Recurse.
3. If no, the bug is triggered by the second half. Restore it and remove the first half.

This reduces a failing input of size $n$ to a minimal failing input in $O(\log n)$ steps.

### Delta Debugging

A more systematic variant of binary search on bugs:

1. Start with the full failing input.
2. Try removing chunks of decreasing size ($n/2$, $n/4$, ...).
3. Keep any removal that preserves the failure.
4. Repeat until no single element can be removed without fixing the bug.

The resulting minimal input makes the root cause obvious.

### Assertion-Based Debugging

Add assertions to verify invariants that your algorithm assumes.

- After sorting: assert the array is non-decreasing.
- After Dijkstra: assert all distances are non-negative.
- After DP: assert the recurrence holds for a few random states.

Assertions catch silent corruption early, before it propagates to produce a wrong final answer.

## Common Debugging Scenarios

### Wrong Answer on Hidden Tests

1. Run stress tests with the brute-force comparison.
2. Check edge cases systematically.
3. Verify integer types (32-bit vs 64-bit).
4. Verify modular arithmetic (is the modulus $10^9 + 7$ or $998244353$?).

### Time Limit Exceeded

1. Verify that your complexity matches the constraint analysis.
2. Check for accidental $O(n^2)$ behavior (e.g., string concatenation in a loop, `vector::erase` from the front).
3. Profile: is I/O the bottleneck? Switch to fast I/O.
4. Check for infinite loops -- add iteration counters.

### Runtime Error

1. Check array bounds -- are you accessing index $-1$ or index $n$?
2. Check division by zero.
3. Check stack overflow from deep recursion.
4. Check null pointer dereference (especially in graph/tree code).

## Testing Workflow Summary

| Step | Action | Time cost |
|---|---|---|
| 1 | Trace first sample by hand | 2--5 min |
| 2 | Run all sample cases | 1 min |
| 3 | Construct and run edge cases | 3--5 min |
| 4 | Stress test (if time permits) | 5--10 min |
| 5 | Submit | -- |

In a contest, allocate roughly 25% of your time per problem to testing. This investment pays off through fewer wrong submissions and lower penalty time.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)

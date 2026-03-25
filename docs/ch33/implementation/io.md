# Input/Output Optimization

In competitive programming, the difference between a Time Limit Exceeded verdict and an Accepted one is sometimes not the algorithm but the I/O method. Standard I/O routines in both C++ and Python carry synchronization and formatting overhead that, on problems with $10^5$--$10^6$ input values, can consume a significant fraction of the time budget. This section covers the most impactful I/O optimizations and when to apply them.

## Why I/O Performance Matters

Consider reading $10^6$ integers. With Python's default `input()` and `int()` calls, this takes roughly 1--2 seconds -- potentially exceeding a 2-second time limit before any computation begins. With `sys.stdin` bulk reading, the same task completes in 0.1--0.2 seconds.

In C++, `cin >> x` with default synchronization is 5--10 times slower than `scanf("%d", &x)` due to stream synchronization with C's `stdio`.

## C++ I/O Optimization

### Disable Stream Synchronization

By default, C++ synchronizes `cin`/`cout` with `scanf`/`printf` for interoperability. Disabling this synchronization gives a significant speedup.

```cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
```

After these two lines, `cin` and `cout` are as fast as `scanf` and `printf`. However, you must not mix C-style (`scanf`/`printf`) and C++-style (`cin`/`cout`) I/O in the same program after disabling synchronization.

### Use '\n' Instead of endl

`endl` flushes the output buffer after every line, which is expensive. Use `'\n'` instead.

```cpp
// Slow: flushes buffer each time
cout << x << endl;

// Fast: no flush
cout << x << '\n';
```

### Bulk Output with printf

For formatting large outputs, `printf` can be faster than `cout` even with synchronization disabled, because it avoids the overhead of C++ stream formatting.

## Python I/O Optimization

### sys.stdin for Fast Input

Replace `input()` with `sys.stdin` for bulk reading.

```python
"""
Demonstration of fast I/O techniques in Python.

Compares standard input() with sys.stdin for reading large inputs.
"""

import sys

# ===================================================================
# Fast Input Methods
# ===================================================================

def read_fast():
    """Read all input at once and split into tokens."""
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    arr = [int(data[i + 1]) for i in range(n)]
    return n, arr

# ===================================================================
# Alternative: Line-by-Line with sys.stdin
# ===================================================================

def read_lines():
    """Read input line by line using sys.stdin."""
    input_fn = sys.stdin.readline
    n = int(input_fn())
    arr = list(map(int, input_fn().split()))
    return n, arr

# ===================================================================
# Fast Output
# ===================================================================

def write_fast(arr):
    """Write output using sys.stdout.write for speed."""
    sys.stdout.write(' '.join(map(str, arr)) + '\n')

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # Example usage with line-by-line reading
    input_fn = sys.stdin.readline
    n = int(input_fn())
    arr = list(map(int, input_fn().split()))
    arr.sort()
    sys.stdout.write(' '.join(map(str, arr)) + '\n')
```

### Key Python I/O Techniques

| Technique | Speed improvement | When to use |
|---|---|---|
| `sys.stdin.readline()` instead of `input()` | 2--3x | Always |
| `sys.stdin.buffer.read()` bulk read | 5--10x | Large input, simple format |
| `sys.stdout.write()` instead of `print()` | 2x | Large output |
| `' '.join(map(str, arr))` | 3--5x vs loop | Printing arrays |

### Avoid Repeated String Concatenation

Building output by appending to a string in a loop creates a new string object each time, giving $O(n^2)$ total cost. Instead, collect all output pieces in a list and join once.

```python
# Slow: O(n^2) string concatenation
result = ""
for x in arr:
    result += str(x) + "\n"
print(result)

# Fast: O(n) join
print('\n'.join(map(str, arr)))
```

## I/O Speed Comparison

Approximate times for reading $10^6$ integers on a typical judge:

| Method | Language | Time |
|---|---|---|
| `cin >> x` (default sync) | C++ | ~1.5 s |
| `cin >> x` (sync disabled) | C++ | ~0.2 s |
| `scanf("%d", &x)` | C++ | ~0.2 s |
| `input()` in loop | Python 3 | ~2.0 s |
| `sys.stdin.readline()` in loop | Python 3 | ~0.7 s |
| `sys.stdin.buffer.read()` bulk | Python 3 | ~0.15 s |

## When I/O Optimization Is Unnecessary

Not every problem needs fast I/O. If the input size is small ($n \le 10^4$) and the algorithm is the bottleneck, standard I/O is fine. Focus I/O optimization effort on problems where:

- $n \ge 10^5$ and the algorithm is $O(n)$ or $O(n \log n)$.
- Multiple test cases multiply the I/O volume.
- The time limit is tight (1 second).

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)

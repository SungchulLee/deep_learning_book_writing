# Properties of Algorithms

Every algorithm should possess these fundamental properties:

$$

\begin{array}{ll}
\text{Correctness} & \text{Produces the right output for every valid input} \\
\text{Efficiency} & \text{Uses resources (time, space) wisely} \\
\text{Finiteness} & \text{Terminates after a finite number of steps} \\
\text{Definiteness} & \text{Each step is precisely defined} \\
\text{Generality} & \text{Solves a class of problems, not just one instance}
\end{array}

$$

## Deterministic vs Non-Deterministic

- **Deterministic**: Same input always produces same output
- **Non-deterministic**: May produce different outputs (randomized algorithms)

```python
def is_sorted(arr):
    """Check if an array is sorted — demonstrates definiteness."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

def main():
    tests = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4], []]
    for arr in tests:
        print(f"{arr} -> sorted: {is_sorted(arr)}")

if __name__ == "__main__":
    main()
```

**Output:**
```
[1, 2, 3, 4] -> sorted: True
[4, 3, 2, 1] -> sorted: False
[1, 3, 2, 4] -> sorted: False
[] -> sorted: True
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)

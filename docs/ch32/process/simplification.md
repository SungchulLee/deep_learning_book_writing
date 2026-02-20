# Simplification

**Simplification** is the technique of solving an easier version of the problem first, then extending the solution to handle the full problem. This is one of the most effective strategies in algorithm design.

## Simplification Strategies

| Strategy | Description | Example |
|---|---|---|
| Reduce dimensions | Solve 1D before 2D | 1D peak finding before 2D |
| Remove constraints | Solve unconstrained version | Knapsack without weight limit |
| Small cases | Solve for $n=1,2,3$ | Identify recurrence patterns |
| Special structure | Assume sorted/tree/DAG | General graph to DAG |
| Relax optimality | Find any solution first | Then optimize |

## Example: 2D Problem via 1D Simplification

**Problem:** Find maximum sum rectangle in a 2D matrix.

**Simplification:** First solve the 1D version (Kadane's algorithm), then extend.

```python
def max_sum_rectangle(matrix):
    if not matrix:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    max_sum = float('-inf')

    for left in range(cols):
        # Compress columns into a single array (1D simplification)
        temp = [0] * rows
        for right in range(left, cols):
            for i in range(rows):
                temp[i] += matrix[i][right]
            # Now solve 1D max subarray (Kadane's)
            current = temp[0]
            best = temp[0]
            for i in range(1, rows):
                current = max(temp[i], current + temp[i])
                best = max(best, current)
            max_sum = max(max_sum, best)
    return max_sum

matrix = [
    [ 1, 2, -1, -4, -20],
    [-8, -3,  4,  2,   1],
    [ 3,  8, 10,  1,   3],
    [-4, -1,  1,  7,  -6]
]
print(max_sum_rectangle(matrix))  # Output: 29
```

**Complexity:** $O(\text{cols}^2 \cdot \text{rows})$ -- the 2D problem is solved by running the 1D algorithm $O(\text{cols}^2)$ times.

## The Simplification Ladder

$$\text{Trivial case} \rightarrow \text{Small case} \rightarrow \text{Special structure} \rightarrow \text{Full problem}$$

Each step should reveal insights that carry forward to the next.

# Reference

- Polya, G. *How to Solve It*, Princeton University Press, 1945.
- Skiena, S. *The Algorithm Design Manual*, Chapter 1, Springer, 2020.

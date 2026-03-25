# N-Queens Problem

The N-Queens problem asks whether $n$ queens can be placed on an $n \times n$
chessboard so that no two queens threaten each other.  Two queens threaten each
other if they share a row, a column, or a diagonal.  This problem is the most
widely used example of backtracking because the constraint structure — three
independent conflict types — maps directly onto the feasibility-check framework,
and the state space tree is easy to visualize.

## Problem Statement

**Input.** A positive integer $n$.

**Output.** An arrangement of $n$ queens on an $n \times n$ board such that no two
queens occupy the same row, column, or diagonal, or a report that no such
arrangement exists.

Because each row must contain exactly one queen, the problem reduces to finding a
permutation $(c_1, c_2, \ldots, c_n)$ of column indices such that no two queens
share a column or diagonal.

## Backtracking Formulation

### State Space Tree

- **Decision $k$** ($k = 1, \ldots, n$): choose the column $c_k \in \{1, \ldots, n\}$
  for the queen in row $k$.
- **Branching factor**: $n$ at every level (before pruning).
- **Full tree size**: $n^n$ leaves without pruning; $n!$ leaves if column uniqueness
  is enforced.

### Feasibility Check

After placing the queen in row $k$ at column $c_k$, check against every previously
placed queen in row $j$ ($1 \leq j < k$):

1. **Column conflict**: $c_k = c_j$.
2. **Diagonal conflict**: $|c_k - c_j| = |k - j|$.

If either condition holds for any $j$, the placement is infeasible and the subtree
is pruned.

The column check ensures that no two queens share a column.  The diagonal check
uses the fact that two cells $(r_1, c_1)$ and $(r_2, c_2)$ lie on the same diagonal
if and only if

$$
|r_1 - r_2| = |c_1 - c_2|
$$

### Constant-Time Feasibility with Auxiliary Arrays

The naive feasibility check iterates over all $k - 1$ previously placed queens,
giving $O(k)$ per node.  Three Boolean arrays reduce this to $O(1)$:

| Array | Indices | Meaning |
|-------|---------|---------|
| `col_used[c]` | $c \in \{1, \ldots, n\}$ | Column $c$ is occupied |
| `diag1[k - c + n - 1]` | main diagonal index | The $\searrow$ diagonal through $(k, c)$ is occupied |
| `diag2[k + c]` | anti-diagonal index | The $\swarrow$ diagonal through $(k, c)$ is occupied |

At row $k$, column $c$ is feasible if and only if all three arrays are False at the
corresponding indices.  Updates during `make_move` and `undo_move` are $O(1)$.

## Algorithm

```
QUEENS(k, n):
    if k > n:
        report solution (c_1, ..., c_n)
        return True

    for c = 1 to n:
        if not col_used[c] and not diag1[k - c + n - 1] and not diag2[k + c]:
            c_k = c
            col_used[c] = True;  diag1[k - c + n - 1] = True;  diag2[k + c] = True
            if QUEENS(k + 1, n):
                return True
            col_used[c] = False; diag1[k - c + n - 1] = False; diag2[k + c] = False

    return False   // no valid column for row k — backtrack
```

## Python Implementation

```python
"""
N-Queens solver using backtracking with O(1) feasibility checks.

Places n queens on an n x n board so that no two queens share
a row, column, or diagonal.
"""


# === Solver ===================================================================

def solve_n_queens(n, find_all=False):
    """Return one (or all) solutions to the n-queens problem.

    Each solution is a list of length n where solution[i] is the column
    (0-indexed) of the queen in row i.
    """
    solutions = []
    placement = [0] * n
    col_used = [False] * n
    diag1 = [False] * (2 * n - 1)   # main diagonals  (row - col + n - 1)
    diag2 = [False] * (2 * n - 1)   # anti-diagonals  (row + col)

    def backtrack(row):
        if row == n:
            solutions.append(placement[:])
            return not find_all          # True = stop after first

        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col
            if not col_used[col] and not diag1[d1] and not diag2[d2]:
                placement[row] = col
                col_used[col] = diag1[d1] = diag2[d2] = True
                if backtrack(row + 1):
                    return True
                col_used[col] = diag1[d1] = diag2[d2] = False

        return False

    backtrack(0)
    return solutions


# === Display ==================================================================

def print_board(solution):
    """Print a chessboard with queens marked as Q."""
    n = len(solution)
    for row in range(n):
        line = ["."] * n
        line[solution[row]] = "Q"
        print(" ".join(line))
    print()


# === Main =====================================================================

if __name__ == "__main__":
    n = 8

    # Find one solution
    results = solve_n_queens(n, find_all=False)
    if results:
        print(f"One solution for {n}-queens:")
        print_board(results[0])

    # Count all solutions
    all_results = solve_n_queens(n, find_all=True)
    print(f"Total solutions for {n}-queens: {len(all_results)}")
```

**Output:**
```
One solution for 8-queens:
Q . . . . . . .
. . . . Q . . .
. . . . . . . Q
. . . . . Q . .
. . Q . . . . .
. . . . . . Q .
. Q . . . . . .
. . . Q . . . .

Total solutions for 8-queens: 92
```

## Complexity Analysis

**Time complexity.** The state space tree has at most $n!$ leaves (since column
reuse is pruned), and the feasibility check at each node is $O(1)$.  The total work
is bounded by

$$
T(n) = O(n!)
$$

In practice, diagonal pruning reduces the tree far below $n!$.  Empirical studies
show that the number of nodes explored grows roughly as $O(c^n)$ for a constant
$c \approx 2.5$, though no closed-form expression for the exact count is known.

**Space complexity.** The recursion depth is $n$, and the auxiliary arrays use
$O(n)$ space, giving $O(n)$ total.

## Known Results

| $n$ | Solutions | Distinct (up to symmetry) |
|-----|-----------|--------------------------|
| 1   | 1         | 1                        |
| 4   | 2         | 1                        |
| 8   | 92        | 12                       |
| 12  | 14200     | 1787                     |
| 14  | 365596    | 45752                    |

No closed-form formula for the number of solutions is known.  The problem is
NP-hard in the general form of placing $n$ non-attacking queens on an $n \times n$
board with some cells pre-occupied, but the standard unconstrained version has a
solution for every $n \geq 4$ (and for $n = 1$).

## Reference

- Garey and Johnson, *Computers and Intractability*, 1979

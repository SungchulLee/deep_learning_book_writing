# Sudoku Solver

Sudoku is a constraint-satisfaction puzzle that maps naturally onto the backtracking
framework.  Every cell must satisfy three independent constraints simultaneously —
row, column, and box uniqueness — making it an ideal testbed for techniques like
incremental feasibility checking, constraint propagation, and variable-ordering
heuristics.  Understanding how backtracking solves Sudoku reveals patterns that
generalize to any constraint-satisfaction problem.

## Problem Statement

**Input.** A $9 \times 9$ grid partially filled with digits from 1 to 9.

**Output.** A completed grid such that every row, every column, and every
$3 \times 3$ box contains each digit from 1 to 9 exactly once.

A well-posed Sudoku puzzle has exactly one solution.  The constraint structure can
be stated formally: let $x_{r,c} \in \{1, \ldots, 9\}$ denote the value in row $r$,
column $c$.  The three constraint families are:

- **Row**: $x_{r,c_1} \neq x_{r,c_2}$ for all $c_1 \neq c_2$.
- **Column**: $x_{r_1,c} \neq x_{r_2,c}$ for all $r_1 \neq r_2$.
- **Box**: within each $3 \times 3$ sub-grid, all nine values are distinct.

## Backtracking Formulation

### State Space Tree

- **Decisions**: fill the empty cells one by one, left-to-right, top-to-bottom
  (or in any fixed order).
- **Branching factor**: up to 9 at each empty cell (before pruning).
- **Full tree size**: up to $9^{k}$ leaves, where $k$ is the number of empty cells
  (typically $k \approx 50$–60 for a standard puzzle).

### Feasibility Check

After placing digit $d$ in cell $(r, c)$, check:

1. Is $d$ already in row $r$?
2. Is $d$ already in column $c$?
3. Is $d$ already in the $3 \times 3$ box containing $(r, c)$?

The box index for cell $(r, c)$ is

$$
b = 3 \left\lfloor \frac{r}{3} \right\rfloor + \left\lfloor \frac{c}{3} \right\rfloor
$$

### Constant-Time Checks with Bitmasks

Maintain three arrays of bitmasks:

| Array | Size | Bit $d$ is set when |
|-------|------|---------------------|
| `row_used[r]` | 9 | Digit $d$ appears in row $r$ |
| `col_used[c]` | 9 | Digit $d$ appears in column $c$ |
| `box_used[b]` | 9 | Digit $d$ appears in box $b$ |

Placing digit $d$ in cell $(r, c)$ is feasible if and only if

$$
\text{row\_used}[r] \;\mathbin{\&}\; (1 \ll d) = 0 \quad\text{and}\quad \text{col\_used}[c] \;\mathbin{\&}\; (1 \ll d) = 0 \quad\text{and}\quad \text{box\_used}[b] \;\mathbin{\&}\; (1 \ll d) = 0
$$

Each check is a single bitwise AND — $O(1)$ per check.

## Algorithm

```
SOLVE_SUDOKU(grid):
    cell = find_next_empty(grid)
    if cell is None:
        return True                     // all cells filled — solution found

    (r, c) = cell
    for d = 1 to 9:
        if is_valid(grid, r, c, d):
            grid[r][c] = d
            update_used(r, c, d)        // set bits
            if SOLVE_SUDOKU(grid):
                return True
            grid[r][c] = 0
            revert_used(r, c, d)        // clear bits

    return False                        // no digit works — backtrack
```

## Python Implementation

```python
"""
Sudoku solver using backtracking with bitmask feasibility checks.

Fills empty cells (marked as 0) one by one, backtracking whenever
no valid digit can be placed.
"""


# === Solver ===================================================================

def solve_sudoku(board):
    """Solve the puzzle in place.  Return True if a solution exists."""
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9

    # Initialize bitmasks from the given clues
    for r in range(9):
        for c in range(9):
            d = board[r][c]
            if d != 0:
                bit = 1 << d
                b = 3 * (r // 3) + (c // 3)
                row_used[r] |= bit
                col_used[c] |= bit
                box_used[b] |= bit

    def backtrack():
        # Find the next empty cell
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    b = 3 * (r // 3) + (c // 3)
                    for d in range(1, 10):
                        bit = 1 << d
                        if not (row_used[r] & bit or
                                col_used[c] & bit or
                                box_used[b] & bit):
                            board[r][c] = d
                            row_used[r] |= bit
                            col_used[c] |= bit
                            box_used[b] |= bit

                            if backtrack():
                                return True

                            board[r][c] = 0
                            row_used[r] ^= bit
                            col_used[c] ^= bit
                            box_used[b] ^= bit

                    return False  # no digit works — backtrack
        return True  # no empty cell — solved

    return backtrack()


# === Display ==================================================================

def print_board(board):
    """Pretty-print a 9x9 Sudoku board."""
    for r in range(9):
        if r > 0 and r % 3 == 0:
            print("------+-------+------")
        row_str = ""
        for c in range(9):
            if c > 0 and c % 3 == 0:
                row_str += "| "
            row_str += str(board[r][c]) + " "
        print(row_str.strip())
    print()


# === Main =====================================================================

if __name__ == "__main__":
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    print("Puzzle:")
    print_board(puzzle)

    if solve_sudoku(puzzle):
        print("Solution:")
        print_board(puzzle)
    else:
        print("No solution exists.")
```

**Output:**
```
Puzzle:
5 3 0 | 0 7 0 | 0 0 0
6 0 0 | 1 9 5 | 0 0 0
0 9 8 | 0 0 0 | 0 6 0
------+-------+------
8 0 0 | 0 6 0 | 0 0 3
4 0 0 | 8 0 3 | 0 0 1
7 0 0 | 0 2 0 | 0 0 6
------+-------+------
0 6 0 | 0 0 0 | 2 8 0
0 0 0 | 4 1 9 | 0 0 5
0 0 0 | 0 8 0 | 0 7 9

Solution:
5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
1 9 8 | 3 4 2 | 5 6 7
------+-------+------
8 5 9 | 7 6 1 | 4 2 3
4 2 6 | 8 5 3 | 7 9 1
7 1 3 | 9 2 4 | 8 5 6
------+-------+------
9 6 1 | 5 3 7 | 2 8 4
2 8 7 | 4 1 9 | 6 3 5
3 4 5 | 2 8 6 | 1 7 9
```

## Optimization: Variable Ordering

The basic algorithm fills cells left-to-right, top-to-bottom.  A significant
improvement is the **most-constrained variable** (MCV) heuristic: always fill the
empty cell that has the fewest remaining legal digits.

The MCV heuristic works because:

- A cell with only one legal digit has **no branching** — the digit is forced.
- A cell with two legal digits has branching factor 2 instead of up to 9.
- Choosing the most constrained cell first maximizes the chance of early failure,
  which prunes larger subtrees higher in the search tree.

With MCV ordering, most well-posed Sudoku puzzles are solved with zero or very few
backtracks.

## Complexity Analysis

**Time complexity.** In the worst case, the algorithm explores up to $9^k$ nodes,
where $k$ is the number of empty cells.  With bitmask feasibility checks each node
costs $O(1)$, giving

$$
T = O(9^k)
$$

In practice, pruning and constraint propagation reduce the search to a tiny fraction
of this bound.  Hard puzzles typically require exploring a few thousand nodes;
most standard puzzles require fewer than one hundred.

**Space complexity.** The recursion depth is at most $k \leq 81$, and the auxiliary
bitmask arrays use $O(1)$ space (fixed at 27 integers).  Total space is $O(1)$
beyond the board itself.

## Constraint Propagation

Pure backtracking can be enhanced with **constraint propagation**, which deduces
forced values without branching:

1. **Naked singles**: if a cell has only one legal digit, assign it immediately.
2. **Hidden singles**: if a digit can appear in only one cell within a row, column,
   or box, assign it to that cell.

After each assignment, propagate constraints to neighboring cells.  If any cell's
legal-digit set becomes empty, backtrack immediately.  This combination of
backtracking and propagation is the basis of efficient Sudoku solvers like
Peter Norvig's well-known Python solver, which handles even the hardest known
puzzles in milliseconds.

## Reference

- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
- Norvig, "Solving Every Sudoku Puzzle," [norvig.com](https://norvig.com/sudoku.html)

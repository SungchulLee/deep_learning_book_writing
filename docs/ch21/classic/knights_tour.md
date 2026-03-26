# Knight's Tour

A chess knight moves in an "L" shape: two squares in one direction and one square perpendicular. The **knight's tour** problem asks whether a knight can visit every square on an $n \times n$ chessboard exactly once. This classical problem provides an elegant application of backtracking with a powerful heuristic — Warnsdorff's rule — that transforms an exponential search into a near-linear one for practical board sizes.

## Problem Statement

Given an $n \times n$ board and a starting position $(r_0, c_0)$, find a sequence of knight moves that visits all $n^2$ squares exactly once. A **closed tour** returns to the starting square; an **open tour** ends on any square.

The eight possible knight moves from position $(r, c)$ are:

$$
(r \pm 1, c \pm 2) \quad \text{and} \quad (r \pm 2, c \pm 1)
$$

## Backtracking Solution

The brute-force approach tries all valid moves from the current position:

1. Mark the current square as visited.
2. For each of the (up to) 8 possible next squares:
    - If the square is on the board and unvisited, recurse.
3. If all $n^2$ squares are visited, the tour is complete.
4. If no move leads to a solution, **backtrack**: unmark the current square and return.

The search tree has branching factor up to 8 and depth $n^2$, giving worst-case $O(8^{n^2})$ — impractical without pruning.

## Warnsdorff's Rule

Warnsdorff's rule is a greedy heuristic that dramatically reduces the search space:

!!! tip "Warnsdorff's Rule"
    At each step, move to the neighboring square with the **fewest onward moves** (i.e., the smallest degree in the remaining graph). Ties are broken arbitrarily.

This "most constrained first" heuristic works because visiting constrained squares early avoids dead ends later. For boards up to about $76 \times 76$, Warnsdorff's rule almost always finds a tour without backtracking.

## Complexity

| Aspect | Value |
|---|---|
| Brute force | $O(8^{n^2})$ worst case |
| With Warnsdorff's | Near $O(n^2)$ in practice |
| Space | $O(n^2)$ for the board |
| Existence | Tours exist for all $n \ge 5$ |

## Python Implementation

```python
"""
Knight's Tour — Backtracking with Warnsdorff's Heuristic.

Finds a knight's tour on an n x n board using backtracking,
optionally accelerated by Warnsdorff's rule.
"""


# === Knight Moves ===

MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


# === Utility Functions ===

def is_valid(r: int, c: int, n: int, board: list[list[int]]) -> bool:
    """Check if position (r, c) is on the board and unvisited."""
    return 0 <= r < n and 0 <= c < n and board[r][c] == -1


def count_onward_moves(r: int, c: int, n: int, board: list[list[int]]) -> int:
    """Count valid onward moves from position (r, c)."""
    count = 0
    for dr, dc in MOVES:
        if is_valid(r + dr, c + dc, n, board):
            count += 1
    return count


# === Backtracking with Warnsdorff's Rule ===

def knights_tour(n: int, start_r: int = 0, start_c: int = 0) -> list[list[int]] | None:
    """Find a knight's tour on an n x n board.

    Args:
        n: Board size.
        start_r, start_c: Starting position.

    Returns:
        Board with move numbers (0 to n^2-1), or None if no tour exists.
    """
    board = [[-1] * n for _ in range(n)]
    board[start_r][start_c] = 0

    def backtrack(r: int, c: int, move_num: int) -> bool:
        if move_num == n * n:
            return True

        # Get valid next moves, sorted by Warnsdorff's rule
        next_moves = []
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc, n, board):
                onward = count_onward_moves(nr, nc, n, board)
                next_moves.append((onward, nr, nc))

        next_moves.sort()  # fewest onward moves first

        for _, nr, nc in next_moves:
            board[nr][nc] = move_num
            if backtrack(nr, nc, move_num + 1):
                return True
            board[nr][nc] = -1

        return False

    if backtrack(start_r, start_c, 1):
        return board
    return None


# === Display Board ===

def print_board(board: list[list[int]]) -> None:
    """Print the board with move numbers."""
    n = len(board)
    width = len(str(n * n - 1))
    for row in board:
        print(" ".join(str(cell).rjust(width) for cell in row))


# === Main ===

if __name__ == "__main__":
    n = 6
    result = knights_tour(n)
    if result:
        print(f"Knight's tour on {n}x{n} board:")
        print_board(result)
    else:
        print(f"No tour found for {n}x{n} board")
```

## Worked Example

On a $5 \times 5$ board starting at $(0, 0)$, one valid tour (move numbers):

```
 0 11  8 19 16
 7 18 15 10  3
12  1 20  5 22
17  6 23  2  9
24 13  4 21 14
```

The knight starts at the top-left (move 0), visits all 25 squares, and ends at $(4, 0)$ (move 24). This is an open tour since move 24 is not a knight's move away from move 0.

## Reference

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
- Warnsdorff, H. C. (1823). *Des Rosselsprunges einfachste und allgemeinste Losung*.

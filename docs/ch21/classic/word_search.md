# Word Search

Given a 2D grid of characters and a target word, the **word search** problem determines whether the word can be formed by following a path of adjacent cells (horizontally or vertically), where each cell is used at most once. This problem is a natural application of backtracking on a grid: the search branches at each step into multiple neighboring cells, and dead ends are pruned as soon as a character mismatch is detected.

## Problem Statement

Given an $m \times n$ grid of characters `board` and a string `word`, return `True` if `word` exists in the grid. A valid path connects horizontally or vertically adjacent cells, and no cell may be reused within a single path.

## Backtracking Strategy

For each cell $(r, c)$ in the grid that matches the first character of `word`:

1. Mark $(r, c)$ as visited.
2. Recursively try to match the remaining characters by moving to each unvisited adjacent cell (up, down, left, right).
3. If the entire word is matched, return `True`.
4. If no adjacent cell extends the match, **backtrack**: unmark $(r, c)$ and return `False`.

The feasibility check at each step — comparing the current cell's character to the next character in the word — prunes branches immediately upon mismatch.

## Complexity

| Aspect | Value |
|---|---|
| Time (worst case) | $O(m \cdot n \cdot 3^L)$ where $L$ is the word length |
| Space | $O(L)$ recursion stack |
| Branching factor | 3 (excluding the cell we came from) |

The $3^L$ factor arises because at each step there are at most 3 unvisited neighbors (the fourth is the cell we came from). In practice, character mismatches prune most branches.

## Python Implementation

```python
"""
Word Search — Backtracking on a 2D Grid.

Determines whether a target word can be found in a character grid
by following a path of adjacent cells without reusing any cell.
"""


# === Word Search ===

def word_search(board: list[list[str]], word: str) -> bool:
    """Return True if word exists in the grid.

    Args:
        board: m x n grid of characters.
        word: Target word to search for.
    """
    if not board or not word:
        return False

    m, n = len(board), len(board[0])

    def backtrack(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            return True

        if (r < 0 or r >= m or c < 0 or c >= n
                or board[r][c] != word[idx]):
            return False

        # Mark as visited by temporarily modifying the cell
        original = board[r][c]
        board[r][c] = "#"

        # Explore all four directions
        found = (
            backtrack(r + 1, c, idx + 1)
            or backtrack(r - 1, c, idx + 1)
            or backtrack(r, c + 1, idx + 1)
            or backtrack(r, c - 1, idx + 1)
        )

        # Backtrack: restore the cell
        board[r][c] = original
        return found

    for r in range(m):
        for c in range(n):
            if board[r][c] == word[0] and backtrack(r, c, 0):
                return True

    return False


# === Find All Occurrences ===

def find_word_paths(
    board: list[list[str]], word: str
) -> list[list[tuple[int, int]]]:
    """Find all distinct paths that form the word."""
    if not board or not word:
        return []

    m, n = len(board), len(board[0])
    results: list[list[tuple[int, int]]] = []

    def backtrack(r: int, c: int, idx: int, path: list[tuple[int, int]]) -> None:
        if idx == len(word):
            results.append(path[:])
            return

        if (r < 0 or r >= m or c < 0 or c >= n
                or board[r][c] != word[idx]):
            return

        original = board[r][c]
        board[r][c] = "#"
        path.append((r, c))

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            backtrack(r + dr, c + dc, idx + 1, path)

        path.pop()
        board[r][c] = original

    for r in range(m):
        for c in range(n):
            if board[r][c] == word[0]:
                backtrack(r, c, 0, [])

    return results


# === Main ===

if __name__ == "__main__":
    board = [
        ["A", "B", "C", "E"],
        ["S", "F", "C", "S"],
        ["A", "D", "E", "E"],
    ]

    test_words = ["ABCCED", "SEE", "ABCB"]
    for w in test_words:
        # Create a fresh copy for each test
        b = [row[:] for row in board]
        print(f"'{w}': {word_search(b, w)}")
    # Output:
    # 'ABCCED': True
    # 'SEE': True
    # 'ABCB': False
```

## Worked Example

For the board:

```
A B C E
S F C S
A D E E
```

Searching for "ABCCED":

1. Start at $(0,0)$: 'A' matches. Mark visited.
2. Move right to $(0,1)$: 'B' matches.
3. Move right to $(0,2)$: 'C' matches.
4. Move down to $(1,2)$: 'C' matches.
5. Move down to $(2,2)$: 'E' matches.
6. Move left to $(2,1)$: 'D' matches. Word found.

Searching for "ABCB": After matching "ABC" at positions $(0,0) \to (0,1) \to (0,2)$, the next 'B' requires revisiting $(0,1)$, which is already on the path. No alternative exists, so the search returns `False`.

## Reference

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.

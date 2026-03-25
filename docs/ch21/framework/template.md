# Backtracking Template

Every backtracking algorithm shares the same recursive skeleton: build a partial
solution one decision at a time, check constraints after each decision, and undo the
decision (backtrack) whenever the partial solution can no longer lead to a valid
answer.  Recognizing this common structure lets us write a single generic template
and then specialize it for problems as different as N-Queens, Sudoku, and graph
coloring.

## Generic Pseudocode

The template below captures the essential control flow of backtracking.  The three
customization points — `is_solution`, `candidates`, and `make_move`/`undo_move` —
are the only parts that change from one problem to the next.

```
BACKTRACK(state, decisions):
    if is_solution(state):
        process(state)          # record, print, or count the solution
        return

    for choice in candidates(state, decisions):
        if is_valid(state, choice):
            make_move(state, choice)
            BACKTRACK(state, decisions + 1)
            undo_move(state, choice)   # restore state before trying next choice
```

Each element of the template serves a specific purpose:

| Component | Role |
|-----------|------|
| `state` | The current partial solution (e.g., board configuration, assignment vector) |
| `decisions` | How many decisions have been made so far (current depth in the state space tree) |
| `is_solution` | Returns `True` when all $n$ decisions are complete |
| `candidates` | Generates the possible values for the next decision |
| `is_valid` | Checks whether `choice` is consistent with the current partial solution |
| `make_move` | Extends `state` by incorporating `choice` |
| `undo_move` | Reverses `make_move`, restoring `state` to its previous form |

## Python Implementation

The following generic implementation demonstrates the template in Python.  Concrete
problems override the helper methods.

```python
"""
Generic backtracking template.

Provides a base class that encapsulates the backtracking control flow.
Subclasses override five methods to solve specific combinatorial problems.
"""


# === Generic backtracking framework ==========================================

class BacktrackingSolver:
    """Base class for backtracking algorithms.

    Subclasses must implement:
        is_solution, process, candidates, make_move, undo_move
    """

    def __init__(self):
        self.solutions = []

    def solve(self, state, depth=0):
        """Run the backtracking search starting from *state* at *depth*."""
        if self.is_solution(state, depth):
            self.process(state)
            return
        for choice in self.candidates(state, depth):
            if self.is_valid(state, choice):
                self.make_move(state, choice)
                self.solve(state, depth + 1)
                self.undo_move(state, choice)

    # --- Customization points (override in subclasses) ---

    def is_solution(self, state, depth):
        raise NotImplementedError

    def process(self, state):
        raise NotImplementedError

    def candidates(self, state, depth):
        raise NotImplementedError

    def is_valid(self, state, choice):
        raise NotImplementedError

    def make_move(self, state, choice):
        raise NotImplementedError

    def undo_move(self, state, choice):
        raise NotImplementedError
```

## Anatomy of the Template

### Decision Point

At each recursive call the algorithm stands at a **decision point** — a node in the
state space tree.  The `candidates` function enumerates the branches leaving that
node.  In a permutation problem the candidates at depth $k$ are the $n - k$ elements
not yet used; in an $m$-coloring problem they are the $m$ available colors.

### Constraint Check

The `is_valid` function performs the **feasibility check**.  It determines whether
adding `choice` to the current partial solution violates any constraint.  A well-designed
`is_valid` runs in $O(1)$ or $O(k)$ time (where $k$ is the current depth) so that
the per-node overhead stays small relative to the tree traversal.

### State Modification and Restoration

The `make_move` / `undo_move` pair ensures that the algorithm leaves the state exactly
as it found it after exploring each subtree.  This **undo semantics** is what
distinguishes backtracking from a plain DFS that simply discards visited nodes:

1. `make_move(state, choice)` — mutate `state` in place.
2. Recurse into the child subtree.
3. `undo_move(state, choice)` — reverse the mutation so siblings see the original state.

!!! warning "Forgetting to undo"

    The most common backtracking bug is an incomplete `undo_move`.  If any
    auxiliary data structure (e.g., a set of used elements, a conflict counter)
    is updated in `make_move` but not reversed in `undo_move`, later branches
    see a corrupted state and produce wrong results.

### Solution Detection

The `is_solution` function checks whether the current depth equals $n$ (all
decisions made).  In optimization variants it may also check whether the current
solution improves the best known objective.

## Finding One vs. All Solutions

The template above finds **all** solutions.  Two common variants modify the
control flow:

**Find the first solution** — return immediately after `process`:

```
BACKTRACK(state, decisions):
    if is_solution(state):
        process(state)
        return True             # signal: stop searching

    for choice in candidates(state, decisions):
        if is_valid(state, choice):
            make_move(state, choice)
            if BACKTRACK(state, decisions + 1):
                return True     # propagate the stop signal
            undo_move(state, choice)
    return False
```

**Find the optimal solution** — maintain a global best and prune:

```
BACKTRACK(state, decisions, best):
    if is_solution(state):
        if objective(state) > best.value:
            best.value = objective(state)
            best.solution = copy(state)
        return

    for choice in candidates(state, decisions):
        if is_valid(state, choice):
            if bound(state, choice) <= best.value:
                continue        # prune: cannot improve
            make_move(state, choice)
            BACKTRACK(state, decisions + 1, best)
            undo_move(state, choice)
```

The optimization variant adds a **bounding step** that previews whether the subtree
can possibly beat the incumbent solution.  This bounding step is the bridge between
backtracking and branch-and-bound.

## Time Complexity

Let $b$ be the average branching factor and $n$ the number of decisions.  Without
pruning, the template visits every node in the state space tree:

$$
T(n) = O\!\left(\sum_{k=0}^{n} b^k\right) = O(b^n)
$$

Pruning reduces the effective branching factor.  If the feasibility check eliminates
a fraction $p$ of branches at each level, the effective branching factor drops to
$b(1 - p)$ and the running time becomes $O\!\bigl((b(1-p))^n\bigr)$.  The
stronger the pruning, the closer the practical running time is to polynomial — though
the worst case remains exponential for NP-hard problems.

## Reference

- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)

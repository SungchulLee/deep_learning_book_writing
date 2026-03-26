# Monte Carlo Tree Search

In game-playing and sequential decision problems, the search tree can be
astronomically large — chess has roughly $10^{120}$ game states. **Monte
Carlo Tree Search (MCTS)** handles this by focusing computational effort
on the most promising parts of the tree, using random simulations to
estimate the value of each node. MCTS powered AlphaGo's historic victory
over a world champion Go player and remains central to modern game AI.

## Core Idea

MCTS builds a search tree incrementally. Each iteration consists of four
phases that traverse from the root to a leaf, then propagate information
back up:

1. **Selection** — Starting from the root, descend through the tree by
   choosing children according to a tree policy (e.g., UCB1) until
   reaching a node that has unexplored children.
2. **Expansion** — Add one (or more) child nodes for an unexplored action.
3. **Simulation** (Rollout) — From the new node, play a random game to
   completion using a default policy (often uniform random moves).
4. **Backpropagation** — Update the visit count and win statistics of
   every node along the path from the new node back to the root.

## UCB1: Balancing Exploration and Exploitation

The **Upper Confidence Bound for Trees (UCT)** selects the child that
maximizes:

$$
\text{UCB1}(v) = \frac{w_v}{n_v} + c \sqrt{\frac{\ln N}{n_v}}
$$

where:

- $w_v$ = number of wins (or total reward) at node $v$
- $n_v$ = number of visits to node $v$
- $N$ = number of visits to the parent of $v$
- $c$ = exploration constant (often $\sqrt{2}$ by theory)

The first term is the **exploitation** component (average win rate).
The second term is the **exploration** component (favors less-visited nodes).

!!! note "Theoretical Guarantee"
    UCB1 applied to trees converges to the optimal action as the number
    of iterations grows. The regret of UCB1 grows only as
    $O(\ln n / \Delta)$ where $\Delta$ is the gap between the best and
    second-best arm values.

## Algorithm in Detail

### Selection

From the root, repeatedly select the child with the highest UCB1 value
until reaching a node where some children have not been visited.

### Expansion

Pick one unvisited child and add it to the tree.

### Simulation

Play out the game from the expanded node using random moves (or a
heuristic policy) until reaching a terminal state. Record the outcome.

### Backpropagation

Walk back from the expanded node to the root. For each node on the path:

- Increment $n_v$ by 1.
- Increment $w_v$ by the simulation result (1 for a win, 0 for a loss,
  0.5 for a draw).

After enough iterations, choose the root's child with the highest visit
count as the best move.

## Worked Example

Consider a tic-tac-toe position where X has two moves: A and B.

After 100 MCTS iterations:
- Move A: $n = 60$, $w = 42$ (win rate $= 0.70$)
- Move B: $n = 40$, $w = 32$ (win rate $= 0.80$)

With $c = \sqrt{2}$ and $N = 100$:

$$
\text{UCB1}(A) = 0.70 + \sqrt{2} \sqrt{\frac{\ln 100}{60}} = 0.70 + 0.393 = 1.093
$$

$$
\text{UCB1}(B) = 0.80 + \sqrt{2} \sqrt{\frac{\ln 100}{40}} = 0.80 + 0.481 = 1.281
$$

Move B has higher UCB1, so the next iteration explores B's subtree.
Despite A having more visits, B's higher win rate and exploration bonus
steer MCTS toward it.

## Implementation

```python
"""
Monte Carlo Tree Search for simple two-player games.

Implements the four MCTS phases (selection, expansion, simulation,
backpropagation) with UCB1 as the tree policy.
"""

import math
import random
from collections import defaultdict


# === MCTS Node ===

class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = None

    def ucb1(self, c=1.414):
        """Compute the UCB1 value for this node."""
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


# === Simple Game: Nim ===

class NimGame:
    """Simple Nim game: players take 1-3 stones from a pile.

    The player who takes the last stone wins.
    """

    def __init__(self, stones=10):
        self.stones = stones
        self.current_player = 0  # 0 or 1

    def clone(self):
        g = NimGame(self.stones)
        g.current_player = self.current_player
        return g

    def get_actions(self):
        return [i for i in range(1, min(4, self.stones + 1))]

    def apply(self, action):
        self.stones -= action
        self.current_player = 1 - self.current_player

    def is_terminal(self):
        return self.stones <= 0

    def winner(self):
        if self.stones <= 0:
            return 1 - self.current_player  # Last player to move wins
        return None


# === MCTS Algorithm ===

def mcts(game, iterations=1000, c=1.414):
    """Run MCTS and return the best action.

    Args:
        game: the current game state.
        iterations: number of MCTS iterations.
        c: exploration constant for UCB1.

    Returns:
        The best action to take from the current state.
    """
    root = MCTSNode(game.clone())
    root.untried_actions = game.get_actions()

    for _ in range(iterations):
        node = root
        state = game.clone()

        # Selection
        while node.untried_actions is not None and \
              len(node.untried_actions) == 0 and \
              len(node.children) > 0:
            node = max(node.children, key=lambda n: n.ucb1(c))
            state.apply(node.action)

        # Expansion
        if node.untried_actions and len(node.untried_actions) > 0:
            action = random.choice(node.untried_actions)
            state.apply(action)
            child = MCTSNode(state.clone(), parent=node, action=action)
            child.untried_actions = state.get_actions() if not state.is_terminal() else []
            node.untried_actions.remove(action)
            node.children.append(child)
            node = child

        # Simulation (random rollout)
        sim_state = state.clone()
        while not sim_state.is_terminal():
            actions = sim_state.get_actions()
            if not actions:
                break
            sim_state.apply(random.choice(actions))

        # Backpropagation
        winner = sim_state.winner()
        while node is not None:
            node.visits += 1
            if winner is not None:
                if winner == game.current_player:
                    node.wins += 1.0
                else:
                    node.wins += 0.0
            else:
                node.wins += 0.5
            node = node.parent

    # Return the most-visited child's action
    best_child = max(root.children, key=lambda n: n.visits)
    return best_child.action, best_child.visits, best_child.wins


# === Main ===

if __name__ == "__main__":
    random.seed(42)

    game = NimGame(stones=10)
    print(f"Nim with {game.stones} stones")

    action, visits, wins = mcts(game, iterations=5000)
    print(f"Best action: take {action} stone(s)")
    print(f"  Visits: {visits}, Win rate: {wins/visits:.3f}")

    # Show all root children stats
    root = MCTSNode(game.clone())
    root.untried_actions = game.get_actions()
    for _ in range(5000):
        node = root
        state = game.clone()
        while node.untried_actions is not None and \
              len(node.untried_actions) == 0 and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            state.apply(node.action)
        if node.untried_actions and len(node.untried_actions) > 0:
            a = random.choice(node.untried_actions)
            state.apply(a)
            child = MCTSNode(state.clone(), parent=node, action=a)
            child.untried_actions = state.get_actions() if not state.is_terminal() else []
            node.untried_actions.remove(a)
            node.children.append(child)
            node = child
        sim = state.clone()
        while not sim.is_terminal():
            acts = sim.get_actions()
            if not acts:
                break
            sim.apply(random.choice(acts))
        w = sim.winner()
        while node is not None:
            node.visits += 1
            node.wins += 1.0 if w == game.current_player else 0.0
            node = node.parent

    print("\nAll moves from root:")
    for child in sorted(root.children, key=lambda n: -n.visits):
        wr = child.wins / child.visits if child.visits > 0 else 0
        print(f"  Take {child.action}: visits={child.visits}, "
              f"win_rate={wr:.3f}")
```

## Variants and Extensions

| Variant | Key Modification |
|---|---|
| UCT | UCB1 applied to trees (standard MCTS) |
| RAVE | Rapid Action Value Estimation — uses move statistics across the tree |
| PUCT | Predictor + UCT — uses a neural network prior (used in AlphaGo/AlphaZero) |
| Progressive widening | Limits branching in large action spaces |

## Complexity

- **Time per iteration:** $O(d)$ where $d$ is the depth of a simulation.
- **Total time:** $O(n \cdot d)$ for $n$ iterations.
- **Space:** $O(n)$ for the tree nodes created.

MCTS is an **anytime algorithm**: it can be stopped at any point and
still return a reasonable action. More iterations yield better results.

## Reference

- Kocsis, L. & Szepesvari, C. "Bandit Based Monte-Carlo Planning." *ECML*, 2006.
- Browne, C. et al. "A Survey of Monte Carlo Tree Search Methods." *IEEE Trans. CI and AI in Games*, 2012.

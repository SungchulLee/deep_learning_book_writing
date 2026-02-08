"""
Chapter 32.1.2: Agent-Environment Interface
============================================
Implements and demonstrates the agent-environment interaction loop,
episodic vs. continuing tasks, and trajectory analysis.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


# =============================================================================
# 1. Abstract Environment Interface
# =============================================================================

class Environment(ABC):
    """Abstract base class defining the agent-environment interface."""

    @abstractmethod
    def reset(self) -> Any:
        """Reset environment and return initial state."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Execute action, return (next_state, reward, done, info).
        This is the fundamental agent-environment interface.
        """
        pass

    @property
    @abstractmethod
    def state_space_size(self) -> int:
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        pass


class Agent(ABC):
    """Abstract base class for an RL agent."""

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """Select action given current state."""
        pass

    def update(self, state, action, reward, next_state, done):
        """Optional: update agent's knowledge."""
        pass


# =============================================================================
# 2. Episodic Environment: Cliff Walking
# =============================================================================

class CliffWalkingEnv(Environment):
    """
    Cliff Walking environment (Sutton & Barto Example 6.6).

    4x12 grid:
    - Start: bottom-left (3,0)
    - Goal: bottom-right (3,11)
    - Cliff: bottom row between start and goal (3,1)-(3,10)
    - Stepping off cliff → reward = -100, reset to start
    - Each step → reward = -1
    - Reaching goal → episode ends

    This is an episodic task with clear terminal state.
    """

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}
        self.state = self.start

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        dr, dc = self.action_deltas[action]
        new_r = np.clip(self.state[0] + dr, 0, self.rows - 1)
        new_c = np.clip(self.state[1] + dc, 0, self.cols - 1)

        new_state = (int(new_r), int(new_c))

        if new_state in self.cliff:
            self.state = self.start
            return self.state, -100.0, False, {"fell_off_cliff": True}
        elif new_state == self.goal:
            self.state = new_state
            return self.state, -1.0, True, {"reached_goal": True}
        else:
            self.state = new_state
            return self.state, -1.0, False, {}

    @property
    def state_space_size(self) -> int:
        return self.rows * self.cols

    @property
    def action_space_size(self) -> int:
        return 4


# =============================================================================
# 3. Continuing Environment: Server Load Balancing
# =============================================================================

class ServerLoadEnv(Environment):
    """
    A continuing task: server load balancing.

    The environment never terminates — the agent must continuously
    balance load across servers.

    State: (load_server1, load_server2, incoming_request_size)
    Action: 0 = route to server 1, 1 = route to server 2
    Reward: -max(load_server1, load_server2)  (penalize imbalance)
    """

    def __init__(self, max_load: float = 100.0, decay_rate: float = 0.1):
        self.max_load = max_load
        self.decay_rate = decay_rate
        self.loads = np.array([0.0, 0.0])
        self.incoming = 0.0

    def reset(self) -> np.ndarray:
        self.loads = np.array([0.0, 0.0])
        self.incoming = np.random.uniform(1, 20)
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Route request to chosen server
        self.loads[action] += self.incoming

        # Natural decay (servers process requests)
        self.loads = np.maximum(0, self.loads - self.decay_rate * self.loads)

        # Reward: negative of max load (encourage balance)
        reward = -np.max(self.loads) / self.max_load

        # New incoming request
        self.incoming = np.random.uniform(1, 20)

        # This is a continuing task — never done
        done = False

        info = {"loads": self.loads.copy(), "balance_ratio": np.min(self.loads) / max(np.max(self.loads), 1e-8)}
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        return np.array([self.loads[0], self.loads[1], self.incoming])

    @property
    def state_space_size(self) -> int:
        return 3  # Continuous

    @property
    def action_space_size(self) -> int:
        return 2


# =============================================================================
# 4. Trajectory Recording and Analysis
# =============================================================================

@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    t: int
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: dict = field(default_factory=dict)


class TrajectoryRecorder:
    """Records and analyzes agent-environment interaction trajectories."""

    def __init__(self):
        self.steps: List[TrajectoryStep] = []
        self.episode_boundaries: List[int] = [0]

    def record(self, t: int, state, action, reward: float, next_state, done: bool, info: dict = None):
        self.steps.append(TrajectoryStep(t, state, action, reward, next_state, done, info or {}))
        if done:
            self.episode_boundaries.append(len(self.steps))

    @property
    def rewards(self) -> List[float]:
        return [s.reward for s in self.steps]

    @property
    def states(self) -> List:
        return [s.state for s in self.steps]

    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """Compute discounted returns G_t for each step."""
        n = len(self.steps)
        returns = [0.0] * n
        G = 0.0
        for i in range(n - 1, -1, -1):
            if self.steps[i].done:
                G = 0.0  # Reset at episode boundary
            G = self.steps[i].reward + gamma * G
            returns[i] = G
        return returns

    def episode_stats(self) -> List[Dict]:
        """Compute statistics for each episode."""
        stats = []
        for i in range(len(self.episode_boundaries) - 1):
            start = self.episode_boundaries[i]
            end = self.episode_boundaries[i + 1]
            ep_rewards = [self.steps[j].reward for j in range(start, end)]
            stats.append({
                "episode": i + 1,
                "length": end - start,
                "total_reward": sum(ep_rewards),
                "mean_reward": np.mean(ep_rewards),
                "min_reward": min(ep_rewards),
                "max_reward": max(ep_rewards),
            })
        return stats


# =============================================================================
# 5. Interaction Loop Implementation
# =============================================================================

def run_interaction_loop(
    env: Environment,
    agent: Agent,
    n_steps: int = 1000,
    max_episode_steps: int = 200,
    verbose: bool = False
) -> TrajectoryRecorder:
    """
    The fundamental agent-environment interaction loop.

    At each step t:
        1. Agent observes state S_t
        2. Agent selects action A_t = π(S_t)
        3. Environment returns (S_{t+1}, R_{t+1}, done)
        4. Agent (optionally) updates its knowledge
    """
    recorder = TrajectoryRecorder()
    state = env.reset()
    episode_step = 0

    for t in range(n_steps):
        # Step 1 & 2: Agent observes state and selects action
        action = agent.select_action(state)

        # Step 3: Environment transition
        next_state, reward, done, info = env.step(action)

        # Record the transition
        recorder.record(t, state, action, reward, next_state, done, info)

        # Step 4: Agent update
        agent.update(state, action, reward, next_state, done)

        if verbose and t < 20:
            print(f"  t={t}: S={state}, A={action}, R={reward:.2f}, S'={next_state}, done={done}")

        # Handle episode boundaries
        episode_step += 1
        if done or episode_step >= max_episode_steps:
            state = env.reset()
            episode_step = 0
        else:
            state = next_state

    return recorder


# =============================================================================
# 6. Example Agents
# =============================================================================

class RandomAgent(Agent):
    """Selects actions uniformly at random."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def select_action(self, state) -> int:
        return np.random.randint(self.n_actions)


class HeuristicCliffAgent(Agent):
    """
    Heuristic agent for cliff walking:
    Goes up first, then right, then down to the goal.
    """

    def __init__(self):
        self.goal = (3, 11)

    def select_action(self, state) -> int:
        r, c = state
        # Phase 1: Go up (away from cliff)
        if r > 0 and c < 11:
            return 0  # UP
        # Phase 2: Go right
        if c < 11:
            return 1  # RIGHT
        # Phase 3: Go down to goal
        if r < 3:
            return 2  # DOWN
        return 1  # Fallback: RIGHT


class LoadBalanceAgent(Agent):
    """Routes requests to the less-loaded server."""

    def select_action(self, state: np.ndarray) -> int:
        # Route to server with lower load
        return 0 if state[0] <= state[1] else 1


# =============================================================================
# 7. Demonstrations
# =============================================================================

def demo_episodic_task():
    """Demonstrate episodic task with cliff walking."""
    print("=" * 60)
    print("Episodic Task: Cliff Walking")
    print("=" * 60)

    env = CliffWalkingEnv()

    # Compare random vs heuristic agent
    for name, agent in [("Random", RandomAgent(4)), ("Heuristic", HeuristicCliffAgent())]:
        recorder = run_interaction_loop(env, agent, n_steps=5000, max_episode_steps=200)
        stats = recorder.episode_stats()

        if stats:
            avg_reward = np.mean([s["total_reward"] for s in stats])
            avg_length = np.mean([s["length"] for s in stats])
            n_episodes = len(stats)
            print(f"\n{name} Agent ({n_episodes} episodes):")
            print(f"  Avg Episode Reward: {avg_reward:.1f}")
            print(f"  Avg Episode Length: {avg_length:.1f}")

    print("\nNote: Episodic task → interaction naturally breaks into episodes")
    print("Each episode: Start → (actions) → Goal or max_steps")


def demo_continuing_task():
    """Demonstrate continuing task with server load balancing."""
    print("\n" + "=" * 60)
    print("Continuing Task: Server Load Balancing")
    print("=" * 60)

    env = ServerLoadEnv()

    for name, agent in [("Random", RandomAgent(2)), ("Balanced", LoadBalanceAgent())]:
        recorder = run_interaction_loop(env, agent, n_steps=1000)
        rewards = recorder.rewards

        print(f"\n{name} Agent (1000 steps, no episodes):")
        print(f"  Mean Reward:  {np.mean(rewards):.4f}")
        print(f"  Total Reward: {sum(rewards):.2f}")

    print("\nNote: Continuing task → no natural episodes, interaction goes on indefinitely")


def demo_trajectory_analysis():
    """Demonstrate trajectory recording and return computation."""
    print("\n" + "=" * 60)
    print("Trajectory Analysis")
    print("=" * 60)

    env = CliffWalkingEnv()
    agent = HeuristicCliffAgent()

    print("\nFirst 15 steps of agent-environment interaction:")
    recorder = run_interaction_loop(env, agent, n_steps=500, max_episode_steps=100, verbose=True)

    # Compute returns
    returns = recorder.compute_returns(gamma=0.99)
    stats = recorder.episode_stats()

    if stats:
        print(f"\nEpisode Statistics (first 5 episodes):")
        for s in stats[:5]:
            print(f"  Episode {s['episode']}: length={s['length']}, "
                  f"total_reward={s['total_reward']:.1f}")


def visualize_cliff_walking():
    """Visualize cliff walking trajectories."""
    env = CliffWalkingEnv()

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    for ax_idx, (name, agent) in enumerate([
        ("Random Agent", RandomAgent(4)),
        ("Heuristic Agent", HeuristicCliffAgent())
    ]):
        ax = axes[ax_idx]

        # Collect first successful episode
        state = env.reset()
        trajectory = [state]
        for _ in range(500):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            trajectory.append(next_state if not info.get("fell_off_cliff") else env.start)
            if done:
                break
            state = next_state if not info.get("fell_off_cliff") else env.start

        # Draw grid
        grid = np.zeros((env.rows, env.cols))
        for (r, c) in env.cliff:
            grid[r, c] = -1  # Cliff
        grid[env.start[0], env.start[1]] = 1  # Start
        grid[env.goal[0], env.goal[1]] = 2  # Goal

        ax.imshow(grid, cmap='RdYlGn', alpha=0.3, aspect='auto')

        # Draw trajectory
        rows = [s[0] for s in trajectory]
        cols = [s[1] for s in trajectory]
        ax.plot(cols, rows, 'b.-', alpha=0.5, markersize=3, linewidth=0.8)
        ax.plot(cols[0], rows[0], 'go', markersize=10, label='Start')
        ax.plot(cols[-1], rows[-1], 'r*', markersize=12, label='End')

        ax.set_title(f"{name} (steps: {len(trajectory)-1})")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend()

        # Mark cliff and goal
        for c in range(1, 11):
            ax.text(c, 3, '☠', ha='center', va='center', fontsize=8)
        ax.text(0, 3, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(11, 3, 'G', ha='center', va='center', fontsize=12, fontweight='bold')

    plt.suptitle("Agent-Environment Interface: Cliff Walking Trajectories", fontsize=14)
    plt.tight_layout()
    plt.savefig("agent_environment_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: agent_environment_trajectories.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    demo_episodic_task()
    demo_continuing_task()
    demo_trajectory_analysis()
    visualize_cliff_walking()

    print("\n✓ Agent-Environment Interface demonstrations complete.")

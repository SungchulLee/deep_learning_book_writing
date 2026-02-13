"""
random_walk.py (Module 03)

Random Walk Simulations
=======================

Location: 06_markov_chain/01_fundamentals/
Difficulty: ⭐⭐ Elementary
Estimated Time: 3-4 hours

Learning Objectives:
- Understand random walks as Markov chains
- Implement 1D and 2D random walks
- Analyze properties: expected position, variance
- Study boundary conditions and first passage times

Mathematical Foundation:
A random walk is a path consisting of steps in random directions.
It's a special case of a Markov chain where:
- States are positions (integers or coordinates)
- Transitions depend only on current position

For 1D symmetric random walk:
- P(X_{n+1} = X_n + 1) = p
- P(X_{n+1} = X_n - 1) = 1-p
- If p = 0.5, it's a simple symmetric random walk

Key Properties:
- E[X_n] = X_0 + n(2p-1) for biased walk
- Var[X_n] = 4np(1-p)
- For symmetric walk: E[X_n] = X_0, Var[X_n] = n
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class RandomWalk1D:
    """
    One-dimensional random walk on the integers.
    
    Mathematical Model:
    X_n = X_0 + Σ_{i=1}^n ξ_i
    where ξ_i are independent random variables with:
    P(ξ_i = +1) = p
    P(ξ_i = -1) = 1-p
    """
    
    def __init__(self, p=0.5, initial_position=0):
        """
        Initialize the random walk.
        
        Parameters:
            p (float): Probability of stepping right (must be in [0,1])
            initial_position (int): Starting position
        
        Mathematical Note:
        - p = 0.5: symmetric (unbiased) random walk
        - p > 0.5: drift to the right
        - p < 0.5: drift to the left
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        
        self.p = p
        self.initial_position = initial_position
        self.current_position = initial_position
        self.history = [initial_position]
    
    def step(self):
        """
        Take one step of the random walk.
        
        Returns:
            int: New position
        
        Mathematical Process:
        With probability p, move right (+1)
        With probability 1-p, move left (-1)
        """
        # Generate step: +1 with probability p, -1 with probability 1-p
        if np.random.random() < self.p:
            step_size = 1
        else:
            step_size = -1
        
        self.current_position += step_size
        self.history.append(self.current_position)
        
        return self.current_position
    
    def simulate(self, n_steps, initial_position=None):
        """
        Simulate n steps of the random walk.
        
        Parameters:
            n_steps (int): Number of steps to take
            initial_position (int): Starting position (if resetting)
        
        Returns:
            list: Sequence of positions
        
        Statistical Properties (for symmetric walk, p=0.5):
        - E[X_n] = X_0 (no expected drift)
        - Var[X_n] = n (variance grows linearly with time)
        - Std[X_n] = √n (standard deviation grows as square root)
        """
        if initial_position is not None:
            self.current_position = initial_position
            self.history = [initial_position]
        
        for _ in range(n_steps):
            self.step()
        
        return self.history
    
    def expected_position(self, n_steps):
        """
        Compute theoretical expected position after n steps.
        
        Parameters:
            n_steps (int): Number of steps
        
        Returns:
            float: Expected position E[X_n]
        
        Mathematical Formula:
        E[X_n] = X_0 + n(2p - 1)
        
        Derivation:
        Each step contributes +1 with prob p, -1 with prob 1-p
        E[ξ_i] = (+1)×p + (-1)×(1-p) = 2p - 1
        E[X_n] = E[X_0 + Σξ_i] = X_0 + n×E[ξ_i] = X_0 + n(2p-1)
        """
        return self.initial_position + n_steps * (2 * self.p - 1)
    
    def variance(self, n_steps):
        """
        Compute theoretical variance after n steps.
        
        Parameters:
            n_steps (int): Number of steps
        
        Returns:
            float: Variance Var[X_n]
        
        Mathematical Formula:
        Var[X_n] = 4np(1-p)
        
        For symmetric walk (p=0.5): Var[X_n] = n
        
        Derivation:
        Var[ξ_i] = E[ξ_i²] - (E[ξ_i])²
        E[ξ_i²] = (+1)²×p + (-1)²×(1-p) = 1
        Var[ξ_i] = 1 - (2p-1)² = 4p(1-p)
        Var[X_n] = n×Var[ξ_i] = 4np(1-p)
        """
        return 4 * n_steps * self.p * (1 - self.p)
    
    def first_passage_time(self, target_position, max_steps=10000):
        """
        Find the first time the walk reaches target_position.
        
        Parameters:
            target_position (int): Target to reach
            max_steps (int): Maximum steps to try
        
        Returns:
            int or None: First passage time, or None if not reached
        
        Mathematical Note:
        First passage time T = min{n ≥ 0 : X_n = target}
        For symmetric walk, E[T] is finite for any target.
        """
        self.current_position = self.initial_position
        
        for step in range(max_steps):
            if self.current_position == target_position:
                return step
            self.step()
        
        return None  # Target not reached within max_steps


class RandomWalk2D:
    """
    Two-dimensional random walk on the integer lattice Z².
    
    Mathematical Model:
    (X_n, Y_n) = (X_0, Y_0) + Σ_{i=1}^n (ξ_i, η_i)
    
    For symmetric walk:
    P(move right) = P(move left) = P(move up) = P(move down) = 1/4
    """
    
    def __init__(self, initial_position=(0, 0)):
        """
        Initialize 2D random walk.
        
        Parameters:
            initial_position (tuple): Starting (x, y) coordinates
        """
        self.initial_position = np.array(initial_position)
        self.current_position = np.array(initial_position)
        self.history = [self.current_position.copy()]
    
    def step(self):
        """
        Take one step in a random cardinal direction.
        
        Returns:
            np.ndarray: New position (x, y)
        
        Mathematical Process:
        Choose one of four directions with equal probability:
        - Right: (+1, 0)
        - Left:  (-1, 0)
        - Up:    (0, +1)
        - Down:  (0, -1)
        """
        # Four possible directions
        directions = np.array([
            [1, 0],   # Right
            [-1, 0],  # Left
            [0, 1],   # Up
            [0, -1]   # Down
        ])
        
        # Choose random direction
        direction = directions[np.random.randint(0, 4)]
        
        self.current_position = self.current_position + direction
        self.history.append(self.current_position.copy())
        
        return self.current_position
    
    def simulate(self, n_steps):
        """
        Simulate n steps of 2D random walk.
        
        Parameters:
            n_steps (int): Number of steps
        
        Returns:
            list: List of (x, y) positions
        
        Statistical Properties:
        - E[distance from origin] ≈ √(2n/π) for large n
        - Probability of return to origin → 0 as n → ∞ (in 2D)
        """
        self.current_position = self.initial_position.copy()
        self.history = [self.current_position.copy()]
        
        for _ in range(n_steps):
            self.step()
        
        return self.history
    
    def distance_from_origin(self):
        """
        Compute Euclidean distance from origin.
        
        Returns:
            float: ||X_n|| = √(x² + y²)
        """
        return np.linalg.norm(self.current_position - self.initial_position)


def example_symmetric_walk():
    """
    Example 1: Symmetric random walk (p = 0.5).
    
    Demonstrates zero expected drift and variance growth.
    """
    print("=" * 70)
    print("Example 1: Symmetric Random Walk (p = 0.5)")
    print("=" * 70)
    
    # Create symmetric walk
    walk = RandomWalk1D(p=0.5, initial_position=0)
    
    # Single realization
    path = walk.simulate(n_steps=100)
    
    print(f"\nSingle path of 100 steps:")
    print(f"  Final position: {path[-1]}")
    print(f"  Maximum position: {max(path)}")
    print(f"  Minimum position: {min(path)}")
    
    # Theoretical vs empirical statistics
    n = 100
    print(f"\nTheoretical properties after {n} steps:")
    print(f"  Expected position: {walk.expected_position(n):.2f}")
    print(f"  Variance: {walk.variance(n):.2f}")
    print(f"  Standard deviation: {np.sqrt(walk.variance(n)):.2f}")
    
    # Run many simulations to verify
    n_simulations = 10000
    final_positions = []
    
    for _ in range(n_simulations):
        walk = RandomWalk1D(p=0.5, initial_position=0)
        path = walk.simulate(100)
        final_positions.append(path[-1])
    
    print(f"\nEmpirical statistics ({n_simulations} simulations):")
    print(f"  Mean final position: {np.mean(final_positions):.2f}")
    print(f"  Variance: {np.var(final_positions):.2f}")
    print(f"  Standard deviation: {np.std(final_positions):.2f}")


def example_biased_walk():
    """
    Example 2: Biased random walk (p ≠ 0.5).
    
    Shows drift in the expected direction.
    """
    print("\n" + "=" * 70)
    print("Example 2: Biased Random Walk (p = 0.6)")
    print("=" * 70)
    
    # Create biased walk
    p = 0.6
    walk = RandomWalk1D(p=p, initial_position=0)
    
    n = 1000
    path = walk.simulate(n_steps=n)
    
    print(f"\nSimulation of {n} steps with p = {p}:")
    print(f"  Final position: {path[-1]}")
    
    # Theoretical expectations
    print(f"\nTheoretical properties:")
    print(f"  Expected drift per step: {2*p - 1:.2f}")
    print(f"  Expected position after {n} steps: {walk.expected_position(n):.2f}")
    print(f"  Variance: {walk.variance(n):.2f}")
    
    # Compare p = 0.3, 0.5, 0.7
    print("\n" + "-" * 70)
    print("Comparing different values of p:")
    print(f"{'p':<8} {'E[X_1000]':<15} {'Var[X_1000]':<15}")
    
    for p_val in [0.3, 0.5, 0.7]:
        walk = RandomWalk1D(p=p_val)
        exp_pos = walk.expected_position(1000)
        var_pos = walk.variance(1000)
        print(f"{p_val:<8.1f} {exp_pos:<15.2f} {var_pos:<15.2f}")


def example_first_passage():
    """
    Example 3: First passage time analysis.
    
    How long does it take to reach a target position?
    """
    print("\n" + "=" * 70)
    print("Example 3: First Passage Time")
    print("=" * 70)
    
    target = 10
    n_simulations = 1000
    
    print(f"\nFinding first passage times to position {target}")
    print("(symmetric walk, p = 0.5)")
    
    passage_times = []
    
    for _ in range(n_simulations):
        walk = RandomWalk1D(p=0.5, initial_position=0)
        fpt = walk.first_passage_time(target, max_steps=10000)
        if fpt is not None:
            passage_times.append(fpt)
    
    if passage_times:
        print(f"\nResults from {len(passage_times)} successful walks:")
        print(f"  Mean first passage time: {np.mean(passage_times):.2f} steps")
        print(f"  Median: {np.median(passage_times):.2f} steps")
        print(f"  Min: {min(passage_times)} steps")
        print(f"  Max: {max(passage_times)} steps")
        print(f"  Did not reach in {n_simulations - len(passage_times)} cases")


def example_2d_walk():
    """
    Example 4: Two-dimensional random walk.
    
    Explores random motion in 2D space.
    """
    print("\n" + "=" * 70)
    print("Example 4: Two-Dimensional Random Walk")
    print("=" * 70)
    
    # Single 2D walk
    walk = RandomWalk2D(initial_position=(0, 0))
    path = walk.simulate(n_steps=1000)
    
    # Extract x and y coordinates
    x_coords = [pos[0] for pos in path]
    y_coords = [pos[1] for pos in path]
    
    print(f"\nSimulation of 1000 steps:")
    print(f"  Final position: ({x_coords[-1]}, {y_coords[-1]})")
    print(f"  Final distance from origin: {walk.distance_from_origin():.2f}")
    print(f"  Max |x|: {max(abs(x) for x in x_coords)}")
    print(f"  Max |y|: {max(abs(y) for y in y_coords)}")
    
    # Analyze distance distribution
    n_simulations = 1000
    final_distances = []
    
    for _ in range(n_simulations):
        walk = RandomWalk2D()
        walk.simulate(1000)
        final_distances.append(walk.distance_from_origin())
    
    print(f"\nDistance statistics ({n_simulations} simulations):")
    print(f"  Mean distance: {np.mean(final_distances):.2f}")
    print(f"  Theoretical approximation: {np.sqrt(2 * 1000 / np.pi):.2f}")


def visualize_random_walks():
    """
    Create visualizations of random walks.
    """
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    # 1D walks comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Multiple symmetric walks
    ax = axes[0, 0]
    for i in range(10):
        walk = RandomWalk1D(p=0.5)
        path = walk.simulate(200)
        ax.plot(path, alpha=0.6, linewidth=1.5)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Position', fontsize=11)
    ax.set_title('10 Symmetric Random Walks (p=0.5)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Biased walks
    ax = axes[0, 1]
    for p_val, color in [(0.3, 'blue'), (0.5, 'green'), (0.7, 'red')]:
        walk = RandomWalk1D(p=p_val)
        path = walk.simulate(200)
        ax.plot(path, color=color, alpha=0.8, linewidth=2, label=f'p={p_val}')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Position', fontsize=11)
    ax.set_title('Biased Random Walks', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: 2D walk
    ax = axes[1, 0]
    walk_2d = RandomWalk2D()
    path_2d = walk_2d.simulate(500)
    
    x_coords = [pos[0] for pos in path_2d]
    y_coords = [pos[1] for pos in path_2d]
    
    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(path_2d)))
    for i in range(len(path_2d) - 1):
        ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=colors[i], linewidth=1.5)
    
    ax.plot(0, 0, 'go', markersize=10, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_title('2D Random Walk (500 steps)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Subplot 4: Distribution of final positions
    ax = axes[1, 1]
    walk = RandomWalk1D(p=0.5)
    final_positions = []
    for _ in range(5000):
        walk = RandomWalk1D(p=0.5)
        path = walk.simulate(100)
        final_positions.append(path[-1])
    
    ax.hist(final_positions, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Position', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Distribution of Final Positions (100 steps, 5000 simulations)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add theoretical normal distribution
    mu = 0
    sigma = np.sqrt(100)
    x = np.linspace(-40, 40, 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, y, 'r-', linewidth=2, label='Normal(0, 100)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/random_walks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Random walk visualizations saved to random_walks.png")


def main():
    """
    Run all random walk examples.
    """
    print("RANDOM WALK SIMULATIONS")
    print("=======================\n")
    
    # Run examples
    example_symmetric_walk()
    example_biased_walk()
    example_first_passage()
    example_2d_walk()
    
    # Create visualizations
    visualize_random_walks()
    
    print("\n" + "=" * 70)
    print("Key Properties of Random Walks:")
    print("=" * 70)
    print("1. Symmetric walk (p=0.5): E[X_n] = X_0, Var[X_n] = n")
    print("2. Biased walk: E[X_n] = X_0 + n(2p-1)")
    print("3. Standard deviation grows as √n")
    print("4. In 1D: symmetric walk is recurrent (returns to origin infinitely often)")
    print("5. In 2D: symmetric walk is recurrent")
    print("6. In 3D: symmetric walk is transient (may never return)")


if __name__ == "__main__":
    main()

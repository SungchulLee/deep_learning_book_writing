"""
00_comprehensive_overview.py

Comprehensive Markov Chain Implementation
==========================================

Location: 06_markov_chain/
Difficulty: All levels (Reference module)
Use: Quick reference combining all core concepts

This module combines all core Markov chain concepts in one place:
- Basic Markov chain simulation
- Multiple methods for stationary distribution
- Visualization and analysis tools
- Practical examples

Learning Objectives:
- Understand and implement Markov chains from scratch
- Compute stationary distributions using 4 different methods
- Compare analytical vs. simulation approaches
- Apply to real-world problems

Mathematical Foundation:
A Markov chain is a sequence X_0, X_1, X_2, ... where:
P(X_{n+1} = j | X_n = i, X_{n-1}, ..., X_0) = P(X_{n+1} = j | X_n = i)

Key concepts:
- Transition matrix P: P[i][j] = P(X_{n+1} = j | X_n = i)
- Stationary distribution π: π = πP
- Ergodicity: irreducible + aperiodic → unique stationary distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg as la


class MarkovChain:
    """
    Complete Markov Chain implementation with simulation and analysis.
    
    This class combines the functionality from multiple approaches:
    1. Basic simulation (step-by-step)
    2. Stationary distribution (analytical and simulation)
    3. Visualization tools
    
    Attributes:
        P: Transition probability matrix
        pi: Initial state distribution
        states: Array of state indices
        current_state: Current state of the chain
        state_history: History of visited states
    """
    
    def __init__(self, transition_probs, initial_distribution=None, 
                 state_names=None, random_seed=None):
        """
        Initialize Markov chain.
        
        Parameters:
            transition_probs: n×n transition matrix P
            initial_distribution: Initial probability distribution (uniform if None)
            state_names: Names for states (optional)
            random_seed: Random seed for reproducibility
        
        Mathematical Requirements:
        - P must be row-stochastic: each row sums to 1
        - All entries P[i][j] ∈ [0, 1]
        - If provided, initial_distribution must sum to 1
        """
        self.P = np.array(transition_probs, dtype=float)
        self.num_states = self.P.shape[0]
        self.states = np.arange(self.num_states)
        
        # Set initial distribution
        if initial_distribution is None:
            self.pi = np.ones(self.num_states) / self.num_states
        else:
            self.pi = np.array(initial_distribution, dtype=float)
        
        # Set state names
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.num_states)]
        else:
            self.state_names = state_names
        
        # Initialize state
        self.current_state = None
        self.state_history = []
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Validate
        self._validate()
    
    def _validate(self):
        """Validate transition matrix and initial distribution."""
        # Check if matrix is square
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Transition matrix must be square")
        
        # Check if rows sum to 1
        row_sums = np.sum(self.P, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Transition matrix rows must sum to 1. Got: {row_sums}")
        
        # Check if all entries are probabilities
        if np.any(self.P < 0) or np.any(self.P > 1):
            raise ValueError("All transition probabilities must be in [0, 1]")
        
        # Check initial distribution
        if not np.isclose(np.sum(self.pi), 1.0):
            raise ValueError(f"Initial distribution must sum to 1. Got: {np.sum(self.pi)}")
    
    def reset(self):
        """Reset to initial state sampled from initial distribution."""
        self.current_state = np.random.choice(self.states, p=self.pi)
        self.state_history = [self.current_state]
        return self.current_state
    
    def step(self):
        """
        Take one step in the Markov chain.
        
        Returns:
            New state
        
        Mathematical Process:
        Sample from categorical distribution P[current_state, :]
        """
        if self.current_state is None:
            self.reset()
        
        # Sample next state
        self.current_state = np.random.choice(
            self.states,
            p=self.P[self.current_state, :]
        )
        
        self.state_history.append(self.current_state)
        return self.current_state
    
    def simulate(self, n_steps, initial_state=None, return_history=True):
        """
        Simulate the Markov chain for n steps.
        
        Parameters:
            n_steps: Number of steps to simulate
            initial_state: Starting state (None for random from pi)
            return_history: Whether to return full path
        
        Returns:
            List of states visited (if return_history=True)
        """
        # Set initial state
        if initial_state is not None:
            self.current_state = initial_state
            self.state_history = [initial_state]
        else:
            self.reset()
        
        # Simulate steps
        for _ in range(n_steps):
            self.step()
        
        if return_history:
            return self.state_history.copy()
        else:
            return self.current_state
    
    # ========================================================================
    # STATIONARY DISTRIBUTION METHODS
    # ========================================================================
    
    def stationary_distribution_eigenvector(self):
        """
        Compute stationary distribution using eigenvector method.
        
        Returns:
            Stationary distribution π
        
        Mathematical Method:
        Find left eigenvector of P with eigenvalue 1:
        π^T is the eigenvector of P^T with eigenvalue 1
        
        Why this works:
        πP = π is equivalent to (P^T)(π^T) = 1·(π^T)
        So π^T is an eigenvector of P^T with eigenvalue 1
        """
        # Compute eigenvalues and eigenvectors of P^T
        eigenvalues, eigenvectors = la.eig(self.P.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        
        # Normalize to get probability distribution
        pi = pi / np.sum(pi)
        
        # Ensure all values are positive
        pi = np.abs(pi)
        pi = pi / np.sum(pi)
        
        return pi
    
    def stationary_distribution_linear_system(self):
        """
        Compute stationary distribution by solving linear system.
        
        Returns:
            Stationary distribution π
        
        Mathematical Method:
        Solve the system:
        1. πP = π  ⟹  π(P - I) = 0  ⟹  (P^T - I)π^T = 0
        2. Σπ_i = 1 (normalization)
        
        We replace one equation with the normalization constraint
        to ensure a unique solution.
        """
        n = self.num_states
        
        # Set up system (P^T - I)π^T = 0
        A = self.P.T - np.eye(n)
        
        # Replace last equation with normalization: Σπ_i = 1
        A[-1, :] = np.ones(n)
        
        # Right-hand side
        b = np.zeros(n)
        b[-1] = 1.0
        
        # Solve linear system
        pi = la.solve(A, b)
        
        return pi
    
    def stationary_distribution_power_iteration(self, max_iter=1000, tol=1e-10):
        """
        Compute stationary distribution using matrix power method.
        
        Parameters:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        
        Returns:
            Stationary distribution π
        
        Mathematical Method:
        For ergodic chains:
        lim_{n→∞} P^n = [π π π ... π]^T (all rows identical)
        
        We compute P^n until convergence.
        """
        P_n = self.P.copy()
        
        for n in range(max_iter):
            P_next = P_n @ self.P
            
            # Check convergence
            if np.max(np.abs(P_next - P_n)) < tol:
                # Extract stationary distribution (any row)
                return P_next[0, :]
            
            P_n = P_next
        
        # If didn't converge, return best estimate
        return P_n[0, :]
    
    def stationary_distribution_simulation(self, n_steps=100000, initial_state=None):
        """
        Estimate stationary distribution using long-run simulation.
        
        Parameters:
            n_steps: Number of simulation steps
            initial_state: Starting state (None for random)
        
        Returns:
            Estimated stationary distribution
        
        Mathematical Justification:
        By the Ergodic Theorem:
        lim_{T→∞} (1/T) Σ_{t=1}^T I{X_t = j} = π_j  (with probability 1)
        
        The time-average frequency equals the stationary probability.
        """
        # Run long simulation
        path = self.simulate(n_steps, initial_state=initial_state)
        
        # Count state frequencies
        state_counts = np.zeros(self.num_states)
        for state in path:
            state_counts[state] += 1
        
        # Convert to probabilities
        pi_empirical = state_counts / len(path)
        
        return pi_empirical
    
    def compare_stationary_methods(self, n_simulation_steps=100000):
        """
        Compare all four methods for computing stationary distribution.
        
        Returns:
            Dictionary with results from all methods
        """
        results = {}
        
        print("Computing stationary distribution using 4 methods...")
        print("="*70)
        
        # Method 1: Eigenvector
        results['eigenvector'] = self.stationary_distribution_eigenvector()
        print("✓ Method 1: Eigenvector approach")
        
        # Method 2: Linear system
        results['linear_system'] = self.stationary_distribution_linear_system()
        print("✓ Method 2: Linear system")
        
        # Method 3: Power iteration
        results['power_iteration'] = self.stationary_distribution_power_iteration()
        print("✓ Method 3: Matrix power iteration")
        
        # Method 4: Simulation
        results['simulation'] = self.stationary_distribution_simulation(n_simulation_steps)
        print(f"✓ Method 4: Simulation ({n_simulation_steps:,} steps)")
        
        print("="*70)
        
        # Display comparison
        print(f"\n{'State':<10s} {'Eigenvec':<12s} {'Linear':<12s} {'Power':<12s} {'Simulation':<12s}")
        print("-"*70)
        
        for i in range(self.num_states):
            print(f"{self.state_names[i]:<10s} "
                  f"{results['eigenvector'][i]:<12.8f} "
                  f"{results['linear_system'][i]:<12.8f} "
                  f"{results['power_iteration'][i]:<12.8f} "
                  f"{results['simulation'][i]:<12.8f}")
        
        return results
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def plot_transition_matrix(self):
        """Visualize transition matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(self.P, cmap='YlOrRd', vmin=0, vmax=1)
        
        ax.set_xticks(range(self.num_states))
        ax.set_yticks(range(self.num_states))
        ax.set_xticklabels(self.state_names)
        ax.set_yticklabels(self.state_names)
        ax.set_xlabel('To State', fontsize=12)
        ax.set_ylabel('From State', fontsize=12)
        ax.set_title('Transition Probability Matrix', fontsize=14)
        
        # Add text annotations
        for i in range(self.num_states):
            for j in range(self.num_states):
                text = ax.text(j, i, f'{self.P[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig
    
    def plot_state_sequence(self, path=None):
        """
        Plot state sequence over time.
        
        Parameters:
            path: State sequence to plot (uses self.state_history if None)
        """
        if path is None:
            path = self.state_history
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.step(range(len(path)), path, where='post', linewidth=2)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title('Markov Chain State Sequence', fontsize=14)
        ax.set_yticks(range(self.num_states))
        ax.set_yticklabels(self.state_names)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_paths(self, n_paths=10, n_steps=50):
        """Plot multiple realizations of the Markov chain."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i in range(n_paths):
            path = self.simulate(n_steps, initial_state=None)
            ax.step(range(len(path)), path, where='post', 
                   alpha=0.6, linewidth=1.5)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title(f'{n_paths} Independent Markov Chain Realizations', fontsize=14)
        ax.set_yticks(range(self.num_states))
        ax.set_yticklabels(self.state_names)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_convergence_to_stationary(self, initial_state=0, max_steps=50):
        """
        Visualize convergence to stationary distribution.
        
        Parameters:
            initial_state: Starting state
            max_steps: Maximum number of steps to show
        """
        # Get stationary distribution
        pi_stationary = self.stationary_distribution_eigenvector()
        
        # Track distribution evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Initial distribution (start at initial_state with probability 1)
        dist = np.zeros(self.num_states)
        dist[initial_state] = 1.0
        
        # Compute distributions at different times
        steps_to_show = [0, 1, 2, 5, 10, 20, max_steps]
        
        for n in steps_to_show:
            # Compute P^n
            if n == 0:
                dist_n = dist
            else:
                P_n = np.linalg.matrix_power(self.P, n)
                dist_n = dist @ P_n
            
            ax.plot(range(self.num_states), dist_n, marker='o', 
                   label=f'n={n}', linewidth=2, markersize=8)
        
        # Add stationary distribution
        ax.plot(range(self.num_states), pi_stationary, 'k--', 
               linewidth=3, label='π (stationary)', markersize=10)
        
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Convergence to Stationary Distribution (Starting from {self.state_names[initial_state]})', 
                    fontsize=14)
        ax.set_xticks(range(self.num_states))
        ax.set_xticklabels(self.state_names)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# EXAMPLE TRANSITION MATRIX GENERATORS
# ============================================================================

class TransitionMatrixExamples:
    """Collection of example transition matrices for testing."""
    
    @staticmethod
    def random_matrix(n_states=4, random_seed=1):
        """
        Generate random transition matrix.
        
        Method: Sample from normal, exponentiate, normalize
        """
        np.random.seed(random_seed)
        P = np.random.normal(0., 1., (n_states, n_states))
        P = np.exp(P)  # Make positive
        P = P / np.sum(P, axis=1, keepdims=True)  # Normalize rows
        return P
    
    @staticmethod
    def symmetric_walk(n_states=4):
        """
        Create symmetric random walk on states 0, 1, ..., n_states-1.
        
        Transition probabilities:
        - Move left with probability 0.4
        - Stay with probability 0.2
        - Move right with probability 0.4
        - Reflect at boundaries
        """
        P = np.zeros((n_states, n_states))
        l, r, s = 0.4, 0.4, 0.2
        
        for i in range(n_states):
            if i > 0:
                P[i, i-1] = l
            if i < n_states - 1:
                P[i, i+1] = r
            P[i, i] = s
            
            # Adjust boundaries
            if i == 0:
                P[i, i] += l
            if i == n_states - 1:
                P[i, i] += r
        
        return P
    
    @staticmethod
    def asymmetric_walk(n_states=4):
        """
        Create asymmetric random walk (biased to the right).
        """
        P = np.zeros((n_states, n_states))
        
        P[0, :] = [0.1, 0.9, 0.0, 0.0]
        P[1, :] = [0.1, 0.6, 0.3, 0.0]
        P[2, :] = [0.0, 0.5, 0.4, 0.1]
        P[3, :] = [0.0, 0.0, 0.7, 0.3]
        
        return P


# ============================================================================
# EXAMPLES AND DEMONSTRATIONS
# ============================================================================

def example_basic_simulation():
    """Example 1: Basic Markov chain simulation."""
    print("="*70)
    print("Example 1: Basic Markov Chain Simulation")
    print("="*70)
    
    # Create symmetric random walk
    P = TransitionMatrixExamples.symmetric_walk(n_states=4)
    mc = MarkovChain(P, state_names=['S0', 'S1', 'S2', 'S3'], random_seed=42)
    
    print("\nTransition Matrix:")
    print(P)
    
    # Simulate
    path = mc.simulate(n_steps=20, initial_state=0)
    print(f"\nSimulated path (21 states including initial):")
    print(path)
    
    # Visualize
    mc.plot_transition_matrix()
    fig_path = os.path.join(os.path.dirname(__file__), 'outputs', 'mc_transition_matrix.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    mc.plot_state_sequence()
    fig_path = os.path.join(os.path.dirname(__file__), 'outputs', 'mc_state_sequence.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Visualizations saved")


def example_stationary_distribution():
    """Example 2: Computing stationary distribution."""
    print("\n" + "="*70)
    print("Example 2: Stationary Distribution - All Methods")
    print("="*70)
    
    # Create chain
    P = TransitionMatrixExamples.asymmetric_walk(n_states=4)
    mc = MarkovChain(P, state_names=['S0', 'S1', 'S2', 'S3'])
    
    print("\nAsymmetric Walk Transition Matrix:")
    print(P)
    print()
    
    # Compare all methods
    results = mc.compare_stationary_methods(n_simulation_steps=100000)
    
    # Verify: π * P should equal π
    pi = results['eigenvector']
    verification = pi @ P
    print(f"\nVerification: ||π*P - π|| = {np.linalg.norm(verification - pi):.2e}")


def example_convergence_visualization():
    """Example 3: Visualize convergence to stationary distribution."""
    print("\n" + "="*70)
    print("Example 3: Convergence to Stationary Distribution")
    print("="*70)
    
    P = TransitionMatrixExamples.symmetric_walk(n_states=4)
    mc = MarkovChain(P, state_names=['S0', 'S1', 'S2', 'S3'])
    
    mc.plot_convergence_to_stationary(initial_state=0, max_steps=50)
    fig_path = os.path.join(os.path.dirname(__file__), 'outputs', 'mc_convergence.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Convergence visualization saved")


def example_multiple_realizations():
    """Example 4: Multiple independent paths."""
    print("\n" + "="*70)
    print("Example 4: Multiple Independent Realizations")
    print("="*70)
    
    P = TransitionMatrixExamples.symmetric_walk(n_states=4)
    mc = MarkovChain(P, state_names=['S0', 'S1', 'S2', 'S3'], random_seed=42)
    
    mc.plot_multiple_paths(n_paths=15, n_steps=50)
    fig_path = os.path.join(os.path.dirname(__file__), 'outputs', 'mc_multiple_paths.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Multiple paths visualization saved")


def main():
    """Run all examples."""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "COMPREHENSIVE MARKOV CHAIN TUTORIAL" + " "*18 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    example_basic_simulation()
    example_stationary_distribution()
    example_convergence_visualization()
    example_multiple_realizations()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Markov chains have the memoryless property")
    print("  2. Stationary distribution can be computed 4 ways:")
    print("     - Eigenvector method (most common)")
    print("     - Linear system solution")
    print("     - Matrix power iteration")
    print("     - Long-run simulation")
    print("  3. All methods give consistent results")
    print("  4. Ergodic chains converge to stationary distribution")
    print("="*70)


if __name__ == "__main__":
    main()

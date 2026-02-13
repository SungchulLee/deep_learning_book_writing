"""
transition_matrices.py (Module 02)

Working with Transition Matrices and State Probabilities
=========================================================

Location: 06_markov_chain/01_fundamentals/
Difficulty: ⭐⭐ Elementary
Estimated Time: 3-4 hours

Learning Objectives:
- Understand transition matrices mathematically
- Compute multi-step transition probabilities
- Analyze matrix powers P^n
- Calculate state probabilities at time n

Mathematical Foundation:
- Transition Matrix P: P[i][j] = P(X_{n+1} = j | X_n = i)
- Chapman-Kolmogorov equation: P^(n) = P^n (matrix power)
- n-step transition probability: P^(n)[i][j] = P(X_n = j | X_0 = i)
- State distribution evolution: π_n = π_0 × P^n
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import matrix_power


class TransitionMatrixAnalyzer:
    """
    Tools for analyzing and computing with transition matrices.
    
    Mathematical Properties:
    1. Stochastic matrix: rows sum to 1
    2. Chapman-Kolmogorov: P^(m+n) = P^m × P^n
    3. Power convergence: lim_{n→∞} P^n may exist
    """
    
    def __init__(self, transition_matrix, state_names=None):
        """
        Initialize the analyzer with a transition matrix.
        
        Parameters:
            transition_matrix (np.ndarray or list): The transition probability matrix
            state_names (list): Optional names for states
        """
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        # Validate
        self._validate_matrix()
    
    def _validate_matrix(self):
        """
        Validate that the matrix is a proper stochastic matrix.
        
        Requirements:
        1. Square matrix
        2. All entries in [0, 1]
        3. Each row sums to 1
        """
        # Check if square
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Transition matrix must be square")
        
        # Check non-negativity and ≤ 1
        if np.any(self.P < 0) or np.any(self.P > 1):
            raise ValueError("All probabilities must be in [0, 1]")
        
        # Check row sums
        row_sums = np.sum(self.P, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Row sums must equal 1. Got: {row_sums}")
    
    def n_step_transition_matrix(self, n):
        """
        Compute the n-step transition matrix P^n.
        
        Parameters:
            n (int): Number of steps
        
        Returns:
            np.ndarray: P^n where P^n[i][j] = P(X_n = j | X_0 = i)
        
        Mathematical Background:
        By Chapman-Kolmogorov equation:
        P^(n)[i][j] = Σ_k P^(m)[i][k] × P^(n-m)[k][j]
        
        This is equivalent to matrix multiplication: P^n = P × P × ... × P (n times)
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return np.eye(self.n_states)  # Identity matrix
        
        # Use scipy's optimized matrix power
        return matrix_power(self.P, n)
    
    def probability_after_n_steps(self, initial_state, target_state, n):
        """
        Calculate probability of being in target_state after n steps from initial_state.
        
        Parameters:
            initial_state (int or str): Starting state
            target_state (int or str): Target state
            n (int): Number of steps
        
        Returns:
            float: P(X_n = target | X_0 = initial)
        
        Mathematical Formula:
        P(X_n = j | X_0 = i) = [P^n]_{i,j}
        """
        # Convert state names to indices if needed
        if isinstance(initial_state, str):
            i = self.state_names.index(initial_state)
        else:
            i = initial_state
        
        if isinstance(target_state, str):
            j = self.state_names.index(target_state)
        else:
            j = target_state
        
        # Compute P^n and extract the probability
        P_n = self.n_step_transition_matrix(n)
        return P_n[i, j]
    
    def state_distribution_after_n_steps(self, initial_distribution, n):
        """
        Compute state probability distribution after n steps.
        
        Parameters:
            initial_distribution (np.ndarray): Initial probability distribution π_0
            n (int): Number of steps
        
        Returns:
            np.ndarray: Distribution π_n where π_n = π_0 × P^n
        
        Mathematical Background:
        If π_0 is a row vector representing initial state probabilities,
        then the distribution after n steps is:
        π_n = π_0 × P^n
        
        Component-wise: π_n[j] = Σ_i π_0[i] × P^n[i][j]
        """
        initial_distribution = np.array(initial_distribution)
        
        # Validate initial distribution
        if not np.isclose(np.sum(initial_distribution), 1.0):
            raise ValueError("Initial distribution must sum to 1")
        
        # Compute P^n
        P_n = self.n_step_transition_matrix(n)
        
        # Matrix multiplication: π_n = π_0 × P^n
        return initial_distribution @ P_n
    
    def analyze_convergence(self, max_steps=100, tolerance=1e-6):
        """
        Analyze whether P^n converges as n → ∞.
        
        Parameters:
            max_steps (int): Maximum number of steps to check
            tolerance (float): Convergence tolerance
        
        Returns:
            dict: Analysis results including convergence status and limit
        
        Mathematical Note:
        For regular Markov chains (some power of P has all positive entries),
        P^n converges to a matrix where all rows are identical,
        representing the stationary distribution.
        """
        results = {
            'converged': False,
            'convergence_step': None,
            'limit_matrix': None,
            'differences': []
        }
        
        P_prev = self.P.copy()
        
        for step in range(1, max_steps + 1):
            P_current = self.P @ P_prev  # P^(n+1) = P × P^n
            
            # Compute maximum difference between consecutive powers
            diff = np.max(np.abs(P_current - P_prev))
            results['differences'].append(diff)
            
            # Check for convergence
            if diff < tolerance:
                results['converged'] = True
                results['convergence_step'] = step
                results['limit_matrix'] = P_current
                break
            
            P_prev = P_current
        
        return results
    
    def visualize_n_step_probabilities(self, initial_state, max_steps=50):
        """
        Visualize how probabilities evolve over n steps.
        
        Parameters:
            initial_state (int or str): Starting state
            max_steps (int): Maximum number of steps to visualize
        """
        if isinstance(initial_state, str):
            i = self.state_names.index(initial_state)
        else:
            i = initial_state
        
        # Compute probabilities for each step
        probabilities = np.zeros((max_steps + 1, self.n_states))
        probabilities[0, i] = 1.0  # Start at initial_state with probability 1
        
        for n in range(1, max_steps + 1):
            P_n = self.n_step_transition_matrix(n)
            probabilities[n, :] = P_n[i, :]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        for j in range(self.n_states):
            plt.plot(range(max_steps + 1), probabilities[:, j], 
                    marker='o', markersize=4, label=self.state_names[j],
                    linewidth=2, alpha=0.7)
        
        plt.xlabel('Number of Steps (n)', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'State Probabilities Over Time (Starting from {self.state_names[i]})',
                 fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/transition_probabilities.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


def example_two_step_computation():
    """
    Example 1: Computing 2-step transition probabilities manually and via matrix power.
    
    Demonstrates the Chapman-Kolmogorov equation.
    """
    print("=" * 70)
    print("Example 1: Two-Step Transition Probability Computation")
    print("=" * 70)
    
    # Simple 2-state chain
    P = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    print("\nTransition Matrix P:")
    print(P)
    
    # Compute P^2 manually using Chapman-Kolmogorov
    print("\nComputing P^2 manually using Chapman-Kolmogorov:")
    print("P^2[0][0] = P[0][0]*P[0][0] + P[0][1]*P[1][0]")
    
    P_2_manual = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            # Chapman-Kolmogorov: P^2[i][j] = Σ_k P[i][k] * P[k][j]
            value = sum(P[i][k] * P[k][j] for k in range(2))
            P_2_manual[i][j] = value
            print(f"P^2[{i}][{j}] = {value:.4f}")
    
    # Compute P^2 using matrix multiplication
    P_2_matrix = P @ P
    
    print("\nP^2 via matrix multiplication:")
    print(P_2_matrix)
    
    print("\nVerification (difference should be ~0):")
    print(np.abs(P_2_manual - P_2_matrix))
    
    # Interpretation
    print("\nInterpretation:")
    print(f"Starting from state 0, probability of being in state 0 after 2 steps: {P_2_matrix[0,0]:.4f}")
    print(f"Starting from state 0, probability of being in state 1 after 2 steps: {P_2_matrix[0,1]:.4f}")


def example_state_distribution_evolution():
    """
    Example 2: Evolution of state distribution over time.
    
    Shows how an initial distribution evolves according to π_n = π_0 × P^n
    """
    print("\n" + "=" * 70)
    print("Example 2: State Distribution Evolution")
    print("=" * 70)
    
    # Three-state weather model
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = np.array([
        [0.7, 0.25, 0.05],
        [0.3, 0.4, 0.3],
        [0.1, 0.4, 0.5]
    ])
    
    analyzer = TransitionMatrixAnalyzer(P, states)
    
    # Start with uniform distribution (equal probability for each state)
    print("\nInitial distribution (uniform):")
    π_0 = np.array([1/3, 1/3, 1/3])
    print(f"π_0 = {π_0}")
    
    # Compute distributions for various steps
    steps_to_show = [1, 2, 5, 10, 20, 50]
    
    print("\nDistribution evolution:")
    print(f"{'Step':<8} {'Sunny':<12} {'Cloudy':<12} {'Rainy':<12}")
    print(f"{'0':<8} {π_0[0]:<12.6f} {π_0[1]:<12.6f} {π_0[2]:<12.6f}")
    
    for n in steps_to_show:
        π_n = analyzer.state_distribution_after_n_steps(π_0, n)
        print(f"{n:<8} {π_n[0]:<12.6f} {π_n[1]:<12.6f} {π_n[2]:<12.6f}")
    
    # Try different initial distributions
    print("\n" + "-" * 70)
    print("Starting from definitely Sunny (π_0 = [1, 0, 0]):")
    print(f"{'Step':<8} {'Sunny':<12} {'Cloudy':<12} {'Rainy':<12}")
    
    π_0_sunny = np.array([1.0, 0.0, 0.0])
    print(f"{'0':<8} {π_0_sunny[0]:<12.6f} {π_0_sunny[1]:<12.6f} {π_0_sunny[2]:<12.6f}")
    
    for n in steps_to_show:
        π_n = analyzer.state_distribution_after_n_steps(π_0_sunny, n)
        print(f"{n:<8} {π_n[0]:<12.6f} {π_n[1]:<12.6f} {π_n[2]:<12.6f}")


def example_convergence_analysis():
    """
    Example 3: Analyzing convergence of P^n as n → ∞.
    
    For regular chains, P^n converges to a limiting matrix.
    """
    print("\n" + "=" * 70)
    print("Example 3: Convergence Analysis")
    print("=" * 70)
    
    # Create a regular Markov chain (all entries of some P^k are positive)
    states = ['A', 'B', 'C']
    P = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ])
    
    analyzer = TransitionMatrixAnalyzer(P, states)
    
    print("\nTransition Matrix P:")
    print(P)
    
    # Analyze convergence
    results = analyzer.analyze_convergence(max_steps=100, tolerance=1e-8)
    
    if results['converged']:
        print(f"\nConvergence achieved at step {results['convergence_step']}")
        print("\nLimiting matrix (all rows identical = stationary distribution):")
        print(results['limit_matrix'])
        
        # Extract stationary distribution (any row of limit matrix)
        stationary = results['limit_matrix'][0, :]
        print(f"\nStationary distribution:")
        for i, state in enumerate(states):
            print(f"  π({state}) = {stationary[i]:.6f}")
    else:
        print("\nDid not converge within 100 steps")
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(1, len(results['differences']) + 1), results['differences'],
                'b-', linewidth=2)
    plt.xlabel('Step n', fontsize=12)
    plt.ylabel('||P^(n+1) - P^n|| (log scale)', fontsize=12)
    plt.title('Convergence of Transition Matrix Powers', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/convergence_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nConvergence plot saved to convergence_plot.png")


def example_specific_probabilities():
    """
    Example 4: Computing specific n-step transition probabilities.
    
    Answers questions like: "What's the probability of being in state B
    after 10 steps if we start in state A?"
    """
    print("\n" + "=" * 70)
    print("Example 4: Specific n-Step Probabilities")
    print("=" * 70)
    
    states = ['Healthy', 'Sick', 'Recovered']
    P = np.array([
        [0.8, 0.2, 0.0],    # Healthy: 80% stay healthy, 20% get sick
        [0.0, 0.5, 0.5],    # Sick: 50% stay sick, 50% recover
        [0.9, 0.0, 0.1]     # Recovered: 90% become healthy, 10% stay recovered
    ])
    
    analyzer = TransitionMatrixAnalyzer(P, states)
    
    print("\nTransition Matrix (Health States):")
    print("             Healthy  Sick  Recovered")
    for i, state in enumerate(states):
        print(f"{state:12s} {P[i]}")
    
    # Answer specific questions
    questions = [
        ("Healthy", "Sick", 1),
        ("Healthy", "Sick", 5),
        ("Healthy", "Recovered", 10),
        ("Sick", "Healthy", 3),
    ]
    
    print("\nSpecific probability queries:")
    for initial, target, steps in questions:
        prob = analyzer.probability_after_n_steps(initial, target, steps)
        print(f"P({target} after {steps} steps | start from {initial}) = {prob:.6f}")


def main():
    """
    Run all examples demonstrating transition matrix operations.
    """
    print("TRANSITION MATRIX ANALYSIS")
    print("==========================\n")
    
    # Run examples
    example_two_step_computation()
    example_state_distribution_evolution()
    example_convergence_analysis()
    example_specific_probabilities()
    
    # Create visualization
    print("\n" + "=" * 70)
    print("Creating Probability Evolution Visualization")
    print("=" * 70)
    
    states = ['State A', 'State B', 'State C']
    P = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ])
    
    analyzer = TransitionMatrixAnalyzer(P, states)
    analyzer.visualize_n_step_probabilities('State A', max_steps=50)
    print("Visualization saved to transition_probabilities.png")
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. P^n[i][j] gives the probability of transitioning from i to j in n steps")
    print("2. Chapman-Kolmogorov: P^(m+n) = P^m × P^n")
    print("3. Distribution evolution: π_n = π_0 × P^n")
    print("4. For regular chains, P^n converges to a limit matrix")
    print("5. The limit matrix has all rows equal to the stationary distribution")


if __name__ == "__main__":
    main()

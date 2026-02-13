"""
stationary_distribution.py (Module 04)

Stationary Distribution Analysis
==================================

Location: 06_markov_chain/02_analysis_methods/
Difficulty: ⭐⭐⭐ Intermediate
Estimated Time: 4-5 hours

Learning Objectives:
- Understand stationary distributions
- Compute stationary distributions using multiple methods
- Analyze conditions for existence and uniqueness
- Study convergence rates

Mathematical Foundation:
A stationary distribution π is a probability distribution satisfying:
π = π × P

Properties:
- π is a left eigenvector of P with eigenvalue 1
- For irreducible, aperiodic chains: unique stationary distribution exists
- lim_{n→∞} P^n converges to rows of π
- Physical interpretation: long-run proportion of time in each state
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig


class StationaryDistributionAnalyzer:
    """
    Tools for computing and analyzing stationary distributions.
    """
    
    def __init__(self, transition_matrix, state_names=None):
        """
        Initialize with transition matrix.
        
        Parameters:
            transition_matrix (np.ndarray): The transition probability matrix P
            state_names (list): Optional state names
        """
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
    
    def compute_via_eigenvector(self):
        """
        Compute stationary distribution via eigenvector method.
        
        Returns:
            np.ndarray: Stationary distribution π
        
        Mathematical Method:
        Find left eigenvector v of P with eigenvalue λ = 1
        Since π × P = π, we have π^T = P^T × π^T
        So π^T is right eigenvector of P^T with eigenvalue 1
        """
        # Compute eigenvalues and eigenvectors of P^T
        eigenvalues, eigenvectors = eig(self.P.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        # Extract corresponding eigenvector
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize to sum to 1 (probability distribution)
        stationary = stationary / np.sum(stationary)
        
        # Ensure all entries are positive
        stationary = np.abs(stationary)
        stationary = stationary / np.sum(stationary)
        
        return stationary
    
    def compute_via_power_iteration(self, max_iter=1000, tol=1e-10):
        """
        Compute via matrix power: lim_{n→∞} P^n
        
        Returns:
            tuple: (stationary distribution, number of iterations)
        
        Mathematical Basis:
        For ergodic chains, P^n converges to a matrix where all rows
        equal the stationary distribution π
        """
        P_n = self.P.copy()
        
        for n in range(1, max_iter):
            P_next = P_n @ self.P
            
            # Check convergence
            if np.max(np.abs(P_next - P_n)) < tol:
                # Extract stationary distribution (any row)
                return P_next[0, :], n
            
            P_n = P_next
        
        # If didn't converge, return best estimate
        return P_n[0, :], max_iter
    
    def compute_via_linear_system(self):
        """
        Compute by solving linear system: π(P - I) = 0, Σπ_i = 1
        
        Returns:
            np.ndarray: Stationary distribution
        
        Mathematical Setup:
        We need to solve:
        1. π × P = π  ⟹  π × (P - I) = 0
        2. Σ π_i = 1 (normalization)
        
        This is equivalent to solving:
        π^T × (P^T - I) = 0
        Σ π_i = 1
        """
        # Set up system: (P^T - I) × π^T = 0 with constraint Σπ_i = 1
        A = (self.P.T - np.eye(self.n_states))
        
        # Replace last equation with normalization constraint
        A[-1, :] = np.ones(self.n_states)
        b = np.zeros(self.n_states)
        b[-1] = 1.0
        
        # Solve linear system
        stationary = np.linalg.solve(A, b)
        
        return stationary
    
    def compute_via_simulation(self, n_steps=100000, initial_state=0):
        """
        Estimate via long-run simulation.
        
        Parameters:
            n_steps (int): Number of simulation steps
            initial_state (int): Starting state
        
        Returns:
            np.ndarray: Estimated stationary distribution
        
        Mathematical Justification:
        By ergodic theorem, the time-average equals the ensemble average:
        lim_{T→∞} (1/T) Σ I{X_t = j} = π_j
        """
        state_counts = np.zeros(self.n_states)
        current_state = initial_state
        
        for _ in range(n_steps):
            state_counts[current_state] += 1
            
            # Transition to next state
            current_state = np.random.choice(
                self.n_states,
                p=self.P[current_state, :]
            )
        
        # Normalize to get probabilities
        return state_counts / n_steps
    
    def check_ergodicity(self):
        """
        Check if chain is ergodic (irreducible and aperiodic).
        
        Returns:
            dict: Results of ergodicity checks
        
        Mathematical Conditions:
        1. Irreducible: can reach any state from any other state
        2. Aperiodic: gcd of return times to any state is 1
        
        Sufficient condition: some P^k has all positive entries
        """
        results = {
            'is_ergodic': False,
            'is_aperiodic': False,
            'is_irreducible': False
        }
        
        # Check if some power of P has all positive entries
        # This guarantees both irreducibility and aperiodicity
        P_power = self.P.copy()
        
        for k in range(1, self.n_states + 1):
            if np.all(P_power > 0):
                results['is_ergodic'] = True
                results['is_aperiodic'] = True
                results['is_irreducible'] = True
                results['power_with_positive_entries'] = k
                break
            P_power = P_power @ self.P
        
        return results


def example_computing_methods():
    """
    Example 1: Compare different methods for computing stationary distribution.
    """
    print("=" * 70)
    print("Example 1: Computing Stationary Distribution - Method Comparison")
    print("=" * 70)
    
    # Three-state chain
    states = ['A', 'B', 'C']
    P = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ])
    
    print("\nTransition Matrix P:")
    print(P)
    
    analyzer = StationaryDistributionAnalyzer(P, states)
    
    # Method 1: Eigenvector
    print("\n" + "-" * 70)
    print("Method 1: Eigenvector Approach")
    π_eig = analyzer.compute_via_eigenvector()
    print("Stationary distribution:")
    for i, state in enumerate(states):
        print(f"  π({state}) = {π_eig[i]:.8f}")
    
    # Verification: π × P should equal π
    verification = π_eig @ P
    print("\nVerification (π × P should equal π):")
    print(f"  Max difference: {np.max(np.abs(verification - π_eig)):.2e}")
    
    # Method 2: Power iteration
    print("\n" + "-" * 70)
    print("Method 2: Matrix Power Iteration")
    π_power, iterations = analyzer.compute_via_power_iteration()
    print(f"Converged in {iterations} iterations")
    print("Stationary distribution:")
    for i, state in enumerate(states):
        print(f"  π({state}) = {π_power[i]:.8f}")
    
    # Method 3: Linear system
    print("\n" + "-" * 70)
    print("Method 3: Linear System Solution")
    π_linear = analyzer.compute_via_linear_system()
    print("Stationary distribution:")
    for i, state in enumerate(states):
        print(f"  π({state}) = {π_linear[i]:.8f}")
    
    # Method 4: Simulation
    print("\n" + "-" * 70)
    print("Method 4: Long-run Simulation (1,000,000 steps)")
    π_sim = analyzer.compute_via_simulation(n_steps=1000000)
    print("Stationary distribution:")
    for i, state in enumerate(states):
        print(f"  π({state}) = {π_sim[i]:.8f}")
    
    # Compare all methods
    print("\n" + "-" * 70)
    print("Comparison of All Methods:")
    print(f"{'State':<8} {'Eigenvec':<12} {'Power':<12} {'Linear':<12} {'Simulation':<12}")
    for i, state in enumerate(states):
        print(f"{state:<8} {π_eig[i]:<12.8f} {π_power[i]:<12.8f} "
              f"{π_linear[i]:<12.8f} {π_sim[i]:<12.8f}")


def example_interpretation():
    """
    Example 2: Physical interpretation of stationary distribution.
    """
    print("\n" + "=" * 70)
    print("Example 2: Physical Interpretation")
    print("=" * 70)
    
    # Queue system: {Empty, 1 customer, 2 customers}
    states = ['Empty', '1 Customer', '2 Customers']
    P = np.array([
        [0.5, 0.4, 0.1],    # From Empty: likely to get a customer
        [0.3, 0.5, 0.2],    # From 1: balanced
        [0.4, 0.4, 0.2]     # From 2: tend to reduce
    ])
    
    print("\nQueue System Transition Matrix:")
    print(f"{'':15s} {'Empty':>12s} {'1 Customer':>12s} {'2 Customers':>12s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:12.4f}" for j in range(len(states)))
        print(f"{state:15s} {row}")
    
    analyzer = StationaryDistributionAnalyzer(P, states)
    π = analyzer.compute_via_eigenvector()
    
    print("\nStationary Distribution (Long-run Proportions):")
    for i, state in enumerate(states):
        print(f"  {state:15s}: π = {π[i]:.6f} ({π[i]*100:.2f}%)")
    
    print("\nInterpretation:")
    print(f"  In the long run:")
    print(f"  - Queue is empty {π[0]*100:.1f}% of the time")
    print(f"  - Queue has 1 customer {π[1]*100:.1f}% of the time")
    print(f"  - Queue has 2 customers {π[2]*100:.1f}% of the time")
    
    # Expected number of customers
    expected_customers = 0*π[0] + 1*π[1] + 2*π[2]
    print(f"\n  Average number of customers in system: {expected_customers:.4f}")


def example_ergodicity():
    """
    Example 3: Check ergodicity conditions.
    """
    print("\n" + "=" * 70)
    print("Example 3: Ergodicity Analysis")
    print("=" * 70)
    
    # Ergodic chain
    print("\nCase 1: Ergodic Chain")
    P_ergodic = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ])
    
    analyzer1 = StationaryDistributionAnalyzer(P_ergodic)
    results1 = analyzer1.check_ergodicity()
    
    print(f"  Is ergodic: {results1['is_ergodic']}")
    if results1['is_ergodic']:
        print(f"  P^{results1['power_with_positive_entries']} has all positive entries")
        print("  ⟹ Stationary distribution exists and is unique")
        π = analyzer1.compute_via_eigenvector()
        print(f"  Stationary: {π}")
    
    # Periodic chain
    print("\nCase 2: Periodic Chain")
    P_periodic = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    
    analyzer2 = StationaryDistributionAnalyzer(P_periodic)
    results2 = analyzer2.check_ergodicity()
    
    print(f"  Is ergodic: {results2['is_ergodic']}")
    print("  This chain cycles: A → B → C → A")
    print("  Stationary distribution exists but convergence doesn't occur")
    π2 = analyzer2.compute_via_eigenvector()
    print(f"  Stationary: {π2}")
    
    # Reducible chain
    print("\nCase 3: Reducible Chain (Two Components)")
    P_reducible = np.array([
        [0.5, 0.5, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 0, 0.7, 0.3],
        [0, 0, 0.3, 0.7]
    ])
    
    analyzer3 = StationaryDistributionAnalyzer(P_reducible)
    results3 = analyzer3.check_ergodicity()
    
    print(f"  Is ergodic: {results3['is_ergodic']}")
    print("  Two separate components: {0,1} and {2,3}")
    print("  Stationary distribution depends on initial state")


def visualize_convergence():
    """
    Visualize convergence to stationary distribution.
    """
    print("\n" + "=" * 70)
    print("Creating Convergence Visualization")
    print("=" * 70)
    
    states = ['A', 'B', 'C']
    P = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ])
    
    analyzer = StationaryDistributionAnalyzer(P, states)
    π_stationary = analyzer.compute_via_eigenvector()
    
    # Track convergence from different initial distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Different initial distributions
    initial_dists = [
        np.array([1.0, 0.0, 0.0]),  # Start at A
        np.array([0.0, 1.0, 0.0]),  # Start at B
        np.array([0.0, 0.0, 1.0]),  # Start at C
        np.array([1/3, 1/3, 1/3])   # Uniform
    ]
    
    labels = ['Start at A', 'Start at B', 'Start at C', 'Uniform']
    colors = ['red', 'blue', 'green', 'purple']
    
    # Plot 1: Distance to stationary distribution
    ax = axes[0]
    
    for init_dist, label, color in zip(initial_dists, labels, colors):
        distances = []
        dist = init_dist.copy()
        
        for n in range(50):
            # Compute distance to stationary distribution
            distance = np.linalg.norm(dist - π_stationary)
            distances.append(distance)
            
            # Update distribution
            dist = dist @ P
        
        ax.semilogy(distances, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Step n', fontsize=12)
    ax.set_ylabel('||π_n - π*|| (log scale)', fontsize=12)
    ax.set_title('Convergence to Stationary Distribution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Evolution of state probabilities
    ax = axes[1]
    
    init_dist = np.array([1.0, 0.0, 0.0])  # Start at A
    steps = range(51)
    state_probs = {state: [] for state in states}
    
    dist = init_dist.copy()
    for n in steps:
        for i, state in enumerate(states):
            state_probs[state].append(dist[i])
        dist = dist @ P
    
    for i, state in enumerate(states):
        ax.plot(steps, state_probs[state], marker='o', markersize=3,
               label=state, linewidth=2)
        # Add stationary distribution line
        ax.axhline(y=π_stationary[i], linestyle='--', color=f'C{i}', alpha=0.5)
    
    ax.set_xlabel('Step n', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('State Probabilities Over Time (Starting from A)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/stationary_convergence.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Convergence visualization saved")


def main():
    """
    Run all examples.
    """
    print("STATIONARY DISTRIBUTION ANALYSIS")
    print("=================================\n")
    
    example_computing_methods()
    example_interpretation()
    example_ergodicity()
    visualize_convergence()
    
    print("\n" + "=" * 70)
    print("Key Theoretical Results:")
    print("=" * 70)
    print("1. Stationary distribution satisfies: π = π × P")
    print("2. For ergodic chains: unique stationary distribution exists")
    print("3. Ergodic = irreducible + aperiodic")
    print("4. P^n converges to π for ergodic chains")
    print("5. Long-run proportion in state j equals π_j")


if __name__ == "__main__":
    main()

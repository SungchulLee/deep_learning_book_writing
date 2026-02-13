"""
absorbing_chains.py (Module 05)

Absorbing Markov Chains
========================

Location: 06_markov_chain/02_analysis_methods/
Difficulty: ⭐⭐⭐ Intermediate
Estimated Time: 3-4 hours

Learning Objectives:
- Understand absorbing states and chains
- Compute absorption probabilities
- Calculate expected time to absorption
- Analyze the fundamental matrix

Mathematical Foundation:
An absorbing state is a state that, once entered, cannot be left.
State i is absorbing if P[i][i] = 1.

An absorbing chain has:
1. At least one absorbing state
2. Every state can reach an absorbing state

Canonical Form of P:
    ┌       ┐
P = │ Q  R  │  where:
    │ 0  I  │
    └       ┘
- Q: transitions between transient states
- R: transitions from transient to absorbing
- I: identity matrix (absorbing states)
- 0: zero matrix (can't leave absorbing states)

Key Quantities:
- Fundamental matrix: N = (I - Q)^{-1}
- N[i][j] = expected visits to transient state j starting from i
- Expected steps to absorption: t = N × 1 (column vector of ones)
- Absorption probabilities: B = N × R
"""

import numpy as np
import matplotlib.pyplot as plt


class AbsorbingMarkovChain:
    """
    Analysis tools for absorbing Markov chains.
    """
    
    def __init__(self, transition_matrix, state_names=None):
        """
        Initialize absorbing chain.
        
        Parameters:
            transition_matrix (np.ndarray): Transition matrix
            state_names (list): Optional state names
        """
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        # Identify absorbing and transient states
        self._identify_states()
        
        # Reorder if needed
        self._reorder_canonical()
    
    def _identify_states(self):
        """
        Identify which states are absorbing.
        
        A state i is absorbing if P[i][i] = 1 (and all other P[i][j] = 0)
        """
        self.absorbing_indices = []
        self.transient_indices = []
        
        for i in range(self.n_states):
            if np.isclose(self.P[i, i], 1.0) and np.allclose(self.P[i, :i], 0.0) and np.allclose(self.P[i, i+1:], 0.0):
                self.absorbing_indices.append(i)
            else:
                self.transient_indices.append(i)
        
        self.n_transient = len(self.transient_indices)
        self.n_absorbing = len(self.absorbing_indices)
    
    def _reorder_canonical(self):
        """
        Reorder states into canonical form: transient first, then absorbing.
        
        Creates Q, R matrices and computes fundamental matrix N.
        """
        if self.n_absorbing == 0:
            raise ValueError("No absorbing states found")
        
        # Reorder states
        reordered_indices = self.transient_indices + self.absorbing_indices
        
        # Reorder transition matrix
        P_canonical = self.P[np.ix_(reordered_indices, reordered_indices)]
        
        # Extract Q and R
        self.Q = P_canonical[:self.n_transient, :self.n_transient]
        self.R = P_canonical[:self.n_transient, self.n_transient:]
        
        # Store reordered names
        self.transient_names = [self.state_names[i] for i in self.transient_indices]
        self.absorbing_names = [self.state_names[i] for i in self.absorbing_indices]
    
    def fundamental_matrix(self):
        """
        Compute fundamental matrix N = (I - Q)^{-1}.
        
        Returns:
            np.ndarray: Fundamental matrix N
        
        Mathematical Interpretation:
        N[i][j] = expected number of times in transient state j,
                  starting from transient state i, before absorption
        
        Derivation:
        Let M[i][j] = E[# visits to j | start at i]
        M[i][j] = δ_{ij} + Σ_k P[i][k] × M[k][j]
        In matrix form: M = I + Q × M
        Solving: M = (I - Q)^{-1} = N
        """
        I = np.eye(self.n_transient)
        self.N = np.linalg.inv(I - self.Q)
        return self.N
    
    def expected_steps_to_absorption(self):
        """
        Compute expected number of steps until absorption from each state.
        
        Returns:
            dict: Expected steps for each transient state
        
        Mathematical Formula:
        t = N × 1 (where 1 is column vector of ones)
        
        Interpretation:
        t[i] = expected steps to absorption starting from transient state i
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        # Multiply N by column vector of ones
        ones = np.ones((self.n_transient, 1))
        t = self.N @ ones
        
        # Return as dictionary
        result = {}
        for i, name in enumerate(self.transient_names):
            result[name] = t[i, 0]
        
        return result
    
    def absorption_probabilities(self):
        """
        Compute probability of absorption into each absorbing state.
        
        Returns:
            dict: For each transient state, probabilities of absorbing into each absorbing state
        
        Mathematical Formula:
        B = N × R
        
        Interpretation:
        B[i][j] = probability of being absorbed into absorbing state j,
                  starting from transient state i
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        self.B = self.N @ self.R
        
        # Return as nested dictionary
        result = {}
        for i, trans_name in enumerate(self.transient_names):
            result[trans_name] = {}
            for j, abs_name in enumerate(self.absorbing_names):
                result[trans_name][abs_name] = self.B[i, j]
        
        return result
    
    def variance_steps_to_absorption(self):
        """
        Compute variance of steps to absorption.
        
        Returns:
            dict: Variance for each transient state
        
        Mathematical Formula:
        Var[T_i] = (2N - I) × t - t²
        where t is the expected steps vector
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        ones = np.ones((self.n_transient, 1))
        t = self.N @ ones
        
        I = np.eye(self.n_transient)
        variance_vec = (2 * self.N - I) @ t - t**2
        
        result = {}
        for i, name in enumerate(self.transient_names):
            result[name] = variance_vec[i, 0]
        
        return result


def example_simple_gambler():
    """
    Example 1: Gambler's ruin problem.
    
    A gambler starts with $2. Each round: win $1 (p=0.5) or lose $1 (1-p=0.5).
    Game ends when reaching $0 (broke) or $4 (target).
    """
    print("=" * 70)
    print("Example 1: Gambler's Ruin")
    print("=" * 70)
    
    # States: $0, $1, $2, $3, $4
    # Absorbing: $0 (broke), $4 (win)
    # Transient: $1, $2, $3
    
    states = ['$0 (Broke)', '$1', '$2', '$3', '$4 (Win)']
    
    # Transition matrix (p = 0.5 for fair game)
    P = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # $0: stay broke
        [0.5, 0.0, 0.5, 0.0, 0.0],  # $1: go to $0 or $2
        [0.0, 0.5, 0.0, 0.5, 0.0],  # $2: go to $1 or $3
        [0.0, 0.0, 0.5, 0.0, 0.5],  # $3: go to $2 or $4
        [0.0, 0.0, 0.0, 0.0, 1.0]   # $4: stay at win
    ])
    
    print("\nTransition Matrix:")
    print(P)
    
    chain = AbsorbingMarkovChain(P, states)
    
    print(f"\nAbsorbing states: {chain.absorbing_names}")
    print(f"Transient states: {chain.transient_names}")
    
    # Fundamental matrix
    print("\n" + "-" * 70)
    print("Fundamental Matrix N (expected visits):")
    N = chain.fundamental_matrix()
    print(f"{'':8s} " + " ".join(f"{s:8s}" for s in chain.transient_names))
    for i, name in enumerate(chain.transient_names):
        row = " ".join(f"{N[i,j]:8.4f}" for j in range(len(chain.transient_names)))
        print(f"{name:8s} {row}")
    
    # Expected steps to absorption
    print("\n" + "-" * 70)
    print("Expected Steps to Absorption:")
    expected_steps = chain.expected_steps_to_absorption()
    for state, steps in expected_steps.items():
        print(f"  Starting from {state}: {steps:.4f} steps")
    
    # Absorption probabilities
    print("\n" + "-" * 70)
    print("Absorption Probabilities:")
    absorption_probs = chain.absorption_probabilities()
    for trans_state in chain.transient_names:
        print(f"\n  Starting from {trans_state}:")
        for abs_state in chain.absorbing_names:
            prob = absorption_probs[trans_state][abs_state]
            print(f"    P(absorb at {abs_state}) = {prob:.6f}")
    
    # Variance
    print("\n" + "-" * 70)
    print("Variance of Steps to Absorption:")
    variances = chain.variance_steps_to_absorption()
    for state, var in variances.items():
        print(f"  Starting from {state}: {var:.4f} (std = {np.sqrt(var):.4f})")


def example_disease_model():
    """
    Example 2: Disease progression model.
    
    States: Healthy, Infected, Recovered, Dead
    Absorbing: Recovered, Dead
    """
    print("\n" + "=" * 70)
    print("Example 2: Disease Progression Model")
    print("=" * 70)
    
    states = ['Healthy', 'Infected', 'Recovered', 'Dead']
    
    P = np.array([
        [0.7, 0.3, 0.0, 0.0],    # Healthy: may get infected
        [0.0, 0.4, 0.5, 0.1],    # Infected: recover, stay sick, or die
        [0.0, 0.0, 1.0, 0.0],    # Recovered: absorbing
        [0.0, 0.0, 0.0, 1.0]     # Dead: absorbing
    ])
    
    print("\nTransition Matrix:")
    print(f"{'':12s} {'Healthy':>10s} {'Infected':>10s} {'Recovered':>10s} {'Dead':>10s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
        print(f"{state:12s} {row}")
    
    chain = AbsorbingMarkovChain(P, states)
    
    print(f"\nAbsorbing states: {chain.absorbing_names}")
    print(f"Transient states: {chain.transient_names}")
    
    # Expected time to absorption
    expected_steps = chain.expected_steps_to_absorption()
    print("\nExpected time until recovery or death:")
    for state, steps in expected_steps.items():
        print(f"  From {state}: {steps:.4f} days")
    
    # Absorption probabilities
    absorption_probs = chain.absorption_probabilities()
    print("\nFinal outcome probabilities:")
    for trans_state in chain.transient_names:
        print(f"\n  Starting from {trans_state}:")
        for abs_state in chain.absorbing_names:
            prob = absorption_probs[trans_state][abs_state]
            print(f"    {abs_state}: {prob:.4f} ({prob*100:.2f}%)")


def visualize_absorption():
    """
    Visualize absorption process through simulation.
    """
    print("\n" + "=" * 70)
    print("Creating Absorption Visualization")
    print("=" * 70)
    
    # Gambler's ruin
    states_idx = {'$0': 0, '$1': 1, '$2': 2, '$3': 3, '$4': 4}
    P = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    
    # Simulate multiple paths
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sample paths
    ax = axes[0, 0]
    
    for _ in range(20):
        path = [2]  # Start at $2
        current = 2
        
        while current != 0 and current != 4 and len(path) < 100:
            probs = P[current, :]
            current = np.random.choice(5, p=probs)
            path.append(current)
        
        ax.plot(path, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Money ($)', fontsize=11)
    ax.set_title('Sample Paths in Gambler\'s Ruin', fontsize=12)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['$0', '$1', '$2', '$3', '$4'])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Broke')
    ax.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Win')
    ax.legend()
    
    # Plot 2: Distribution of absorption times
    ax = axes[0, 1]
    
    absorption_times = []
    for _ in range(10000):
        steps = 0
        current = 2
        
        while current != 0 and current != 4 and steps < 1000:
            probs = P[current, :]
            current = np.random.choice(5, p=probs)
            steps += 1
        
        absorption_times.append(steps)
    
    ax.hist(absorption_times, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Steps to Absorption', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Distribution of Time to Absorption', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add theoretical mean
    chain = AbsorbingMarkovChain(P, ['$0', '$1', '$2', '$3', '$4'])
    expected = chain.expected_steps_to_absorption()
    theoretical_mean = expected['$2']
    ax.axvline(x=theoretical_mean, color='red', linestyle='--', linewidth=2,
              label=f'Theoretical Mean: {theoretical_mean:.2f}')
    ax.axvline(x=np.mean(absorption_times), color='blue', linestyle='--', linewidth=2,
              label=f'Empirical Mean: {np.mean(absorption_times):.2f}')
    ax.legend()
    
    # Plot 3: Absorption probabilities
    ax = axes[1, 0]
    
    outcomes = {'Win': 0, 'Broke': 0}
    for _ in range(10000):
        current = 2
        
        while current != 0 and current != 4:
            probs = P[current, :]
            current = np.random.choice(5, p=probs)
        
        if current == 4:
            outcomes['Win'] += 1
        else:
            outcomes['Broke'] += 1
    
    labels = list(outcomes.keys())
    values = [outcomes[k] / 10000 for k in labels]
    colors = ['green', 'red']
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Absorption Outcomes (Starting from $2)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Fundamental matrix visualization
    ax = axes[1, 1]
    
    chain = AbsorbingMarkovChain(P, ['$0', '$1', '$2', '$3', '$4'])
    N = chain.fundamental_matrix()
    
    im = ax.imshow(N, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(chain.transient_names)))
    ax.set_yticks(range(len(chain.transient_names)))
    ax.set_xticklabels(chain.transient_names)
    ax.set_yticklabels(chain.transient_names)
    ax.set_xlabel('To State', fontsize=11)
    ax.set_ylabel('From State', fontsize=11)
    ax.set_title('Fundamental Matrix N (Expected Visits)', fontsize=12)
    
    for i in range(len(chain.transient_names)):
        for j in range(len(chain.transient_names)):
            text = ax.text(j, i, f'{N[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/absorbing_chains.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Absorption visualization saved")


def main():
    """
    Run all examples.
    """
    print("ABSORBING MARKOV CHAINS")
    print("=======================\n")
    
    example_simple_gambler()
    example_disease_model()
    visualize_absorption()
    
    print("\n" + "=" * 70)
    print("Key Concepts:")
    print("=" * 70)
    print("1. Absorbing state: P[i][i] = 1")
    print("2. Fundamental matrix: N = (I - Q)^{-1}")
    print("3. Expected steps to absorption: t = N × 1")
    print("4. Absorption probabilities: B = N × R")
    print("5. N[i][j] = expected visits to state j from state i")


if __name__ == "__main__":
    main()

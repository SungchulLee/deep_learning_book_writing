"""
basic_markov_chain.py (Module 01)

Introduction to Markov Chains
==============================

Location: 06_markov_chain/01_fundamentals/
Difficulty: ⭐ Beginner
Estimated Time: 2-3 hours

Learning Objectives:
- Understand the Markov property
- Implement simple discrete-time Markov chains
- Simulate state transitions
- Visualize state sequences

Mathematical Foundation:
A Markov chain is a sequence of random variables X_0, X_1, X_2, ... where:
P(X_{n+1} = j | X_n = i, X_{n-1} = k, ..., X_0 = m) = P(X_{n+1} = j | X_n = i)

This is called the Markov property: the future depends only on the present, not the past.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


class BasicMarkovChain:
    """
    A simple implementation of a discrete-time Markov chain.
    
    Attributes:
        states (list): List of state names
        transition_matrix (np.ndarray): Transition probability matrix
        current_state (int): Index of current state
    """
    
    def __init__(self, states, transition_matrix):
        """
        Initialize the Markov chain.
        
        Parameters:
            states (list): Names of states (e.g., ['A', 'B', 'C'])
            transition_matrix (np.ndarray): Transition probability matrix P
                                          where P[i][j] = P(X_{n+1}=j | X_n=i)
        
        Mathematical Note:
        - Each row of transition_matrix must sum to 1
        - All entries must be non-negative
        """
        self.states = states
        self.transition_matrix = np.array(transition_matrix)
        self.n_states = len(states)
        
        # Validate transition matrix
        self._validate_transition_matrix()
        
        # Initialize at a random state
        self.current_state = np.random.randint(0, self.n_states)
    
    def _validate_transition_matrix(self):
        """
        Validate that the transition matrix is stochastic.
        
        A matrix is stochastic if:
        1. All entries are non-negative
        2. Each row sums to 1 (probability distribution)
        """
        # Check dimensions
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix must be {self.n_states}x{self.n_states}")
        
        # Check non-negativity
        if np.any(self.transition_matrix < 0):
            raise ValueError("All transition probabilities must be non-negative")
        
        # Check row sums (each should sum to 1)
        row_sums = np.sum(self.transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of transition matrix must sum to 1")
    
    def step(self):
        """
        Perform one step of the Markov chain.
        
        Returns:
            str: Name of the new state
        
        Mathematical Process:
        Given current state i, select next state j with probability P[i][j]
        This is equivalent to sampling from the categorical distribution
        defined by row i of the transition matrix.
        """
        # Get transition probabilities from current state
        probabilities = self.transition_matrix[self.current_state]
        
        # Sample next state according to these probabilities
        # np.random.choice uses the discrete distribution defined by probabilities
        self.current_state = np.random.choice(self.n_states, p=probabilities)
        
        return self.states[self.current_state]
    
    def simulate(self, n_steps, initial_state=None):
        """
        Simulate the Markov chain for n steps.
        
        Parameters:
            n_steps (int): Number of steps to simulate
            initial_state (int or str): Starting state (index or name)
        
        Returns:
            list: Sequence of states visited
        
        Mathematical Note:
        This generates a realization of the stochastic process {X_n}
        """
        # Set initial state if provided
        if initial_state is not None:
            if isinstance(initial_state, str):
                self.current_state = self.states.index(initial_state)
            else:
                self.current_state = initial_state
        
        # Record the state sequence
        state_sequence = [self.states[self.current_state]]
        
        # Simulate n steps
        for _ in range(n_steps):
            state_sequence.append(self.step())
        
        return state_sequence
    
    def get_state_distribution(self, n_steps, n_simulations=10000):
        """
        Estimate the state distribution after n steps using Monte Carlo simulation.
        
        Parameters:
            n_steps (int): Number of steps to take
            n_simulations (int): Number of simulations to run
        
        Returns:
            np.ndarray: Estimated probability distribution over states
        
        Mathematical Background:
        If π_0 is the initial distribution and P is the transition matrix,
        then the distribution after n steps is: π_n = π_0 * P^n
        
        We estimate this by running many simulations and counting.
        """
        # Count final states across all simulations
        final_state_counts = np.zeros(self.n_states)
        
        for _ in range(n_simulations):
            # Run one simulation
            sequence = self.simulate(n_steps)
            final_state = sequence[-1]
            
            # Count the final state
            final_state_index = self.states.index(final_state)
            final_state_counts[final_state_index] += 1
        
        # Convert counts to probabilities
        estimated_distribution = final_state_counts / n_simulations
        
        return estimated_distribution


def example_two_state_chain():
    """
    Example 1: Simple two-state Markov chain
    
    States: {0, 1} or {Off, On}
    Transition matrix:
        From\To  Off   On
        Off      0.7   0.3
        On       0.4   0.6
    
    Interpretation:
    - If currently Off, 70% chance to stay Off, 30% chance to turn On
    - If currently On, 40% chance to turn Off, 60% chance to stay On
    """
    print("=" * 60)
    print("Example 1: Two-State Markov Chain (Off/On)")
    print("=" * 60)
    
    # Define states
    states = ['Off', 'On']
    
    # Define transition matrix
    # P[i][j] = probability of going from state i to state j
    transition_matrix = [
        [0.7, 0.3],  # From Off: 70% stay Off, 30% go to On
        [0.4, 0.6]   # From On: 40% go to Off, 60% stay On
    ]
    
    # Create Markov chain
    mc = BasicMarkovChain(states, transition_matrix)
    
    print("\nTransition Matrix:")
    print(transition_matrix)
    print("\nSimulating 20 steps...")
    
    # Simulate starting from 'Off'
    sequence = mc.simulate(n_steps=20, initial_state='Off')
    print(f"\nState sequence: {sequence}")
    
    # Estimate long-term distribution
    print("\nEstimating state distribution after 100 steps...")
    distribution = mc.get_state_distribution(n_steps=100)
    
    print(f"Estimated probabilities:")
    for state, prob in zip(states, distribution):
        print(f"  P({state}) = {prob:.4f}")


def example_three_state_chain():
    """
    Example 2: Three-state weather model
    
    States: {Sunny, Cloudy, Rainy}
    
    This models simplified weather transitions:
    - Sunny tends to stay sunny or become cloudy
    - Cloudy can go any direction
    - Rainy tends to become cloudy or stay rainy
    """
    print("\n" + "=" * 60)
    print("Example 2: Three-State Weather Model")
    print("=" * 60)
    
    # Define states
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    # Define transition matrix
    transition_matrix = [
        [0.7, 0.25, 0.05],  # From Sunny
        [0.3, 0.4, 0.3],     # From Cloudy
        [0.1, 0.4, 0.5]      # From Rainy
    ]
    
    # Create Markov chain
    mc = BasicMarkovChain(states, transition_matrix)
    
    print("\nTransition Matrix:")
    print("        Sunny  Cloudy  Rainy")
    for i, state in enumerate(states):
        print(f"{state:7s} {transition_matrix[i]}")
    
    # Simulate multiple days
    print("\nSimulating 30 days starting from Sunny...")
    sequence = mc.simulate(n_steps=30, initial_state='Sunny')
    
    # Print in a readable format (10 days per line)
    for i in range(0, len(sequence), 10):
        day_sequence = sequence[i:i+10]
        print(f"Days {i:2d}-{min(i+9, len(sequence)-1):2d}: {day_sequence}")
    
    # Count state frequencies
    print("\nState frequencies in simulation:")
    for state in states:
        count = sequence.count(state)
        frequency = count / len(sequence)
        print(f"  {state:7s}: {count:2d}/{len(sequence)} = {frequency:.3f}")


def visualize_state_sequence(sequence, title="Markov Chain State Sequence"):
    """
    Visualize a state sequence over time.
    
    Parameters:
        sequence (list): List of state names
        title (str): Title for the plot
    """
    # Get unique states and map to integers
    unique_states = sorted(list(set(sequence)))
    state_to_int = {state: i for i, state in enumerate(unique_states)}
    
    # Convert sequence to integers
    int_sequence = [state_to_int[state] for state in sequence]
    
    # Create plot
    plt.figure(figsize=(12, 4))
    plt.step(range(len(int_sequence)), int_sequence, where='post', linewidth=2)
    plt.yticks(range(len(unique_states)), unique_states)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'markov_sequence.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to markov_sequence.png")


def main():
    """
    Main function to run all examples.
    """
    print("BASIC MARKOV CHAIN SIMULATIONS")
    print("================================\n")
    
    # Run examples
    example_two_state_chain()
    example_three_state_chain()
    
    # Create a visualization
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)
    
    states = ['A', 'B', 'C']
    transition_matrix = [
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ]
    
    mc = BasicMarkovChain(states, transition_matrix)
    sequence = mc.simulate(n_steps=50, initial_state='A')
    visualize_state_sequence(sequence, "Three-State Markov Chain Simulation")
    
    print("\n" + "=" * 60)
    print("Exercises for Students:")
    print("=" * 60)
    print("1. Modify the two-state chain to model a light bulb (Working/Broken)")
    print("2. Create a four-state chain for traffic lights")
    print("3. Experiment with different initial states - does it affect long-term behavior?")
    print("4. Try to create a chain that always returns to the starting state")


if __name__ == "__main__":
    main()

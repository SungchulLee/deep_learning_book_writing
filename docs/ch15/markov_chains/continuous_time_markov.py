"""
continuous_time_markov.py (Module 10)

Continuous-Time Markov Chains (CTMC)
=====================================

Location: 06_markov_chain/03_applications/
Difficulty: ⭐⭐⭐⭐ Advanced
Estimated Time: 3-4 hours

Learning Objectives:
- Understand continuous-time processes
- Compute transition probabilities P(t)
- Analyze generator matrices
- Simulate CTMCs

Mathematical Foundation:
Continuous-time Markov chain: X(t), t ≥ 0
- Transitions can occur at any time
- Holding times are exponentially distributed
- Generator matrix Q: Q[i][j] = transition rate from i to j (i≠j)
- Kolmogorov forward equation: P'(t) = P(t) × Q
- Solution: P(t) = exp(Qt)
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


class ContinuousTimeMarkovChain:
    """Continuous-time Markov chain simulator."""
    
    def __init__(self, generator_matrix, state_names=None):
        """
        Initialize CTMC.
        
        Parameters:
            generator_matrix: Q where Q[i][j] is rate from i to j (i≠j)
        """
        self.Q = np.array(generator_matrix, dtype=float)
        self.n_states = self.Q.shape[0]
        
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
    
    def transition_probabilities(self, t):
        """Compute P(t) = exp(Qt)."""
        return expm(self.Q * t)
    
    def simulate(self, T, initial_state=0):
        """Simulate CTMC until time T."""
        times = [0]
        states = [initial_state]
        current_state = initial_state
        current_time = 0
        
        while current_time < T:
            # Holding time in current state (exponential with rate -Q[i][i])
            rate = -self.Q[current_state, current_state]
            if rate <= 0:
                break
            
            holding_time = np.random.exponential(1/rate)
            current_time += holding_time
            
            if current_time >= T:
                break
            
            # Choose next state
            transition_rates = self.Q[current_state, :].copy()
            transition_rates[current_state] = 0
            probs = transition_rates / transition_rates.sum()
            
            current_state = np.random.choice(self.n_states, p=probs)
            times.append(current_time)
            states.append(current_state)
        
        times.append(T)
        states.append(current_state)
        
        return times, states


# Example: Birth-death process
if __name__ == "__main__":
    print("CONTINUOUS-TIME MARKOV CHAINS")
    print("=" * 70)
    
    # Birth-death process: population can increase or decrease
    Q = np.array([
        [-2, 2, 0],
        [1, -3, 2],
        [0, 1, -1]
    ])
    
    ctmc = ContinuousTimeMarkovChain(Q, ['Low', 'Medium', 'High'])
    
    print("\\nGenerator Matrix Q:")
    print(Q)
    
    print("\\nTransition probabilities P(1.0):")
    P_1 = ctmc.transition_probabilities(1.0)
    print(P_1)
    
    print("\\nSimulation over time [0, 10]:")
    times, states = ctmc.simulate(10, initial_state=1)
    print(f"Number of transitions: {len(times)-1}")

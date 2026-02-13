"""
weather_model.py (Module 07)

Weather Modeling with Markov Chains
====================================

Location: 06_markov_chain/03_applications/
Difficulty: ⭐⭐ Elementary
Estimated Time: 2-3 hours

Learning Objectives:
- Model real-world phenomena using Markov chains
- Estimate transition matrices from data
- Make weather predictions
- Analyze long-term weather patterns

Mathematical Foundation:
Weather can be modeled as a Markov chain if we assume:
- The weather tomorrow depends only on today's weather
- Transition probabilities are time-homogeneous (constant)

This is a simplification, but useful for understanding patterns.

Application:
Given historical weather data, we can:
1. Estimate transition probabilities
2. Predict future weather
3. Calculate long-term frequency of weather types
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class WeatherMarkovChain:
    """
    Markov chain model for weather prediction.
    
    States typically include: Sunny, Cloudy, Rainy, etc.
    """
    
    def __init__(self, states):
        """
        Initialize weather model.
        
        Parameters:
            states (list): Weather state names (e.g., ['Sunny', 'Rainy'])
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.transition_matrix = None
    
    def estimate_from_data(self, weather_sequence):
        """
        Estimate transition matrix from observed weather data.
        
        Parameters:
            weather_sequence (list): Sequence of observed weather states
        
        Returns:
            np.ndarray: Estimated transition matrix
        
        Mathematical Method:
        Maximum Likelihood Estimation (MLE):
        P̂[i][j] = (# transitions from i to j) / (# times in state i)
        
        This is the frequency estimator:
        P̂[i][j] = N_{ij} / Σ_k N_{ik}
        where N_{ij} = number of observed transitions i → j
        """
        # Count transitions
        # transition_counts[i][j] = number of transitions from state i to state j
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for t in range(len(weather_sequence) - 1):
            current_state = weather_sequence[t]
            next_state = weather_sequence[t + 1]
            
            # Convert to indices
            i = self.state_to_idx[current_state]
            j = self.state_to_idx[next_state]
            
            transition_counts[i, j] += 1
        
        # Normalize to get probabilities
        # Each row sums to 1
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        
        # Handle states that never occurred (avoid division by zero)
        row_sums[row_sums == 0] = 1
        
        self.transition_matrix = transition_counts / row_sums
        
        return self.transition_matrix
    
    def predict_next_day(self, current_weather):
        """
        Predict tomorrow's weather (probabilistically).
        
        Parameters:
            current_weather (str): Today's weather
        
        Returns:
            dict: Probability distribution for tomorrow's weather
        
        Mathematical Basis:
        P(X_{t+1} = j | X_t = i) = P[i][j]
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        i = self.state_to_idx[current_weather]
        probabilities = self.transition_matrix[i, :]
        
        # Return as dictionary
        return {state: prob for state, prob in zip(self.states, probabilities)}
    
    def predict_n_days(self, current_weather, n_days):
        """
        Predict weather distribution n days ahead.
        
        Parameters:
            current_weather (str): Current weather state
            n_days (int): Number of days ahead
        
        Returns:
            dict: Probability distribution n days ahead
        
        Mathematical Basis:
        P(X_{t+n} = j | X_t = i) = [P^n]_{i,j}
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        # Create initial distribution (100% current_weather)
        initial_dist = np.zeros(self.n_states)
        initial_dist[self.state_to_idx[current_weather]] = 1.0
        
        # Multiply by P^n
        P_n = np.linalg.matrix_power(self.transition_matrix, n_days)
        future_dist = initial_dist @ P_n
        
        return {state: prob for state, prob in zip(self.states, future_dist)}
    
    def simulate_weather(self, n_days, initial_weather):
        """
        Simulate weather sequence for n days.
        
        Parameters:
            n_days (int): Number of days to simulate
            initial_weather (str): Starting weather
        
        Returns:
            list: Simulated weather sequence
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        sequence = [initial_weather]
        current_idx = self.state_to_idx[initial_weather]
        
        for _ in range(n_days):
            # Sample next state according to transition probabilities
            probs = self.transition_matrix[current_idx, :]
            next_idx = np.random.choice(self.n_states, p=probs)
            
            sequence.append(self.states[next_idx])
            current_idx = next_idx
        
        return sequence
    
    def stationary_distribution(self, method='eigenvector'):
        """
        Compute the stationary distribution.
        
        Parameters:
            method (str): 'eigenvector' or 'power'
        
        Returns:
            dict: Stationary probabilities for each state
        
        Mathematical Background:
        Stationary distribution π satisfies: π = π × P
        This means π is a left eigenvector of P with eigenvalue 1.
        
        Physical Interpretation:
        Long-run proportion of time spent in each weather state.
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        if method == 'eigenvector':
            # Find left eigenvector with eigenvalue 1
            # P^T × v = 1 × v, so we find eigenvector of P^T
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            
            # Find eigenvector corresponding to eigenvalue 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            
            # Normalize to sum to 1
            stationary = stationary / stationary.sum()
        
        elif method == 'power':
            # Compute P^n for large n
            P_n = np.linalg.matrix_power(self.transition_matrix, 1000)
            stationary = P_n[0, :]  # Any row gives the stationary distribution
        
        return {state: prob for state, prob in zip(self.states, stationary)}


def example_simple_weather_model():
    """
    Example 1: Simple three-state weather model.
    
    Demonstrates estimation from data and prediction.
    """
    print("=" * 70)
    print("Example 1: Three-State Weather Model")
    print("=" * 70)
    
    # Observed weather data (30 days)
    observed_weather = [
        'Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy',
        'Sunny', 'Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy',
        'Cloudy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Cloudy',
        'Cloudy', 'Rainy', 'Rainy', 'Rainy', 'Cloudy', 'Sunny',
        'Sunny', 'Cloudy', 'Rainy', 'Cloudy', 'Sunny', 'Sunny'
    ]
    
    print(f"\nObserved weather sequence ({len(observed_weather)} days):")
    print(observed_weather)
    
    # Count frequencies
    counter = Counter(observed_weather)
    print(f"\nObserved frequencies:")
    for state, count in sorted(counter.items()):
        print(f"  {state}: {count}/{len(observed_weather)} = {count/len(observed_weather):.3f}")
    
    # Create model and estimate transitions
    states = ['Sunny', 'Cloudy', 'Rainy']
    model = WeatherMarkovChain(states)
    P = model.estimate_from_data(observed_weather)
    
    print(f"\nEstimated Transition Matrix:")
    print(f"{'':10s} {'Sunny':>10s} {'Cloudy':>10s} {'Rainy':>10s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
        print(f"{state:10s} {row}")
    
    # Make predictions
    print(f"\n" + "-" * 70)
    print("Predictions if today is Sunny:")
    tomorrow = model.predict_next_day('Sunny')
    for state, prob in sorted(tomorrow.items()):
        print(f"  P(Tomorrow = {state} | Today = Sunny) = {prob:.4f}")
    
    print(f"\nPredictions 7 days ahead if today is Sunny:")
    week_ahead = model.predict_n_days('Sunny', 7)
    for state, prob in sorted(week_ahead.items()):
        print(f"  P(Day 7 = {state} | Today = Sunny) = {prob:.4f}")


def example_stationary_distribution():
    """
    Example 2: Computing and interpreting stationary distribution.
    
    Shows long-term weather patterns.
    """
    print("\n" + "=" * 70)
    print("Example 2: Stationary Distribution Analysis")
    print("=" * 70)
    
    # Use a predefined transition matrix
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = np.array([
        [0.7, 0.25, 0.05],   # From Sunny
        [0.3, 0.4, 0.3],      # From Cloudy
        [0.2, 0.3, 0.5]       # From Rainy
    ])
    
    print("\nTransition Matrix:")
    print(f"{'':10s} {'Sunny':>10s} {'Cloudy':>10s} {'Rainy':>10s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
        print(f"{state:10s} {row}")
    
    # Create model
    model = WeatherMarkovChain(states)
    model.transition_matrix = P
    
    # Compute stationary distribution
    print("\n" + "-" * 70)
    print("Stationary Distribution (long-run frequencies):")
    
    # Method 1: Eigenvector
    stationary_eig = model.stationary_distribution(method='eigenvector')
    print("\nUsing eigenvector method:")
    for state, prob in sorted(stationary_eig.items()):
        print(f"  π({state}) = {prob:.6f}")
    
    # Method 2: Power iteration
    stationary_pow = model.stationary_distribution(method='power')
    print("\nUsing matrix power method:")
    for state, prob in sorted(stationary_pow.items()):
        print(f"  π({state}) = {prob:.6f}")
    
    # Verify with simulation
    print("\n" + "-" * 70)
    print("Verification via simulation (10,000 days):")
    
    long_sim = model.simulate_weather(10000, initial_weather='Sunny')
    simulated_freq = Counter(long_sim)
    
    print("\nSimulated frequencies:")
    for state in sorted(states):
        freq = simulated_freq[state] / len(long_sim)
        theoretical = stationary_eig[state]
        print(f"  {state}: {freq:.6f} (theoretical: {theoretical:.6f})")


def example_seasonal_weather():
    """
    Example 3: Generate synthetic seasonal weather data.
    
    Models seasons with different transition probabilities.
    """
    print("\n" + "=" * 70)
    print("Example 3: Seasonal Weather Patterns")
    print("=" * 70)
    
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    # Summer transition matrix (more sunny)
    P_summer = np.array([
        [0.8, 0.15, 0.05],
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2]
    ])
    
    # Winter transition matrix (more rainy)
    P_winter = np.array([
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    
    print("\nSummer Transition Matrix:")
    print(P_summer)
    
    print("\nWinter Transition Matrix:")
    print(P_winter)
    
    # Simulate summer
    model_summer = WeatherMarkovChain(states)
    model_summer.transition_matrix = P_summer
    summer_weather = model_summer.simulate_weather(90, 'Sunny')
    
    # Simulate winter
    model_winter = WeatherMarkovChain(states)
    model_winter.transition_matrix = P_winter
    winter_weather = model_winter.simulate_weather(90, 'Cloudy')
    
    # Compare frequencies
    summer_freq = Counter(summer_weather)
    winter_freq = Counter(winter_weather)
    
    print("\n" + "-" * 70)
    print("Simulated 90-day frequencies:")
    print(f"{'State':<10s} {'Summer':<15s} {'Winter':<15s}")
    
    for state in states:
        s_freq = summer_freq[state] / len(summer_weather)
        w_freq = winter_freq[state] / len(winter_weather)
        print(f"{state:<10s} {s_freq:<15.4f} {w_freq:<15.4f}")


def visualize_weather_model():
    """
    Create visualizations for weather model.
    """
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = np.array([
        [0.7, 0.25, 0.05],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    
    model = WeatherMarkovChain(states)
    model.transition_matrix = P
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Transition matrix heatmap
    ax = axes[0, 0]
    im = ax.imshow(P, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_xlabel('To State', fontsize=11)
    ax.set_ylabel('From State', fontsize=11)
    ax.set_title('Transition Probability Matrix', fontsize=12)
    
    # Add text annotations
    for i in range(len(states)):
        for j in range(len(states)):
            text = ax.text(j, i, f'{P[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    
    # Subplot 2: Sample weather sequence
    ax = axes[0, 1]
    weather_seq = model.simulate_weather(60, 'Sunny')
    
    # Convert to numbers for plotting
    state_to_num = {state: i for i, state in enumerate(states)}
    num_seq = [state_to_num[w] for w in weather_seq]
    
    ax.step(range(len(num_seq)), num_seq, where='post', linewidth=2)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Weather', fontsize=11)
    ax.set_title('Simulated 60-Day Weather Sequence', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Long-term prediction convergence
    ax = axes[1, 0]
    
    days_ahead = range(1, 31)
    sunny_probs = []
    cloudy_probs = []
    rainy_probs = []
    
    for n in days_ahead:
        dist = model.predict_n_days('Sunny', n)
        sunny_probs.append(dist['Sunny'])
        cloudy_probs.append(dist['Cloudy'])
        rainy_probs.append(dist['Rainy'])
    
    ax.plot(days_ahead, sunny_probs, 'o-', label='Sunny', linewidth=2)
    ax.plot(days_ahead, cloudy_probs, 's-', label='Cloudy', linewidth=2)
    ax.plot(days_ahead, rainy_probs, '^-', label='Rainy', linewidth=2)
    
    # Add stationary distribution lines
    stationary = model.stationary_distribution()
    ax.axhline(y=stationary['Sunny'], color='C0', linestyle='--', alpha=0.5)
    ax.axhline(y=stationary['Cloudy'], color='C1', linestyle='--', alpha=0.5)
    ax.axhline(y=stationary['Rainy'], color='C2', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Prediction Convergence to Stationary Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Stationary distribution bar chart
    ax = axes[1, 1]
    stationary = model.stationary_distribution()
    
    colors = ['#FFD700', '#87CEEB', '#4682B4']
    bars = ax.bar(states, [stationary[s] for s in states], color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Long-run Frequency', fontsize=11)
    ax.set_title('Stationary Distribution', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, state in zip(bars, states):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{stationary[state]:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/weather_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Weather model visualizations saved to weather_model.png")


def main():
    """
    Run all weather modeling examples.
    """
    print("WEATHER MODELING WITH MARKOV CHAINS")
    print("====================================\n")
    
    # Run examples
    example_simple_weather_model()
    example_stationary_distribution()
    example_seasonal_weather()
    
    # Create visualizations
    visualize_weather_model()
    
    print("\n" + "=" * 70)
    print("Practical Applications:")
    print("=" * 70)
    print("1. Short-term weather prediction (1-7 days)")
    print("2. Long-term climate pattern analysis")
    print("3. Agricultural planning")
    print("4. Event planning based on weather probabilities")
    print("5. Understanding stationary behavior of weather systems")


if __name__ == "__main__":
    main()

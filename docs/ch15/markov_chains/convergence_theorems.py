"""
================================================================================
CONVERGENCE THEOREMS AND MIXING TIMES
================================================================================

DIFFICULTY LEVEL: INTERMEDIATE-ADVANCED
ESTIMATED TIME: 90-120 minutes
PREREQUISITES: Files 01-03, Understanding of limits and convergence

LEARNING OBJECTIVES:
1. Understand irreducibility and aperiodicity conditions
2. State and prove convergence theorems
3. Define and compute mixing times
4. Analyze convergence rates via eigenvalues
5. Bridge to MCMC: why mixing time matters

MATHEMATICAL FOUNDATIONS:
==========================

DEFINITIONS:

1. IRREDUCIBILITY:
   A chain is IRREDUCIBLE if every state can be reached from every other state.
   Formally: For all i,j, there exists n > 0 such that P^n[i,j] > 0.
   
   INTUITION: The entire state space is "connected" - no isolated parts.

2. APERIODICITY:
   A state i has PERIOD d = gcd{n: P^n[i,i] > 0}.
   If d = 1, state i is APERIODIC.
   
   INTUITION: The chain doesn't get "stuck" in deterministic cycles.

3. ERGODICITY:
   A chain is ERGODIC if it is both irreducible and aperiodic.
   
   This is THE condition needed for convergence!

FUNDAMENTAL CONVERGENCE THEOREM:
--------------------------------
Let P be the transition matrix of an ERGODIC Markov chain. Then:

1. There exists a UNIQUE stationary distribution π
2. For ALL initial states i:
       lim_{n→∞} P^n[i,j] = π[j]  for all j
   
3. The convergence is EXPONENTIAL:
       |P^n[i,j] - π[j]| ≤ C · ρ^n
   where ρ < 1 is the second-largest eigenvalue

MIXING TIME:
------------
The MIXING TIME is the time until the chain is "close" to its stationary 
distribution.

DEFINITION (ε-mixing time):
τ_mix(ε) = min{n : ||P^n(i,·) - π||_{TV} ≤ ε for all i}

where ||·||_{TV} is the total variation distance:
||μ - ν||_{TV} = (1/2) Σ_j |μ[j] - ν[j]|

TYPICAL CHOICE: ε = 1/4 (then τ_mix ≡ τ_mix(1/4))

WHY IT MATTERS FOR MCMC:
- Need to run chain for ≈ τ_mix steps to get approximate samples from π
- Longer mixing time → slower MCMC sampling
- Understanding mixing is crucial for practical MCMC!

SPECTRAL GAP:
-------------
The SPECTRAL GAP is:
   γ = 1 - |λ_2|
where λ_2 is the second-largest eigenvalue of P.

THEOREM: τ_mix ≈ O(1/γ) (inversely proportional to spectral gap)

Larger gap → faster mixing!

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import pandas as pd
from typing import List, Tuple, Dict
import warnings
import os
warnings.filterwarnings('ignore')


# Configure output directory for local execution
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')


################################################################################
# SECTION 1: CHECKING ERGODICITY CONDITIONS
################################################################################

def check_irreducibility(P: np.ndarray, verbose: bool = True) -> bool:
    """
    Check if a Markov chain is irreducible.
    
    ALGORITHM:
    1. Compute P^n for increasing n
    2. Check if all entries eventually become positive
    3. If yes within reasonable time → irreducible
    
    PRACTICAL CHECK:
    Compute P + P^2 + ... + P^n for n = n_states
    If all entries > 0, chain is irreducible.
    
    WHY: If there exists any path from i to j, it will appear within n steps.
    
    Parameters:
    -----------
    P : np.ndarray
        Transition matrix
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    is_irreducible : bool
    """
    
    n_states = P.shape[0]
    
    # Compute sum of powers: Σ_{k=1}^n P^k
    P_sum = np.zeros_like(P)
    P_k = P.copy()
    
    for k in range(1, n_states + 1):
        P_sum += P_k
        P_k = P_k @ P
    
    # Check if all entries are positive
    is_irreducible = np.all(P_sum > 0)
    
    if verbose:
        print("\nChecking Irreducibility:")
        print("-" * 60)
        print(f"States can reach each other: {is_irreducible}")
        
        if not is_irreducible:
            # Find unreachable pairs
            unreachable = np.argwhere(P_sum == 0)
            print(f"\nUnreachable state pairs:")
            for i, j in unreachable[:5]:  # Show first 5
                print(f"  State {i} → State {j}")
            if len(unreachable) > 5:
                print(f"  ... and {len(unreachable) - 5} more")
    
    return is_irreducible


def check_aperiodicity(P: np.ndarray, verbose: bool = True) -> bool:
    """
    Check if a Markov chain is aperiodic.
    
    PRACTICAL CHECK:
    A sufficient condition for aperiodicity: P[i,i] > 0 for at least one i.
    (If any state has self-loop, chain is aperiodic)
    
    RIGOROUS CHECK:
    Compute gcd of {n: P^n[i,i] > 0} for each state.
    If all gcds = 1, chain is aperiodic.
    
    For simplicity, we use the sufficient condition.
    
    Parameters:
    -----------
    P : np.ndarray
        Transition matrix
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    is_aperiodic : bool
    """
    
    # Check diagonal entries
    has_self_loop = np.any(np.diag(P) > 0)
    
    if verbose:
        print("\nChecking Aperiodicity:")
        print("-" * 60)
        
        if has_self_loop:
            self_loop_states = np.where(np.diag(P) > 0)[0]
            print(f"Has self-loops at states: {self_loop_states}")
            print(f"✓ Chain is APERIODIC (sufficient condition)")
        else:
            print("No self-loops detected")
            print("Need more sophisticated check for periodicity...")
            # Could implement gcd check here for completeness
            print("(Assuming aperiodic for now - more rigorous check omitted)")
    
    return has_self_loop  # Simplified check


def check_ergodicity(P: np.ndarray, verbose: bool = True) -> Dict[str, bool]:
    """
    Check if a Markov chain is ergodic (irreducible + aperiodic).
    
    ERGODICITY is THE key condition for convergence theorems!
    
    Parameters:
    -----------
    P : np.ndarray
        Transition matrix
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    results : dict
        Dictionary with 'irreducible', 'aperiodic', 'ergodic' keys
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("ERGODICITY CHECK")
        print("=" * 80)
    
    is_irr = check_irreducibility(P, verbose=verbose)
    is_aper = check_aperiodicity(P, verbose=verbose)
    
    is_erg = is_irr and is_aper
    
    if verbose:
        print("\n" + "-" * 60)
        print("SUMMARY:")
        print(f"  Irreducible: {is_irr}")
        print(f"  Aperiodic:   {is_aper}")
        print(f"  ✓ ERGODIC:   {is_erg}")
        print("-" * 60)
    
    return {
        'irreducible': is_irr,
        'aperiodic': is_aper,
        'ergodic': is_erg
    }


################################################################################
# SECTION 2: MIXING TIME ANALYSIS
################################################################################

def compute_total_variation_distance(pi1: np.ndarray, pi2: np.ndarray) -> float:
    """
    Compute total variation distance between two distributions.
    
    DEFINITION:
    ||μ - ν||_{TV} = (1/2) Σ_i |μ[i] - ν[i]|
    
    INTERPRETATION:
    - Maximum difference in probability of any event
    - Ranges from 0 (identical) to 1 (disjoint support)
    - Standard metric for comparing probability distributions
    
    Parameters:
    -----------
    pi1, pi2 : np.ndarray
        Probability distributions
        
    Returns:
    --------
    tv_distance : float
        Total variation distance
    """
    return 0.5 * np.sum(np.abs(pi1 - pi2))


def compute_mixing_time(P: np.ndarray,
                       pi_stationary: np.ndarray,
                       epsilon: float = 0.25,
                       max_steps: int = 10000) -> Tuple[int, np.ndarray]:
    """
    Compute ε-mixing time for a Markov chain.
    
    DEFINITION:
    τ_mix(ε) = min{n : ||P^n(i,·) - π||_{TV} ≤ ε for ALL starting states i}
    
    ALGORITHM:
    1. For n = 1, 2, 3, ...
    2. Compute P^n
    3. For each starting state i, compute TV distance to π
    4. If max TV distance ≤ ε for all i, return n
    
    Parameters:
    -----------
    P : np.ndarray
        Transition matrix
    pi_stationary : np.ndarray
        Stationary distribution
    epsilon : float
        Tolerance threshold (standard: 0.25 or 0.01)
    max_steps : int
        Maximum steps to check
        
    Returns:
    --------
    mixing_time : int
        Number of steps to reach ε-mixing
    tv_distances : np.ndarray
        Maximum TV distance at each step
    """
    
    n_states = P.shape[0]
    tv_distances = []
    
    P_n = P.copy()
    
    for n in range(1, max_steps + 1):
        # Compute maximum TV distance over all starting states
        max_tv = 0.0
        for i in range(n_states):
            # Distribution after n steps starting from state i
            pi_n = P_n[i, :]
            tv = compute_total_variation_distance(pi_n, pi_stationary)
            max_tv = max(max_tv, tv)
        
        tv_distances.append(max_tv)
        
        # Check if mixed
        if max_tv <= epsilon:
            return n, np.array(tv_distances)
        
        # Compute next power
        P_n = P_n @ P
    
    warnings.warn(f"Did not reach ε={epsilon} mixing in {max_steps} steps")
    return max_steps, np.array(tv_distances)


def compute_spectral_gap(P: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute spectral gap of transition matrix.
    
    DEFINITION:
    Spectral gap = 1 - |λ_2|
    where λ_2 is the second-largest eigenvalue (in magnitude)
    
    INTERPRETATION:
    - Larger gap → faster convergence
    - Gap = 0 → no convergence (e.g., periodic chains)
    - Theoretical mixing time: τ_mix ≈ O(1/gap)
    
    Parameters:
    -----------
    P : np.ndarray
        Transition matrix
        
    Returns:
    --------
    gap : float
        Spectral gap
    eigenvalues : np.ndarray
        All eigenvalues (sorted by magnitude)
    """
    
    # Compute eigenvalues
    eigenvalues = linalg.eigvals(P)
    
    # Sort by magnitude (absolute value)
    eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]
    
    # Largest should be 1 (up to numerical error)
    lambda_1 = eigenvalues[0].real
    assert np.abs(lambda_1 - 1.0) < 1e-6, "Largest eigenvalue should be 1"
    
    # Second-largest (in magnitude)
    lambda_2 = np.abs(eigenvalues[1])
    
    # Spectral gap
    gap = 1.0 - lambda_2
    
    return gap, eigenvalues


def analyze_mixing(P: np.ndarray,
                  pi_stationary: np.ndarray,
                  states: List[str]):
    """
    Comprehensive mixing analysis.
    
    Computes:
    1. Spectral gap
    2. Mixing times for different ε
    3. Convergence rates from different starting states
    """
    
    print("\n" + "=" * 80)
    print("MIXING TIME ANALYSIS")
    print("=" * 80)
    
    # Spectral gap
    gap, eigenvalues = compute_spectral_gap(P)
    
    print(f"\nSpectral Gap: {gap:.6f}")
    print(f"\nTop 5 eigenvalues (by magnitude):")
    for i, lam in enumerate(eigenvalues[:5]):
        print(f"  λ_{i+1} = {lam:.6f}")
    
    # Mixing times for different tolerances
    print("\n" + "-" * 60)
    print("MIXING TIMES:")
    print("-" * 60)
    
    epsilons = [0.5, 0.25, 0.1, 0.01]
    mixing_times = {}
    
    for eps in epsilons:
        t_mix, _ = compute_mixing_time(P, pi_stationary, epsilon=eps, max_steps=1000)
        mixing_times[eps] = t_mix
        print(f"τ_mix({eps:.2f}) = {t_mix:4d} steps")
    
    # Theoretical prediction from spectral gap
    if gap > 0:
        t_mix_theory = np.log(1/0.25) / gap  # Rough estimate
        print(f"\nTheoretical estimate (ε=0.25): ≈ {t_mix_theory:.1f} steps")
        print(f"Actual: {mixing_times[0.25]} steps")
    
    return mixing_times, gap, eigenvalues


################################################################################
# SECTION 3: CONVERGENCE VISUALIZATION
################################################################################

def visualize_convergence_rate(P: np.ndarray,
                               pi_stationary: np.ndarray,
                               states: List[str],
                               max_steps: int = 100):
    """
    Visualize convergence to stationary distribution.
    
    Shows:
    1. TV distance over time (log scale)
    2. Component-wise convergence
    3. Comparison with exponential decay rate
    """
    
    n_states = len(states)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Compute TV distances and component errors
    tv_distances_by_start = np.zeros((n_states, max_steps))
    
    for start_idx in range(n_states):
        pi_t = np.zeros(n_states)
        pi_t[start_idx] = 1.0
        
        for t in range(max_steps):
            pi_t = pi_t @ P
            tv = compute_total_variation_distance(pi_t, pi_stationary)
            tv_distances_by_start[start_idx, t] = tv
    
    # Plot 1: TV distance (log scale)
    ax1 = axes[0, 0]
    for start_idx, state_name in enumerate(states):
        ax1.semilogy(range(max_steps), tv_distances_by_start[start_idx, :],
                    marker='o', markersize=3, label=f'Start: {state_name}',
                    linewidth=2, alpha=0.7)
    
    # Add theoretical exponential decay
    gap, _ = compute_spectral_gap(P)
    if gap > 0:
        theoretical_decay = np.exp(-gap * np.arange(max_steps))
        ax1.semilogy(range(max_steps), theoretical_decay,
                    'k--', linewidth=2, label='Theoretical exp(-γt)', alpha=0.5)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Total Variation Distance', fontsize=12)
    ax1.set_title('Convergence Rate (Log Scale)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Maximum TV distance
    ax2 = axes[0, 1]
    max_tv = np.max(tv_distances_by_start, axis=0)
    ax2.semilogy(range(max_steps), max_tv, 'b-', linewidth=3, label='Max over starts')
    
    # Mark mixing times
    for eps in [0.5, 0.25, 0.1]:
        idx = np.where(max_tv <= eps)[0]
        if len(idx) > 0:
            t_mix = idx[0]
            ax2.axvline(x=t_mix, color='r', linestyle='--', alpha=0.5)
            ax2.text(t_mix, 0.5, f'ε={eps}', rotation=90, va='bottom')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Max TV Distance', fontsize=12)
    ax2.set_title('Worst-Case Convergence', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Component-wise convergence (from one start state)
    ax3 = axes[1, 0]
    start_idx = 0
    pi_t = np.zeros(n_states)
    pi_t[start_idx] = 1.0
    
    component_evolution = np.zeros((max_steps, n_states))
    for t in range(max_steps):
        pi_t = pi_t @ P
        component_evolution[t, :] = pi_t
    
    for j in range(n_states):
        ax3.plot(range(max_steps), component_evolution[:, j],
                marker='o', markersize=3, label=f'π[{states[j]}]',
                linewidth=2, alpha=0.7)
        ax3.axhline(y=pi_stationary[j], color='gray', linestyle='--', alpha=0.3)
    
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Component Convergence (start: {states[start_idx]})',
                 fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Eigenvalue spectrum
    ax4 = axes[1, 1]
    gap, eigenvalues = compute_spectral_gap(P)
    
    # Plot on complex plane
    for i, lam in enumerate(eigenvalues):
        if i == 0:
            ax4.plot(lam.real, lam.imag, 'ro', markersize=15, 
                    label='λ₁ = 1', zorder=3)
        else:
            ax4.plot(lam.real, lam.imag, 'bo', markersize=10, alpha=0.7)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    
    # Highlight second eigenvalue
    lam_2 = eigenvalues[1]
    ax4.plot(lam_2.real, lam_2.imag, 'go', markersize=15,
            label=f'λ₂ = {lam_2:.3f}', zorder=3)
    
    ax4.set_xlabel('Real', fontsize=12)
    ax4.set_ylabel('Imaginary', fontsize=12)
    ax4.set_title(f'Eigenvalue Spectrum (gap = {gap:.4f})',
                 fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    return fig


################################################################################
# SECTION 4: EXAMPLES
################################################################################

def example_fast_mixing_chain():
    """
    Example: Chain with fast mixing (large spectral gap).
    """
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: FAST MIXING CHAIN")
    print("=" * 80)
    print("\nChain with strong connectivity and self-loops")
    
    # Strongly connected with large self-loop probabilities
    P = np.array([[0.8, 0.1, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.1, 0.1, 0.8]])
    
    states = ['A', 'B', 'C']
    
    # Check ergodicity
    check_ergodicity(P)
    
    # Find stationary distribution
    from scipy.linalg import eig
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = eigenvectors[:, idx].real
    pi = pi / pi.sum()
    
    print(f"\nStationary distribution: {dict(zip(states, np.round(pi, 4)))}")
    
    # Mixing analysis
    mixing_times, gap, _ = analyze_mixing(P, pi, states)
    
    return P, pi, states


def example_slow_mixing_chain():
    """
    Example: Chain with slow mixing (small spectral gap).
    """
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: SLOW MIXING CHAIN")
    print("=" * 80)
    print("\nChain with weak connectivity (bottleneck structure)")
    
    # Two clusters weakly connected
    P = np.array([[0.45, 0.45, 0.1],
                  [0.45, 0.45, 0.1],
                  [0.1,  0.1,  0.8]])
    
    states = ['A₁', 'A₂', 'B']
    
    # Check ergodicity
    check_ergodicity(P)
    
    # Find stationary distribution
    from scipy.linalg import eig
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = eigenvectors[:, idx].real
    pi = pi / pi.sum()
    
    print(f"\nStationary distribution: {dict(zip(states, np.round(pi, 4)))}")
    
    # Mixing analysis
    mixing_times, gap, _ = analyze_mixing(P, pi, states)
    
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("Bottleneck between {A₁,A₂} and {B} slows mixing!")
    print("-" * 60)
    
    return P, pi, states


################################################################################
# MAIN DEMONSTRATION
################################################################################

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║             CONVERGENCE THEOREMS AND MIXING TIMES                      ║
    ║                    Educational Module 04                               ║
    ║                Difficulty: INTERMEDIATE-ADVANCED                       ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Example 1: Fast mixing
    P_fast, pi_fast, states_fast = example_fast_mixing_chain()
    
    fig1 = visualize_convergence_rate(P_fast, pi_fast, states_fast, max_steps=50)
    plt.savefig(f'{OUTPUT_DIR}/04_fast_mixing_convergence.png',
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Example 2: Slow mixing
    P_slow, pi_slow, states_slow = example_slow_mixing_chain()
    
    fig2 = visualize_convergence_rate(P_slow, pi_slow, states_slow, max_steps=200)
    plt.savefig(f'{OUTPUT_DIR}/04_slow_mixing_convergence.png',
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. ERGODICITY = Irreducibility + Aperiodicity
       → Required for convergence to unique π
       → Check by examining graph structure
    
    2. MIXING TIME τ_mix(ε):
       → Time to get within ε of stationary distribution
       → Depends on spectral gap: τ_mix ≈ O(1/gap)
       → Critical for MCMC efficiency!
    
    3. SPECTRAL GAP γ = 1 - |λ_2|:
       → Larger gap → faster mixing
       → Related to "connectivity" of state space
       → Bottlenecks slow down mixing
    
    4. CONVERGENCE RATE:
       → Exponential: ||P^n(i,·) - π||_{TV} ≤ C·ρⁿ
       → Rate ρ = |λ_2| < 1
       → Can visualize and measure empirically
    
    5. WHY THIS MATTERS FOR MCMC:
       → Need to run chain for ≈ τ_mix steps before sampling
       → Slow mixing → impractical MCMC
       → Next module: use Markov chains to sample from ANY distribution!
    
    ✓ Files saved to {os.path.abspath(OUTPUT_DIR)}/
    
    NEXT STEP (File 05): Sampling applications - THE BRIDGE TO MCMC!
    
    CRITICAL QUESTION FOR NEXT MODULE:
    "Given a target distribution π(x) we want to sample from,
     can we DESIGN a Markov chain with stationary distribution π?
     If yes, how fast does it mix?"
    
    This is the CORE QUESTION of MCMC methods!
    """)
    
    print("\n" + "=" * 80)
    print("END OF MODULE 04")
    print("=" * 80)

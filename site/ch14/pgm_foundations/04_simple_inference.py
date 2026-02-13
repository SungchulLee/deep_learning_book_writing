"""
Simple Inference in Bayesian Networks
======================================

This module covers basic inference techniques for Bayesian Networks,
focusing on exact inference by enumeration.

Learning Objectives:
-------------------
1. Understand what inference means in probabilistic models
2. Learn inference by enumeration (brute force approach)
3. Perform marginal queries: P(X)
4. Perform conditional queries: P(X|E)
5. Understand rejection sampling and its limitations

Mathematical Foundations:
------------------------
Marginal inference: P(X) = Σ_{Y} P(X, Y)
Conditional inference: P(X|E) = P(X, E) / P(E)
                              = Σ_{Y} P(X, E, Y) / Σ_{X,Y} P(X, E, Y)

Author: Educational ML Team
Level: Beginner
Prerequisites: 01_pgm_fundamentals.py, 02_bayesian_networks_basics.py
"""

import numpy as np
from typing import Dict, List, Set, Optional
from itertools import product
import time
from collections import defaultdict

# Import from previous modules
import sys
sys.path.append('..')
from beginner.bayesian_networks_basics import BayesianNetwork, build_weather_network


class InferenceByEnumeration:
    """
    Performs exact inference by enumerating all possible configurations.
    
    This is the most straightforward inference method:
    1. Enumerate all possible assignments to all variables
    2. Sum probabilities of assignments consistent with evidence
    3. Normalize to get conditional probabilities
    
    Time complexity: O(2^n) for n binary variables (exponential!)
    Space complexity: O(2^n)
    
    While inefficient, this method is:
    - Exact (not approximate)
    - Easy to understand
    - Useful for small networks
    - Good baseline for testing other algorithms
    """
    
    def __init__(self, bn: BayesianNetwork):
        """
        Initialize the inference engine.
        
        Args:
            bn: Bayesian Network to perform inference on
        """
        self.bn = bn
        self.query_cache = {}  # Cache for repeated queries
    
    def query_marginal(self, 
                      query_vars: List[str],
                      evidence: Optional[Dict[str, int]] = None,
                      use_cache: bool = True) -> Dict[tuple, float]:
        """
        Compute P(query_vars | evidence) by enumeration.
        
        Args:
            query_vars: Variables to compute probability distribution over
            evidence: Dictionary of observed variable values (can be None)
            use_cache: Whether to use cached results
        
        Returns:
            Dictionary mapping each assignment of query_vars to its probability
        
        Example:
            # Compute P(Rain | Cloudy=1)
            result = inference.query_marginal(
                query_vars=['Rain'],
                evidence={'Cloudy': 1}
            )
            # Returns: {(0,): 0.2, (1,): 0.8}
        """
        if evidence is None:
            evidence = {}
        
        # Create cache key
        query_key = (tuple(sorted(query_vars)), tuple(sorted(evidence.items())))
        if use_cache and query_key in self.query_cache:
            return self.query_cache[query_key]
        
        print(f"\nComputing P({', '.join(query_vars)}" + 
              (f" | {', '.join(f'{k}={v}' for k, v in evidence.items())})" if evidence else ")"))
        print("-" * 60)
        
        # Get all variables in the network
        all_vars = list(self.bn.graph.nodes())
        
        # Separate variables into: query, evidence, and hidden
        hidden_vars = [v for v in all_vars if v not in query_vars and v not in evidence]
        
        print(f"Query variables: {query_vars}")
        print(f"Evidence variables: {list(evidence.keys())}")
        print(f"Hidden variables: {hidden_vars}")
        
        # Initialize result dictionary
        result = defaultdict(float)
        
        # Get cardinalities
        query_cards = [self.bn.cardinalities[v] for v in query_vars]
        hidden_cards = [self.bn.cardinalities[v] for v in hidden_vars]
        
        total_configurations = np.prod(query_cards) * np.prod(hidden_cards)
        print(f"\nEnumerating {int(total_configurations)} configurations...")
        
        # Enumerate all assignments to query and hidden variables
        for query_values in product(*[range(c) for c in query_cards]):
            # For each query assignment, sum over all hidden variable assignments
            query_assignment = dict(zip(query_vars, query_values))
            
            for hidden_values in product(*[range(c) for c in hidden_cards]):
                hidden_assignment = dict(zip(hidden_vars, hidden_values))
                
                # Complete assignment
                full_assignment = {**query_assignment, **hidden_assignment, **evidence}
                
                # Compute P(full_assignment) using the Bayesian network
                prob = self.bn.compute_joint_probability(full_assignment)
                
                # Add to the result for this query assignment
                result[query_values] += prob
        
        # Normalize to get conditional probability P(query | evidence)
        total = sum(result.values())
        
        if total > 0:
            result = {k: v/total for k, v in result.items()}
        else:
            print("Warning: Evidence has probability 0!")
            # Uniform distribution
            result = {k: 1.0/len(result) for k in result.keys()}
        
        # Cache the result
        if use_cache:
            self.query_cache[query_key] = dict(result)
        
        return dict(result)
    
    def query_single_variable(self, 
                             variable: str,
                             evidence: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Convenience method to query a single variable.
        
        Args:
            variable: Variable to query
            evidence: Evidence dictionary
        
        Returns:
            NumPy array of probabilities for each value of the variable
        
        Example:
            # Compute P(Rain | Cloudy=1)
            probs = inference.query_single_variable('Rain', {'Cloudy': 1})
            print(f"P(Rain=0|Cloudy=1) = {probs[0]}")
            print(f"P(Rain=1|Cloudy=1) = {probs[1]}")
        """
        result_dict = self.query_marginal([variable], evidence)
        
        # Convert to array
        cardinality = self.bn.cardinalities[variable]
        probs = np.zeros(cardinality)
        
        for value_tuple, prob in result_dict.items():
            probs[value_tuple[0]] = prob
        
        return probs
    
    def most_probable_explanation(self, 
                                  evidence: Dict[str, int]) -> Dict[str, int]:
        """
        Find the most probable complete assignment given evidence.
        
        This is also called MAP (Maximum A Posteriori) inference.
        We find: argmax_X P(X | evidence)
        
        Args:
            evidence: Observed variables
        
        Returns:
            Most probable assignment to all non-evidence variables
        
        Example:
            # Given that grass is wet, what's the most likely explanation?
            mpe = inference.most_probable_explanation({'WetGrass': 1})
            # Might return: {'Cloudy': 1, 'Rain': 1, 'Sprinkler': 0}
        """
        print(f"\nFinding Most Probable Explanation given evidence:")
        print(f"Evidence: {evidence}")
        print("-" * 60)
        
        # Get all non-evidence variables
        all_vars = list(self.bn.graph.nodes())
        query_vars = [v for v in all_vars if v not in evidence]
        
        # Enumerate all assignments
        best_assignment = None
        best_prob = -1
        
        query_cards = [self.bn.cardinalities[v] for v in query_vars]
        
        for query_values in product(*[range(c) for c in query_cards]):
            assignment = dict(zip(query_vars, query_values))
            full_assignment = {**assignment, **evidence}
            
            prob = self.bn.compute_joint_probability(full_assignment)
            
            if prob > best_prob:
                best_prob = prob
                best_assignment = assignment
        
        print(f"\nMost probable assignment: {best_assignment}")
        print(f"Probability: {best_prob:.6f}")
        
        return best_assignment


class RejectionSampling:
    """
    Approximate inference using rejection sampling.
    
    Rejection sampling:
    1. Generate samples from the prior P(X)
    2. Reject samples that don't match the evidence
    3. Compute statistics from accepted samples
    
    Pros:
    - Simple to implement
    - Unbiased estimator
    
    Cons:
    - Can be very inefficient if evidence is unlikely
    - Rejection rate can be very high
    """
    
    def __init__(self, bn: BayesianNetwork):
        """Initialize rejection sampling."""
        self.bn = bn
    
    def query(self,
             query_vars: List[str],
             evidence: Dict[str, int],
             num_samples: int = 10000) -> Dict[tuple, float]:
        """
        Perform inference using rejection sampling.
        
        Args:
            query_vars: Variables to query
            evidence: Evidence dictionary
            num_samples: Number of samples to generate
        
        Returns:
            Approximate probability distribution over query variables
        """
        print(f"\nRejection Sampling: {num_samples} samples")
        print("-" * 60)
        
        accepted_samples = []
        rejected = 0
        
        # Generate samples
        for _ in range(num_samples):
            sample = self.bn.forward_sample()
            
            # Check if sample matches evidence
            matches = all(sample[var] == val for var, val in evidence.items())
            
            if matches:
                accepted_samples.append(sample)
            else:
                rejected += 1
        
        print(f"Accepted: {len(accepted_samples)}")
        print(f"Rejected: {rejected}")
        print(f"Acceptance rate: {len(accepted_samples)/num_samples:.2%}")
        
        if len(accepted_samples) == 0:
            print("Warning: No samples accepted! Try more samples or check evidence.")
            return {}
        
        # Compute probabilities from accepted samples
        result = defaultdict(int)
        
        for sample in accepted_samples:
            key = tuple(sample[var] for var in query_vars)
            result[key] += 1
        
        # Normalize
        total = sum(result.values())
        result = {k: v/total for k, v in result.items()}
        
        return dict(result)


def demonstrate_simple_queries():
    """
    Demonstrate simple inference queries on the weather network.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Simple Inference Queries")
    print("="*70)
    
    # Build network
    bn = build_weather_network()
    inference = InferenceByEnumeration(bn)
    
    # Query 1: Marginal probability P(Rain)
    print("\n" + "="*70)
    print("Query 1: What's the probability of rain?")
    print("="*70)
    
    rain_probs = inference.query_single_variable('Rain')
    print(f"\nP(Rain=0) = {rain_probs[0]:.4f}")
    print(f"P(Rain=1) = {rain_probs[1]:.4f}")
    
    # Query 2: Conditional probability P(Rain | Cloudy=1)
    print("\n" + "="*70)
    print("Query 2: What's the probability of rain given it's cloudy?")
    print("="*70)
    
    rain_given_cloudy = inference.query_single_variable('Rain', {'Cloudy': 1})
    print(f"\nP(Rain=0 | Cloudy=1) = {rain_given_cloudy[0]:.4f}")
    print(f"P(Rain=1 | Cloudy=1) = {rain_given_cloudy[1]:.4f}")
    
    print("\nNote: Rain is more likely when it's cloudy (0.8 vs 0.2)")
    
    # Query 3: Multiple variables P(Rain, Sprinkler | WetGrass=1)
    print("\n" + "="*70)
    print("Query 3: Given wet grass, what caused it?")
    print("="*70)
    
    result = inference.query_marginal(['Rain', 'Sprinkler'], {'WetGrass': 1})
    
    print("\nP(Rain, Sprinkler | WetGrass=1):")
    for (rain, sprinkler), prob in sorted(result.items()):
        print(f"  Rain={rain}, Sprinkler={sprinkler}: {prob:.4f}")
    
    print("\nInterpretation:")
    print("- Most likely: Rain=1, Sprinkler=0 (rain caused it)")
    print("- Second: Rain=0, Sprinkler=1 (sprinkler caused it)")
    print("- Least likely: Rain=0, Sprinkler=0 (shouldn't happen but numerical errors)")


def demonstrate_mpe():
    """
    Demonstrate Most Probable Explanation (MPE) inference.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Most Probable Explanation (MPE)")
    print("="*70)
    
    bn = build_weather_network()
    inference = InferenceByEnumeration(bn)
    
    # MPE given wet grass
    print("\n" + "="*70)
    print("Scenario: You observe that the grass is wet")
    print("Question: What's the most likely complete explanation?")
    print("="*70)
    
    mpe = inference.most_probable_explanation({'WetGrass': 1})
    
    print("\nInterpretation:")
    if mpe.get('Rain') == 1:
        print("- It probably rained")
    if mpe.get('Cloudy') == 1:
        print("- It was probably cloudy")
    if mpe.get('Sprinkler') == 1:
        print("- The sprinkler was probably on")
    else:
        print("- The sprinkler was probably off")


def compare_exact_vs_approximate():
    """
    Compare exact inference with rejection sampling.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Exact vs Approximate Inference")
    print("="*70)
    
    bn = build_weather_network()
    
    # Exact inference
    print("\n" + "-"*70)
    print("EXACT INFERENCE (Enumeration)")
    print("-"*70)
    
    exact_inference = InferenceByEnumeration(bn)
    
    start = time.time()
    exact_result = exact_inference.query_single_variable('Rain', {'WetGrass': 1})
    exact_time = time.time() - start
    
    print(f"\nP(Rain | WetGrass=1):")
    print(f"  Rain=0: {exact_result[0]:.4f}")
    print(f"  Rain=1: {exact_result[1]:.4f}")
    print(f"Time: {exact_time:.6f} seconds")
    
    # Approximate inference
    print("\n" + "-"*70)
    print("APPROXIMATE INFERENCE (Rejection Sampling)")
    print("-"*70)
    
    rejection_sampling = RejectionSampling(bn)
    
    start = time.time()
    approx_result = rejection_sampling.query(['Rain'], {'WetGrass': 1}, num_samples=10000)
    approx_time = time.time() - start
    
    print(f"\nP(Rain | WetGrass=1):")
    print(f"  Rain=0: {approx_result.get((0,), 0):.4f}")
    print(f"  Rain=1: {approx_result.get((1,), 0):.4f}")
    print(f"Time: {approx_time:.6f} seconds")
    
    # Compare
    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    print(f"\nError in P(Rain=1 | WetGrass=1):")
    error = abs(exact_result[1] - approx_result.get((1,), 0))
    print(f"  {error:.4f} ({error/exact_result[1]*100:.2f}% relative error)")
    
    print(f"\nNote: Rejection sampling is approximate but can be much faster")
    print(f"for large networks. However, it can be very inefficient if")
    print(f"evidence is unlikely (low acceptance rate).")


def main():
    """
    Main demonstration function.
    """
    print("\n" + "="*70)
    print("SIMPLE INFERENCE IN BAYESIAN NETWORKS")
    print("="*70)
    
    print("\nTopics covered:")
    print("1. Inference by enumeration (exact)")
    print("2. Marginal queries: P(X) and P(X|E)")
    print("3. Most Probable Explanation (MPE)")
    print("4. Rejection sampling (approximate)")
    
    # Run demonstrations
    demonstrate_simple_queries()
    demonstrate_mpe()
    compare_exact_vs_approximate()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Inference = computing P(Query | Evidence)")
    print("\n2. Enumeration is exact but exponentially expensive:")
    print("   - Time: O(2^n) for n binary variables")
    print("   - Practical only for small networks")
    
    print("\n3. Approximate methods (like rejection sampling):")
    print("   - Trade accuracy for speed")
    print("   - Useful for large networks")
    print("   - Accuracy improves with more samples")
    
    print("\n4. More efficient exact methods exist:")
    print("   - Variable elimination (next module!)")
    print("   - Junction tree algorithm")
    print("   - These exploit network structure")
    
    print("\n" + "="*70)
    print("Next: Learn about efficient inference algorithms!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

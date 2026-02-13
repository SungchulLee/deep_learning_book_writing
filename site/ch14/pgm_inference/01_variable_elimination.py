"""
Variable Elimination Algorithm
===============================

This module implements the Variable Elimination algorithm, an efficient
exact inference method that exploits the structure of Bayesian Networks.

Learning Objectives:
-------------------
1. Understand why enumeration is inefficient
2. Learn the Variable Elimination algorithm
3. Understand factor operations (product, marginalization)
4. Learn about elimination orderings and their impact
5. Implement efficient exact inference

Mathematical Foundations:
------------------------
Variable Elimination works by:
1. Convert CPTs to factors
2. For each variable to eliminate:
   - Collect all factors containing that variable
   - Multiply them together (factor product)
   - Sum out the variable (marginalization)
3. Multiply remaining factors and normalize

Time complexity: O(n * d^(w+1)) where:
- n = number of variables
- d = maximum cardinality
- w = induced width (depends on elimination order)

Author: Educational ML Team
Level: Intermediate
Prerequisites: 01-04 from beginner level
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import time


class Factor:
    """
    Represents a factor (potential function) over a set of variables.
    
    A factor φ(X1, ..., Xk) is a function that maps each assignment
    of values to (X1, ..., Xk) to a non-negative number.
    
    Factors are the fundamental data structure for variable elimination.
    """
    
    def __init__(self,
                 variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[np.ndarray] = None):
        """
        Initialize a factor.
        
        Args:
            variables: List of variable names in this factor
            cardinalities: Dict of cardinalities for all variables
            values: NumPy array of factor values (optional)
        """
        self.variables = sorted(variables)  # Keep sorted for consistency
        self.cardinalities = cardinalities
        
        shape = tuple(cardinalities[var] for var in self.variables)
        
        if values is None:
            self.values = np.ones(shape)
        else:
            self.values = np.array(values)
            assert self.values.shape == shape, \
                f"Shape mismatch: {self.values.shape} vs {shape}"
    
    def multiply(self, other: 'Factor') -> 'Factor':
        """
        Compute the factor product: φ1 × φ2.
        
        Factor multiplication combines two factors by multiplying
        their values for each consistent assignment.
        
        For example, if φ1(A,B) and φ2(B,C), then:
        (φ1 × φ2)(A,B,C) = φ1(A,B) × φ2(B,C)
        
        Args:
            other: Another factor to multiply with
        
        Returns:
            Product factor over the union of variables
        """
        # Find union of variables
        new_variables = sorted(set(self.variables + other.variables))
        
        # Create result factor
        result = Factor(new_variables, self.cardinalities)
        
        # For each assignment to the new variables
        for assignment in self._enumerate_assignments(new_variables):
            # Get values from both factors
            val1 = self._get_value(assignment)
            val2 = other._get_value(assignment)
            
            # Multiply
            result._set_value(assignment, val1 * val2)
        
        return result
    
    def marginalize(self, variables_to_eliminate: List[str]) -> 'Factor':
        """
        Sum out (marginalize) variables from the factor.
        
        Marginalization computes: φ'(X) = Σ_Y φ(X, Y)
        
        Args:
            variables_to_eliminate: Variables to sum out
        
        Returns:
            New factor with variables eliminated
        """
        # Variables remaining after elimination
        remaining_vars = [v for v in self.variables 
                         if v not in variables_to_eliminate]
        
        if not remaining_vars:
            # Sum over all variables = single number
            return Factor([], self.cardinalities, 
                         np.array([np.sum(self.values)]))
        
        # Create result factor
        result = Factor(remaining_vars, self.cardinalities)
        
        # For each assignment to remaining variables
        for remaining_assignment in result._enumerate_assignments(remaining_vars):
            # Sum over all assignments to eliminated variables
            total = 0.0
            
            elim_cards = [self.cardinalities[v] for v in variables_to_eliminate]
            for elim_values in product(*[range(c) for c in elim_cards]):
                elim_assignment = dict(zip(variables_to_eliminate, elim_values))
                full_assignment = {**remaining_assignment, **elim_assignment}
                
                total += self._get_value(full_assignment)
            
            result._set_value(remaining_assignment, total)
        
        return result
    
    def reduce(self, evidence: Dict[str, int]) -> 'Factor':
        """
        Reduce a factor by fixing evidence variables.
        
        Creates a new factor that only considers assignments
        consistent with the evidence.
        
        Args:
            evidence: Dict of observed variable values
        
        Returns:
            Reduced factor
        """
        # Variables not in evidence
        remaining_vars = [v for v in self.variables if v not in evidence]
        
        if not remaining_vars:
            # All variables observed
            return Factor([], self.cardinalities, 
                         np.array([self._get_value(evidence)]))
        
        # Create result factor
        result = Factor(remaining_vars, self.cardinalities)
        
        # For each assignment to remaining variables
        for assignment in result._enumerate_assignments(remaining_vars):
            full_assignment = {**assignment, **evidence}
            value = self._get_value(full_assignment)
            result._set_value(assignment, value)
        
        return result
    
    def normalize(self) -> 'Factor':
        """Normalize factor so values sum to 1."""
        total = np.sum(self.values)
        if total > 0:
            return Factor(self.variables, self.cardinalities, 
                         self.values / total)
        return self
    
    def _get_value(self, assignment: Dict[str, int]) -> float:
        """Get factor value for an assignment."""
        if not self.variables:
            return self.values[0]
        
        index = tuple(assignment[var] for var in self.variables)
        return self.values[index]
    
    def _set_value(self, assignment: Dict[str, int], value: float):
        """Set factor value for an assignment."""
        if not self.variables:
            self.values[0] = value
        else:
            index = tuple(assignment[var] for var in self.variables)
            self.values[index] = value
    
    def _enumerate_assignments(self, variables: List[str]):
        """Generate all assignments to variables."""
        if not variables:
            yield {}
            return
        
        cards = [self.cardinalities[var] for var in variables]
        for values in product(*[range(c) for c in cards]):
            yield dict(zip(variables, values))
    
    def __str__(self) -> str:
        """String representation."""
        if not self.variables:
            return f"Factor([]) = {self.values[0]:.4f}"
        return f"Factor({', '.join(self.variables)})\nShape: {self.values.shape}"


class VariableElimination:
    """
    Implements the Variable Elimination algorithm for exact inference.
    
    This is the workhorse algorithm for exact inference in Bayesian Networks.
    It's much more efficient than enumeration because it exploits the
    factored structure of the distribution.
    """
    
    def __init__(self, bn):
        """
        Initialize Variable Elimination.
        
        Args:
            bn: BayesianNetwork instance
        """
        self.bn = bn
        
    def _create_factors_from_cpts(self) -> List[Factor]:
        """
        Convert all CPTs in the network to factors.
        
        Returns:
            List of factors, one for each variable
        """
        factors = []
        
        for variable in self.bn.graph.nodes():
            cpt = self.bn.get_cpt(variable)
            parents = cpt.parents
            
            # Create factor over variable and its parents
            factor_vars = parents + [variable]
            factor = Factor(factor_vars, self.bn.cardinalities)
            
            # Copy CPT values to factor
            for assignment in factor._enumerate_assignments(factor_vars):
                parent_vals = {p: assignment[p] for p in parents}
                var_val = assignment[variable]
                prob = cpt.get_probability(var_val, parent_vals)
                factor._set_value(assignment, prob)
            
            factors.append(factor)
        
        return factors
    
    def _choose_elimination_order(self,
                                  variables_to_eliminate: Set[str],
                                  strategy: str = 'min_neighbors') -> List[str]:
        """
        Choose an elimination order.
        
        The elimination order greatly affects efficiency!
        Different strategies:
        - min_neighbors: Eliminate variable with fewest neighbors first
        - min_fill: Minimize edges added to graph
        - weighted_min_fill: Consider variable cardinalities
        
        Args:
            variables_to_eliminate: Variables to order
            strategy: Ordering strategy
        
        Returns:
            List of variables in elimination order
        """
        if strategy == 'min_neighbors':
            # Simple greedy strategy: eliminate variable with fewest neighbors
            order = []
            graph = self.bn.graph.to_undirected()
            remaining = set(variables_to_eliminate)
            
            while remaining:
                # Find variable with minimum neighbors in remaining set
                min_var = min(remaining, 
                             key=lambda v: len(set(graph.neighbors(v)) & remaining))
                order.append(min_var)
                remaining.remove(min_var)
            
            return order
        else:
            # Default: just use sorted order
            return sorted(variables_to_eliminate)
    
    def query(self,
             query_vars: List[str],
             evidence: Optional[Dict[str, int]] = None,
             elimination_order: Optional[List[str]] = None,
             verbose: bool = True) -> Dict[tuple, float]:
        """
        Perform inference using Variable Elimination.
        
        Args:
            query_vars: Variables to compute distribution over
            evidence: Observed variable values
            elimination_order: Custom elimination order (optional)
            verbose: Whether to print progress
        
        Returns:
            Probability distribution over query variables
        """
        if evidence is None:
            evidence = {}
        
        if verbose:
            print(f"\nVariable Elimination Query:")
            print(f"Query: P({', '.join(query_vars)}" +
                  (f" | {', '.join(f'{k}={v}' for k, v in evidence.items())})" 
                   if evidence else ")"))
            print("-" * 60)
        
        # Step 1: Create initial factors from CPTs
        factors = self._create_factors_from_cpts()
        
        if verbose:
            print(f"\nStep 1: Created {len(factors)} initial factors from CPTs")
        
        # Step 2: Reduce factors with evidence
        if evidence:
            factors = [f.reduce(evidence) for f in factors]
            if verbose:
                print(f"Step 2: Reduced factors with evidence {evidence}")
        
        # Step 3: Determine variables to eliminate
        all_vars = set(self.bn.graph.nodes())
        variables_to_eliminate = all_vars - set(query_vars) - set(evidence.keys())
        
        if verbose:
            print(f"Step 3: Variables to eliminate: {sorted(variables_to_eliminate)}")
        
        # Step 4: Choose elimination order
        if elimination_order is None:
            elimination_order = self._choose_elimination_order(variables_to_eliminate)
        
        if verbose:
            print(f"Step 4: Elimination order: {elimination_order}")
        
        # Step 5: Eliminate variables one by one
        if verbose:
            print(f"\nStep 5: Eliminating variables...")
        
        for i, var in enumerate(elimination_order, 1):
            if verbose:
                print(f"\n  Eliminating {var} ({i}/{len(elimination_order)}):")
            
            # Find factors containing this variable
            relevant_factors = [f for f in factors if var in f.variables]
            other_factors = [f for f in factors if var not in f.variables]
            
            if verbose:
                print(f"    - Found {len(relevant_factors)} factors containing {var}")
            
            if relevant_factors:
                # Multiply relevant factors
                product_factor = relevant_factors[0]
                for f in relevant_factors[1:]:
                    product_factor = product_factor.multiply(f)
                
                if verbose:
                    print(f"    - Multiplied factors: scope = {product_factor.variables}")
                
                # Marginalize out the variable
                marginalized_factor = product_factor.marginalize([var])
                
                if verbose:
                    print(f"    - Marginalized out {var}: scope = {marginalized_factor.variables}")
                
                # Update factor list
                factors = other_factors + [marginalized_factor]
            else:
                factors = other_factors
        
        # Step 6: Multiply remaining factors
        if verbose:
            print(f"\nStep 6: Multiplying {len(factors)} remaining factors...")
        
        result_factor = factors[0]
        for f in factors[1:]:
            result_factor = result_factor.multiply(f)
        
        # Step 7: Normalize
        result_factor = result_factor.normalize()
        
        if verbose:
            print(f"Step 7: Normalized result")
        
        # Convert factor to dictionary
        result = {}
        for assignment in result_factor._enumerate_assignments(query_vars):
            key = tuple(assignment[var] for var in query_vars)
            result[key] = result_factor._get_value(assignment)
        
        return result


def demonstrate_variable_elimination():
    """Demonstrate Variable Elimination on a simple network."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Variable Elimination")
    print("="*70)
    
    # Import and build weather network
    import sys
    sys.path.append('..')
    from beginner.bayesian_networks_basics import build_weather_network
    
    bn = build_weather_network()
    
    # Create inference engines
    ve = VariableElimination(bn)
    
    # Query 1: Simple query
    print("\n" + "="*70)
    print("Query 1: P(Rain | WetGrass=1)")
    print("="*70)
    
    result = ve.query(['Rain'], {'WetGrass': 1}, verbose=True)
    
    print(f"\nResult:")
    print(f"P(Rain=0 | WetGrass=1) = {result[(0,)]:.4f}")
    print(f"P(Rain=1 | WetGrass=1) = {result[(1,)]:.4f}")
    
    # Query 2: Multiple query variables
    print("\n" + "="*70)
    print("Query 2: P(Rain, Sprinkler | WetGrass=1)")
    print("="*70)
    
    result = ve.query(['Rain', 'Sprinkler'], {'WetGrass': 1}, verbose=True)
    
    print(f"\nResult:")
    for (rain, sprinkler), prob in sorted(result.items()):
        print(f"P(Rain={rain}, Sprinkler={sprinkler} | WetGrass=1) = {prob:.4f}")


def compare_enumeration_vs_ve():
    """
    Compare efficiency of enumeration vs variable elimination.
    """
    print("\n" + "="*70)
    print("COMPARISON: Enumeration vs Variable Elimination")
    print("="*70)
    
    import sys
    sys.path.append('..')
    from beginner.bayesian_networks_basics import build_weather_network
    from beginner.simple_inference import InferenceByEnumeration
    
    bn = build_weather_network()
    
    # Enumeration
    print("\nEnumeration:")
    enum = InferenceByEnumeration(bn)
    start = time.time()
    enum_result = enum.query_marginal(['Rain'], {'WetGrass': 1}, use_cache=False)
    enum_time = time.time() - start
    print(f"Time: {enum_time:.6f} seconds")
    
    # Variable Elimination
    print("\nVariable Elimination:")
    ve = VariableElimination(bn)
    start = time.time()
    ve_result = ve.query(['Rain'], {'WetGrass': 1}, verbose=False)
    ve_time = time.time() - start
    print(f"Time: {ve_time:.6f} seconds")
    
    # Compare
    print(f"\nSpeedup: {enum_time/ve_time:.2f}x faster")
    print(f"\nNote: For small networks the difference is small,")
    print(f"but for larger networks VE can be orders of magnitude faster!")


def main():
    """Main demonstration."""
    print("\n" + "="*70)
    print("VARIABLE ELIMINATION ALGORITHM")
    print("="*70)
    
    print("\nVariable Elimination is the standard algorithm for exact")
    print("inference in Bayesian Networks. It exploits the factored")
    print("structure to avoid exponential blowup (when possible).")
    
    demonstrate_variable_elimination()
    compare_enumeration_vs_ve()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. VE is much more efficient than enumeration")
    print("\n2. Key operations:")
    print("   - Factor product (multiply factors)")
    print("   - Marginalization (sum out variables)")
    
    print("\n3. Elimination order matters!")
    print("   - Good order: polynomial time")
    print("   - Bad order: exponential time")
    print("   - Finding optimal order is NP-hard")
    
    print("\n4. Time complexity: O(n * d^(w+1))")
    print("   - w = induced width (depends on order)")
    print("   - For trees: w=1, very efficient!")
    print("   - For dense graphs: can be expensive")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

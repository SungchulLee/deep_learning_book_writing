"""
Structure Learning for Bayesian Networks
========================================

This module covers learning the structure (graph) of Bayesian Networks from data.

Learning Objectives:
-------------------
1. Understand the structure learning problem
2. Learn constraint-based methods (PC algorithm)
3. Learn score-based methods (hill climbing with BIC)
4. Understand the challenges of structure learning
5. Apply structure learning to real data

Mathematical Foundations:
------------------------
Structure learning problem: Given data D, find structure G that best explains D

Two main approaches:
1. Constraint-based: Test conditional independencies
   - PC algorithm: Tests d-separations
   - Based on statistical independence tests

2. Score-based: Optimize a scoring function
   - Score: BIC, AIC, BDe, etc.
   - Search: Hill climbing, greedy, genetic algorithms
   
BIC Score: BIC(G, D) = log P(D | G, θ_MLE) - (d/2) log n
where d = number of parameters, n = number of samples

Author: Educational ML Team
Level: Advanced
Prerequisites: All beginner and intermediate modules
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
import pandas as pd
from scipy.stats import chi2_contingency
from collections import defaultdict


class IndependenceTest:
    """
    Statistical tests for (conditional) independence.
    
    Uses chi-squared test for independence.
    """
    
    @staticmethod
    def test_independence(data: pd.DataFrame,
                         var1: str,
                         var2: str,
                         alpha: float = 0.05) -> bool:
        """
        Test if var1 and var2 are independent.
        
        Uses chi-squared test of independence.
        
        Args:
            data: DataFrame containing the variables
            var1, var2: Variable names
            alpha: Significance level
        
        Returns:
            True if independent (fail to reject H0), False otherwise
        """
        # Create contingency table
        contingency = pd.crosstab(data[var1], data[var2])
        
        # Chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # If p-value > alpha, fail to reject null hypothesis (independence)
        return p_value > alpha
    
    @staticmethod
    def test_conditional_independence(data: pd.DataFrame,
                                     var1: str,
                                     var2: str,
                                     given: List[str],
                                     alpha: float = 0.05) -> bool:
        """
        Test if var1 and var2 are conditionally independent given `given`.
        
        Tests: X ⊥ Y | Z
        
        Args:
            data: DataFrame
            var1, var2: Variables to test
            given: Conditioning variables
            alpha: Significance level
        
        Returns:
            True if conditionally independent, False otherwise
        """
        if not given:
            return IndependenceTest.test_independence(data, var1, var2, alpha)
        
        # For each value of conditioning variables, test independence
        # If independent for all values, then conditionally independent
        
        # Group by conditioning variables
        grouped = data.groupby(given)
        
        p_values = []
        for name, group in grouped:
            if len(group) < 5:  # Skip groups that are too small
                continue
            
            # Test independence in this group
            contingency = pd.crosstab(group[var1], group[var2])
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                p_values.append(p_value)
            except:
                continue
        
        if not p_values:
            return True  # Not enough data, assume independent
        
        # Use minimum p-value (most conservative)
        min_p_value = min(p_values)
        return min_p_value > alpha


class PCAlgorithm:
    """
    Implements the PC (Peter-Clark) algorithm for structure learning.
    
    PC algorithm is a constraint-based method that:
    1. Starts with a complete undirected graph
    2. Removes edges based on conditional independence tests
    3. Orients edges based on v-structures
    4. Propagates orientations
    
    Returns a CPDAG (Completed Partially Directed Acyclic Graph).
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize PC algorithm.
        
        Args:
            alpha: Significance level for independence tests
        """
        self.alpha = alpha
        self.tester = IndependenceTest()
    
    def learn_structure(self, data: pd.DataFrame) -> nx.Graph:
        """
        Learn Bayesian Network structure using PC algorithm.
        
        Args:
            data: DataFrame with samples
        
        Returns:
            Undirected graph (skeleton) of the network
        """
        variables = list(data.columns)
        n = len(variables)
        
        print(f"\nPC Algorithm: Learning structure from {len(data)} samples")
        print(f"Variables: {variables}")
        print("-" * 70)
        
        # Step 1: Start with complete undirected graph
        graph = nx.Graph()
        graph.add_nodes_from(variables)
        for i in range(n):
            for j in range(i+1, n):
                graph.add_edge(variables[i], variables[j])
        
        print(f"\nStep 1: Started with complete graph ({graph.number_of_edges()} edges)")
        
        # Step 2: Remove edges based on conditional independence
        # Test with conditioning sets of increasing size
        separating_sets = {}  # Store separating sets for later
        
        for order in range(n):
            print(f"\nStep 2.{order+1}: Testing conditional independence with |S| = {order}")
            
            edges_to_remove = []
            
            for edge in list(graph.edges()):
                var1, var2 = edge
                
                # Get potential conditioning sets (neighbors of var1 or var2)
                neighbors = set(graph.neighbors(var1)) | set(graph.neighbors(var2))
                neighbors -= {var1, var2}
                
                # Test all conditioning sets of size `order`
                if len(neighbors) >= order:
                    for cond_set in combinations(neighbors, order):
                        cond_list = list(cond_set)
                        
                        # Test conditional independence
                        if self.tester.test_conditional_independence(
                            data, var1, var2, cond_list, self.alpha
                        ):
                            edges_to_remove.append(edge)
                            separating_sets[edge] = cond_list
                            print(f"  {var1} ⊥ {var2} | {{{', '.join(cond_list)}}}")
                            break
            
            # Remove edges
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            
            if edges_to_remove:
                print(f"  Removed {len(edges_to_remove)} edges")
            
            # If no more edges to test, stop
            if graph.number_of_edges() == 0:
                break
        
        print(f"\nStep 3: Final skeleton has {graph.number_of_edges()} edges")
        
        return graph


class ScoreBasedLearning:
    """
    Score-based structure learning using hill climbing with BIC score.
    
    Score-based methods:
    1. Define a scoring function (BIC, AIC, BDe, etc.)
    2. Search for structure that maximizes score
    3. Use heuristic search (hill climbing, genetic algorithms, etc.)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize score-based learning.
        
        Args:
            data: Training data
        """
        self.data = data
        self.variables = list(data.columns)
        self.n_samples = len(data)
        
        # Compute statistics for scoring
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Precompute statistics for efficient scoring."""
        self.counts = {}
        
        for var in self.variables:
            # Count occurrences
            self.counts[var] = self.data[var].value_counts().to_dict()
    
    def compute_bic_score(self, graph: nx.DiGraph) -> float:
        """
        Compute BIC score for a given DAG.
        
        BIC = log P(D | G, θ_MLE) - (d/2) log n
        
        where:
        - P(D | G, θ_MLE) is the likelihood with MLE parameters
        - d is the number of free parameters
        - n is the number of samples
        
        Args:
            graph: Directed acyclic graph
        
        Returns:
            BIC score (higher is better)
        """
        score = 0.0
        num_parameters = 0
        
        for var in graph.nodes():
            parents = list(graph.predecessors(var))
            
            # Compute local BIC score for this variable
            if not parents:
                # No parents: just prior distribution
                local_counts = self.data[var].value_counts()
                n_local = local_counts.sum()
                
                # Log-likelihood
                for count in local_counts:
                    if count > 0:
                        p = count / n_local
                        score += count * np.log(p)
                
                # Penalty term
                cardinality = len(local_counts)
                num_parameters += cardinality - 1
            
            else:
                # Has parents: conditional distribution
                # Group by parent values
                parent_groups = self.data.groupby(parents)[var]
                
                for parent_values, group in parent_groups:
                    counts = group.value_counts()
                    n_local = counts.sum()
                    
                    # Log-likelihood for this parent configuration
                    for count in counts:
                        if count > 0:
                            p = count / n_local
                            score += count * np.log(p)
                
                # Penalty term
                var_card = self.data[var].nunique()
                parent_configs = len(parent_groups)
                num_parameters += parent_configs * (var_card - 1)
        
        # BIC penalty
        penalty = (num_parameters / 2) * np.log(self.n_samples)
        bic = score - penalty
        
        return bic
    
    def hill_climbing(self, max_parents: int = 3, max_iterations: int = 100) -> nx.DiGraph:
        """
        Learn structure using hill climbing search.
        
        Hill climbing:
        1. Start with empty graph (or random graph)
        2. At each step, try all single-edge modifications
        3. Make the modification that most improves the score
        4. Stop when no improvement possible
        
        Args:
            max_parents: Maximum number of parents per node
            max_iterations: Maximum number of iterations
        
        Returns:
            Learned DAG
        """
        print(f"\nHill Climbing Structure Learning")
        print(f"Max parents: {max_parents}, Max iterations: {max_iterations}")
        print("-" * 70)
        
        # Start with empty graph
        current_graph = nx.DiGraph()
        current_graph.add_nodes_from(self.variables)
        
        current_score = self.compute_bic_score(current_graph)
        print(f"Initial BIC score: {current_score:.2f}")
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            
            best_graph = None
            best_score = current_score
            best_operation = None
            
            # Try all possible single-edge operations
            
            # 1. Add edge
            for var1 in self.variables:
                for var2 in self.variables:
                    if var1 == var2:
                        continue
                    
                    # Check if edge already exists
                    if current_graph.has_edge(var1, var2):
                        continue
                    
                    # Check if adding would exceed max parents
                    if len(list(current_graph.predecessors(var2))) >= max_parents:
                        continue
                    
                    # Try adding edge
                    new_graph = current_graph.copy()
                    new_graph.add_edge(var1, var2)
                    
                    # Check if still acyclic
                    if not nx.is_directed_acyclic_graph(new_graph):
                        continue
                    
                    # Compute score
                    score = self.compute_bic_score(new_graph)
                    
                    if score > best_score:
                        best_score = score
                        best_graph = new_graph
                        best_operation = f"Add {var1} -> {var2}"
            
            # 2. Remove edge
            for var1, var2 in current_graph.edges():
                new_graph = current_graph.copy()
                new_graph.remove_edge(var1, var2)
                
                score = self.compute_bic_score(new_graph)
                
                if score > best_score:
                    best_score = score
                    best_graph = new_graph
                    best_operation = f"Remove {var1} -> {var2}"
            
            # 3. Reverse edge
            for var1, var2 in current_graph.edges():
                # Check if reversing would exceed max parents
                if len(list(current_graph.predecessors(var1))) >= max_parents:
                    continue
                
                new_graph = current_graph.copy()
                new_graph.remove_edge(var1, var2)
                new_graph.add_edge(var2, var1)
                
                # Check if still acyclic
                if not nx.is_directed_acyclic_graph(new_graph):
                    continue
                
                score = self.compute_bic_score(new_graph)
                
                if score > best_score:
                    best_score = score
                    best_graph = new_graph
                    best_operation = f"Reverse {var1} -> {var2}"
            
            # Check if we found improvement
            if best_graph is None:
                print("No improvement found. Converged!")
                break
            
            print(f"Best operation: {best_operation}")
            print(f"BIC score: {current_score:.2f} -> {best_score:.2f} (Δ = {best_score - current_score:.2f})")
            
            current_graph = best_graph
            current_score = best_score
        
        print(f"\nFinal graph has {current_graph.number_of_edges()} edges")
        print(f"Final BIC score: {current_score:.2f}")
        
        return current_graph


def demonstrate_structure_learning():
    """Demonstrate structure learning on synthetic data."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Structure Learning")
    print("="*70)
    
    # Generate synthetic data from a known structure
    print("\nGenerating synthetic data from known structure...")
    
    # Create a simple network: A -> B -> C, A -> C
    np.random.seed(42)
    n_samples = 1000
    
    # Sample A (prior)
    A = np.random.binomial(1, 0.5, n_samples)
    
    # Sample B given A
    B = np.random.binomial(1, 0.3 + 0.4 * A)
    
    # Sample C given A and B
    prob_C = 0.2 + 0.3 * A + 0.4 * B
    C = np.random.binomial(1, prob_C)
    
    # Create DataFrame
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    
    print(f"Generated {n_samples} samples")
    print(f"True structure: A -> B -> C, A -> C")
    
    # Learn structure using hill climbing
    print("\n" + "="*70)
    print("Score-Based Learning (Hill Climbing)")
    print("="*70)
    
    learner = ScoreBasedLearning(data)
    learned_graph = learner.hill_climbing(max_parents=2, max_iterations=20)
    
    # Visualize learned structure
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(learned_graph)
    nx.draw(learned_graph, pos,
            with_labels=True,
            node_color='lightgreen',
            node_size=3000,
            font_size=16,
            font_weight='bold',
            arrows=True,
            arrowsize=30,
            edge_color='gray',
            width=2)
    plt.title("Learned Bayesian Network Structure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nLearned edges:", list(learned_graph.edges()))


def main():
    """Main demonstration."""
    print("\n" + "="*70)
    print("STRUCTURE LEARNING FOR BAYESIAN NETWORKS")
    print("="*70)
    
    print("\nStructure learning is the problem of discovering")
    print("the graph structure from observational data.")
    
    demonstrate_structure_learning()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Two main approaches:")
    print("   - Constraint-based (PC algorithm): Test independencies")
    print("   - Score-based: Optimize scoring function")
    
    print("\n2. Structure learning is hard!")
    print("   - Super-exponential space of DAGs")
    print("   - NP-hard in general")
    print("   - Need heuristic search methods")
    
    print("\n3. Challenges:")
    print("   - Observational equivalence (multiple DAGs fit data equally)")
    print("   - Limited data (statistical power)")
    print("   - Computational complexity")
    
    print("\n4. Practical considerations:")
    print("   - Incorporate domain knowledge")
    print("   - Limit search space (max parents)")
    print("   - Use appropriate scoring functions")
    print("   - Validate learned structures")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

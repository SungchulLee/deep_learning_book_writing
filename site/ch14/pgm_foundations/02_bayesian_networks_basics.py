"""
Bayesian Networks - Basics
===========================

This module introduces Bayesian Networks, the most common type of
Probabilistic Graphical Model for representing causal relationships.

Learning Objectives:
-------------------
1. Understand what Bayesian Networks are and their components
2. Learn to represent conditional probability tables (CPTs)
3. Build simple Bayesian networks from scratch
4. Compute joint probabilities using the chain rule
5. Perform basic queries on Bayesian networks

Mathematical Foundations:
------------------------
A Bayesian Network is a tuple (G, P) where:
- G = (V, E) is a directed acyclic graph (DAG)
- P = {P(Xi | Parents(Xi))} is a set of conditional probability distributions

The network represents the joint distribution:
    P(X1, ..., Xn) = ∏ P(Xi | Parents(Xi))

Author: Educational ML Team
Level: Beginner
Prerequisites: 01_pgm_fundamentals.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional, Union
from itertools import product
import pandas as pd


class ConditionalProbabilityTable:
    """
    Represents a Conditional Probability Table (CPT) for a variable.
    
    A CPT specifies P(Variable | Parents) for all combinations of
    parent values. This is the fundamental building block of
    Bayesian Networks.
    
    Attributes:
        variable: Name of the variable this CPT describes
        parents: List of parent variable names
        cardinalities: Dict mapping all variables to their cardinalities
        table: NumPy array containing conditional probabilities
    """
    
    def __init__(self,
                 variable: str,
                 parents: List[str],
                 cardinalities: Dict[str, int],
                 table: Optional[np.ndarray] = None):
        """
        Initialize a CPT.
        
        Args:
            variable: Name of the variable (e.g., 'Rain')
            parents: List of parent variable names (e.g., ['Cloudy'])
            cardinalities: Dict of cardinalities for all variables
            table: Optional CPT values. Shape should be 
                   (*parent_cardinalities, variable_cardinality)
                   If None, initializes with uniform distribution
        
        Example:
            # CPT for P(Rain | Cloudy)
            # Rain is binary, Cloudy is binary
            cpt = ConditionalProbabilityTable(
                variable='Rain',
                parents=['Cloudy'],
                cardinalities={'Rain': 2, 'Cloudy': 2},
                table=np.array([[0.8, 0.2],  # P(Rain | Cloudy=0)
                               [0.2, 0.8]])  # P(Rain | Cloudy=1)
            )
        """
        self.variable = variable
        self.parents = parents
        self.cardinalities = cardinalities
        
        # Determine the shape of the CPT
        # Shape: (card(parent1), card(parent2), ..., card(variable))
        parent_cards = [cardinalities[p] for p in parents]
        var_card = cardinalities[variable]
        shape = tuple(parent_cards + [var_card])
        
        if table is None:
            # Initialize with uniform distribution
            self.table = np.ones(shape) / var_card
        else:
            self.table = np.array(table)
            # Verify shape
            assert self.table.shape == shape, \
                f"Table shape {self.table.shape} doesn't match expected {shape}"
            # Verify that each conditional distribution sums to 1
            self._verify_normalized()
    
    def _verify_normalized(self):
        """
        Verify that conditional probabilities sum to 1.
        
        For each configuration of parent values, the probabilities
        over the variable's values should sum to 1.
        """
        # Sum over the last axis (the variable itself)
        sums = np.sum(self.table, axis=-1)
        if not np.allclose(sums, 1.0):
            print(f"Warning: CPT for {self.variable} is not normalized!")
            print(f"Sums: {sums}")
    
    def get_probability(self, 
                       variable_value: int,
                       parent_values: Dict[str, int]) -> float:
        """
        Get P(Variable=value | Parents=parent_values).
        
        Args:
            variable_value: Value of the variable (integer index)
            parent_values: Dict mapping parent names to their values
        
        Returns:
            Conditional probability
        
        Example:
            # Get P(Rain=1 | Cloudy=0)
            prob = cpt.get_probability(1, {'Cloudy': 0})
        """
        # Build index tuple for accessing the table
        index = []
        for parent in self.parents:
            index.append(parent_values[parent])
        index.append(variable_value)
        
        return self.table[tuple(index)]
    
    def set_probability(self,
                       variable_value: int,
                       parent_values: Dict[str, int],
                       probability: float):
        """
        Set P(Variable=value | Parents=parent_values).
        
        Args:
            variable_value: Value of the variable
            parent_values: Dict mapping parent names to their values
            probability: Probability value to set
        """
        index = []
        for parent in self.parents:
            index.append(parent_values[parent])
        index.append(variable_value)
        
        self.table[tuple(index)] = probability
    
    def sample(self, parent_values: Dict[str, int]) -> int:
        """
        Sample a value for the variable given parent values.
        
        This is useful for forward sampling in Bayesian networks.
        
        Args:
            parent_values: Dict mapping parent names to their values
        
        Returns:
            Sampled value (integer)
        
        Example:
            # Sample Rain value given Cloudy=1
            rain_value = cpt.sample({'Cloudy': 1})
        """
        # Get the conditional distribution
        index = tuple(parent_values[p] for p in self.parents)
        probabilities = self.table[index]
        
        # Sample from the categorical distribution
        return np.random.choice(len(probabilities), p=probabilities)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert CPT to a readable DataFrame.
        
        This is useful for visualization and debugging.
        
        Returns:
            DataFrame with parent values and conditional probabilities
        """
        rows = []
        
        # Generate all combinations of parent values
        parent_cards = [self.cardinalities[p] for p in self.parents]
        
        if not self.parents:
            # No parents - just a prior distribution
            for var_val in range(self.cardinalities[self.variable]):
                row = {self.variable: var_val, 'Probability': self.table[var_val]}
                rows.append(row)
        else:
            for parent_combo in product(*[range(c) for c in parent_cards]):
                for var_val in range(self.cardinalities[self.variable]):
                    row = {}
                    # Add parent values
                    for parent, pval in zip(self.parents, parent_combo):
                        row[parent] = pval
                    # Add variable value
                    row[self.variable] = var_val
                    # Add probability
                    index = parent_combo + (var_val,)
                    row['Probability'] = self.table[index]
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def __str__(self) -> str:
        """String representation of the CPT."""
        if not self.parents:
            return f"P({self.variable})\n{self.to_dataframe().to_string(index=False)}"
        else:
            parent_str = ', '.join(self.parents)
            return f"P({self.variable} | {parent_str})\n{self.to_dataframe().to_string(index=False)}"


class BayesianNetwork:
    """
    Represents a Bayesian Network: a DAG with associated CPTs.
    
    A Bayesian Network consists of:
    1. A directed acyclic graph (DAG) structure
    2. Conditional probability tables (CPTs) for each node
    
    The network represents: P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))
    
    Attributes:
        graph: NetworkX DiGraph representing the structure
        cpts: Dict mapping variable names to their CPTs
        cardinalities: Dict mapping variable names to cardinalities
    """
    
    def __init__(self):
        """Initialize an empty Bayesian Network."""
        self.graph = nx.DiGraph()
        self.cpts: Dict[str, ConditionalProbabilityTable] = {}
        self.cardinalities: Dict[str, int] = {}
    
    def add_variable(self, name: str, cardinality: int):
        """
        Add a variable (node) to the network.
        
        Args:
            name: Variable name (e.g., 'Weather', 'Traffic')
            cardinality: Number of possible values (e.g., 2 for binary)
        
        Example:
            bn = BayesianNetwork()
            bn.add_variable('Rain', 2)  # Binary variable
            bn.add_variable('Season', 4)  # 4 seasons
        """
        self.graph.add_node(name)
        self.cardinalities[name] = cardinality
    
    def add_edge(self, parent: str, child: str):
        """
        Add a directed edge (causal relationship) between variables.
        
        Args:
            parent: Parent variable name
            child: Child variable name
        
        Raises:
            ValueError: If edge would create a cycle
        
        Example:
            bn.add_edge('Rain', 'WetGrass')  # Rain causes WetGrass
        """
        self.graph.add_edge(parent, child)
        
        # Verify no cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent, child)
            raise ValueError(f"Adding edge {parent}->{child} creates a cycle!")
    
    def set_cpt(self, 
                variable: str,
                table: np.ndarray):
        """
        Set the conditional probability table for a variable.
        
        Args:
            variable: Variable name
            table: NumPy array with conditional probabilities
                   Shape should match (card(parent1), ..., card(parentN), card(variable))
        
        Example:
            # For binary Rain with binary Cloudy as parent
            bn.set_cpt('Rain', np.array([[0.8, 0.2],   # P(Rain | Cloudy=0)
                                         [0.3, 0.7]]))  # P(Rain | Cloudy=1)
        """
        parents = list(self.graph.predecessors(variable))
        parents.sort()  # Consistent ordering
        
        cpt = ConditionalProbabilityTable(
            variable=variable,
            parents=parents,
            cardinalities=self.cardinalities,
            table=table
        )
        
        self.cpts[variable] = cpt
    
    def get_cpt(self, variable: str) -> ConditionalProbabilityTable:
        """
        Get the CPT for a variable.
        
        Args:
            variable: Variable name
        
        Returns:
            The conditional probability table
        """
        if variable not in self.cpts:
            raise ValueError(f"No CPT defined for variable {variable}")
        return self.cpts[variable]
    
    def compute_joint_probability(self, assignment: Dict[str, int]) -> float:
        """
        Compute P(X1=x1, X2=x2, ..., Xn=xn) for a complete assignment.
        
        Uses the chain rule factorization:
        P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))
        
        Args:
            assignment: Complete assignment of values to all variables
        
        Returns:
            Joint probability
        
        Example:
            # Compute P(Cloudy=1, Rain=1, WetGrass=1)
            prob = bn.compute_joint_probability({
                'Cloudy': 1,
                'Rain': 1,
                'WetGrass': 1
            })
        """
        probability = 1.0
        
        # Multiply conditional probabilities for each variable
        for variable in self.graph.nodes():
            cpt = self.get_cpt(variable)
            parents = list(self.graph.predecessors(variable))
            
            # Get parent values
            parent_values = {p: assignment[p] for p in parents}
            
            # Get variable value
            var_value = assignment[variable]
            
            # Multiply by P(variable | parents)
            prob = cpt.get_probability(var_value, parent_values)
            probability *= prob
        
        return probability
    
    def forward_sample(self) -> Dict[str, int]:
        """
        Generate a sample from the joint distribution using forward sampling.
        
        Forward sampling follows the topological order:
        1. Sample each variable in topological order
        2. Sample from P(Xi | Parents(Xi)) using already-sampled parent values
        
        Returns:
            Complete assignment sampled from P(X1,...,Xn)
        
        Example:
            # Generate 1000 samples from the network
            samples = [bn.forward_sample() for _ in range(1000)]
        """
        assignment = {}
        
        # Sample in topological order (parents before children)
        for variable in nx.topological_sort(self.graph):
            cpt = self.get_cpt(variable)
            parents = list(self.graph.predecessors(variable))
            
            # Get parent values from already-sampled variables
            parent_values = {p: assignment[p] for p in parents}
            
            # Sample this variable
            assignment[variable] = cpt.sample(parent_values)
        
        return assignment
    
    def visualize(self, figsize: Tuple[int, int] = (12, 8), show_cpts: bool = False):
        """
        Visualize the Bayesian Network structure and optionally CPTs.
        
        Args:
            figsize: Figure size
            show_cpts: Whether to display CPT values
        """
        plt.figure(figsize=figsize)
        
        # Layout
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        except:
            pos = nx.spring_layout(self.graph)
        
        # Draw nodes and edges
        nx.draw(self.graph, pos,
                with_labels=True,
                node_color='lightcoral',
                node_size=3000,
                font_size=12,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                width=2)
        
        # Add cardinalities as node labels
        labels = {node: f"{node}\n(card={self.cardinalities[node]})" 
                 for node in self.graph.nodes()}
        pos_labels = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
        nx.draw_networkx_labels(self.graph, pos_labels, labels, 
                               font_size=9, font_color='darkred')
        
        plt.title("Bayesian Network Structure", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Display CPTs if requested
        if show_cpts:
            print("\n" + "="*70)
            print("CONDITIONAL PROBABILITY TABLES")
            print("="*70)
            for variable in nx.topological_sort(self.graph):
                cpt = self.get_cpt(variable)
                print(f"\n{cpt}")
                print("-"*70)


def build_weather_network() -> BayesianNetwork:
    """
    Build a simple weather prediction Bayesian Network.
    
    Network structure:
        Cloudy -> Rain -> WetGrass
        Cloudy -> Sprinkler -> WetGrass
    
    This is a classic example that demonstrates:
    - Common cause (Cloudy affects both Rain and Sprinkler)
    - Multiple causes (WetGrass can be wet due to Rain OR Sprinkler)
    
    Returns:
        Complete Bayesian Network with CPTs
    """
    print("\nBuilding Weather Network...")
    print("-" * 70)
    
    bn = BayesianNetwork()
    
    # Add variables (all binary: 0=False, 1=True)
    bn.add_variable('Cloudy', 2)
    bn.add_variable('Sprinkler', 2)
    bn.add_variable('Rain', 2)
    bn.add_variable('WetGrass', 2)
    
    # Add edges (causal relationships)
    bn.add_edge('Cloudy', 'Sprinkler')  # Cloudy affects Sprinkler usage
    bn.add_edge('Cloudy', 'Rain')        # Cloudy affects Rain
    bn.add_edge('Sprinkler', 'WetGrass') # Sprinkler can wet grass
    bn.add_edge('Rain', 'WetGrass')      # Rain can wet grass
    
    # Set CPTs
    
    # P(Cloudy) - Prior probability
    # [P(Cloudy=0), P(Cloudy=1)]
    bn.set_cpt('Cloudy', np.array([0.5, 0.5]))
    
    # P(Sprinkler | Cloudy)
    # Less likely to use sprinkler when cloudy
    bn.set_cpt('Sprinkler', np.array([
        [0.5, 0.5],  # P(Sprinkler | Cloudy=0)
        [0.9, 0.1]   # P(Sprinkler | Cloudy=1) - unlikely when cloudy
    ]))
    
    # P(Rain | Cloudy)
    # More likely to rain when cloudy
    bn.set_cpt('Rain', np.array([
        [0.8, 0.2],  # P(Rain | Cloudy=0) - unlikely when not cloudy
        [0.2, 0.8]   # P(Rain | Cloudy=1) - likely when cloudy
    ]))
    
    # P(WetGrass | Sprinkler, Rain)
    # Grass is wet if Sprinkler is on OR it's raining
    # Order: [Sprinkler=0, Rain=0], [Sprinkler=0, Rain=1],
    #        [Sprinkler=1, Rain=0], [Sprinkler=1, Rain=1]
    bn.set_cpt('WetGrass', np.array([
        [[1.0, 0.0],   # Sprinkler=0, Rain=0: Grass dry
         [0.1, 0.9]],  # Sprinkler=0, Rain=1: Grass wet (rain)
        [[0.1, 0.9],   # Sprinkler=1, Rain=0: Grass wet (sprinkler)
         [0.01, 0.99]] # Sprinkler=1, Rain=1: Grass very wet (both)
    ]))
    
    print("Network built successfully!")
    print(f"Variables: {list(bn.graph.nodes())}")
    print(f"Edges: {list(bn.graph.edges())}")
    
    return bn


def demonstrate_joint_probability():
    """
    Demonstrate computing joint probabilities in a Bayesian Network.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Computing Joint Probabilities")
    print("="*70)
    
    bn = build_weather_network()
    
    # Compute some joint probabilities
    test_cases = [
        {'Cloudy': 0, 'Sprinkler': 0, 'Rain': 0, 'WetGrass': 0},
        {'Cloudy': 1, 'Sprinkler': 0, 'Rain': 1, 'WetGrass': 1},
        {'Cloudy': 1, 'Sprinkler': 1, 'Rain': 1, 'WetGrass': 1},
    ]
    
    print("\nComputing joint probabilities for different scenarios:")
    print("-" * 70)
    
    for i, assignment in enumerate(test_cases, 1):
        prob = bn.compute_joint_probability(assignment)
        
        # Create readable description
        desc = ", ".join([f"{var}={'Yes' if val else 'No'}" 
                         for var, val in assignment.items()])
        
        print(f"\nScenario {i}: {desc}")
        print(f"P(assignment) = {prob:.6f}")
        
        # Show factorization
        print("\nFactorization:")
        print(f"  = P(Cloudy={assignment['Cloudy']})")
        print(f"  × P(Sprinkler={assignment['Sprinkler']} | Cloudy={assignment['Cloudy']})")
        print(f"  × P(Rain={assignment['Rain']} | Cloudy={assignment['Cloudy']})")
        print(f"  × P(WetGrass={assignment['WetGrass']} | Sprinkler={assignment['Sprinkler']}, Rain={assignment['Rain']})")


def demonstrate_sampling():
    """
    Demonstrate forward sampling from a Bayesian Network.
    
    Forward sampling is a Monte Carlo method to approximate
    the joint distribution.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Forward Sampling")
    print("="*70)
    
    bn = build_weather_network()
    
    # Generate samples
    num_samples = 10000
    print(f"\nGenerating {num_samples} samples from the network...")
    
    samples = [bn.forward_sample() for _ in range(num_samples)]
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(samples)
    
    print("\nFirst 10 samples:")
    print(df.head(10).to_string(index=False))
    
    # Compute empirical probabilities
    print("\n" + "-"*70)
    print("Empirical vs. True Probabilities")
    print("-"*70)
    
    # Check P(Cloudy)
    empirical_cloudy = df['Cloudy'].mean()
    print(f"\nP(Cloudy=1):")
    print(f"  True: 0.500")
    print(f"  Empirical: {empirical_cloudy:.3f}")
    
    # Check P(WetGrass)
    empirical_wet = df['WetGrass'].mean()
    # Compute true probability by summing over all configurations
    true_wet = 0.0
    for assignment in [dict(zip(['Cloudy', 'Sprinkler', 'Rain', 'WetGrass'], combo))
                      for combo in product([0,1], repeat=4)
                      if combo[3] == 1]:  # WetGrass=1
        true_wet += bn.compute_joint_probability(assignment)
    
    print(f"\nP(WetGrass=1):")
    print(f"  True: {true_wet:.3f}")
    print(f"  Empirical: {empirical_wet:.3f}")
    
    # Conditional probability: P(Rain=1 | Cloudy=1)
    cloudy_samples = df[df['Cloudy'] == 1]
    empirical_rain_given_cloudy = cloudy_samples['Rain'].mean()
    
    print(f"\nP(Rain=1 | Cloudy=1):")
    print(f"  True: 0.800")
    print(f"  Empirical: {empirical_rain_given_cloudy:.3f}")
    
    print("\nNote: With more samples, empirical probabilities converge to true values!")


def build_student_network() -> BayesianNetwork:
    """
    Build a student performance Bayesian Network.
    
    This network models factors affecting a student's exam grade:
    - Difficulty of the exam
    - Student's intelligence
    - Student's grade (depends on both)
    - Recommendation letter quality (depends on grade)
    
    Network structure:
        Difficulty -> Grade <- Intelligence
        Grade -> Letter
    
    Returns:
        Complete Bayesian Network
    """
    print("\nBuilding Student Network...")
    print("-" * 70)
    
    bn = BayesianNetwork()
    
    # Add variables
    bn.add_variable('Difficulty', 2)   # 0=Easy, 1=Hard
    bn.add_variable('Intelligence', 2)  # 0=Low, 1=High
    bn.add_variable('Grade', 3)         # 0=A, 1=B, 2=C
    bn.add_variable('Letter', 2)        # 0=Weak, 1=Strong
    
    # Add edges
    bn.add_edge('Difficulty', 'Grade')
    bn.add_edge('Intelligence', 'Grade')
    bn.add_edge('Grade', 'Letter')
    
    # Set CPTs
    
    # P(Difficulty)
    bn.set_cpt('Difficulty', np.array([0.6, 0.4]))
    
    # P(Intelligence)
    bn.set_cpt('Intelligence', np.array([0.7, 0.3]))
    
    # P(Grade | Intelligence, Difficulty)
    # [Difficulty=0, Intelligence=0] -> mostly B and C
    # [Difficulty=0, Intelligence=1] -> mostly A and B
    # [Difficulty=1, Intelligence=0] -> mostly C
    # [Difficulty=1, Intelligence=1] -> mostly B
    bn.set_cpt('Grade', np.array([
        [[0.3, 0.4, 0.3],   # Difficulty=0, Intelligence=0
         [0.9, 0.08, 0.02]], # Difficulty=0, Intelligence=1
        [[0.05, 0.25, 0.7],  # Difficulty=1, Intelligence=0
         [0.5, 0.3, 0.2]]    # Difficulty=1, Intelligence=1
    ]))
    
    # P(Letter | Grade)
    # Better grades lead to stronger letters
    bn.set_cpt('Letter', np.array([
        [0.1, 0.9],  # Grade=A -> Strong letter
        [0.4, 0.6],  # Grade=B -> Moderate letter
        [0.9, 0.1]   # Grade=C -> Weak letter
    ]))
    
    print("Student network built successfully!")
    return bn


def main():
    """
    Main function demonstrating Bayesian Network concepts.
    """
    print("\n" + "="*70)
    print("BAYESIAN NETWORKS - BASICS")
    print("="*70)
    
    print("\nTopics covered:")
    print("1. Building Bayesian Networks")
    print("2. Conditional Probability Tables (CPTs)")
    print("3. Computing joint probabilities")
    print("4. Forward sampling")
    
    # Build and visualize weather network
    print("\n" + "="*70)
    print("Example 1: Weather Network")
    print("="*70)
    bn = build_weather_network()
    bn.visualize(show_cpts=True)
    
    # Demonstrate joint probability computation
    demonstrate_joint_probability()
    
    # Demonstrate sampling
    demonstrate_sampling()
    
    # Build and visualize student network
    print("\n" + "="*70)
    print("Example 2: Student Network")
    print("="*70)
    student_bn = build_student_network()
    student_bn.visualize(show_cpts=True)
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Bayesian Networks = DAG + CPTs")
    print("2. CPTs specify P(Variable | Parents)")
    print("3. Joint distribution: P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))")
    print("4. Forward sampling follows topological order")
    print("5. Network structure encodes conditional independence")
    
    print("\n" + "="*70)
    print("Next: Learn about inference in Bayesian Networks!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

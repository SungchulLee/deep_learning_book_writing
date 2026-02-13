"""
Probabilistic Graphical Models - Fundamentals
==============================================

This module introduces the foundational concepts of Probabilistic Graphical Models (PGMs).

Learning Objectives:
-------------------
1. Understand what probabilistic graphical models are and why they're useful
2. Learn about graphical representations of probability distributions
3. Master the concepts of independence and conditional independence
4. Understand d-separation in directed graphs
5. Learn how to represent joint probability distributions compactly

Mathematical Foundations:
------------------------
- Joint probability: P(X1, X2, ..., Xn)
- Conditional probability: P(X|Y) = P(X,Y) / P(Y)
- Chain rule: P(X1,...,Xn) = ∏ P(Xi | X1,...,Xi-1)
- Independence: P(X,Y) = P(X)P(Y)
- Conditional independence: P(X,Y|Z) = P(X|Z)P(Y|Z)

Author: Educational ML Team
Level: Beginner
Prerequisites: Basic probability theory
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
from itertools import product


class ProbabilityDistribution:
    """
    Represents a discrete probability distribution over a set of random variables.
    
    This class provides basic operations for working with probability distributions,
    including marginalization, conditioning, and independence testing.
    
    Attributes:
        variables: List of random variable names
        cardinalities: Dictionary mapping variable names to their cardinalities
        values: NumPy array containing probability values
    """
    
    def __init__(self, 
                 variables: List[str], 
                 cardinalities: Dict[str, int],
                 values: Optional[np.ndarray] = None):
        """
        Initialize a probability distribution.
        
        Args:
            variables: List of variable names (e.g., ['X', 'Y', 'Z'])
            cardinalities: Dict of cardinalities (e.g., {'X': 2, 'Y': 3, 'Z': 2})
            values: Optional probability values array. If None, uniform distribution
        
        Example:
            # Create a distribution over two binary variables
            dist = ProbabilityDistribution(
                variables=['X', 'Y'],
                cardinalities={'X': 2, 'Y': 2},
                values=np.array([[0.3, 0.2], [0.4, 0.1]])
            )
        """
        self.variables = variables
        self.cardinalities = cardinalities
        
        # Calculate the shape of the distribution array
        shape = tuple(cardinalities[var] for var in variables)
        
        if values is None:
            # Initialize with uniform distribution
            total_size = np.prod(shape)
            self.values = np.ones(shape) / total_size
        else:
            self.values = np.array(values)
            # Ensure the distribution is normalized
            self.values = self.values / np.sum(self.values)
        
        # Validate that the shape matches
        assert self.values.shape == shape, \
            f"Values shape {self.values.shape} doesn't match expected {shape}"
    
    def marginalize(self, variables_to_keep: List[str]) -> 'ProbabilityDistribution':
        """
        Marginalize out variables not in variables_to_keep.
        
        Marginalization is the process of summing out variables to obtain
        a distribution over a subset of variables.
        
        Mathematical definition:
            P(X) = Σ_Y P(X, Y)
        
        Args:
            variables_to_keep: List of variables to keep in the distribution
        
        Returns:
            New ProbabilityDistribution over the specified variables
        
        Example:
            # If we have P(X, Y, Z), we can get P(X, Y) by marginalizing out Z
            joint = ProbabilityDistribution(['X', 'Y', 'Z'], ...)
            marginal = joint.marginalize(['X', 'Y'])  # This gives P(X, Y)
        """
        # Find which variables to sum out
        variables_to_sum = [var for var in self.variables if var not in variables_to_keep]
        
        # If no variables to sum out, return a copy
        if not variables_to_sum:
            return ProbabilityDistribution(
                self.variables, 
                self.cardinalities, 
                self.values.copy()
            )
        
        # Find the axes corresponding to variables to sum out
        axes_to_sum = tuple(self.variables.index(var) for var in variables_to_sum)
        
        # Perform marginalization by summing over the appropriate axes
        marginalized_values = np.sum(self.values, axis=axes_to_sum)
        
        # Create new cardinalities dict
        new_cardinalities = {var: self.cardinalities[var] for var in variables_to_keep}
        
        return ProbabilityDistribution(
            variables_to_keep,
            new_cardinalities,
            marginalized_values
        )
    
    def condition(self, evidence: Dict[str, int]) -> 'ProbabilityDistribution':
        """
        Condition the distribution on observed evidence.
        
        Conditioning is the process of fixing certain variables to observed values
        and renormalizing the distribution.
        
        Mathematical definition:
            P(X|Y=y) = P(X, Y=y) / P(Y=y)
        
        Args:
            evidence: Dictionary mapping variable names to observed values
                     (e.g., {'X': 0, 'Y': 1})
        
        Returns:
            New ProbabilityDistribution conditioned on the evidence
        
        Example:
            # Condition P(X, Y, Z) on Y=1 to get P(X, Z | Y=1)
            joint = ProbabilityDistribution(['X', 'Y', 'Z'], ...)
            conditional = joint.condition({'Y': 1})
        """
        # Create a slice that selects the appropriate values
        slice_indices = []
        remaining_variables = []
        remaining_cardinalities = {}
        
        for var in self.variables:
            if var in evidence:
                # This variable is observed, select its value
                slice_indices.append(evidence[var])
            else:
                # This variable is not observed, keep all values
                slice_indices.append(slice(None))
                remaining_variables.append(var)
                remaining_cardinalities[var] = self.cardinalities[var]
        
        # Extract the conditioned values
        conditioned_values = self.values[tuple(slice_indices)]
        
        # Normalize to make it a proper probability distribution
        # P(X|Y=y) = P(X,Y=y) / Σ_X P(X,Y=y)
        total = np.sum(conditioned_values)
        if total > 0:
            conditioned_values = conditioned_values / total
        else:
            # If total is 0, the evidence is impossible
            print("Warning: Evidence has probability 0!")
            conditioned_values = np.ones_like(conditioned_values) / conditioned_values.size
        
        return ProbabilityDistribution(
            remaining_variables,
            remaining_cardinalities,
            conditioned_values
        )
    
    def is_independent(self, var1: str, var2: str, threshold: float = 1e-6) -> bool:
        """
        Test if two variables are independent.
        
        Two variables X and Y are independent if:
            P(X, Y) = P(X) * P(Y) for all values of X and Y
        
        Args:
            var1: Name of first variable
            var2: Name of second variable
            threshold: Numerical threshold for equality testing
        
        Returns:
            True if variables are independent, False otherwise
        
        Example:
            dist = ProbabilityDistribution(['X', 'Y'], ...)
            if dist.is_independent('X', 'Y'):
                print("X and Y are independent")
        """
        # Get marginal distributions
        p_var1 = self.marginalize([var1])
        p_var2 = self.marginalize([var2])
        
        # Get joint distribution over just these two variables
        p_joint = self.marginalize([var1, var2])
        
        # Compute the product of marginals: P(X) * P(Y)
        # We need to compute the outer product
        idx1 = p_joint.variables.index(var1)
        idx2 = p_joint.variables.index(var2)
        
        # Reshape arrays for broadcasting
        shape1 = [1] * len(p_joint.variables)
        shape1[idx1] = self.cardinalities[var1]
        
        shape2 = [1] * len(p_joint.variables)
        shape2[idx2] = self.cardinalities[var2]
        
        marginal_product = (
            p_var1.values.reshape(shape1) * 
            p_var2.values.reshape(shape2)
        )
        
        # Check if P(X,Y) ≈ P(X)P(Y)
        difference = np.abs(p_joint.values - marginal_product)
        return np.all(difference < threshold)
    
    def is_conditionally_independent(self, 
                                    var1: str, 
                                    var2: str, 
                                    given: List[str],
                                    threshold: float = 1e-6) -> bool:
        """
        Test if two variables are conditionally independent given others.
        
        X and Y are conditionally independent given Z if:
            P(X, Y | Z) = P(X | Z) * P(Y | Z) for all values
        
        This can be checked by verifying that for all values z of Z:
            P(X, Y, Z=z) = P(X, Z=z) * P(Y, Z=z) / P(Z=z)
        
        Args:
            var1: Name of first variable
            var2: Name of second variable
            given: List of conditioning variable names
            threshold: Numerical threshold for equality testing
        
        Returns:
            True if conditionally independent, False otherwise
        
        Example:
            # Test if X ⊥ Y | Z (X and Y are independent given Z)
            if dist.is_conditionally_independent('X', 'Y', ['Z']):
                print("X and Y are conditionally independent given Z")
        """
        # We need to check this for all values of the conditioning variables
        # Get all possible assignments to the conditioning variables
        given_cardinalities = [self.cardinalities[var] for var in given]
        
        for given_values in product(*[range(card) for card in given_cardinalities]):
            # Create evidence dictionary
            evidence = {var: val for var, val in zip(given, given_values)}
            
            # Condition on this evidence
            conditioned = self.condition(evidence)
            
            # Check if var1 and var2 are independent in the conditioned distribution
            if not conditioned.is_independent(var1, var2, threshold):
                return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of the distribution."""
        return f"P({', '.join(self.variables)})\nShape: {self.values.shape}\nValues:\n{self.values}"


class DirectedGraph:
    """
    Represents a directed acyclic graph (DAG) for Bayesian networks.
    
    A DAG is a graph with directed edges that contains no cycles.
    This is the fundamental structure underlying Bayesian networks.
    
    Attributes:
        graph: NetworkX DiGraph object
    """
    
    def __init__(self):
        """Initialize an empty directed graph."""
        self.graph = nx.DiGraph()
    
    def add_node(self, node: str):
        """
        Add a node (random variable) to the graph.
        
        Args:
            node: Name of the node/variable
        """
        self.graph.add_node(node)
    
    def add_edge(self, parent: str, child: str):
        """
        Add a directed edge from parent to child.
        
        The edge represents a direct probabilistic dependency:
        the parent influences the child.
        
        Args:
            parent: Name of the parent node
            child: Name of the child node
        
        Raises:
            ValueError: If adding the edge would create a cycle
        """
        # Check if adding this edge would create a cycle
        self.graph.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent, child)
            raise ValueError(f"Adding edge {parent}->{child} would create a cycle!")
    
    def get_parents(self, node: str) -> List[str]:
        """
        Get the parents of a node.
        
        Parents are nodes that have directed edges pointing to this node.
        In probability terms, these are the variables that the node
        directly depends on.
        
        Args:
            node: Name of the node
        
        Returns:
            List of parent node names
        """
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """
        Get the children of a node.
        
        Children are nodes that this node has directed edges pointing to.
        
        Args:
            node: Name of the node
        
        Returns:
            List of child node names
        """
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """
        Get all ancestors of a node.
        
        Ancestors include parents, parents of parents, etc.
        These are all nodes from which there is a directed path to this node.
        
        Args:
            node: Name of the node
        
        Returns:
            Set of ancestor node names
        """
        return nx.ancestors(self.graph, node)
    
    def get_descendants(self, node: str) -> Set[str]:
        """
        Get all descendants of a node.
        
        Descendants include children, children of children, etc.
        These are all nodes to which there is a directed path from this node.
        
        Args:
            node: Name of the node
        
        Returns:
            Set of descendant node names
        """
        return nx.descendants(self.graph, node)
    
    def topological_order(self) -> List[str]:
        """
        Get a topological ordering of the nodes.
        
        A topological ordering is an ordering of nodes such that
        for every edge (u,v), u comes before v in the ordering.
        
        This is useful for:
        1. Computing joint probabilities using the chain rule
        2. Forward sampling
        3. Various inference algorithms
        
        Returns:
            List of nodes in topological order
        """
        return list(nx.topological_sort(self.graph))
    
    def is_d_separated(self, 
                      X: Set[str], 
                      Y: Set[str], 
                      Z: Set[str]) -> bool:
        """
        Test if X and Y are d-separated given Z.
        
        D-separation (directional separation) is a graphical criterion for
        conditional independence in Bayesian networks.
        
        If X and Y are d-separated given Z, then X ⊥ Y | Z
        (X is conditionally independent of Y given Z).
        
        D-separation rules:
        1. Chain: X -> Z -> Y: X and Y are d-separated given Z
        2. Fork: X <- Z -> Y: X and Y are d-separated given Z
        3. Collider: X -> Z <- Y: X and Y are NOT d-separated given Z
                     But ARE d-separated if Z is NOT observed
        
        Args:
            X: Set of nodes in first group
            Y: Set of nodes in second group
            Z: Set of conditioning nodes
        
        Returns:
            True if X and Y are d-separated given Z
        
        Example:
            graph = DirectedGraph()
            graph.add_edge('X', 'Z')
            graph.add_edge('Z', 'Y')
            # X and Y are d-separated given Z (chain structure)
            is_sep = graph.is_d_separated({'X'}, {'Y'}, {'Z'})  # True
        """
        # NetworkX provides a d-separation test
        return nx.d_separated(self.graph, X, Y, Z)
    
    def visualize(self, title: str = "Directed Graph", figsize: Tuple[int, int] = (10, 6)):
        """
        Visualize the directed graph.
        
        Args:
            title: Title for the plot
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Use hierarchical layout for better visualization
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        except:
            pos = nx.spring_layout(self.graph)
        
        # Draw the graph
        nx.draw(self.graph, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=12,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                width=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def demonstrate_independence():
    """
    Demonstrate the concept of independence with examples.
    
    This function creates distributions that exhibit independence
    and conditional independence, helping visualize these concepts.
    """
    print("=" * 70)
    print("DEMONSTRATION: Independence vs Conditional Independence")
    print("=" * 70)
    
    # Example 1: Independent variables
    # Two coin flips - they are independent
    print("\nExample 1: Two Independent Coin Flips")
    print("-" * 70)
    
    # P(Coin1, Coin2) where both are fair and independent
    # P(C1=H, C2=H) = 0.25, P(C1=H, C2=T) = 0.25, etc.
    independent_dist = ProbabilityDistribution(
        variables=['Coin1', 'Coin2'],
        cardinalities={'Coin1': 2, 'Coin2': 2},
        values=np.array([[0.25, 0.25],  # Coin1=0 (Tails)
                        [0.25, 0.25]])   # Coin1=1 (Heads)
    )
    
    print(f"Joint distribution P(Coin1, Coin2):")
    print(independent_dist.values)
    print(f"\nAre Coin1 and Coin2 independent? {independent_dist.is_independent('Coin1', 'Coin2')}")
    
    # Example 2: Dependent variables
    # Weather affects whether someone carries an umbrella
    print("\n\nExample 2: Dependent Variables (Weather and Umbrella)")
    print("-" * 70)
    
    # P(Rain, Umbrella)
    # People are more likely to carry umbrella when it rains
    dependent_dist = ProbabilityDistribution(
        variables=['Rain', 'Umbrella'],
        cardinalities={'Rain': 2, 'Umbrella': 2},
        values=np.array([[0.50, 0.05],  # Rain=0 (No rain)
                        [0.05, 0.40]])   # Rain=1 (Rain)
    )
    
    print(f"Joint distribution P(Rain, Umbrella):")
    print(dependent_dist.values)
    print(f"\nAre Rain and Umbrella independent? {dependent_dist.is_independent('Rain', 'Umbrella')}")
    
    # Example 3: Conditional independence
    # X -> Z -> Y structure (chain)
    print("\n\nExample 3: Conditional Independence (Chain Structure)")
    print("-" * 70)
    print("Structure: X -> Z -> Y")
    print("X and Y are conditionally independent given Z")
    
    # Create a distribution that exhibits conditional independence
    # P(X, Z, Y) where X affects Z, and Z affects Y
    chain_values = np.zeros((2, 2, 2))
    # Build using chain rule: P(X,Z,Y) = P(X) P(Z|X) P(Y|Z)
    
    # P(X)
    p_x = np.array([0.6, 0.4])
    
    # P(Z|X) - X influences Z
    p_z_given_x = np.array([[0.8, 0.2],  # Z|X=0
                            [0.3, 0.7]])  # Z|X=1
    
    # P(Y|Z) - Z influences Y
    p_y_given_z = np.array([[0.7, 0.3],  # Y|Z=0
                            [0.2, 0.8]])  # Y|Z=1
    
    for x in range(2):
        for z in range(2):
            for y in range(2):
                chain_values[x, z, y] = p_x[x] * p_z_given_x[x, z] * p_y_given_z[z, y]
    
    chain_dist = ProbabilityDistribution(
        variables=['X', 'Z', 'Y'],
        cardinalities={'X': 2, 'Z': 2, 'Y': 2},
        values=chain_values
    )
    
    print(f"\nAre X and Y independent? {chain_dist.is_independent('X', 'Y')}")
    print(f"Are X and Y conditionally independent given Z? "
          f"{chain_dist.is_conditionally_independent('X', 'Y', ['Z'])}")
    
    print("\nIntuition: Once we know Z, knowing X doesn't give us additional")
    print("information about Y. All the influence from X to Y goes through Z.")


def demonstrate_d_separation():
    """
    Demonstrate d-separation with different graph structures.
    
    D-separation is a key concept for understanding conditional independence
    in Bayesian networks.
    """
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION: D-Separation in Different Structures")
    print("=" * 70)
    
    # Structure 1: Chain (X -> Z -> Y)
    print("\nStructure 1: Chain (X -> Z -> Y)")
    print("-" * 70)
    chain = DirectedGraph()
    chain.add_node('X')
    chain.add_node('Z')
    chain.add_node('Y')
    chain.add_edge('X', 'Z')
    chain.add_edge('Z', 'Y')
    
    print("Graph: X -> Z -> Y")
    print(f"X ⊥ Y | Z? {chain.is_d_separated({'X'}, {'Y'}, {'Z'})} (should be True)")
    print(f"X ⊥ Y | ∅? {chain.is_d_separated({'X'}, {'Y'}, set())} (should be False)")
    print("\nIntuition: Information flows from X to Y through Z.")
    print("If we observe Z, the path is blocked.")
    
    # Structure 2: Fork (X <- Z -> Y)
    print("\n\nStructure 2: Fork (X <- Z -> Y)")
    print("-" * 70)
    fork = DirectedGraph()
    fork.add_node('X')
    fork.add_node('Z')
    fork.add_node('Y')
    fork.add_edge('Z', 'X')
    fork.add_edge('Z', 'Y')
    
    print("Graph: X <- Z -> Y")
    print(f"X ⊥ Y | Z? {fork.is_d_separated({'X'}, {'Y'}, {'Z'})} (should be True)")
    print(f"X ⊥ Y | ∅? {fork.is_d_separated({'X'}, {'Y'}, set())} (should be False)")
    print("\nIntuition: Z is a common cause of X and Y.")
    print("If we observe Z, X and Y become independent.")
    
    # Structure 3: Collider (X -> Z <- Y)
    print("\n\nStructure 3: Collider (X -> Z <- Y)")
    print("-" * 70)
    collider = DirectedGraph()
    collider.add_node('X')
    collider.add_node('Z')
    collider.add_node('Y')
    collider.add_edge('X', 'Z')
    collider.add_edge('Y', 'Z')
    
    print("Graph: X -> Z <- Y")
    print(f"X ⊥ Y | Z? {collider.is_d_separated({'X'}, {'Y'}, {'Z'})} (should be False)")
    print(f"X ⊥ Y | ∅? {collider.is_d_separated({'X'}, {'Y'}, set())} (should be True)")
    print("\nIntuition: Z is a common effect of X and Y.")
    print("If we DON'T observe Z, X and Y are independent.")
    print("If we DO observe Z, X and Y become dependent (explaining away effect).")
    
    # Visualize all three structures
    print("\n\nVisualizing all three structures...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (graph, title) in zip(axes, 
                                   [(chain.graph, "Chain: X → Z → Y"),
                                    (fork.graph, "Fork: X ← Z → Y"),
                                    (collider.graph, "Collider: X → Z ← Y")]):
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, ax=ax,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=14,
                font_weight='bold',
                arrows=True,
                arrowsize=20)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def demonstrate_factorization():
    """
    Demonstrate how Bayesian networks factor joint distributions.
    
    Key insight: A Bayesian network over variables X1, ..., Xn
    represents the joint distribution as:
    
    P(X1, ..., Xn) = ∏ P(Xi | Parents(Xi))
    
    This factorization makes computation much more efficient.
    """
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION: Factorization in Bayesian Networks")
    print("=" * 70)
    
    print("\nConsider a simple alarm system:")
    print("- Burglary and Earthquake are independent events")
    print("- Alarm goes off if Burglary OR Earthquake occurs")
    print("- John and Mary call if they hear the Alarm")
    print("\nStructure: Burglary -> Alarm <- Earthquake")
    print("           Alarm -> JohnCalls")
    print("           Alarm -> MaryCalls")
    
    # Create the graph
    graph = DirectedGraph()
    for node in ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']:
        graph.add_node(node)
    
    graph.add_edge('Burglary', 'Alarm')
    graph.add_edge('Earthquake', 'Alarm')
    graph.add_edge('Alarm', 'JohnCalls')
    graph.add_edge('Alarm', 'MaryCalls')
    
    print("\n\nNaive joint distribution representation:")
    print("-" * 70)
    print("Without structure: P(B, E, A, J, M)")
    print("Number of parameters: 2^5 - 1 = 31 independent parameters")
    print("(We need to store probability for each of 32 possible combinations)")
    
    print("\n\nFactorized representation using Bayesian network:")
    print("-" * 70)
    print("P(B, E, A, J, M) = P(B) × P(E) × P(A|B,E) × P(J|A) × P(M|A)")
    print("\nNumber of parameters:")
    print("- P(B): 1 parameter (probability of burglary)")
    print("- P(E): 1 parameter (probability of earthquake)")
    print("- P(A|B,E): 4 parameters (2×2 combinations of B and E)")
    print("- P(J|A): 2 parameters (2 values of A)")
    print("- P(M|A): 2 parameters (2 values of A)")
    print("Total: 1 + 1 + 4 + 2 + 2 = 10 parameters")
    print("\nSpace savings: 31 vs 10 parameters (68% reduction!)")
    
    print("\n\nThis factorization also enables efficient inference:")
    print("- We can compute conditional probabilities efficiently")
    print("- We can perform reasoning with incomplete information")
    print("- We can identify independence relationships")
    
    # Visualize the network
    graph.visualize("Alarm Network: Factorized Representation")


def main():
    """
    Main function to run all demonstrations.
    
    This provides a comprehensive introduction to PGM fundamentals
    through concrete examples and visualizations.
    """
    print("\n" + "=" * 70)
    print("PROBABILISTIC GRAPHICAL MODELS - FUNDAMENTALS")
    print("=" * 70)
    print("\nThis module introduces the core concepts of PGMs:")
    print("1. Probability distributions and their operations")
    print("2. Independence and conditional independence")
    print("3. Graphical representations (directed graphs)")
    print("4. D-separation")
    print("5. Factorization of joint distributions")
    
    # Run demonstrations
    demonstrate_independence()
    demonstrate_d_separation()
    demonstrate_factorization()
    
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. PGMs provide a compact representation of joint distributions")
    print("   using graph structure to encode independence relationships.")
    
    print("\n2. Independence and conditional independence are different:")
    print("   - Independent: P(X,Y) = P(X)P(Y)")
    print("   - Conditionally independent: P(X,Y|Z) = P(X|Z)P(Y|Z)")
    
    print("\n3. D-separation is a graphical test for conditional independence:")
    print("   - Chain & Fork: Z blocks the path when observed")
    print("   - Collider: Z blocks the path when NOT observed")
    
    print("\n4. Factorization enables efficient computation:")
    print("   P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))")
    
    print("\n5. These concepts are fundamental to all graphical models:")
    print("   - Bayesian networks (next module)")
    print("   - Markov random fields")
    print("   - Factor graphs")
    print("   - And many more...")
    
    print("\n" + "=" * 70)
    print("Next: Learn how to build and use Bayesian networks!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

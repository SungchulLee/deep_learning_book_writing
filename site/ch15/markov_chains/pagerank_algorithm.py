"""
pagerank_algorithm.py (Module 08)

PageRank Algorithm (Google's Original Algorithm)
=================================================

Location: 06_markov_chain/03_applications/
Difficulty: ⭐⭐⭐ Intermediate
Estimated Time: 3-4 hours

Learning Objectives:
- Understand PageRank as a Markov chain application
- Implement the PageRank algorithm
- Handle teleportation and damping factors
- Rank web pages by importance

Mathematical Foundation:
PageRank models web surfing as a random walk on the web graph:
- States = web pages
- Transitions = clicking links uniformly at random
- With probability α: follow random link
- With probability (1-α): jump to random page (teleportation)

PageRank equation:
PR(p) = (1-α)/N + α × Σ_{q→p} PR(q)/L(q)

where:
- PR(p) = PageRank of page p
- α = damping factor (typically 0.85)
- N = total number of pages
- q→p means page q links to page p
- L(q) = number of outgoing links from q

In matrix form: r = (1-α)/N × e + α × P^T × r
where r is the PageRank vector (stationary distribution)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class PageRank:
    """
    Implementation of PageRank algorithm.
    """
    
    def __init__(self, adjacency_matrix, damping_factor=0.85):
        """
        Initialize PageRank calculator.
        
        Parameters:
            adjacency_matrix (np.ndarray): A[i][j] = 1 if page i links to page j
            damping_factor (float): Probability of following a link (typically 0.85)
        
        Mathematical Setup:
        We construct the Google matrix:
        G = α × P^T + (1-α) × E
        where P^T is the column-stochastic link matrix
        and E is the uniform teleportation matrix
        """
        self.A = np.array(adjacency_matrix, dtype=float)
        self.n_pages = self.A.shape[0]
        self.alpha = damping_factor
        
        # Construct transition matrix
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """
        Build the PageRank transition matrix.
        
        Mathematical Process:
        1. Create link matrix P from adjacency matrix A
           P[i][j] = A[i][j] / L(i) if L(i) > 0
           where L(i) = number of outgoing links from page i
        
        2. Handle dangling nodes (pages with no outgoing links)
           Replace their rows with uniform distribution
        
        3. Add teleportation:
           G = α × P^T + (1-α)/N × E
           where E is all-ones matrix
        """
        # Count outgoing links for each page
        out_degrees = self.A.sum(axis=1)
        
        # Create transition matrix P
        # P[i][j] = probability of going from i to j by following links
        P = np.zeros_like(self.A)
        
        for i in range(self.n_pages):
            if out_degrees[i] > 0:
                # Normalize by number of outgoing links
                P[i, :] = self.A[i, :] / out_degrees[i]
            else:
                # Dangling node: uniform distribution
                P[i, :] = 1.0 / self.n_pages
        
        # Transpose to get column-stochastic matrix
        P_T = P.T
        
        # Add teleportation (Google matrix)
        # G = α × P^T + (1-α)/N × E
        E = np.ones((self.n_pages, self.n_pages)) / self.n_pages
        self.G = self.alpha * P_T + (1 - self.alpha) * E
    
    def compute_pagerank_power_iteration(self, max_iter=100, tol=1e-8):
        """
        Compute PageRank using power iteration method.
        
        Parameters:
            max_iter (int): Maximum iterations
            tol (float): Convergence tolerance
        
        Returns:
            tuple: (pagerank vector, number of iterations)
        
        Mathematical Method:
        Power iteration: r^{(k+1)} = G × r^{(k)}
        Start with r^{(0)} = 1/N × e (uniform distribution)
        Iterate until convergence: ||r^{(k+1)} - r^{(k)}|| < tol
        """
        # Initialize with uniform distribution
        r = np.ones(self.n_pages) / self.n_pages
        
        for iteration in range(max_iter):
            r_new = self.G @ r
            
            # Check convergence
            if np.linalg.norm(r_new - r, ord=1) < tol:
                return r_new, iteration + 1
            
            r = r_new
        
        return r, max_iter
    
    def compute_pagerank_eigenvector(self):
        """
        Compute PageRank using eigenvector method.
        
        Returns:
            np.ndarray: PageRank vector
        
        Mathematical Method:
        PageRank is the dominant eigenvector of G:
        G × r = λ × r where λ = 1
        
        We find the eigenvector corresponding to eigenvalue 1
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.G.T)
        
        # Find eigenvector with eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        r = np.real(eigenvectors[:, idx])
        
        # Normalize to sum to 1
        r = r / r.sum()
        
        return r
    
    def rank_pages(self, pagerank_vector, page_names=None):
        """
        Rank pages by PageRank score.
        
        Parameters:
            pagerank_vector (np.ndarray): PageRank scores
            page_names (list): Optional page names
        
        Returns:
            list: Sorted list of (page, score) tuples
        """
        if page_names is None:
            page_names = [f"Page {i}" for i in range(self.n_pages)]
        
        # Create list of (page, score) tuples
        page_scores = list(zip(page_names, pagerank_vector))
        
        # Sort by score (descending)
        page_scores.sort(key=lambda x: x[1], reverse=True)
        
        return page_scores


def example_simple_web_graph():
    """
    Example 1: Simple web graph with 4 pages.
    
    Link structure:
    A → B, C
    B → C
    C → A
    D → A, B, C  (D is authority page)
    """
    print("=" * 70)
    print("Example 1: Simple Web Graph (4 Pages)")
    print("=" * 70)
    
    # Adjacency matrix
    # A[i][j] = 1 if page i links to page j
    pages = ['A', 'B', 'C', 'D']
    A = np.array([
        [0, 1, 1, 0],  # A links to B, C
        [0, 0, 1, 0],  # B links to C
        [1, 0, 0, 0],  # C links to A
        [1, 1, 1, 0]   # D links to A, B, C
    ])
    
    print("\nAdjacency Matrix:")
    print(f"{'':5s} " + " ".join(f"{p:3s}" for p in pages))
    for i, page in enumerate(pages):
        row = " ".join(f"{int(A[i,j]):3d}" for j in range(len(pages)))
        print(f"{page:5s} {row}")
    
    print("\nLink structure:")
    for i, page_i in enumerate(pages):
        links_to = [pages[j] for j in range(len(pages)) if A[i,j] == 1]
        if links_to:
            print(f"  {page_i} → {', '.join(links_to)}")
    
    # Compute PageRank
    pr = PageRank(A, damping_factor=0.85)
    
    print("\n" + "-" * 70)
    print("Computing PageRank...")
    
    # Method 1: Power iteration
    r_power, iterations = pr.compute_pagerank_power_iteration()
    print(f"\nPower iteration (converged in {iterations} iterations):")
    for page, score in zip(pages, r_power):
        print(f"  {page}: {score:.6f}")
    
    # Method 2: Eigenvector
    r_eig = pr.compute_pagerank_eigenvector()
    print(f"\nEigenvector method:")
    for page, score in zip(pages, r_eig):
        print(f"  {page}: {score:.6f}")
    
    # Rank pages
    print("\n" + "-" * 70)
    print("Page Rankings:")
    ranked = pr.rank_pages(r_power, pages)
    for rank, (page, score) in enumerate(ranked, 1):
        print(f"  {rank}. {page}: {score:.6f}")
    
    print("\nInterpretation:")
    print("  Page A has highest PageRank because:")
    print("  - It's linked by C (which is linked by B and A)")
    print("  - It's linked by D (authority page)")


def example_damping_factor_effect():
    """
    Example 2: Effect of damping factor on PageRank.
    
    Shows how α affects the ranking.
    """
    print("\n" + "=" * 70)
    print("Example 2: Effect of Damping Factor")
    print("=" * 70)
    
    pages = ['A', 'B', 'C', 'D']
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 0]
    ])
    
    print("\nComparing different damping factors:")
    print(f"{'α':<8s} " + " ".join(f"{p:>12s}" for p in pages))
    
    for alpha in [0.5, 0.75, 0.85, 0.95]:
        pr = PageRank(A, damping_factor=alpha)
        r, _ = pr.compute_pagerank_power_iteration()
        
        row = " ".join(f"{score:12.6f}" for score in r)
        print(f"{alpha:<8.2f} {row}")
    
    print("\nObservation:")
    print("  Higher α → more influence from link structure")
    print("  Lower α → closer to uniform distribution")


def example_larger_network():
    """
    Example 3: Larger network with 8 pages.
    
    Demonstrates ranking in a more complex structure.
    """
    print("\n" + "=" * 70)
    print("Example 3: Larger Network (8 Pages)")
    print("=" * 70)
    
    pages = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Create more complex link structure
    A = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],  # A → B, C
        [1, 0, 1, 1, 0, 0, 0, 0],  # B → A, C, D
        [0, 0, 0, 1, 1, 0, 0, 0],  # C → D, E
        [0, 0, 0, 0, 1, 1, 0, 0],  # D → E, F
        [0, 0, 0, 0, 0, 1, 1, 0],  # E → F, G
        [0, 0, 0, 0, 0, 0, 1, 1],  # F → G, H
        [0, 0, 0, 0, 0, 0, 0, 1],  # G → H
        [1, 0, 0, 0, 0, 0, 0, 0]   # H → A (creates cycle)
    ])
    
    pr = PageRank(A, damping_factor=0.85)
    r, iterations = pr.compute_pagerank_power_iteration()
    
    print(f"\nPageRank scores (converged in {iterations} iterations):")
    ranked = pr.rank_pages(r, pages)
    
    for rank, (page, score) in enumerate(ranked, 1):
        bar = '█' * int(score * 500)
        print(f"  {rank}. {page}: {score:.6f} {bar}")


def visualize_pagerank():
    """
    Visualize PageRank using network graph.
    """
    print("\n" + "=" * 70)
    print("Creating PageRank Visualization")
    print("=" * 70)
    
    pages = ['A', 'B', 'C', 'D', 'E']
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ])
    
    pr = PageRank(A, damping_factor=0.85)
    r, _ = pr.compute_pagerank_power_iteration()
    
    # Create network graph
    G = nx.DiGraph()
    for i, page in enumerate(pages):
        G.add_node(page)
    
    for i in range(len(pages)):
        for j in range(len(pages)):
            if A[i,j] == 1:
                G.add_edge(pages[i], pages[j])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Network with PageRank sizes
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42)
    
    # Node sizes proportional to PageRank
    node_sizes = [r[i] * 5000 for i in range(len(pages))]
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes,
           node_color='lightblue', font_size=12, font_weight='bold',
           arrows=True, arrowsize=20, edge_color='gray', width=2)
    
    ax.set_title('Web Graph (Node size = PageRank)', fontsize=13)
    
    # Plot 2: PageRank bar chart
    ax = axes[1]
    
    ranked = pr.rank_pages(r, pages)
    pages_sorted = [p for p, _ in ranked]
    scores_sorted = [s for _, s in ranked]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pages)))
    bars = ax.barh(pages_sorted, scores_sorted, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('PageRank Score', fontsize=12)
    ax.set_title('PageRank Rankings', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, score in zip(bars, scores_sorted):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{score:.4f}',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pagerank.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("PageRank visualization saved")


def main():
    """
    Run all PageRank examples.
    """
    print("PAGERANK ALGORITHM")
    print("==================\n")
    
    example_simple_web_graph()
    example_damping_factor_effect()
    example_larger_network()
    visualize_pagerank()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. PageRank = stationary distribution of random surfer")
    print("2. Damping factor (typically 0.85) balances link-following and teleportation")
    print("3. Pages with many incoming links from important pages rank higher")
    print("4. Power iteration typically converges in ~50-100 iterations")


if __name__ == "__main__":
    main()

"""
Dijkstra with Heap
"""

class Graph:
    
    def __init__(self, num_vertices):
        """
        graph with given number of vertices
        num_vertices : number of vertices
        """
        self.num_vertices = num_vertices 
        self.edges = {i : [] for i in range(self.num_vertices)}
        self.dist = [None] * self.num_vertices
        self.not_selected = set(range(self.num_vertices))
        self.src = None
        self.selected_vertex = None
 
    def addEdge(self, u, v, w):
        """
        add a directional edge from u to v with weight w
        u : start vertex
        v : end vertex
        w : weight, which can be negative
        """
        self.edges[u].append((v, w))
            
    def initializeDistance(self, src):
        self.src = src
        self.dist = [float("inf")] * self.num_vertices
        self.dist[self.src] = 0
        
    def printShortestDistance(self):
        print(f"Distance from Source {self.src}")
        for i in range(self.num_vertices):
            print(f"{i}\t\t{self.dist[i]:>2}")
                
    def relaxEdge(self, u, v, w):
        if self.dist[u] != float("Inf") and self.dist[u] + w < self.dist[v]:
            self.dist[v] = self.dist[u] + w
     
    def runDijkstra(self, src):
        """
        finds shortest distances from src to all other vertices using Bellman-Ford algorithm
        """
        # Step 1: Initialize distances from src to all other vertices as infinite.
        # dist from src to src is 0.
        self.initializeDistance(src)
        
        # Step 2: Relax all edges |V| - 1 times. 
        # A shortest path from src to any other vertex can have at-most |V| - 1 edges.
        # These edges can be stabilized in relaxations of |V| - 1 times.
        for _ in range(self.num_vertices - 1):
            self.selectVertexNearSourceFromNotSelected()              
            self.not_selected.remove(self.selected_vertex)       
            for v, w in self.edges[self.selected_vertex]:
                self.relaxEdge(self.selected_vertex, v, w)
                
    def selectVertexNearSourceFromNotSelected(self):
        min_dist_over_non_selected = float('inf')
        for u in self.not_selected:
            try:
                dist = min([w for (v, w) in self.edges[u]])
            except ValueError: # if there is no edge
                continue # we skip
            if dist < min_dist_over_non_selected:
                self.selected_vertex = u
                min_dist_over_non_selected = dist
        

def main():
    g = Graph(5) # number of vertices is 5
    
    g.addEdge(0, 1, 1) # add some edges from u to v with weights w
    g.addEdge(0, 2, 4) # add some edges from u to v with weights w
    g.addEdge(1, 2, 3) # add some edges from u to v with weights w
    g.addEdge(1, 3, 2) # add some edges from u to v with weights w
    g.addEdge(1, 4, 2) # add some edges from u to v with weights w
    g.addEdge(3, 2, 5) # add some edges from u to v with weights w
    g.addEdge(3, 1, 1) # add some edges from u to v with weights w
    g.addEdge(4, 3, 3) # add some edges from u to v with weights w

    g.runDijkstra(0) # run Dijkstra with src 0
    
    g.printShortestDistance() # print shorest distance from src
    
    

# === Main ===
if __name__ == "__main__":
    main()

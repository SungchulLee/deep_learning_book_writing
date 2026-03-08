# Bellman-Ford Code

```python
class Graph:
    
    def __init__(self, num_vertices):
        """
        graph with given number of vertices
        num_vertices : number of vertices
        """
        self.num_vertices = num_vertices 
        self.edges = []
        self.dist = [None] * self.num_vertices
        self.src = None
 
    def addEdge(self, u, v, w):
        """
        add a directional edge from u to v with weight w
        u : start vertex
        v : end vertex
        w : weight, which can be negative
        """
        self.edges.append([u, v, w])
            
    def hasNoNegativeCycle(self):
        """
        Bellman Ford computes all shortest distances from the give source 
        in self.num_vertices - 1 steps of relaxation
        if graph doesn't contain negative weight cycle. 
        If we can relax further, then there is a negative weight cycle.
        """
        for u, v, w in self.edges:
            if self.dist[u] != float("Inf") and self.dist[u] + w < self.dist[v]:
                return False # there is a negative cycle
        return True # no negative cycle
            
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
     
    def runBellmanFord(self, src):
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
            for u, v, w in self.edges:
                self.relaxEdge(u, v, w)
        

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

    g.runBellmanFord(0) # run BellmanFord with src 0
    
    if g.hasNoNegativeCycle(): 
        g.printShortestDistance() # print shorest distance from src
    else:
        print("There is a negative cycle and hence no solution.")
    
    
if __name__ == "__main__":
    main()
```

**Output:**
```
Distance from Source 0
0		 0
1		 1
2		 4
3		 3
4		 3
```

# Reference

[[알고리즘] 제16-1강 최단경로(shortest path problem)](https://www.youtube.com/watch?v=QH-Btq8SgLQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=38)

[3.6 Dijkstra Algorithm - Single Source Shortest Path - Greedy Method](https://www.youtube.com/watch?v=XB4MIexjvY0&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=45)

[Bellman–Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)

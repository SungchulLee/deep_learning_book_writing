# Adjacency List


$$
\text{Graph Representation}\left\{\begin{array}{l}
\text{Adjacency Matrix}\\
\text{Incidence Matrix}\\
\text{Adjacency List}\\
\text{Edge List}\\
\end{array}\right.
$$


```python
# implementation of an undirected graph using Adjacency Lists
class Vertex:
	def __init__(self, n):
		self.name = n
		self.neighbors = list()
	
	def add_neighbor(self, v, weight):
		if v not in self.neighbors:
			self.neighbors.append((v, weight))
			self.neighbors.sort()

class Graph:
	vertices = {}
	
	def add_vertex(self, vertex):
		if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
			self.vertices[vertex.name] = vertex
			return True
		else:
			return False
	
	def add_edge(self, u, v, weight=0):
		if u in self.vertices and v in self.vertices:
			# my YouTube video shows a silly for loop here, but this is a much faster way to do it
			self.vertices[u].add_neighbor(v, weight)
			self.vertices[v].add_neighbor(u, weight)
			return True
		else:
			return False
			
	def print_graph(self):
		for key in sorted(list(self.vertices.keys())):
			print(key + str(self.vertices[key].neighbors))

            
def main():
    g = Graph()
    # print(str(len(g.vertices)))
    a = Vertex('A')
    g.add_vertex(a)
    g.add_vertex(Vertex('B'))
    for i in range(ord('A'), ord('K')):
        g.add_vertex(Vertex(chr(i)))

    edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ', 'GJ', 'HI']
    for edge in edges:
        g.add_edge(edge[:1], edge[1:])

    g.print_graph()
    
    
if __name__ == "__main__":
    main()
```

**Output:**
```
A[('B', 0), ('E', 0)]
B[('A', 0), ('F', 0)]
C[('G', 0)]
D[('E', 0), ('H', 0)]
E[('A', 0), ('D', 0), ('H', 0)]
F[('B', 0), ('G', 0), ('I', 0), ('J', 0)]
G[('C', 0), ('F', 0), ('J', 0)]
H[('D', 0), ('E', 0), ('I', 0)]
I[('F', 0), ('H', 0)]
J[('F', 0), ('G', 0)]
```


# Reference

Python: 2 Ways to Represent GRAPHS [youtube](https://www.youtube.com/watch?v=HDUzBEG1GlA&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=7) [graph_adjacency-list.py](https://github.com/joeyajames/Python/blob/master/graph_adjacency-list.py)

[Graph Representation in Data Structure | C++ Java Python3](https://www.youtube.com/watch?v=TDXDhcSl0UM&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=2)

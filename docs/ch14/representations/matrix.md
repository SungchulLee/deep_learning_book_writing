# Adjacency Matrix


$$
\text{Graph Representation}\left\{\begin{array}{l}
\text{Adjacency Matrix}\\
\text{Incidence Matrix}\\
\text{Adjacency List}\\
\text{Edge List}\\
\end{array}\right.
$$


```python
# implementation of an undirected graph using Adjacency Matrix, 
# with weighted or unweighted edges
# its definitely work
class Vertex:
	def __init__(self, n):
		self.name = n

class Graph:
	vertices = {}
	edges = []
	edge_indices = {}
	
	def add_vertex(self, vertex):
		if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
			self.vertices[vertex.name] = vertex
			for row in self.edges:
				row.append(0)
			self.edges.append([0] * (len(self.edges)+1))
			self.edge_indices[vertex.name] = len(self.edge_indices)
			return True
		else:
			return False
	
	def add_edge(self, u, v, weight=1):
		if u in self.vertices and v in self.vertices:
			self.edges[self.edge_indices[u]][self.edge_indices[v]] = weight
			self.edges[self.edge_indices[v]][self.edge_indices[u]] = weight
			return True
		else:
			return False
			
	def print_graph(self):
		for v, i in sorted(self.edge_indices.items()):
			print(v + ' ', end='')
			for j in range(len(self.edges)):
				print(self.edges[i][j], end='')
			print(' ')    

            
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
A 0100100000 
B 1000010000 
C 0000001000 
D 0000100100 
E 1001000100 
F 0100001011 
G 0010010001 
H 0001100010 
I 0000010100 
J 0000011000
```


# Reference

Python: 2 Ways to Represent GRAPHS [youtube](https://www.youtube.com/watch?v=HDUzBEG1GlA&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=7) [graph_adjacency-matrix.py](https://github.com/joeyajames/Python/blob/master/graph_adjacency-matrix.py)

[Graph Representation in Data Structure | C++ Java Python3](https://www.youtube.com/watch?v=TDXDhcSl0UM&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=2)

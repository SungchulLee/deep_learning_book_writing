# 이웃 행렬
$$
\text{Graph Representation}\left\{\begin{array}{l}
\text{Adjacency Matrix}\\
\text{Incidence Matrix}\\
\text{Adjacency List}\\
\text{Edge List}\\
\end{array}\right.
$$

```python
# 이웃 행렬을 쓴 무방향 그래프 구현,
# 무게 있는 변과 무게 없는 변 모두
# 확실히 굴러간다
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

**출력:**
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

# 참고 문헌

Python: 2 Ways to Represent GRAPHS [youtube](https://www.youtube.com/watch?v=HDUzBEG1GlA&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=7) [graph_adjacency-matrix.py](https://github.com/joeyajames/Python/blob/master/graph_adjacency-matrix.py)

[Graph Representation in Data Structure | C++ Java Python3](https://www.youtube.com/watch?v=TDXDhcSl0UM&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=2)

## 연습문제

**연습문제 1.**
무방향 그래프의 이웃 행렬이 대칭임을 증명하여라.

??? success "연습문제 1 풀이"
    무방향 그래프에서 변 $\{u, v\}$은 $(u, v)$과 $(v, u)$이 모두 있음을 뜻한다. 그러므로 $A[u][v] = 1$일 때 그리고 그때만 $A[v][u] = 1$이며, 따라서 $A = A^\top$이다. $\square$

---

**연습문제 2.**
이웃 행렬 $A$에서 항목 $(A^k)_{ij}$이 꼭짓점 $i$에서 꼭짓점 $j$까지 길이 $k$인 걸음의 개수를 셈을 보여라.

??? success "연습문제 2 풀이"
    **바탕 경우**($k = 1$): $(A^1)_{ij} = A_{ij}$이며, $i$에서 $j$로 가는 변이 있으면(길이 1인 걸음) 1, 아니면 0이다.

    **귀납 걸음**: $(A^{k-1})_{ij}$이 길이 $k-1$인 걸음을 센다고 놓자. 그러면 $(A^k)_{ij} = \sum_{m} (A^{k-1})_{im} \cdot A_{mj}$이다. 항마다 $i$에서 $m$까지 길이 $k-1$인 걸음을 세고 거기에 변 $(m, j)$을 이어 붙인 것이다. 가운데 꼭짓점 $m$ 전부에 걸쳐 합하면 $i$에서 $j$까지 길이 $k$인 걸음의 총 개수가 나온다. $\square$

---

**연습문제 3.**
그래프가 성길 때($|E| = O(|V|)$) 무게 있는 그래프를 이웃 행렬로 저장할 때와 이웃 목록으로 저장할 때의 공간 복잡도는 얼마인가?

??? success "연습문제 3 풀이"
    이웃 행렬은 변의 개수와 상관없이 늘 $O(|V|^2)$ 공간을 쓴다. $|E| = O(|V|)$인 성긴 그래프에서 이웃 목록은 $O(|V| + |E|) = O(|V|)$ 공간을 쓴다. 행렬은 항목 $O(|V|^2) - O(|V|) \approx O(|V|^2)$개를 0으로 채우며 헤프게 쓴다. 이를테면 꼭짓점 1000개의 나무는 변이 999개만 필요한데(목록 항목 $\approx 2000$개) 행렬은 항목이 $1{,}000{,}000$개이다.

---

**연습문제 4.**
(이웃 행렬로 주어진) 그래프에 제 고리가 있는지 가려내는 함수를 구현하여라.

??? success "연습문제 4 풀이"
    ```python
    def has_self_loop(matrix):
        for i in range(len(matrix)):
            if matrix[i][i] != 0:
                return True
        return False
    ```
    이 함수는 행렬의 대각선 항목을 $O(n)$ 시간에 살핀다. 꼭짓점 $i$의 제 고리는 $A[i][i] \neq 0$을 뜻한다.

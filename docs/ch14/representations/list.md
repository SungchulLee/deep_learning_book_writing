# 이웃 목록
$$
\text{Graph Representation}\left\{\begin{array}{l}
\text{Adjacency Matrix}\\
\text{Incidence Matrix}\\
\text{Adjacency List}\\
\text{Edge List}\\
\end{array}\right.
$$

```python
# 이웃 목록을 쓴 무방향 그래프 구현
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
			# 내 유튜브 영상에서는 여기에 어수룩한 for 되풀이를 쓰지만 이 길이 훨씬 빠르다
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

**출력:**
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

# 참고 문헌

Python: 2 Ways to Represent GRAPHS [youtube](https://www.youtube.com/watch?v=HDUzBEG1GlA&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=7) [graph_adjacency-list.py](https://github.com/joeyajames/Python/blob/master/graph_adjacency-list.py)

[Graph Representation in Data Structure | C++ Java Python3](https://www.youtube.com/watch?v=TDXDhcSl0UM&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=2)

## 연습문제

**연습문제 1.**
꼭짓점 $n$개와 변 $m$개인 무방향 그래프의 이웃 목록 표현이 주어졌을 때, 기억 공간을 모두 얼마나 쓰는가? 답을 뒷받침하여라.

??? success "연습문제 1 풀이"
    꼭짓점마다 제 이웃의 목록을 갖는다. 무방향 그래프에서 변 $(u,v)$은 두 번 나타난다. $u$의 목록에 한 번, $v$의 목록에 한 번이다. 저장은 목록 머리 $n$개와 이웃 항목 $2m$개로 공간이 $O(n + m)$이다.

---

**연습문제 2.**
이웃 목록을 이웃 행렬로 바꾸는 함수를 구현하여라.

??? success "연습문제 2 풀이"
    ```python
    def adj_list_to_matrix(adj, n):
        matrix = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                matrix[u][v] = 1
        return matrix
    ```
    모든 변을 한 번 훑으므로 바꾸는 데 $O(n + m)$ 시간이 들고, 행렬에 $O(n^2)$ 공간을 쓴다.

---

**연습문제 3.**
이웃 목록에서 변 $(u, v)$이 있는지 살피는 시간 복잡도는 얼마인가? 이를 어떻게 나아지게 할 수 있는가?

??? success "연습문제 3 풀이"
    기본 이웃 목록은 이웃을 파이썬 목록에 담으므로 변이 있는지 살피려면 $u$의 이웃 목록을 훑어야 하고 $O(\deg(u))$ 시간이 든다. 이웃 목록을 해시 집합으로 바꾸면 고르게 나눠 $O(1)$으로 나아진다. 대신 상수 인자가 조금 커지고 해시 표 때문에 기억 공간이 더 든다.

---

**연습문제 4.**
무방향 그래프에서 모든 이웃 목록의 길이의 합이 $2|E|$임을 증명하여라.

??? success "연습문제 4 풀이"
    무방향 변 $\{u, v\}$마다 $u$의 이웃 목록에 항목 하나, $v$의 이웃 목록에 항목 하나를 보태므로 변마다 항목이 2개이다. 모든 꼭짓점에 걸쳐 합하면 악수 보조정리에 따라 $\sum_{v \in V} |\text{adj}[v]| = \sum_{v \in V} \deg(v) = 2|E|$이다. $\square$

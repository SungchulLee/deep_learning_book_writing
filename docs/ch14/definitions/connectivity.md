# 이어짐
```python
from functools import lru_cache as cache
```

```python
class Graph:
    
    def __init__(self, edges):
        self.edges = edges
        self.dict = self.edges_to_dict() 
        
    def edges_to_dict(self): 
        graph_dict = {}
        for start, end in self.edges:
            if start in graph_dict:
                graph_dict[start].append(end)
            else:
                graph_dict[start] = [end]
        return graph_dict 

    @cache
    def get_paths(self, start, end):
        if start == end: return [[start]]
        if start not in self.dict: return [[]]
        paths = []
        for node in self.dict[start]:
            new_paths = self.get_paths(node, end)
            for p in new_paths:
                paths.append([start]+p)
        return paths
    
    @cache
    def get_shortest_path(self, start, end):
        if start == end: return [start]
        if start not in self.dict: return []
        for idx, node in enumerate(self.dict[start]):
            path_segment = self.get_shortest_path(node, end)
            path_segment_length = len(path_segment)
            if idx==0:
                shortest_path_segment = path_segment
                shortest_path_segment_length = path_segment_length
            else:
                if path_segment_length < shortest_path_segment_length:
                    shortest_path_segment = path_segment
                    shortest_path_segment_length = path_segment_length
        return [start] + shortest_path_segment
```

```python
if __name__ == '__main__':

    routes = [
        ("Mumbai", "Paris"),
        ("Mumbai", "Dubai"),
        ("Paris", "Dubai"),
        ("Paris", "New York"),
        ("Dubai", "New York"),
        ("New York", "Toronto"),
    ]
    
    route_graph = Graph(routes)

    start = "Mumbai"
    end = "New York"
    print(f"All paths between: {start} and {end}: ", route_graph.get_paths(start, end))
    print(f"Shortest path between {start} and {end}: ", route_graph.get_shortest_path(start,end))
```

**출력:**
```
All paths between: Mumbai and New York:  [['Mumbai', 'Paris', 'Dubai', 'New York'], ['Mumbai', 'Paris', 'New York'], ['Mumbai', 'Dubai', 'New York']]
Shortest path between Mumbai and New York:  ['Mumbai', 'Paris', 'New York']
```

# 참고 문헌

[Graph Introduction - Data Structures & Algorithms Tutorials In Python #12](https://www.youtube.com/watch?v=j0IYCyBdzfA&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=12)

## 연습문제

**연습문제 1.**
$G$을 꼭짓점 6개의 무방향 그래프라 하자. $G$이 이어져 있음을 보장하는 데 필요한 변의 최소 개수는 얼마인가?

??? success "연습문제 1 풀이"
    꼭짓점 $n$개의 이어진 그래프에는 적어도 변 $n - 1$개가 필요하다(뻗음 나무). $n = 6$이면 최소는 변 5개이다. 그러나 변 5개로도 이어지지 않은 그래프를 만들 수 있다(이를테면 5-고리에 외톨이 꼭짓점 하나). 이어짐을 *보장하는* 변의 최소 개수는 $\binom{n-1}{2} + 1 = \binom{5}{2} + 1 = 11$이다. 꼭짓점 $n$개에 변이 $\binom{n-1}{2}$개보다 많으면 반드시 이어지기 때문이다(외톨이 꼭짓점 하나에 꼭짓점 $n-1$개의 완전 그래프를 더한 것의 변이 꼭 $\binom{n-1}{2}$개이다).

---

**연습문제 2.**
이어진 그래프 $G$에서 가장 긴 길 짝마다 적어도 꼭짓점 하나를 함께 가짐을 증명하여라.

??? success "연습문제 2 풀이"
    어긋냄을 위해 $P_1$과 $P_2$이 꼭짓점을 하나도 함께 갖지 않는 가장 긴 길 둘이라고 하자. $G$이 이어져 있으므로 $P_1$ 위 어떤 꼭짓점에서 $P_2$ 위 어떤 꼭짓점까지 길 $Q$이 있다. $P_1$, $Q$, $P_2$의 일부를 이어 붙이면 $P_1$과 $P_2$ 둘보다 긴 길을 지을 수 있어 그것들이 가장 긴 길이라는 가정과 어긋난다. $\square$

---

**연습문제 3.**
이웃 목록으로 나타낸 무방향 그래프의 이어진 덩이 개수를 셈하는 함수를 적어라.

??? success "연습문제 3 풀이"
    ```python
    from collections import deque

    def count_components(adj, n):
        visited = [False] * n
        components = 0
        for start in range(n):
            if not visited[start]:
                components += 1
                queue = deque([start])
                visited[start] = True
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
        return components
    ```
    BFS마다 이어진 덩이 하나를 살펴본다. 전체 시간은 $O(V + E)$이다.

---

**연습문제 4.**
이어진 그래프의 꼭짓점 $v$을 지웠을 때 그래프가 끊어지면 $v$을 **자르는 꼭짓점**이라 한다. 꼭짓점이 $n \geq 3$개이고 자르는 꼭짓점이 없는 이어진 그래프에는 고리가 있음을 보여라.

??? success "연습문제 4 풀이"
    $G$에 자르는 꼭짓점이 없으면 $G$은 2-이어짐(두 겹 이어짐)이다. 꼭짓점이 적어도 3개인 2-이어짐 그래프는 꼭짓점 짝 $u, w$마다 안쪽 꼭짓점이 겹치지 않는 길 둘이 $u$에서 $w$까지 있음을 만족한다. 이 두 길을 이어 붙이면 고리가 된다. $\square$

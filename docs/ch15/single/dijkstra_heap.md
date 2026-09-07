# 힙을 쓴 데이크스트라
```python
class Graph:
    
    def __init__(self, num_vertices):
        """
        주어진 꼭짓점 수의 그래프
        num_vertices : 꼭짓점의 개수
        """
        self.num_vertices = num_vertices 
        self.edges = {i : [] for i in range(self.num_vertices)}
        self.dist = [None] * self.num_vertices
        self.not_selected = set(range(self.num_vertices))
        self.src = None
        self.selected_vertex = None
 
    def addEdge(self, u, v, w):
        """
        u에서 v로 무게 w인 방향 변 더하기
        u : 시작 꼭짓점
        v : 끝 꼭짓점
        w : 무게, 음수일 수 있다
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
        벨먼-포드 알고리즘으로 src에서 다른 모든 꼭짓점까지의 최단 거리를 찾는다
        """
        # 걸음 1: src에서 다른 모든 꼭짓점까지의 거리를 무한으로 잡기.
        # src에서 src까지의 거리는 0이다.
        self.initializeDistance(src)
        
        # 걸음 2: 모든 변을 |V| - 1번 늦추기.
        # src에서 다른 어느 꼭짓점으로 가는 최단 경로의 변은 많아야 |V| - 1개다.
        # 이 변들은 |V| - 1번의 늦추기로 안정될 수 있다.
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
            except ValueError: # 변이 없으면
                continue # 건너뜀
            if dist < min_dist_over_non_selected:
                self.selected_vertex = u
                min_dist_over_non_selected = dist
        

def main():
    g = Graph(5) # 꼭짓점 수가 5
    
    g.addEdge(0, 1, 1) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(0, 2, 4) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(1, 2, 3) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(1, 3, 2) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(1, 4, 2) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(3, 2, 5) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(3, 1, 1) # u에서 v로 무게 w인 변 몇 개 더하기
    g.addEdge(4, 3, 3) # u에서 v로 무게 w인 변 몇 개 더하기

    g.runDijkstra(0) # src 0으로 데이크스트라 돌리기
    
    g.printShortestDistance() # src에서의 최단 거리 찍기
    
    
if __name__ == "__main__":
    main()
```

**출력:**
```
Distance from Source 0
0		 0
1		 1
2		 4
3		 3
4		 3
```

# 참고 문헌

[3.6 Dijkstra Algorithm - Single Source Shortest Path - Greedy Method](https://www.youtube.com/watch?v=XB4MIexjvY0&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=45)

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

[Dijkstra algorithm | Single source shortest path algorithm](https://www.youtube.com/watch?v=Sj5Z-jaE2x0)

[Dijkstra algorithm | Code implementation](https://www.youtube.com/watch?v=t2d-XYuPfg0)

[743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)

[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

## 연습문제

**연습문제 1.**
꼭짓점 4개와 변 $A \xrightarrow{2} B$, $A \xrightarrow{5} C$, $B \xrightarrow{1} C$, $B \xrightarrow{4} D$, $C \xrightarrow{1} D$을 갖는 그래프에서 꼭짓점 $A$부터 이 알고리즘이 굴러가는 것을 따라가라.

??? success "연습문제 1 풀이"
    처음 거리: $d(A) = 0$, $d(B) = \infty$, $d(C) = \infty$, $d(D) = \infty$.

    $A$에서 다루기: $d(B) = 2$, $d(C) = 5$.
    $B$에서 다루기(잠정 거리가 가장 작다): $d(C) = \min(5, 2+1) = 3$, $d(D) = 2+4 = 6$.
    $C$에서 다루기: $d(D) = \min(6, 3+1) = 4$.
    $D$에서 다루기: 새로 고침 없음.

    $A$에서의 마지막 최단 거리: $d(A)=0, d(B)=2, d(C)=3, d(D)=4$.

---

**연습문제 2.**
그래프에 음의 변 무게가 있으면 데이크스트라 알고리즘이 왜 무너지는지 설명하여라. 구체적인 어긋냄 보기를 들어라.

??? success "연습문제 2 풀이"
    데이크스트라 알고리즘은 앞으로 어떤 늦추기도 그것을 낫게 할 수 없다고 놓고 잠정 거리가 가장 작은 꼭짓점을 욕심껏 확정한다. 음의 변이 있으면 아직 다루지 않은 꼭짓점을 지나는 더 긴 길이 나중에 더 짧아질 수 있다. 이를테면 변 $A \xrightarrow{1} B$, $A \xrightarrow{3} C$, $C \xrightarrow{-4} B$에서 데이크스트라는 $B$을 거리 1으로 확정하지만, 길 $A \to C \to B$의 거리는 $3 + (-4) = -1 < 1$이다.

---

**연습문제 3.**
이진 힙을 쓴 데이크스트라 알고리즘의 시간 복잡도는 얼마인가? 피보나치 힙을 쓰면?

??? success "연습문제 3 풀이"
    **이진 힙**을 쓰면 최소 꺼내기 $V$번마다 $O(\log V)$, 열쇠 낮추기 $E$번마다 $O(\log V)$이 들어 $O((V + E) \log V)$이다. **피보나치 힙**을 쓰면 최소 꺼내기가 고르게 나눠 $O(\log V)$, 열쇠 낮추기가 고르게 나눠 $O(1)$이라 $O(V \log V + E)$이다. $E = \Theta(V^2)$인 빽빽한 그래프에서는 피보나치 힙이 이론상 낫다.

---

**연습문제 4.**
모든 변을 $|V| - 1$번 늦춘 뒤 벨먼-포드 알고리즘이 최단 경로를 맞게 셈함을 증명하여라.

??? success "연습문제 4 풀이"
    최단 경로 안 변의 개수 $k$에 대한 귀납으로 증명한다. $i$번 되풀이한 뒤 알고리즘은 변을 많아야 $i$개 쓰는 최단 경로를 모두 맞게 셈해 두었다. **바탕:** 0번 되풀이한 뒤 $d(s) = 0$이 맞다(변 0개짜리 길). **걸음:** $i$번 되풀이한 뒤 변이 $\leq i$개인 최단 경로가 모두 맞다고 놓자. $i+1$번째 되풀이에서 변이 $i+1$개인 최단 경로 $s \to \cdots \to u \to v$을 생각하자. $u$까지의 부분 길은 변이 $i$개이고 가정에 따라 맞다. 변 $(u,v)$을 늦추면 $d(v) = d(u) + w(u,v)$이 되고 이것이 가장 좋다. 최단 경로의 변이 많아야 $|V|-1$개이므로(음의 고리가 없다) $|V|-1$번 되풀이하면 넉넉하다. $\square$

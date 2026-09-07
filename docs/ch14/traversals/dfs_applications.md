# DFS의 쓰임새

깊이 우선 찾기는 가장 쓸모가 넓은 그래프 알고리즘 가운데 하나이다. 되돌아오기 전에 되도록 깊이 살펴보는 힘 덕분에 BFS이 따라올 수 없는 성질을 갖는다. 곧 고리를 자연스럽게 알아내고, 그래프 짜임을 드러내는 앞/뒤 차례를 셈하며, 위상 정렬과 강하게 이어진 덩이 알고리즘의 바탕을 놓는다. 이 쪽은 DFS의 핵심 쓰임새를 보인다.

## 앞 차례와 뒤 차례 번호 매기기

많은 그래프 알고리즘이 어느 꼭짓점이 다른 꼭짓점의 조상인지 자손인지 알아야 한다. DFS의 시각 도장이 바로 그 정보를 준다. DFS은 꼭짓점마다 시각 도장 둘을 찍는다. 곧 꼭짓점을 처음 들를 때의 **찾은 때**(앞 차례)와 그 자손을 모두 살펴본 뒤의 **마친 때**(뒤 차례)이다. 이 시각 도장이 DFS 나무의 되돌이 짜임을 담으며 DFS 기반 알고리즘 대부분의 열쇠가 된다.

꼭짓점이 $n$개인 방향 그래프에서 앞 번호와 뒤 번호는 $1$부터 $2n$까지이며, 꼭짓점 짝 $u$과 $v$마다 구간 $[\text{pre}(u), \text{post}(u)]$과 $[\text{pre}(v), \text{post}(v)]$은 서로 겹치지 않거나 한쪽이 다른 쪽을 담는다. 이 **겹쳐 넣기 성질**은 DFS의 되돌이 성격에서 곧바로 따라 나온다. 곧 $v$이 $u$의 자손이면 $v$은 $u$보다 나중에 찾아지고 $u$보다 먼저 마친다.

```python
"""
앞 차례와 뒤 차례 번호를 매기는 DFS.

앞/뒤 시각 도장은 DFS 나무의 조상-자손 관계를 드러내며
고리 알아내기와 위상 정렬을 가능하게 한다.
"""

# === 시각 도장을 찍는 DFS ===================================================

def dfs_timestamps(graph):
    """꼭짓점마다 앞 차례와 뒤 차례 번호를 셈한다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        방향 그래프의 이웃 목록.

    반환값
    -------
    pre : dict[int, int]
        꼭짓점마다 찾은 때.
    post : dict[int, int]
        꼭짓점마다 마친 때.
    """
    pre = {}
    post = {}
    clock = [0]  # 겹친 함수를 위한 바꿀 수 있는 세개

    def explore(u):
        clock[0] += 1
        pre[u] = clock[0]
        for v in graph[u]:
            if v not in pre:
                explore(v)
        clock[0] += 1
        post[u] = clock[0]

    for vertex in graph:
        if vertex not in pre:
            explore(vertex)

    return pre, post


# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    pre, post = dfs_timestamps(graph)
    print("Vertex | Pre | Post")
    print("-------|-----|-----")
    for v in sorted(pre):
        print(f"   {v}   |  {pre[v]}  |  {post[v]}")
```

**출력:**
```
Vertex | Pre | Post
-------|-----|-----
   0   |  1  |  8
   1   |  2  |  5
   2   |  6  |  7
   3   |  3  |  4
```

꼭짓점 0의 구간 $[1, 8]$이 가장 넓으며, 이는 DFS 나무에서 다른 모든 꼭짓점이 그 자손임을 확인해 준다.

## 고리 알아내기

고리를 알아내는 일은 여러 자리에서 꼭 필요하다. 기댐 풀기(이를테면 빌드 얼개, 꾸러미 관리자)는 기댐이 돌고 돌면 무너지고, 많은 알고리즘이 입력에 고리가 없다고 놓는다. DFS은 만나는 변의 짜임으로 자연스러운 고리 검정을 준다.

방향 그래프에 고리가 있을 때 그리고 그때만 DFS이 **되돌이 변**을 만난다. 이는 꼭짓점에서 DFS 나무의 조상으로 가는 변이다. 되돌이 변 $(u, v)$은 $v$이 찾아졌으나(앞 번호가 있으나) 아직 마치지 않았을 때(뒤 번호가 없을 때) 알아본다. 마찬가지로 $v$이 지금 되돌이 더미에 있을 때이다.

```python
"""
DFS으로 방향 그래프에서 고리 알아내기.

(아직 되돌이 더미에 있는 꼭짓점으로 가는) 되돌이 변은 고리가 있음을 알린다.
"""

# === 고리 알아내기 ==========================================================

def has_cycle(graph):
    """방향 그래프에 고리가 있으면 True를 되돌린다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        방향 그래프의 이웃 목록.

    반환값
    -------
    bool
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def explore(u):
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:
                return True  # 되돌이 변 찾음
            if color[v] == WHITE and explore(v):
                return True
        color[u] = BLACK
        return False

    return any(color[v] == WHITE and explore(v) for v in graph)


# === 메인 =====================================================================

if __name__ == "__main__":
    dag = {0: [1, 2], 1: [3], 2: [3], 3: []}
    cyclic = {0: [1], 1: [2], 2: [0]}

    print(f"DAG has cycle? {has_cycle(dag)}")
    print(f"Cyclic graph has cycle? {has_cycle(cyclic)}")
```

**출력:**
```
DAG has cycle? False
Cyclic graph has cycle? True
```

## 위상 정렬 맛보기

일감에 앞서 해야 할 것이 있으면(졸업 전에 들어야 할 과목, 이음 전에 해야 할 컴파일 걸음) 모든 기댐을 지키는 차례가 필요하다. DFS은 방향 비순환 그래프에 우아한 풀이를 준다.

DAG에서 **위상 차례**는 변마다 앞선 꼭짓점에서 뒤선 꼭짓점을 가리키도록 꼭짓점을 늘어놓는다. DFS은 꼭짓점을 뒤 차례의 거꾸로로 적어 위상 차례를 만든다. 꼭짓점 $u$에서 $v$으로 가는 변이 있으면 (그래프에 고리가 없으므로) $u$이 $v$보다 나중에 마치므로, 마친 차례를 뒤집으면 $u$이 $v$보다 앞에 놓인다.

!!! note "온전한 다룸"
    칸의 BFS 기반 판을 아우르는 온전한 위상 정렬 알고리즘은 다음 장의 위상 정렬 마당에서 다룬다.

```python
"""
DFS 뒤 차례의 거꾸로로 하는 위상 정렬.

DAG에서만 올바르며, 여럿일 수 있는 올바른 차례 가운데 하나를 낸다.
"""

# === 위상 정렬 ==============================================================

def topological_sort(graph):
    """DAG의 위상 차례를 되돌린다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        DAG의 이웃 목록.

    반환값
    -------
    list[int]
        위상 차례로 늘어놓은 꼭짓점.
    """
    visited = set()
    order = []

    def explore(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                explore(v)
        order.append(u)

    for vertex in graph:
        if vertex not in visited:
            explore(vertex)

    order.reverse()
    return order


# === 메인 =====================================================================

if __name__ == "__main__":
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"Topological order: {topological_sort(dag)}")
```

**출력:**
```
Topological order: [0, 2, 1, 3, 4]
```

이 차례에서 DAG의 변마다 왼쪽에서 오른쪽을 가리키므로 올바름이 확인된다.

## 무방향 그래프의 이어진 덩이

BFS과 마찬가지로 DFS도 다녀가지 않은 꼭짓점마다 찾기를 띄워 이어진 덩이를 낱낱이 셀 수 있다. 다녀가지 않은 꼭짓점에서의 DFS 부름마다 온전한 덩이 하나를 찾아낸다. DFS 기반 길도 똑같이 $O(V + E)$ 시간에 돌지만 덩이 안에서 꼭짓점을 들르는 차례가 BFS과 다르다. 곧 DFS은 형제를 살펴보기 전에 깊이 파고들고, BFS은 층층이 넓혀 간다.

## 요약

| 쓰임새 | 쓰는 DFS의 핵심 성질 | 시간 |
|---|---|---|
| 앞/뒤 번호 매기기 | DFS의 되돌이 짜임 | $O(V + E)$ |
| 고리 알아내기 | 잿빛 꼭짓점으로 가는 되돌이 변 | $O(V + E)$ |
| 위상 정렬 | DAG에서 뒤 차례의 거꾸로 | $O(V + E)$ |
| 이어진 덩이 | DFS 숲이 그래프를 나눈다 | $O(V + E)$ |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
- Dasgupta, S., Papadimitriou, C., & Vazirani, U. (2006). *Algorithms*, 3-4장.

## 연습문제

**연습문제 1.**
무방향 그래프에서 모든 이음점(자르는 꼭짓점)을 찾는 데 DFS을 쓰는 법을 밝혀라. 시간 복잡도는 얼마인가?

??? success "연습문제 1 풀이"
    DFS을 돌리며 찾은 때 $\text{disc}[v]$과 낮은 값 $\text{low}[v]$($v$의 부분 나무에서 되돌이 변으로 닿을 수 있는 가장 이른 찾은 때)을 셈한다. 다음이면 꼭짓점 $u$이 이음점이다. (1) $u$이 DFS 나무의 뿌리이고 자식이 둘 이상이거나, (2) $u$이 뿌리가 아니고 $\text{low}[v] \geq \text{disc}[u]$인 자식 $v$이 있다($v$의 부분 나무 안 어떤 꼭짓점도 $u$을 거치지 않고는 $u$의 조상에 닿을 수 없다). 셈 전체가 DFS 한 번으로 $O(V + E)$ 시간에 돈다. $\square$

---

**연습문제 2.**
DFS의 뒤 차례를 거꾸로 하면 왜 DAG의 위상 정렬이 되는지 설명하여라. 그래프에 고리가 있으면 왜 무너지는가?

??? success "연습문제 2 풀이"
    DAG에서 DFS을 하면 꼭짓점 $u$이 마칠 때(검게 될 때) $u$에서 닿는 모든 꼭짓점이 이미 마쳤다. 변 $(u, v)$마다 꼭짓점 $v$이 $u$보다 먼저 마친다. 마친 때의 거꾸로 차례로 꼭짓점을 적으면 변 $(u, v)$마다 $u$이 $v$보다 앞에 놓여 올바른 위상 차례가 된다. 그래프에 고리 $v_0 \to v_1 \to \cdots \to v_k \to v_0$이 있으면 올바른 위상 차례가 없다(꼭짓점마다 다음 것보다 앞서야 하므로 어긋난다). DFS은 이를 되돌이 변으로 알아낸다. $\square$

---

**연습문제 3.**
DFS으로 무방향 그래프의 이어진 덩이 개수를 세어라. 알고리즘을 적어라.

??? success "연습문제 3 풀이"
    세개 $c = 0$과 다녀감 배열을 첫걸음 잡는다. 다녀가지 않은 꼭짓점 $v$마다 $c$을 올리고 $v$에서 DFS을 돌려 닿는 꼭짓점을 모두 다녀갔다고 표시한다. 모든 꼭짓점을 다룬 뒤 $c$이 이어진 덩이의 개수와 같다.

    ```python
    def count_components(adj, n):
        visited = [False] * n
        count = 0
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
        for v in range(n):
            if not visited[v]:
                dfs(v)
                count += 1
        return count
    ```

    시간 복잡도: $O(V + E)$. $\square$

---

**연습문제 4.**
방향 그래프의 DFS 숲은 변을 나무 변, 되돌이 변, 앞선 변, 가로 변으로 나눈다. 무방향 그래프의 DFS이 앞선 변이나 가로 변을 만들 수 있는가? 왜 그런가?

??? success "연습문제 4 풀이"
    만들 수 없다. 무방향 그래프의 DFS에서는 나무가 아닌 변이 모두 되돌이 변이다. 꼭짓점 $u$에서 DFS이 이웃 $v$을 살필 때 $v$이 이미 다녀갔고($v$이 $u$의 어버이가 아니고) 그렇다면 $v$은 DFS 나무에서 $u$의 조상이어야 한다. $v$이 다른 가지에 있었다면 DFS이 $v$을 다룰 때 변 $(v, u)$을 살펴 $u$을 $v$의 자손으로 만들었을 것이기 때문이다. 그러므로 무방향 DFS에서는 가로 변과 앞선 변이 생길 수 없다. $\square$

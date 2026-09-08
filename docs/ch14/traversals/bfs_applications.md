# BFS의 쓰임새

너비 우선 찾기는 꼭짓점을 모두 들르는 일만 하지 않는다. BFS이 샘에서의 거리가 커지는 차례로 꼭짓점을 살펴보므로, 근본이 되는 여러 그래프 문제를 자연스럽게 푼다. 이 쪽은 가장 중요한 쓰임새를 보인다. 곧 무게 없는 그래프의 최단 경로, 이분성 검정, 이어진 덩이 찾기, 층 차례 돌아보기이다.

---

## 1. 무게 없는 그래프의 최단 경로

많은 그래프 문제가 꼭짓점 둘 사이의 가장 적은 변을 찾는 일로 줄어든다. 이를테면 사회 망에서 두 사람 사이의 "떨어진 촌수"가 바로 무게 없는 최단 경로이다. BFS은 꼭짓점을 층층이 살펴보므로 이 문제를 가장 좋게 푼다. 곧 꼭짓점 $v$에 처음 닿았을 때 쓴 변의 개수가 가장 적다.

무게 없는 그래프에서는 변마다 값이 같으므로 경로 위 변의 개수가 곧 경로의 길이이다. 가는 길에 꼭짓점마다 앞선 꼭짓점을 적어 두면 실제 최단 경로를 되살릴 수 있다. 시간 복잡도는 $O(V + E)$ 그대로이며, 여기서 $V$은 꼭짓점의 개수, $E$은 변의 개수이다.

```python
"""
BFS으로 무게 없는 그래프에서 최단 경로 찾기.

BFS이 최단 거리를 어떻게 저절로 셈하는지, 그리고 앞선 꼭짓점을
좇으면 어떻게 경로를 되살릴 수 있는지 보인다.
"""

from collections import deque

# === BFS으로 최단 경로 ======================================================

def bfs_shortest_path(graph, source, target):
    """무게 없는 그래프에서 source에서 target까지의 최단 경로를 되돌린다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록 표현.
    source : int
        시작 꼭짓점.
    target : int
        도착 꼭짓점.

    반환값
    -------
    list[int] | None
        최단 경로를 꼭짓점 목록으로, 닿을 수 없으면 None.
    """
    if source == target:
        return [source]

    visited = {source}
    predecessor = {source: None}
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                predecessor[neighbor] = node
                if neighbor == target:
                    # 앞선 꼭짓점을 따라가며 길 되살리기
                    path = []
                    current = target
                    while current is not None:
                        path.append(current)
                        current = predecessor[current]
                    return path[::-1]
                queue.append(neighbor)
    return None

# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2], 4: [2]}
    path = bfs_shortest_path(graph, 0, 3)
    print(f"Shortest path from 0 to 3: {path}")
    print(f"Distance: {len(path) - 1} edges")
```

**출력:**
```
Shortest path from 0 to 3: [0, 1, 3]
Distance: 2 edges
```

앞선 꼭짓점으로 되살리면 줄에 온전한 경로를 담는 대신 $O(V)$ 기억 공간만 쓰므로 전체 공간 복잡도가 $O(V)$으로 유지된다.

---

## 2. 이분성 검정

실전 문제 가운데 많은 것이 물건을 두 무리로 나누되 그 사이에 제약을 두어야 한다. 일 나눠 주기, 두 빛깔로 그래프 칠하기, 부딪힘 없는 일정 짜기가 모두 바탕 그래프가 이분인지에 달렸다. BFS은 이 성질을 선형 시간에 살피는 단순한 검정을 준다.

그래프의 꼭짓점 묶음을 두 무리로 나누어 변마다 한 무리의 꼭짓점과 다른 무리의 꼭짓점을 잇게 할 수 있으면 그 그래프는 **이분**이다. 마찬가지로, 그래프가 이분일 때 그리고 그때만 길이가 홀수인 고리가 없다. BFS 기반 알고리즘은 층을 돌아보며 빛깔을 번갈아 준다. 어떤 변이라도 같은 빛깔의 꼭짓점 둘을 잇는다면 그 그래프는 이분이 아니다.

```python
"""
BFS 두 빛깔 칠하기로 이분성 검정하기.

같은 빛깔의 꼭짓점을 잇는 변이 없도록 BFS이 두 빛깔을
줄 수 있으면 그 그래프는 이분이다.
"""

from collections import deque

# === 이분성 살피기 ===========================================================

def is_bipartite(graph):
    """무방향 그래프가 이분인지 살핀다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록 표현.

    반환값
    -------
    bool
        그래프가 이분이면 True, 아니면 False.
    """
    color = {}
    for start in graph:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True

# === 메인 =====================================================================

if __name__ == "__main__":
    bipartite_graph = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    non_bipartite = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    print(f"Square cycle bipartite? {is_bipartite(bipartite_graph)}")
    print(f"Triangle bipartite? {is_bipartite(non_bipartite)}")
```

**출력:**
```
Square cycle bipartite? True
Triangle bipartite? False
```

네모 고리(꼭짓점 4개가 이루는 고리)는 빛깔 갈래가 $\{0, 2\}$과 $\{1, 3\}$이므로 이분이다. 세모는 꼭짓점 셋이 서로 다 이어져 길이가 홀수인 고리가 생기므로 이분이 아니다.

---

## 3. 이어진 덩이

어느 꼭짓점끼리 서로 닿을 수 있는지 아는 일이 그래프를 뜯어보는 첫걸음일 때가 많다. 이를테면 컴퓨터 망에서 이어진 덩이는 서로 말을 주고받을 수 있는 기계의 무리에 해당한다. BFS은 이 무리를 곧바로 찾아내는 길을 준다.

무방향 그래프에서 **이어진 덩이**는 어느 짝 사이에도 길이 있는, 더 키울 수 없는 꼭짓점 묶음이다. 아직 다녀가지 않은 꼭짓점에서 BFS을 하면 그 덩이 전체를 찾아낸다. 모든 꼭짓점을 훑으며 다녀가지 않은 꼭짓점마다 BFS을 띄우면 모든 덩이를 통틀어 $O(V + E)$ 시간에 낱낱이 셀 수 있다.

```python
"""
BFS으로 이어진 덩이 찾기.

다녀가지 않은 꼭짓점에서의 BFS 부름마다 온전한 덩이 하나를 찾아낸다.
"""

from collections import deque

# === 이어진 덩이 ============================================================

def connected_components(graph):
    """무방향 그래프의 이어진 덩이를 모두 찾는다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록 표현.

    반환값
    -------
    list[list[int]]
        안쪽 목록마다 한 덩이의 꼭짓점을 담는다.
    """
    visited = set()
    components = []

    for vertex in graph:
        if vertex not in visited:
            component = []
            queue = deque([vertex])
            visited.add(vertex)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

    return components

# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1], 1: [0],
        2: [3, 4], 3: [2, 4], 4: [2, 3],
        5: [],
    }
    comps = connected_components(graph)
    for i, comp in enumerate(comps):
        print(f"Component {i}: {comp}")
```

**출력:**
```
Component 0: [0, 1]
Component 1: [2, 3, 4]
Component 2: [5]
```

---

## 4. 층별 순회

경로와 덩이를 찾는 일 말고도, 샘에서의 거리로 꼭짓점을 묶어야 하는 문제라면 BFS이 자연스러운 고름이다. 층 차례 돌아보기는 꼭짓점을 거리 갈래로 또렷이 갈라 주며, 층마다 나무의 깊이에 해당하는 나무 알고리즘에서 특히 쓸모 있다.

```python
"""
BFS으로 층 차례 돌아보기.

샘에서의 거리(층)로 꼭짓점을 묶는다.
"""

from collections import deque

# === 층 차례 돌아보기 ========================================================

def level_order(graph, source):
    """BFS 층으로 묶은 꼭짓점을 되돌린다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록 표현.
    source : int
        시작 꼭짓점.

    반환값
    -------
    list[list[int]]
        안쪽 목록마다 샘에서 그 거리에 있는 꼭짓점을 담는다.
    """
    visited = {source}
    queue = deque([source])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(current_level)

    return levels

# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}
    levels = level_order(graph, 0)
    for depth, lvl in enumerate(levels):
        print(f"Level {depth}: {lvl}")
```

**출력:**
```
Level 0: [0]
Level 1: [1, 2]
Level 2: [3, 4, 5]
```

---

## 5. BFS 쓰임새 간추림

| 쓰임새 | 핵심 생각 | 시간 |
|---|---|---|
| 최단 경로(무게 없음) | 처음 들름 = 가장 적은 변 | $O(V + E)$ |
| 이분성 검정 | BFS 층으로 두 빛깔 칠하기 | $O(V + E)$ |
| 이어진 덩이 | 다녀가지 않은 꼭짓점마다 BFS | $O(V + E)$ |
| 층 차례 돌아보기 | 샘에서의 거리로 묶기 | $O(V + E)$ |

네 쓰임새 모두 BFS 한 번 훑기(또는 그래프를 상수 번 훑기) 위에 곧바로 세워지므로 선형 시간에 돈다.

---

## 연습문제

**연습문제 1.**
BFS으로 나무의 지름(가장 긴 최단 경로)을 찾아라. BFS을 두 번 쓰는 길을 밝히고 그것이 맞음을 증명하여라.

??? success "연습문제 1 풀이"
    (1) 아무 꼭짓점 $s$에서 BFS을 돌리고 $s$에서 가장 먼 꼭짓점을 $u$이라 하자. (2) $u$에서 BFS을 돌리고 $u$에서 가장 먼 꼭짓점을 $v$이라 하자. 거리 $d(u, v)$이 지름이다. **증명**: 실제 지름 짝을 $(a, b)$이라 하자. 걸음 1을 마치면 $u$은 $s$에서 가장 멀다. 어긋냄으로 $d(u, v) \geq d(a, b)$임을 보일 수 있다. 곧 $d(u, v) < d(a, b)$이면 $s$에서 $a$이나 $b$까지의 거리가 $d(s, u)$을 넘어 $u$을 고른 것과 어긋난다. BFS 두 번 모두 $O(V + E)$이 드므로 전체도 $O(V + E)$이다. $\square$

---

**연습문제 2.**
샘 꼭짓점 $s$에서 거리 $k$ 안에 있는 꼭짓점을 모두 찾는 데 BFS을 쓰는 법을 밝혀라. 시간 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    $s$에서 거리를 지키며 BFS을 돌린다. 거리가 $k$을 넘는 꼭짓점은 더 이상 줄에 넣지 않는다. 곧 $d(u) = k$인 꼭짓점 $u$을 꺼낼 때 이웃은 살피되 하나도 줄에 넣지 않는다(거리가 $k + 1$이 되므로). $d(v) \leq k$인 꼭짓점을 모두 되돌린다. 시간 복잡도: $O(V_k + E_k)$이며 여기서 $V_k$과 $E_k$은 거리 $k$ 안의 꼭짓점과 변으로, $k$이 작으면 $O(V + E)$보다 훨씬 적을 수 있다. $\square$

---

**연습문제 3.**
미로를 칸이 열려 있거나 막혀 있는 격자로 나타낸다. BFS이 들머리에서 날머리까지 최단 경로를 찾는 법을 밝혀라.

??? success "연습문제 3 풀이"
    미로를 속뜻 그래프로 본뜬다. 곧 열린 칸마다 꼭짓점이고, 변은 이웃한 열린 칸(위, 아래, 왼쪽, 오른쪽)을 잇는다. 들머리 칸에서 BFS을 돌린다. 날머리 칸에 처음 닿았을 때의 BFS 거리가 최단 경로의 길이이다. 경로를 되살리려면 칸마다 어버이 가리개를 지키고 날머리에서 들머리로 거슬러 간다. 변마다 무게가 같으므로(한 걸음) BFS이 가장 좋은 풀이를 찾는다. 시간 복잡도: $R \times C$ 격자에서 $O(R \cdot C)$. $\square$

---

**연습문제 4.**
BFS이 무방향 그래프가 이분인지 어떻게 알아내는지 설명하여라. 알고리즘이 층을 가로지르는 변을 만나면 어떻게 되는가?

??? success "연습문제 4 풀이"
    샘에 빛깔 0을 주고 BFS 층마다 빛깔을 번갈아 준다. 곧 거리가 짝수인 꼭짓점은 빛깔 0, 홀수인 꼭짓점은 빛깔 1이다. 변 $(u, v)$을 살필 때 $v$이 아직 다녀가지 않았으면 반대 빛깔을 준다. $v$이 이미 다녀갔고 $u$과 빛깔이 같으면 그 그래프는 이분이 아니다(이 "같은 층" 변이 홀수 고리를 만든다). $v$이 반대 빛깔이면 그 변은 어긋나지 않는다. 이분 그래프에서는 층이 두 조각을 번갈아 오가므로 층을 가로지르는 변은 늘 다른 빛깔의 꼭짓점을 잇는다. $\square$

## 정리하며

이 마당은 무게 없는 그래프의 최단 경로、이분성 검정、이어진 덩이、층별 순회을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.

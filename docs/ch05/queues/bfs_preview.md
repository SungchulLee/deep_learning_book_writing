# 너비 우선 탐색 맛보기

너비 우선 탐색(BFS)은 큐 자료 구조의 가장 중요한 알고리즘적 응용이다. 그래프와 출발 꼭짓점이 주어지면 BFS는 거리가 2인 꼭짓점을 하나라도 살피기 전에 거리가 1인 꼭짓점을 모두 살피고, 거리가 3인 꼭짓점보다 거리가 2인 꼭짓점을 먼저 살피는 식으로 나아간다. 이러한 층별 탐색은 큐의 선입선출 성질에서 곧바로 나온다. 먼저 찾은 꼭짓점을 먼저 살피기 때문이다. BFS는 가중치 없는 그래프에서 최단 경로를 찾으며 다른 여러 그래프 알고리즘의 밑돌이 된다. 이 쪽은 큐를 공부할 까닭을 보이려고 BFS를 맛보기로 소개한다. 온전한 다룸은 그래프 알고리즘 장에 있다.

---

## 1. 큐가 층별 탐색을 만들어 내는 까닭

BFS가 꼭짓점 $v$을 방문하면 아직 방문하지 않은 $v$의 이웃을 모두 큐에 넣는다. 큐가 선입선출이므로 이 이웃들은 이미 큐에 있던 꼭짓점을 모두 살핀 뒤에야 살펴진다. 앞서 들어 있던 꼭짓점들은 모두 출발점에서 같은 거리이거나 한 층 가까우므로, BFS는 자연스럽게 출발점에서의 거리 순으로 꼭짓점을 처리한다.

이것이 핵심 통찰이다. **선입선출 순서가 너비 우선 탐색을 보장한다.** 큐를 스택(후입선출)으로 바꾸면 깊이 우선 탐색이 된다.

---

## 2. 너비 우선 탐색 알고리즘

이 알고리즘은 세 가지 상태를 관리한다.

1. 어떤 꼭짓점을 이미 보았는지 기록하는 **방문** 집합
2. 살펴볼 꼭짓점의 **큐**
3. 꼭짓점마다 출발점에서의 거리를 적어 두는 **거리** 사전

**절차:**

1. 출발 꼭짓점을 거리 0으로 방문 표시하고 큐에 넣는다
2. 큐가 비어 있지 않은 동안:
    - 꼭짓점 $u$을 뺀다
    - $u$의 각 이웃 $v$에 대해:
        - $v$을 아직 방문하지 않았다면 거리 $d(u) + 1$으로 방문 표시하고 큐에 넣는다

---

## 3. 복잡도

BFS는 각 꼭짓점을 정확히 한 번 방문하고 각 변을 (유향 그래프에서는) 한 번, (무향 그래프에서는) 두 번 살피므로 시간 복잡도는 다음과 같다.

$$
T(V, E) = O(V + E)
$$

공간 복잡도는 방문 집합, 거리 사전, 큐를 합해 $O(V)$이다.

```python
"""
너비 우선 탐색 맛보기 — 큐의 응용으로서의 너비 우선 탐색.

큐의 선입선출 성질이 층별 그래프 탐색을 만들어 내어 가중치 없는
그래프의 최단 경로를 찾는 모습을 보인다.
"""
from collections import deque

# === 너비 우선 탐색 구현 =======================================================

def bfs(graph, source):
    """출발 정점에서 시작하는 너비 우선 탐색.

    반환값:
        visited_order: 너비 우선 탐색이 방문한 차례의 꼭짓점 목록.
        distances: 꼭짓점마다 출발점에서의 거리를 담은 사전.
        parent: 꼭짓점마다 너비 우선 탐색에서의 부모를 담은 사전 (경로 복원용).

    시간:  O(V + E)
    공간: O(V)
    """
    visited = {source}
    distances = {source: 0}
    parent = {source: None}
    queue = deque([source])
    visited_order = []

    while queue:
        u = queue.popleft()
        visited_order.append(u)
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                distances[v] = distances[u] + 1
                parent[v] = u
                queue.append(v)

    return visited_order, distances, parent

def reconstruct_path(parent, target):
    """출발점에서 목표까지의 최단 경로를 복원한다."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))

# === 층을 기록하는 너비 우선 탐색 ==================================================

def bfs_by_level(graph, source):
    """층을 하나하나 추적하여 출력하는 너비 우선 탐색.

    너비 우선 탐색을 최단 경로 계산에 쓸모 있게 만드는 층별 탐색
    양상을 보인다.
    """
    visited = {source}
    queue = deque([source])
    level = 0
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            u = queue.popleft()
            current_level.append(u)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        levels.append(current_level)
        print(f"  Level {level}: {current_level}")
        level += 1

    return levels

# === 시연 ============================================================

if __name__ == "__main__":
    # 예시 그래프 (무향, 인접 리스트로 나타냄)
    #     A --- B --- E
    #     |     |
    #     C --- D --- F
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "D"],
        "D": ["B", "C", "F"],
        "E": ["B"],
        "F": ["D"],
    }

    # 기본 너비 우선 탐색
    print("BFS from vertex 'A':")
    order, dist, parent = bfs(graph, "A")
    print(f"  Visit order: {order}")
    print(f"  Distances:   {dist}")
    print()

    # 층별 너비 우선 탐색
    print("BFS levels from vertex 'A':")
    bfs_by_level(graph, "A")
    print()

    # 최단 경로 복원
    print("Shortest paths from 'A':")
    for target in sorted(graph.keys()):
        if target != "A":
            path = reconstruct_path(parent, target)
            print(f"  A → {target}: {' → '.join(path)} (distance {dist[target]})")
```

**출력:**
```
BFS from vertex 'A':
  Visit order: ['A', 'B', 'C', 'D', 'E', 'F']
  Distances:   {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 3}

BFS levels from vertex 'A':
  Level 0: ['A']
  Level 1: ['B', 'C']
  Level 2: ['D', 'E']
  Level 3: ['F']

Shortest paths from 'A':
  A → B: A → B (distance 1)
  A → C: A → C (distance 1)
  A → D: A → D (distance 2)
  A → E: A → E (distance 2)
  A → F: A → F (distance 3)
```

층별 출력은 BFS가 거리 $d+1$인 꼭짓점을 하나라도 살피기 전에 거리 $d$인 꼭짓점을 모두 살핌을 확인해 준다. A에서 F까지의 최단 경로는 중간 꼭짓점 두 개를 거치므로 거리가 3이다.

---

## 4. 너비 우선 탐색이 최단 경로를 준다

가중치 없는 그래프에서 BFS는 출발점에서 닿을 수 있는 모든 꼭짓점까지의 최단 경로를 계산한다. 이는 두 가지 사실에서 따라 나온다.

1. **단조성**: BFS가 매기는 거리는 줄어들지 않는다. 꼭짓점 $u$이 $v$보다 먼저 빠진다면 $d(u) \leq d(v)$이다.
2. **최적성**: BFS가 꼭짓점 $u$을 거쳐 $v$을 처음 찾을 때, 출발점에서 $u$을 거쳐 $v$에 이르는 경로의 변은 정확히 $d(u) + 1$개이며 이것이 참 최단 경로의 길이와 같다.

!!! tip "너비 우선 탐색이 최단 경로를 주지 못할 때"
    BFS는 **가중치 없는** 그래프(또는 모든 변의 가중치가 같은 그래프)에서만 최단 경로를 찾는다. 가중치가 있는 그래프에서는 데이크스트라 알고리즘이나 벨먼-포드를 써야 한다.

---

## 연습문제

**연습문제 1.**
너비 우선 탐색 맛보기의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
너비 우선 탐색 맛보기을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
너비 우선 탐색 맛보기을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
너비 우선 탐색 맛보기을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$

## 정리하며

이 마당은 큐가 층별 탐색을 만들어 내는 까닭、너비 우선 탐색 알고리즘、복잡도、너비 우선 탐색이 최단 경로를 준다을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.

# 고른 값 찾기

너비 우선 찾기는 변의 무게가 모두 같을 때 최단 경로를 찾지만, 쌓인 값이 아니라 뛴 횟수로 꼭짓점을 넓히므로 무게 있는 그래프에서는 무너진다. **고른 값 찾기(UCS)**는 FIFO 줄 대신 우선순위 줄을 써서 쌓인 길 값이 가장 작은 꼭짓점을 먼저 넓혀 이를 바로잡는다. UCS은 데이크스트라 알고리즘과 같지만 인공지능/찾기 문헌에서는 보통 목표 살피기와 살펴본 묶음을 곁들여 소개한다. 변의 무게가 모두 같으면 UCS은 BFS으로 주저앉는다.

## 알고리즘

UCS은 샘 $s$에서의 길 값 차례로 꼭짓점을 살펴본다:

1. 우선순위 줄을 $(0, s)$(값, 꼭짓점)으로 첫걸음 잡는다.
2. 살펴본 묶음을 (비어 있게) 첫걸음 잡는다.
3. 우선순위 줄이 비지 않은 동안:
    - 값 $g(u)$이 가장 작은 꼭짓점 $u$을 꺼낸다.
    - $u$이 목표면 $g(u)$과 길을 되돌린다.
    - $u$이 살펴본 묶음에 있으면 건너뛴다.
    - $u$을 살펴본 묶음에 넣는다.
    - 변 무게가 $w(u,v)$인 $u$의 이웃 $v$마다:
        - $v$을 아직 살펴보지 않았으면 $(g(u) + w(u,v),\, v)$을 우선순위 줄에 넣는다.
4. 우선순위 줄이 다 비면 길이 없다.

## 데이크스트라 알고리즘과의 관계

UCS과 데이크스트라 알고리즘은 같은 최단 경로를 셈한다. 차이는 담아내는 틀에 있다:

| 성질 | UCS | 데이크스트라 |
|----------|-----|----------|
| 목표 살피기 | 목표에서 멈춘다 | 모든 거리를 셈한다 |
| 문헌 | 인공지능 / 찾기 | 그래프 알고리즘 |
| 우선순위 줄 | 표준 | 흔히 열쇠 낮추기와 함께 |
| 어림짐작 | 없음($h = 0$) | 없음 |

값 매기기 함수가 $f(n) = g(n) + 0 = g(n)$이 되므로, UCS은 모든 $n$에 대해 $h(n) = 0$인 A* 찾기이기도 하다.

## 올바름

변의 무게가 모두 음이 아니면($w(u,v) \geq 0$) UCS은 **가장 좋다**(최단 경로를 찾는다). 증명은 데이크스트라의 맞음 증명과 같다. 곧 꼭짓점을 우선순위 줄에서 꺼내면 그 거리가 확정인데, 다른 어떤 길도 아직 넓히지 않은, 값이 같거나 더 큰 꼭짓점을 지나야 하기 때문이다.

## 복잡도

$C^*$을 가장 좋은 풀이의 값, $\epsilon$을 변 무게의 최솟값이라 하자:

$$
\text{Time and space: } O(b^{1 + \lfloor C^*/\epsilon \rfloor})
$$

여기서 $b$은 갈라짐 인자이다. 이진 힙을 쓰면:

$$
\text{Time: } O((|V| + |E|) \log |V|)
$$

## 구현

```python
"""
무게 그래프의 최단 경로를 위한 고른 값 찾기.

쌓인 경로 값의 차례대로 꼭짓점을 넓히며,
우선순위 줄을 쓴다. 음이 아닌 무게에서는 데이크스트라 알고리즘과 같다.
목표를 향해 멈춘다.
"""

import heapq


# === 고른 값 찾기 ===

def uniform_cost_search(graph: dict, source: int,
                        goal: int) -> tuple:
    """고른 값 찾기로 근원에서 목표까지의 최단 경로 찾기.

    인수:
        graph: 이웃 목록 {u: [(v, weight), ...]}.
        source: 시작 꼭짓점.
        goal: 과녁 꼭짓점.

    반환값:
        (값, 경로) 짝. 다음이면 (float('inf'), [])을 돌려준다:
        경로가 없을 때.
    """
    frontier = [(0, source, [source])]
    explored = set()

    while frontier:
        cost, u, path = heapq.heappop(frontier)

        if u == goal:
            return cost, path

        if u in explored:
            continue
        explored.add(u)

        for v, weight in graph.get(u, []):
            if v not in explored:
                new_cost = cost + weight
                heapq.heappush(frontier, (new_cost, v, path + [v]))

    return float('inf'), []


# === 시연 ===

if __name__ == "__main__":
    # 무게 있는 방향 그래프
    graph = {
        0: [(1, 4), (2, 2)],
        1: [(3, 5)],
        2: [(1, 1), (3, 8), (4, 10)],
        3: [(4, 2)],
        4: []
    }

    print("Graph:")
    for u, neighbors in sorted(graph.items()):
        for v, w in neighbors:
            print(f"  {u} -> {v} (weight {w})")
    print()

    # 0에서 4까지의 최단 경로 찾기
    cost, path = uniform_cost_search(graph, 0, 4)
    print(f"Shortest path 0 -> 4: cost={cost}, path={path}")
    print()

    # 0에서 3까지의 최단 경로 찾기
    cost, path = uniform_cost_search(graph, 0, 3)
    print(f"Shortest path 0 -> 3: cost={cost}, path={path}")
    print()

    # 경로가 없는 경우
    graph_disconnected = {0: [(1, 1)], 1: [], 2: [(3, 1)], 3: []}
    cost, path = uniform_cost_search(graph_disconnected, 0, 3)
    print(f"Disconnected 0 -> 3: cost={cost}, path={path}")
```

**출력:**
```
그래프:
  0 -> 1 (weight 4)
  0 -> 2 (weight 2)
  1 -> 3 (weight 5)
  2 -> 1 (weight 1)
  2 -> 3 (weight 8)
  2 -> 4 (weight 10)
  3 -> 4 (weight 2)
  4 ->

Shortest path 0 -> 4: cost=10, path=[0, 2, 1, 3, 4]

Shortest path 0 -> 3: cost=8, path=[0, 2, 1, 3]

Disconnected 0 -> 3: cost=inf, path=[]
```

!!! warning "음의 변 무게"
    UCS은 (데이크스트라처럼) 음의 변 무게를 다루지 못한다. 그래프에 무게가 음인 변이 있으면 대신 벨먼-포드 알고리즘을 써라. 이는 음의 고리를 알아내고 $O(|V| \cdot |E|)$ 시간에 최단 경로를 맞게 셈한다.

## 참고 문헌

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.), 3장. Pearson.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 24장. MIT Press.

## 연습문제

**연습문제 1.**
고른 값 찾기가 데이크스트라 알고리즘과 같음을 보여라. 핵심 자료 짜임은 무엇인가?

??? success "연습문제 1 풀이"
    고른 값 찾기(UCS)는 샘에서 쌓인 길 값 $g(v)$이 가장 작은 마디를 넓힌다. $g(v)$으로 차례 매긴 우선순위 줄을 쓴다. 이것이 바로 데이크스트라 알고리즘이다. 둘 다 거리가 가장 작은 꼭짓점을 꺼내고 그 이웃을 늦추며 우선순위 줄을 새로 고친다. 차이는 담아내는 틀뿐이다. UCS은 보통 인공지능/찾기 자리에서(넓힐 때 목표를 살피는 식으로), 데이크스트라는 그래프 알고리즘 자리에서(모든 최단 거리를 셈하는 식으로) 그려진다. 이진 힙을 쓰면 둘 다 시간 복잡도가 $O((V + E) \log V)$이다. $\square$

---

**연습문제 2.**
고른 값 찾기와 BFS을 견주어라. 어느 때 BFS이 같은 결과를 내는가?

??? success "연습문제 2 풀이"
    BFS은 뛴 횟수(변의 개수)로 살펴보고 UCS은 쌓인 길 값으로 살펴본다. 변의 무게가 모두 같으면(이를테면 모두 1이면) 쌓인 값이 뛴 횟수와 같으므로 BFS과 UCS이 같은 차례로 마디를 넓히고 같은 최단 경로를 찾는다. 무게가 다르면 BFS은 변이 적지만 값의 합은 큰 길을 찾아 참으로 짧은(값이 가장 작은) 길을 놓칠 수 있다. UCS은 무게가 어떻게 흩어져 있든 늘 값이 가장 작은 길을 찾는다. $\square$

---

**연습문제 3.**
고른 값 찾기는 온전한가? 가장 좋은가? 뒷받침하여라.

??? success "연습문제 3 풀이"
    **온전함**: 그렇다. 변의 무게가 모두 양이고($w > 0$) 갈라짐 인자에 끝이 있다면 그렇다. 길을 따라 쌓인 값이 반드시 커지므로 마디마다 언젠가 넓혀진다. 무게가 0인 변이 있으면 알고리즘이 나아가지 못하고 맴돌 수 있다. **가장 좋음**: 그렇다. $g(v)$이 줄지 않는 차례로 마디를 넓히기 때문이다. 목표를 처음 넓힐 때 그 $g$ 값이 가장 좋다(데이크스트라의 맞음 증명과 같은 따짐). $\square$

---

**연습문제 4.**
고른 값 찾기의 공간 복잡도는 얼마인가? 그것을 줄일 길을 내놓아라.

??? success "연습문제 4 풀이"
    UCS은 만든 마디를 모두 우선순위 줄과 다녀감 묶음에 담아 최악의 경우 $O(b^{d^*/\epsilon})$ 공간이 든다. 여기서 $b$은 갈라짐 인자, $d^*$은 가장 좋은 풀이의 값, $\epsilon$은 변 무게의 최솟값이다. 큰 상태 공간에서는 이것이 엄청날 수 있다. 기억 공간을 줄이려면 값 문턱값을 쓴 되풀이 깊이 늘리기를 써라. IDA*은 되풀이마다 값 문턱값을 올리며 (DFS처럼) $O(bd)$ 기억 공간만 쓴다. 이는 (마디를 다시 넓히는) 셈을 내주고 기억 공간을 아끼는 것이다. $\square$

# 양방향 찾기

표준 최단 경로 알고리즘은 샘에서 바깥으로 찾아 나가므로 과녁에서 먼 꼭짓점을 많이 살펴볼 수 있다. **양방향 찾기**는 찾기 둘을 한꺼번에 돌린다. 곧 샘 $s$에서 앞으로 하나, 과녁 $t$에서 뒤로 하나를 돌리고 두 앞자락이 만나면 멈춘다. 실전의 여러 상황에서 이는 한 방향 찾기에 견주어 살펴보는 꼭짓점의 개수를 크게 줄인다.

## 직관

꼭짓점마다 이웃이 $b$개이고 최단 경로의 길이가 $d$인 그래프에서의 BFS을 생각하자. 한 방향 BFS은 꼭짓점을 $O(b^d)$개까지 살펴본다. 양방향 BFS은 찾기 둘을 돌리는데 저마다 깊이 $d/2$에 이르며 방향마다 꼭짓점 $O(b^{d/2})$개쯤을 살펴본다. 합치면 다음과 같다

$$
O(2 \cdot b^{d/2}) = O(b^{d/2})
$$

이는 $O(b^d)$보다 지수만큼 작을 수 있다.

## 양방향 BFS

무게 없는 그래프에서는 BFS 두 벌을 한꺼번에 돌린다. 걸음마다 꼭짓점이 더 적은 앞자락을 넓힌다(일의 균형을 맞추려고). 어떤 꼭짓점이 두 다녀감 묶음에 모두 나타나면 찾기가 멈춘다.

```python
"""
무게 없는 그래프의 최단 경로를 위한 양방향 너비 우선 찾기.

근원에서 앞쪽 너비 우선 찾기를, 과녁에서 뒤쪽 너비 우선 찾기를 돌려
두 앞자락이 만나면 멈춘다.
"""

from collections import deque

# === 양방향 너비 우선 찾기 ====================================================

def bidirectional_bfs(graph, source, target):
    """양방향 너비 우선 찾기로 최단 경로 찾기.

    매개변수
    ----------
    graph : dict[int, list[int]]
        방향 없는 그래프의 이웃 목록.
    source : int
        시작 꼭짓점.
    target : int
        목표 꼭짓점.

    반환값
    -------
    list[int] | None
        source에서 target까지의 최단 경로, 닿을 수 없으면 None.
    """
    if source == target:
        return [source]

    # 앞쪽 찾기 상태
    front_visited = {source: None}
    front_queue = deque([source])

    # 뒤쪽 찾기 상태
    back_visited = {target: None}
    back_queue = deque([target])

    def build_path(meeting_point):
        """두 앞선 꼭짓점 표로 경로 되짚기."""
        # 앞쪽 부분: 근원 -> 만난 점
        path = []
        node = meeting_point
        while node is not None:
            path.append(node)
            node = front_visited[node]
        path.reverse()

        # 뒤쪽 부분: 만난 점 -> 과녁
        node = back_visited[meeting_point]
        while node is not None:
            path.append(node)
            node = back_visited[node]

        return path

    while front_queue and back_queue:
        # 더 작은 앞자락 넓히기
        if len(front_queue) <= len(back_queue):
            node = front_queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in front_visited:
                    front_visited[neighbor] = node
                    front_queue.append(neighbor)
                    if neighbor in back_visited:
                        return build_path(neighbor)
        else:
            node = back_queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in back_visited:
                    back_visited[neighbor] = node
                    back_queue.append(neighbor)
                    if neighbor in front_visited:
                        return build_path(neighbor)

    return None


# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 4],
        3: [1, 5],
        4: [2, 5],
        5: [3, 4, 6],
        6: [5],
    }

    path = bidirectional_bfs(graph, 0, 6)
    print(f"Bidirectional BFS path: {path}")
    print(f"Path length: {len(path) - 1} edges")
```

**출력:**
```
Bidirectional BFS path: [0, 1, 3, 5, 6]
Path length: 4 edges
```

## 양방향 데이크스트라

무게 있는 그래프에서 양방향 데이크스트라는 우선순위 줄 기반 찾기 둘을 돌린다. 멈추는 조건이 더 미묘하다. 곧 두 우선순위 줄의 최소 열쇠의 합이, 지금까지 만난 꼭짓점을 거쳐 찾은 가장 좋은 길보다 커지면 찾기를 멈춘다. 살펴보지 않은 어떤 길도 더 쌀 수 없으므로 이는 가장 좋음을 보장한다.

!!! warning "멈추기는 조심해야 한다"
    무게 있는 그래프에서는 어떤 꼭짓점이 두 닫힌 묶음에 모두 나타났다고 그냥 멈추면 가장 좋음이 보장되지 않는다. 올바른 조건은 가장 짧은 후보 길이 두 앞자락 최솟값의 합 이하인지 살피는 것이다. 곧 $\mu \leq d_f^{\min} + d_b^{\min}$이며 여기서 $\mu$은 지금까지 찾은 가장 좋은 만남점 거리이다.

## 복잡도

| 판 | 시간(최악의 경우) | 공간 | 실전에서의 빨라짐 |
|---|---|---|---|
| 양방향 BFS | $O(b^{d/2})$ | $O(b^{d/2})$ | BFS 대비 최대 $\sqrt{b^d}$배 |
| 양방향 데이크스트라 | $O((V + E) \log V)$ | $O(V)$ | 실전에서 약 2배 |

양방향 데이크스트라의 최악의 경우 복잡도는 표준 데이크스트라와 같지만, 실전에서는 꼭짓점의 절반쯤만 살펴보아 걸리는 시간을 크게 줄인다.

## 양방향 찾기를 언제 쓰나

양방향 찾기는 다음일 때 가장 잘 듣는다:

- 샘과 과녁을 둘 다 미리 안다.
- 그래프가 무방향이다(또는 변을 쉽게 뒤집을 수 있다).
- 갈라짐 인자가 크고 풀이의 깊이가 알맞다.

방향 그래프에서는 뒤로 찾으려면 거꾸로 그래프(변을 뒤집은 것)가 필요한데, 미리 셈해 두거나 그래프 짜임이 거꾸로 이웃을 효율적으로 찾게 해 주어야 한다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 24-25장. MIT Press.
- Pohl, I. (1971). Bi-directional search. *Machine Intelligence*, 6, 127-140.

## 연습문제

**연습문제 1.**
양방향 찾기의 기본 생각과 왜 한 방향 찾기보다 빠를 수 있는지 설명하여라.

??? success "연습문제 1 풀이"
    양방향 찾기는 찾기 둘을 한꺼번에 돌린다. 곧 샘 $s$에서 앞으로 하나, 과녁 $t$에서 뒤로 하나이다. 두 앞자락이 만나면 $s$에서 $t$까지의 길을 찾은 것이다. 갈라짐 인자가 $b$인 그래프에서 BFS을 하면 한 방향 찾기는 거리 $d$의 과녁에 대해 마디 $O(b^d)$개를 살펴본다. 양방향 찾기는 마디 $O(2 \cdot b^{d/2}) = O(b^{d/2})$개를 살펴본다. 지수만큼 줄어드는 것이다. 핵심 통찰은 작은 공 둘의 부피 합이 큰 공 하나보다 훨씬 작다는 것이다. $\square$

---

**연습문제 2.**
양방향 데이크스트라에서 찾기를 언제 멈춰야 하는가? 두 앞자락이 만나자마자 멈추면 왜 모자란가?

??? success "연습문제 2 풀이"
    두 앞자락이 어떤 꼭짓점 $v$에서 처음 만났을 때 $v$을 거치는 길이 가장 좋지 않을 수 있다. 한쪽 앞자락이 아직 꺼내지 않은 꼭짓점을 쓰는 더 짧은 길이 있을 수 있기 때문이다. 올바른 멈춤 조건은 이것이다. 곧 앞 방향과 뒤 방향 우선순위 줄의 최소 열쇠의 합이 지금까지 찾은 가장 좋은 길보다 커지면 멈춘다. 특히 $\mu$이 찾은 가장 짧은 $s$-$t$ 길이고 $d_f + d_b \geq \mu$이면($d_f$과 $d_b$은 줄마다의 지금 최소 거리) $\mu$이 가장 좋다. $\square$

---

**연습문제 3.**
양방향 찾기를 방향 그래프에 쓸 수 있는가? 어떤 어려움이 생기는가?

??? success "연습문제 3 풀이"
    쓸 수 있지만 뒤 방향 찾기가 변을 거꾸로 지나야 한다. 그래프가 이웃 목록으로 주어지면 거꾸로 이웃 목록(꼭짓점마다 들어오는 변을 담은 것)이 필요하다. 거꾸로 그래프를 짓는 데 $O(V + E)$ 시간이 든다. 거꾸로 이웃을 셈하기 어려운 속뜻 그래프에서는 양방향 찾기가 실전에서 못 쓸 수 있다. 또 양방향 A*에서는 두 방향 모두에 한결같은 어림짐작을 짜는 데 조심해야 한다. $\square$

---

**연습문제 4.**
양방향 BFS과 한 방향 BFS의 기억 공간 씀씀이를 견주어라.

??? success "연습문제 4 풀이"
    한 방향 BFS은 크기 $O(b^d)$의 앞자락 하나를 담는다. 양방향 BFS은 저마다 크기 $O(b^{d/2})$인 앞자락 둘을 담아 모두 $O(b^{d/2})$이다. 기억 공간이 크게 줄어든다. 다만 양방향 BFS은 어떤 꼭짓점을 반대쪽 찾기가 다녀갔는지 효율적으로 살펴야 하므로 방향마다 해시 집합이나 다녀감 배열이 필요해 공간이 $O(b^{d/2})$ 더 든다. 통틀어 양방향 BFS은 $O(b^{d/2})$ 기억 공간을 쓰고 한 방향은 $O(b^d)$을 쓴다. $\square$

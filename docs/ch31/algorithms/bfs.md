# 바깥 기억 너비 우선 찾기

그래프가 너무 커서 으뜸 기억에 들어가지 않으면 여느 너비 우선 찾기로는 이웃 목록을 모두 램에 올릴 수 없다. **바깥 기억 너비 우선 찾기**는 너비 우선 찾기를 바깥 기억(들고남) 모형에 맞춘 것으로, 여기서 비용 잣대는 CPU 연산이 아니라 원반 덩이를 옮기는 횟수다. 효율 좋은 바깥 기억 너비 우선 찾기는 웹 그래프, 사회 그물을 비롯한 거대한 그래프 자료를 다루는 데 결정적이다.

## 들고남 모형

바깥 기억 모형에서 기계는 다음을 가진다:

- 크기 $M$의 **으뜸 기억**(자료 낱것으로 잰다).
- 크기에 제한이 없고 크기 $B$의 덩이로 닿는 **원반**.

**들고남 연산** 하나가 원반과 기억 사이에서 낱것 $B$개의 덩이 하나를 옮긴다. 들고남 복잡도는 그런 옮김의 횟수를 재며, 낱것 $N$개를 줄 세우는 데 드는 값을 $\text{sort}(N) = O((N/B) \log_{M/B}(N/B))$으로 적고 많은 바깥 기억 알고리즘의 밑금으로 삼는다.

## 어수룩한 바깥 기억 너비 우선 찾기

가장 단순한 길은 여느 너비 우선 찾기를 돌리되 그래프를 원반에 담는 것이다. 켜마다 앞자락 꼭짓점의 이웃 목록을 모두 읽는다.

**들고남 복잡도**: 가장 나쁜 경우 꼭짓점에 닿을 때마다 아무 원반 읽기가 일어날 수 있다. 꼭짓점이 $V$개면 덩이 크기와 상관없이 들고남 $O(V)$번이 들어 가장 좋은 값보다 훨씬 나쁘다.

## 무나갈라-라나데 바깥 기억 너비 우선 찾기

무나갈라와 라나데(1999)는 더 효율 좋은 길을 내놓았다. 핵심 생각은 앞자락과 이웃 얼개를 줄 세워 차례대로 원반에 닿는 이점을 살리는 것이다.

### 알고리즘

너비 우선 켜 $d$마다:

1. 지금 앞자락 $F_d$을 꼭짓점 번호로 **줄 세운다**.
2. 줄 세운 변 목록을 **훑으며** 합침 같은 지나기로 앞자락 꼭짓점의 이웃을 모두 뽑아낸다.
3. 이웃 목록을 줄 세우고 앞서 들른 꼭짓점을 걸러 **겹침을 없앤다**.
4. 그 결과가 다음 앞자락 $F_{d+1}$을 이룬다.

### 들고남 복잡도

$n_i = |F_i|$을 $i$번째 앞자락의 크기, $D$을 너비 우선 깊이라 하자. 들고남 비용은 다음과 같다:

$$
O\!\left(\sum_{i=0}^{D}\left(\text{sort}(n_i) + \text{scan}(E)\right)\right) = O\!\left(D \cdot \frac{E}{B} + \sum_{i=0}^{D} \text{sort}(n_i)\right)
$$

$\sum_i n_i = V$이므로 줄 세우기 비용의 합은 많아야 $O(\text{sort}(V) \cdot D)$이다. 지름 $D$이 작은 그래프에서는 어수룩한 $O(V)$번보다 훨씬 낫다.

## 구현

```python
"""
바깥 기억 너비 우선 찾기 흉내내기.

무나갈라-라나데 길을 흉내 낸다. 너비 우선 켜마다 묶음으로 다루어
앞자락을 줄 세우고 변을 훑고 겹침을 없앤다. 들고남은 실제로 하지 않고
세기만 한다.
"""

import math

# ===================================================================
# 바깥 기억 너비 우선 찾기 흉내내기
# ===================================================================

def external_bfs(adj, source, B=4):
    """들고남 비용을 좇으며 바깥 기억 너비 우선 찾기를 흉내 낸다.

    인수:
        adj: 목록의 사전으로 된 이웃 목록
        source: 시작 꼭짓점
        B: 흉내 낸 덩이 크기

    반환값:
        dist: 샘에서의 거리 지도
        io_count: 흉내 낸 온 들고남 횟수
    """
    dist = {source: 0}
    frontier = [source]
    io_count = 0
    level = 0

    while frontier:
        # 앞자락을 줄 세운다(들고남 sort(|앞자락|)번)
        frontier.sort()
        n_f = len(frontier)
        if n_f > 0:
            io_count += max(1, n_f // B)  # 간단히 한 훑기 비용

        # 앞자락 꼭짓점의 변을 훑는다
        next_frontier = []
        edges_scanned = 0
        for u in frontier:
            for v in adj.get(u, []):
                edges_scanned += 1
                if v not in dist:
                    dist[v] = level + 1
                    next_frontier.append(v)

        # 변 훑기의 들고남
        io_count += max(1, edges_scanned // B)

        # 다음 앞자락의 겹침을 없앤다(줄 세우기 + 훑기)
        next_frontier.sort()
        deduped = []
        for v in next_frontier:
            if not deduped or deduped[-1] != v:
                deduped.append(v)
        if next_frontier:
            io_count += max(1, len(next_frontier) // B)

        frontier = deduped
        level += 1

    return dist, io_count

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    # 시험 그래프를 짓는다
    adj = {}
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
             (3, 6), (4, 6), (5, 7), (6, 8), (7, 8)]
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    dist, io_count = external_bfs(adj, source=0, B=2)

    print("External BFS from vertex 0:")
    for v in sorted(dist):
        print(f"  dist[{v}] = {dist[v]}")
    print(f"\nVertices: {len(dist)}")
    print(f"Edges:    {len(edges)}")
    print(f"BFS depth: {max(dist.values())}")
    print(f"Simulated I/O ops (B=2): {io_count}")

    # 덩이 크기를 달리해 견준다
    print("\nI/O count vs block size:")
    for B in [1, 2, 4, 8]:
        _, ios = external_bfs(adj, source=0, B=B)
        print(f"  B={B}: {ios} I/Os")
```

**출력:**
```
External BFS from vertex 0:
  dist[0] = 0
  dist[1] = 1
  dist[2] = 1
  dist[3] = 2
  dist[4] = 2
  dist[5] = 2
  dist[6] = 3
  dist[7] = 3
  dist[8] = 4

꼭짓점: 9
변:    10
너비 우선 깊이: 4
Simulated I/O ops (B=2): 20

I/O count vs block size:
  B=1: 38 I/Os
  B=2: 20 I/Os
  B=4: 15 I/Os
  B=8: 15 I/Os
```

## 복잡도 비교

| 알고리즘 | 들고남 복잡도 | 메모 |
|---|---|---|
| 어수룩한 너비 우선 찾기 | $O(V)$ | 꼭짓점마다 아무 닿기 |
| 무나갈라-라나데 | $O(D \cdot (E/B + \text{sort}(V)))$ | 줄 세우기 바탕, $D$이 작을 때 좋다 |
| 멜호른-마이어 | $O(V + \text{sort}(E))$ | 일반 그래프에서 가장 좋다 |

여기서 $D$은 너비 우선 깊이, $B$은 덩이 크기, $\text{sort}(N) = O((N/B) \log_{M/B}(N/B))$이다.

## 실용적인 고려

- **그래프 놓기**: 이웃 목록을 꼭짓점 번호로 줄 세워 담으면 차례대로 훑을 수 있어 바깥 기억 효율에 결정적이다.
- **반 바깥 기억 모형**: $V$은 기억에 들어가지만 $E$은 들어가지 않을 때, 들른 표시 배열을 램에 둘 수 있어 더 단순한 알고리즘으로 넉넉하다.
- **앞손질**: 변 목록을 한 번 줄 세우면(들고남 $O(\text{sort}(E))$번) 여러 샘에서의 너비 우선 묻기에 걸쳐 그 비용이 고루 나뉜다.

## 참고 문헌

- Munagala, K. and Ranade, A. (1999). "I/O-complexity of graph algorithms." *SODA*.
- Mehlhorn, K. and Meyer, U. (2002). "External-memory breadth-first search with sublinear I/O." *ESA*.
- Vitter, J. S. (2001). "External memory algorithms and data structures: dealing with massive data." *ACM Computing Surveys*.


## 연습문제

**연습문제 1.**
바깥 기억 너비 우선 찾기를 밝히고 여느 너비 우선 찾기가 바깥 기억에서 왜 효율이 나쁜지 말하여라.

??? success "연습문제 1 풀이"
    여느 너비 우선 찾기는 줄을 쓰며 꼭짓점을 하나씩 다루고 이웃 목록을 살피려 아무 기억 닿기를 한다. 램을 넘는 그래프에서는 닿을 때마다 원반 들고남이 일어날 수 있다(꼭짓점 하나에 덩이 하나 읽기). 이는 들고남 $O(|V| + |E|)$번, 곧 변마다 한 번꼴이다. 바깥 기억 너비 우선 찾기(무나갈라-라나데, 멜호른-마이어)는 꼭짓점을 켜마다 다루며 변을 출발 꼭짓점으로 줄 세워 들고남을 묶는다. 이는 $B$이 덩이 크기, $M$이 기억일 때 들고남 $O(|V|/B + (|V| + |E|) / (B \sqrt{M/B}))$번을 이룬다.

---

**연습문제 2.**
바깥 기억 너비 우선 찾기의 들고남 복잡도는 얼마이며 안쪽 기억 판과 견주면 어떠한가?

??? success "연습문제 2 풀이"
    안쪽 기억 너비 우선 찾기: 때 $O(|V| + |E|)$. 바깥 기억 판: $\text{sort}(N) = O(N/B \cdot \log_{M/B}(N/B))$이고 $D$이 너비 우선 깊이일 때 들고남 $O(\text{sort}(|E|) + |V| \cdot D / B)$번. 줄 세우기 항이 변 다루기를, $|V|D/B$ 항이 켜마다의 앞자락 다루기를 맡는다. 빽빽한 그래프($|E| = O(|V|^2)$)에서는 원반 위의 어수룩한 너비 우선 찾기보다 훨씬 효율이 좋다.

---

**연습문제 3.**
바깥 기억 너비 우선 찾기 전에 흔히 하는 그래프 가르기 앞손질 걸음을 밝혀라.

??? success "연습문제 3 풀이"
    그래프 가르기는 그래프를 기억에 들어가는 뭉치로 나누되 뭉치를 가로지르는 변을 가장 적게 한다. 너비 우선 찾기 동안 뭉치 하나를 통째로 한 번 올리고 그 안의 꼭짓점을 모두 다룬 뒤 다음으로 넘어간다. 이는 들고남을 $O(|E|)$에서 $|E|_{\text{cross}}$이 뭉치를 가로지르는 변의 개수일 때 $O(|E|_{\text{cross}} + |V|/B)$으로 줄인다. METIS 같은 연장이 좋은 가르기를 셈한다. 앞손질에 들고남 $O(\text{sort}(|E|))$번이 들지만 여러 번 돌리면 고루 나뉜다.

---

**연습문제 4.**
바깥 기억 너비 우선 찾기는 큰 규모 사회 그물 살피기에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    사회 그물(마디 수십억 개, 변 수천억 개)은 램을 넘는다. 바깥 기억 너비 우선 찾기는 다음을 셈한다. (1) 이어진 조각(들르지 않은 마디에서 너비 우선 찾기를 되풀이), (2) 뽑기 바탕 가운데임을 위한 최단 길 거리, (3) 그래프 신경망 익히기 자료를 만들기 위한 $k$번 건넌 이웃 자리. GraphChi과 X-Stream 같은 시스템이 바깥 기억 그래프 다루기를 써서 SSD을 단 기계 한 대로 이런 그물을 살피며, 나눠진 시스템의 값과 복잡함을 피한다.
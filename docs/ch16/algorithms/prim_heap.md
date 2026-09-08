# 힙을 쓴 프림

기본 배열 방식 프림 알고리즘은 $O(V^2)$에 돌아가며, 이는 빽빽한 그래프에는 효율적이지만 성긴 그래프에는 낭비이다. 열쇠값이 가장 작은 꼭짓점을 찾는 선형 훑기를 이진 최소 힙(우선순위 줄)으로 바꾸면 EXTRACT-MIN과 DECREASE-KEY이 모두 $O(\log V)$ 시간이 들어 전체 복잡도가 $O(E \log V)$으로 내려간다. 이 쪽에서는 힙 방식 구현을 자세히 보인다.

---

## 1. 힙이 왜 도움이 되나

프림 알고리즘의 되풀이마다 다음이 필요하다:

1. 나무 밖 꼭짓점 가운데 열쇠값이 가장 작은 것을 **찾는다**(EXTRACT-MIN).
2. 더 가벼운 잇는 변을 찾으면 이웃의 열쇠값을 **고친다**(DECREASE-KEY).

정렬하지 않은 배열에서는 EXTRACT-MIN이 $O(V)$, DECREASE-KEY이 $O(1)$이 든다. 꺼내기 $V$번과 줄이기 많아야 $E$번에 걸쳐 모두 $O(V^2 + E) = O(V^2)$이다.

이진 최소 힙에서는 두 연산 모두 $O(\log V)$이 든다. 꺼내기 $V$번과 줄이기 많아야 $E$번에 걸쳐 이어진 그래프에서는($E \ge V - 1$이므로) 모두 $O((V + E) \log V) = O(E \log V)$이 된다.

---

## 2. 구현

아래 파이썬 구현은 최소 힙을 주는 `heapq`을 쓴다. `heapq`은 DECREASE-KEY을 곧바로 받쳐 주지 않으므로 **게으른 지우기** 전략을 쓴다. 곧 고친 열쇠값으로 새 자리를 밀어 넣고, 그 꼭짓점을 꺼낼 때 예전 자리를 묵은 것으로 여긴다.

```python
"""
이진 최소 힙을 쓴 프림의 최소 뻗은 나무 알고리즘.

파이썬의 heapq에 게으른 지우기를 붙여
DECREASE-KEY 연산이 없는 것을 다룬다.
"""

import heapq
from collections import defaultdict

# === 그래프 나타내기 ===

def build_adjacency_list(n, edges):
    """변 목록으로 이웃 목록 세우기."""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj

# === 힙을 쓴 프림 알고리즘 ===

def prim(n, edges, start=0):
    """
    이진 힙을 쓴 프림 알고리즘으로 최소 뻗은 나무 셈하기.

    매개변수
    ----------
    n : int
        꼭짓점의 개수(0부터 n-1까지 이름 붙임).
    edges : list of (u, v, w)
        양 끝이 정수이고 무게가 수인 변 목록.
    start : int
        시작 꼭짓점(기본값 0).

    반환값
    -------
    mst_edges : list of (u, v, w)
        최소 뻗은 나무의 변.
    total_weight : int or float
        최소 뻗은 나무의 전체 무게.
    """
    adj = build_adjacency_list(n, edges)
    in_tree = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n

    key[start] = 0
    # 힙의 자리: (열쇠값, 꼭짓점)
    heap = [(0, start)]
    mst_edges = []
    total_weight = 0

    while heap:
        k, u = heapq.heappop(heap)
        if in_tree[u]:
            continue  # 게으른 지우기: 묵은 자리는 건너뛴다
        in_tree[u] = True
        total_weight += k
        if parent[u] != -1:
            mst_edges.append((parent[u], u, k))

        for v, w in adj[u]:
            if not in_tree[v] and w < key[v]:
                key[v] = w
                parent[v] = u
                heapq.heappush(heap, (w, v))

    return mst_edges, total_weight

# === 보기 ===

if __name__ == "__main__":
    #   0 ---4--- 1
    #   |  \      |
    #   1    3    2
    #   |      \  |
    #   2 ---5--- 3
    edges = [
        (0, 1, 4),
        (0, 2, 1),
        (1, 2, 3),
        (1, 3, 2),
        (2, 3, 5),
    ]
    mst, weight = prim(4, edges, start=0)
    print(f"MST edges: {mst}")
    print(f"Total weight: {weight}")
```

**출력:**
```
MST edges: [(0, 2, 1), (2, 1, 3), (1, 3, 2)]
Total weight: 6
```

---

## 3. 게으른 지우기 풀이

파이썬의 `heapq` 단원은 DECREASE-KEY 연산을 주지 않는다. 힙에 이미 있는 자리를 고치는 대신 고친 열쇠값으로 새 자리를 밀어 넣는다. 이미 나무에 더해진 꼭짓점(`in_tree[u] == True`)을 꺼내면 그냥 건너뛴다. 이 길을 **게으른 지우기**라 한다.

주고받음은 이렇다. 힙에 자리가 $O(V)$개가 아니라 많아야 $O(E)$개 들어갈 수 있다. 힙 연산마다 ($E \le V^2$이므로) $O(\log E) = O(\log V)$이 들므로 점근 복잡도는 그대로 $O(E \log V)$이다.

---

## 4. 복잡도 분석

**시간**: 알고리즘은 `heappush` 연산을 많아야 $E$번, `heappop` 연산을 많아야 $E$번 한다(묵은 자리든 쓸 수 있는 자리든 하나에 한 번씩). 저마다 $O(\log E) = O(\log V)$이 든다. 모두 합하면:

$$
T(V, E) = O(E \log V)
$$

**공간**: 이웃 목록과 힙에 $O(V + E)$.

---

## 5. 피보나치 힙과 견주기

피보나치 힙은 DECREASE-KEY을 고르게 친 $O(1)$ 시간에, EXTRACT-MIN을 고르게 친 $O(\log V)$ 시간에 받쳐 준다. 이러면 프림의 전체 시간이 다음으로 줄어든다:

$$
T(V, E) = O(E + V \log V)
$$

$E = O(V)$인 성긴 그래프에서 피보나치 힙은 $O(V \log V)$을 주어 이진 힙의 $O(V \log V)$을 상수만큼 낫게 한다. $E = \Theta(V^2)$인 빽빽한 그래프에서는 $O(V^2)$을 주어 단순한 배열 구현과 같다. 피보나치 힙은 이론으로는 낫지만 상수가 크고 구현이 복잡해 실전에서 쓰는 일이 드물다.

| 구현 | 시간 | 공간 | 실전 빠르기 |
|---------------|------|-------|-----------------|
| 배열 | $O(V^2)$ | $O(V)$ | 빽빽한 그래프에 가장 좋음 |
| 이진 힙 | $O(E \log V)$ | $O(V + E)$ | 두루 쓰기에 가장 좋음 |
| 피보나치 힙 | $O(E + V \log V)$ | $O(V + E)$ | 실전에서 더 빠른 일이 드묾 |

---

## 연습문제

**연습문제 1.**
이진 힙의 열쇠값 줄이기 연산을 설명하고 그것이 프림 알고리즘에 왜 꼭 필요한지 말하여라.

??? success "연습문제 1 풀이"
    무게가 $w$인 새 변 $(u, v)$을 찾았는데 $w <$ 지금의 $v$의 열쇠값이면 힙에서 $v$의 열쇠값을 줄여야 한다. 이진 힙에서 이는 (1) 열쇠값을 고치고 (2) 힙 성질이 되살아날 때까지 그 원소를 위로 올리는(어버이와 맞바꾸는) 것을 뜻한다. 이는 $O(\log V)$ 시간이 든다. 열쇠값 줄이기가 없으면 $v$의 새 자리를 넣어야 하고(게으른 지우기) 힙에 자리가 $O(E)$개 생겨 $O(E \log E)$ 시간이 든다. 열쇠값 줄이기는 힙 크기를 $O(V)$으로 지켜 $O((V + E) \log V)$을 준다. $\square$

---

**연습문제 2.**
프림 알고리즘에 피보나치 힙을 쓸 때의 강점을 설명하여라. 그 결과 시간 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    피보나치 힙은 열쇠값 줄이기를 (이진 힙의 $O(\log V)$에 견주어) 고르게 친 $O(1)$ 시간에, 최소 꺼내기를 고르게 친 $O(\log V)$에 받쳐 준다. 프림에서는 최소 꺼내기 $V$번이 $O(V \log V)$을, 열쇠값 줄이기 $E$번이 $O(E)$을 보탠다. 모두 합하면 $O(E + V \log V)$이다. 빽빽한 그래프에서는 이것이 $O(V^2)$으로 배열과 같고, 성긴 그래프에서는 $O(V \log V)$이다. 다만 피보나치 힙은 상수가 크고 구현이 복잡해 실전에서는 이진 힙이 더 빠른 일이 많다. $\square$

---

**연습문제 3.**
파이썬의 `heapq` 단원으로 프림의 한 판을 구현하여라. 열쇠값 줄이기 연산이 없는 것을 어떻게 다루는가?

??? success "연습문제 3 풀이"
    파이썬의 `heapq`은 열쇠값 줄이기를 받쳐 주지 않는다. 게으른 지우기를 쓴다. 곧 예전 것을 없애지 않고 새 (무게, 꼭짓점) 짝을 밀어 넣고, 꺼낼 때 이미 들어간 꼭짓점의 자리는 건너뛴다:

    ```python
    import heapq
    def prim(adj, n):
        in_mst = [False] * n
        heap = [(0, 0)]  # (무게, 꼭짓점)
        total = 0
        while heap:
            w, u = heapq.heappop(heap)
            if in_mst[u]:
                continue
            in_mst[u] = True
            total += w
            for v, wt in adj[u]:
                if not in_mst[v]:
                    heapq.heappush(heap, (wt, v))
        return total
    ```

    힙에 자리가 많아야 $O(E)$개 들어갈 수 있으므로 복잡도는 $O(E \log E) = O(E \log V)$이다. $\square$

---

**연습문제 4.**
무게가 $[1, W]$ 범위의 정수인 그래프에서 프림 알고리즘을 $O(E \log V)$보다 낫게 할 수 있는가?

??? success "연습문제 4 풀이"
    그렇다. 통 줄(무게를 자리 번호로 삼는 통 $W$개의 배열)을 쓴다. 통마다 그 열쇠값을 갖는 꼭짓점을 담는다. 최소 꺼내기는 통을 차례로 훑으므로 꺼내기마다 $O(W)$이 들지만 변마다 통 사이를 한 번만 옮긴다. 모두 합하면 $O(E + VW)$이다. $W$이 작으면(이를테면 $W = O(V)$이면) $O(E + V^2)$이 되어 배열 방식과 같다. $W$이 아주 작으면(상수이면) $O(V + E)$의 선형 시간이 된다. 이것이 실전 구현에서 쓰는 판 엠데 보아스 방식 또는 통 방식이다. $\square$

## 정리하며

이 마당은 힙이 왜 도움이 되나、구현、게으른 지우기 풀이、복잡도 분석을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 23장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [1584. Min Cost to Connect All Points -- LeetCode](https://leetcode.com/problems/min-cost-to-connect-all-points/)

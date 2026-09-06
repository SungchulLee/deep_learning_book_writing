# 마구잡이 최소 자름(카거 알고리즘)

방향 없는 그래프의 최소 자름, 곧 없애면 그래프가 끊기는 가장 작은 모서리 모임을 찾는 일은 그물 믿음성과 그래프 가르기의 옛부터의 문제이다. 최대 흐름에 바탕한 정해진 알고리즘은 $O(n^3)$이나 그보다 낫지만, 카거의 마구잡이 오그리기 알고리즘은 놀랍도록 단순한 방식으로 같은 것을 이룬다. 곧 꼭짓점이 둘만 남을 때까지 아무 모서리를 거듭 오그린다.

## 문제 서술

꼭짓점이 $n = |V|$개인 방향 없는 겹그래프 $G = (V, E)$이 주어질 때 **자름**은 $V$을 비지 않은 두 모임 $S$과 $\bar{S}$으로 가르는 것이다. 자름의 **크기**는 그 가름을 가로지르는 모서리의 수이다. **최소 자름**은 크기가 가장 작은 자름이다.

## 카거의 오그리기 알고리즘

### 모서리 오그리기

모서리 $(u, v)$을 **오그린다**는 것은 $u$과 $v$을 하나의 큰 꼭짓점으로 합치는 것이다. $u$과 $v$ 사이의 모서리는 모두 없애고 다른 꼭짓점으로 가는 모서리는 남긴다(나란한 모서리가 생길 수 있는 겹그래프가 된다). 제 고리는 없앤다.

### 알고리즘

1. 그래프의 꼭짓점이 2개를 넘는 동안:
    - 모서리를 고르게 아무렇게나 고른다.
    - 그것을 오그린다.
2. 남은 큰 꼭짓점 둘이 자름을 정한다. 그 사이의 모서리 수를 돌려준다.

### 한 번 돌리기

한 번 돌릴 때마다 자름 하나가 나온다(반드시 최소는 아니다). 핵심 통찰은 한 번에 최소 자름을 찾을 확률이 놀랍도록 높다는 것이다.

## 확률 살피기

**정리.** 카거 알고리즘을 한 번 돌리면 적어도 $\binom{n}{2}^{-1} = \frac{2}{n(n-1)}$의 확률로 최소 자름을 돌려준다.

**밝힘.** $C$을 크기 $k$인 최소 자름이라 하자. 꼭짓점이 $t$개 남은 어느 걸음에서든 그래프에는 모서리가 적어도 $kt/2$개 있다(모든 꼭짓점의 차수가 $\ge k$이기 때문이다). 최소 자름 모서리를 오그릴 확률은 많아야 다음과 같다.

$$
\frac{k}{kt/2} = \frac{2}{t}
$$

$n - 2$번의 오그리기 내내 최소 자름 모서리를 *하나도* 오그리지 않을 확률은 다음과 같다.

$$
\prod_{t=n}^{3} \left(1 - \frac{2}{t}\right)
= \prod_{t=3}^{n} \frac{t-2}{t}
= \frac{1 \cdot 2}{(n-1) \cdot n}
= \frac{2}{n(n-1)}
$$

$\square$

## 성공 확률 높이기

알고리즘을 $T$번 돌려 찾은 가장 작은 자름을 돌려준다. $T$번 모두 최소 자름을 *놓칠* 확률은 다음과 같다.

$$
\left(1 - \frac{2}{n(n-1)}\right)^T \le e^{-2T / n(n-1)}
$$

$T = \binom{n}{2} \ln n = \frac{n(n-1)}{2} \ln n$으로 두면 못 이룰 확률이 많아야 $1/n$이 된다.

$$
e^{-\ln n} = \frac{1}{n}
$$

**온 시간:** $O(n^2 \cdot T) = O(n^4 \log n)$.

## 카거-스타인의 개선

카거-스타인 알고리즘은 앞선 오그리기가 더 안전하다는(최소 자름 모서리를 자를 확률이 낮다는) 점을 살려 도는 시간을 줄인다. 꼭짓점이 $\lceil n/\sqrt{2} \rceil + 1$개가 될 때까지 오그린 뒤 서로 매이지 않은 되돌이 부르기 둘로 갈라진다.

$$
T(n) = O(n^2 \log^3 n)
$$

이는 빽빽한 그래프에서 순진한 $O(n^4 \log n)$과 정해진 최대 흐름 방식보다 모두 빠르다.

## 구현

```python
"""
카거의 마구잡이 최소 자름 알고리즘.

아무 모서리를 거듭 오그려 방향 없는 그래프의
최소 자름을 찾는다.
"""

import random
import copy


# === 그래프 표현 ===

def make_graph(n, edges):
    """이웃 목록 겹그래프를 만든다.

    Returns a dict mapping vertex -> list of neighbors (with repeats
    for parallel edges).
    """
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# === 모서리 오그리기 ===

def contract(adj, u, v):
    """Contract edge (u, v) by merging v into u.

    제 고리를 없애고 v의 모서리를 모두 u으로 돌린다.
    """
    # v의 이웃을 모두 u으로 옮긴다
    for w in adj[v]:
        if w != u:
            adj[u].append(w)
            adj[w] = [u if x == v else x for x in adj[w]]

    # 제 고리를 없앤다
    adj[u] = [x for x in adj[u] if x != u]

    # v을 없앤다
    del adj[v]


# === 카거 알고리즘(한 번 돌리기) ===

def karger_once(adj):
    """카거의 오그리기 알고리즘 한 번 돌리기.

    Returns the cut size (number of edges between the final 2 vertices).
    """
    adj = copy.deepcopy(adj)

    while len(adj) > 2:
        # 아무 모서리를 고른다
        u = random.choice(list(adj.keys()))
        if not adj[u]:
            break
        v = random.choice(adj[u])
        contract(adj, u, v)

    # 자름 크기는 남은 두 꼭짓점 사이의 모서리 수이다
    vertices = list(adj.keys())
    if len(vertices) < 2:
        return float("inf")
    return len(adj[vertices[0]])


# === 카거 거듭 돌리기 ===

def karger_min_cut(n, edges, trials=None):
    """카거 알고리즘을 여러 번 돌려 최소 자름을 찾는다.

    인수:
        n: 꼭짓점의 수.
        edges: list of (u, v) edge tuples.
        trials: number of repetitions (default: n^2 * ln(n) / 2).

    반환값:
        찾은 최소 자름의 크기.
    """
    import math
    if trials is None:
        trials = max(int(n * n * math.log(n) / 2), 10)

    adj = make_graph(n, edges)
    min_cut = float("inf")

    for _ in range(trials):
        cut = karger_once(adj)
        min_cut = min(min_cut, cut)

    return min_cut


# === 메인 ===

if __name__ == "__main__":
    random.seed(42)

    # 최소 자름 = 2로 알려진 단순한 그래프
    #   0 -- 1
    #   |    |
    #   2 -- 3
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    n = 4
    print(f"Graph: {n} vertices, edges = {edges}")
    print(f"Min-cut (50 trials): {karger_min_cut(n, edges, trials=50)}")

    # 최소 자름 = 1인 그래프(다리)
    #   0 -- 1 -- 2
    edges2 = [(0, 1), (1, 2)]
    print(f"\nBridge graph: min-cut = {karger_min_cut(3, edges2, trials=50)}")

    # 더 빽빽한 그래프
    edges3 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    print(f"\nDenser graph: min-cut = {karger_min_cut(5, edges3, trials=100)}")
```

**출력:**
```
Graph: 4 vertices, edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
Min-cut (50 trials): 2

Bridge graph: min-cut = 1

Denser graph: min-cut = 2
```

## 복잡도 요약

| 알고리즘 | 시간 | 성공 확률 |
|---|---|---|
| 카거 한 번 돌리기 | $O(n^2)$ | $\ge 2/(n(n-1))$ |
| 카거 거듭 돌리기 | $O(n^4 \log n)$ | $\ge 1 - 1/n$ |
| 카거-스타인 | $O(n^2 \log^3 n)$ | $\ge 1 - 1/n$ |
| 정해진 방법(슈퇴어-바그너) | $O(nm + n^2 \log n)$ | $1$ |

## 참고 문헌

- Karger, D. R. "Global Min-Cuts in RNC, and Other Ramifications of a Simple Min-Cut Algorithm." *SODA*, 1993.
- Karger, D. R. & Stein, C. "A New Approach to the Minimum Cut Problem." *JACM*, 1996.

## 연습문제

**연습문제 1.**
마구잡이 최소 자름(카거 알고리즘)의 핵심 마구잡이 재주와 그것이 정해진 방식보다 나은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    마구잡이 최소 자름(카거 알고리즘)은 마구잡이를 써서 정해진 알고리즘이 마주칠 수 있는 가장 나쁜 들임을 피한다. 아무렇게나 고르므로 알고리즘의 솜씨가 들임의 짜임이 아니라 제 동전 던지기에 달린다. 그래서 모든 들임에 대해 참인 센 기댓값 시간이나 높은 확률의 보장을 흔히 얻으며, 짓궂거나 병리적인 경우를 걱정할 까닭이 없어진다. $\square$

---

**연습문제 2.**
마구잡이 최소 자름(카거 알고리즘)의 기댓값 시간 복잡도는 얼마인가? 가장 나쁜 경우의 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    기댓값 시간 복잡도는 흔히 $O(n)$이나 $O(n \log n)$이며 높은 확률로 이룬다. 가장 나쁜 경우는 다항식만큼 더 나쁠 수 있지만(예컨대 $O(n^2)$) 그럴 확률은 무시할 만큼 작다. 기댓값과 가장 나쁜 경우의 틈이 마구잡이의 값이며, 가장 나쁜 움직임이 일어날 확률은 들임 크기에 따라 지수로 줄어든다. $\square$

---

**연습문제 3.**
마구잡이 최소 자름(카거 알고리즘)은 라스베이거스 알고리즘인가 몬테카를로 알고리즘인가? 그 차이를 설명하라.

??? success "연습문제 3 풀이"
    **라스베이거스**: 늘 옳은 결과를 내며 도는 시간이 아무 변수이다(기댓값이 다항식). **몬테카를로**: 늘 다항식 시간에 돌지만 결과가 어떤 가둔 확률로 틀릴 수 있다. 마구잡이 최소 자름(카거 알고리즘)은 옳음을 보장하느냐 도는 시간을 보장하느냐에 따라 이 가운데 하나에 든다. 이 가름이 어긋날 확률을 어떻게 다룰지 정한다. $\square$

---

**연습문제 4.**
마구잡이 최소 자름(카거 알고리즘)에서 마구잡이를 없애거나 솜씨가 나쁠 확률을 줄이는 법을 설명하라.

??? success "연습문제 4 풀이"
    방책은 다음과 같다. (1) **거듭 해 보기**: 알고리즘을 여러 번 돌려 가장 좋거나 많은 쪽 결과를 택하면 어긋날 확률이 지수로 줄어든다. (2) **마구잡이 없애기**: 조건부 기댓값이나 흩는 함수 무리로 아무 고르기를 정해진 고르기로 바꾼다. (3) **키우기**: 몬테카를로 알고리즘에서는 $k$번 되풀이해 어긋남을 $2^{-k}$으로 줄인다. (4) **비슷 마구잡이 만들개**: 알고리즘이 보기에 "마구잡이처럼 보이는" 정해진 차례를 쓴다. $\square$

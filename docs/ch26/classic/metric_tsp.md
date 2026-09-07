# 잣대 떠돌이 장수 문제 어림

일반 떠돌이 장수 문제는 P = NP가 아니라면 어떤 상수 갑절로도 어림할 수 없다. 그러나 거리가 **삼각 부등식**을 채우면, 곧 곧바로 가는 것이 돌아가는 것보다 결코 나쁘지 않으면 상수 갑절 어림이 가능해진다. 이 특별한 경우를 **잣대 떠돌이 장수 문제**라 부른다.

## 문제의 정의

삼각 부등식을 채우는 음이 아닌 모서리 무게를 지닌 온전 그래프 $G = (K_n, w)$이 주어질 때

$$
w(u, v) \le w(u, x) + w(x, v) \quad \forall\, u, v, x \in V
$$

온 무게가 가장 작은 해밀턴 돌기(모든 꼭짓점을 꼭 한 번씩 들르는 나들이)를 찾아라.

## 최소 뻗음 나무에 바탕한 2 어림

**직관.** 최소 뻗음 나무는 온 모서리 무게가 가장 작게 모든 꼭짓점을 잇는다. 최소 뻗음 나무를 따라 돌면 모든 꼭짓점을 들른다(거듭 들르기도 한다). 삼각 부등식 덕분에 거듭 들르는 꼭짓점을 비용을 늘리지 않고 건너뛸 수 있다.

**알고리즘:**

1. $G$의 최소 뻗음 나무 $T$을 셈한다
2. $T$을 깊이 우선으로 걸으며 처음 들른 차례로 꼭짓점을 적는다
3. 이 차례를 해밀턴 돌기로 돌려준다

!!! tip "정리"
    최소 뻗음 나무 두 배 알고리즘은 비용이 많아야 $2 \cdot \text{OPT}$인 나들이를 낸다.

**밝힘.** $C^*$을 가장 좋은 나들이라 하자. $C^*$에서 아무 모서리나 지우면 뻗음 나무가 되므로 $w(T) \le w(C^*)$이다. $T$의 깊이 우선 걷기는 모서리마다 꼭 두 번 지나므로 비용이 $2 \cdot w(T) \le 2 \cdot w(C^*)$인 닫힌 걷기가 된다. 삼각 부등식으로 거듭 들르는 꼭짓점을 건너뛰면 비용이 줄기만 하므로 나오는 해밀턴 돌기의 비용은 많아야 다음과 같다.

$$
2 \cdot w(T) \le 2 \cdot \text{OPT} \qquad \square
$$

## 크리스토피데스-세르듀코프 알고리즘

**직관.** 최소 뻗음 나무 걷기는 홀수 차수 꼭짓점 때문에 되돌아가게 되어 비용이 남는다. 홀수 차수 꼭짓점에 무게가 가장 작은 온전 짝짓기를 더하면 오일러 그래프가 되어 모서리를 거듭 지나지 않고 훑을 수 있다.

**알고리즘:**

1. $G$의 최소 뻗음 나무 $T$을 셈한다
2. $O$을 $T$의 홀수 차수 꼭짓점 모임이라 하자(늘 짝수 개이다)
3. $O$의 꼭짓점에 무게가 가장 작은 온전 짝짓기 $M$을 찾는다
4. $T$과 $M$을 아울러 겹그래프 $H$을 만든다. 이제 모든 꼭짓점의 차수가 짝수이다
5. $H$의 오일러 돌기를 찾는다
6. 거듭 들르는 꼭짓점을 건너뛰어 해밀턴 돌기를 얻는다

!!! tip "정리(Christofides, 1976)"
    이 알고리즘은 비용이 많아야 $\frac{3}{2} \cdot \text{OPT}$인 나들이를 낸다.

**밝힘.** 최소 뻗음 나무는 $w(T) \le \text{OPT}$을 채운다. 짝짓기에서는 가장 좋은 나들이를 $O$의 꼭짓점으로 줄이면 $O$ 위의 해밀턴 돌기가 된다. 이 돌기는 (모서리를 번갈아) 온전 짝짓기 둘로 가를 수 있고 저마다 비용이 많아야 $\text{OPT}/2$이다. 최소 짝짓기 $M$은 다음을 채운다.

$$
w(M) \le \frac{\text{OPT}}{2}
$$

$T \cup M$의 오일러 돌기는 비용이 $w(T) + w(M)$이다. 건너뛴 뒤:

$$
w(\text{tour}) \le w(T) + w(M) \le \text{OPT} + \frac{\text{OPT}}{2}
= \frac{3}{2}\,\text{OPT} \qquad \square
$$

이 3/2 비율은 Karlin, Klein, Oveis Gharan(2021)이 조금 개선하기까지 거의 50년 동안 가장 좋은 것으로 남아 있었다.

## 구현

```python
"""
잣대 떠돌이 장수 문제: 최소 뻗음 나무에 바탕한 2 어림.
"""

import heapq
from collections import defaultdict


# === 프림의 최소 뻗음 나무 ==================================================

def prim_mst(n, adj):
    """프림 알고리즘으로 최소 뻗음 나무를 셈한다. 최소 뻗음 나무의 이웃 목록을 돌려준다."""
    visited = [False] * n
    mst = defaultdict(list)
    # (무게, 꼭짓점, 어버이)
    heap = [(0, 0, -1)]
    total = 0

    while heap:
        w, u, parent = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        if parent >= 0:
            mst[parent].append(u)
            mst[u].append(parent)
        for v, wt in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (wt, v, u))

    return mst, total


# === 깊이 우선 앞차례 나들이 =================================================

def dfs_preorder(mst, n):
    """해밀턴 돌기 차례를 얻으려 최소 뻗음 나무를 깊이 우선 앞차례로 훑는다."""
    visited = [False] * n
    order = []
    stack = [0]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in reversed(mst[u]):
            if not visited[v]:
                stack.append(v)

    return order


# === 최소 뻗음 나무에 바탕한 2 어림 ==========================================

def metric_tsp_2approx(n, dist):
    """
    최소 뻗음 나무 두 배로 구한 잣대 떠돌이 장수 문제의 2 어림.

    dist: 삼각 부등식을 채우는 n x n 거리 행렬.
    (나들이 비용, 나들이 차례)을 돌려준다.
    """
    # 이웃 관계를 세운다
    adj = defaultdict(list)
    for u in range(n):
        for v in range(u + 1, n):
            adj[u].append((v, dist[u][v]))
            adj[v].append((u, dist[u][v]))

    mst, mst_cost = prim_mst(n, adj)
    tour = dfs_preorder(mst, n)

    # 나들이 비용을 셈한다
    cost = sum(dist[tour[i]][tour[i + 1]] for i in range(n - 1))
    cost += dist[tour[-1]][tour[0]]

    return cost, tour


# === 보여 주기 ===============================================================

if __name__ == "__main__":
    # 유클리드 거리를 지닌 고을 4개(삼각 부등식이 참이다)
    import math

    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist[i][j] = math.sqrt(dx * dx + dy * dy)

    cost, tour = metric_tsp_2approx(n, dist)
    opt = 4.0  # 정사각형 둘레
    print(f"Tour: {tour}")
    print(f"Tour cost: {cost:.4f}")
    print(f"Optimal:   {opt:.4f}")
    print(f"Ratio:     {cost / opt:.4f}")
```

## 요약

| 알고리즘 | 비율 | 시간 |
|---|---|---|
| 최소 뻗음 나무 두 배 | $2$ | $O(n^2)$ |
| 크리스토피데스-세르듀코프 | $3/2$ | $O(n^3)$ |
| 칼린-클라인-오베이스가란(2021) | $3/2 - \delta$ | 다항식 |

삼각 부등식은 꼭 필요하다. 그것이 없으면 P = NP가 아닌 한 어떤 다항식 시간 알고리즘도 상수 어림 비율을 이룰 수 없다.

## 참고 문헌

- Christofides, N. "Worst-Case Analysis of a New Heuristic for the Travelling Salesman Problem." Technical Report 388, CMU, 1976.
- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001. Chapter 3.

## 연습문제

**연습문제 1.**
잣대 떠돌이 장수 문제 어림의 어림 알고리즘을 설명하고 그 어림 보장을 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 다항식 시간에 돌며 가장 좋은 값의 밝힐 수 있는 갑절 안에 드는 풀이를 낸다. 어림 비율은 알고리즘이 내놓은 것을 가장 좋은 값의 아래 한계(가장 작게 하기)나 위 한계(가장 크게 하기), 곧 선형 계획 느슨하게 하기 값이나 조합 한계, 문제의 짜임 성질과 이어 밝힌다. $\square$

---

**연습문제 2.**
잣대 떠돌이 장수 문제 어림의 어림 비율을 밝히는 데 어떤 아래 한계 재주를 쓰는가?

??? success "연습문제 2 풀이"
    밝힘은 흔히 알고리즘의 풀이를 느슨하게 한 한계(선형 계획 느슨하게 하기, 분수 풀이, 조합 아래 한계)와 견준다. 가장 작게 하기에서는 $ALG \leq \rho \cdot LP^* \leq \rho \cdot OPT$이다. 가장 크게 하기에서는 $ALG \geq OPT / \rho$이다. 아래 한계는 효율 좋게 셈할 수 있고 쓸모 있는 비율을 줄 만큼 빡빡해야 한다. $\square$

---

**연습문제 3.**
잣대 떠돌이 장수 문제 어림의 어림 비율을 더 좋게 할 수 있는가? 알려진 어려움 결과는 무엇인가?

??? success "연습문제 3 풀이"
    어림 비율이 얼마나 빡빡한지는 복잡도 이론의 가정(P $\neq$ NP, 하나뿐인 놀이 추측 등)에 달렸다. 어떤 문제에서는 단순한 욕심쟁이나 반올림 알고리즘이 여느 가정 아래 이미 가장 좋다. 다른 문제에서는 가장 좋은 알고리즘과 가장 센 어려움 결과 사이에 틈이 있어 아직 풀리지 않은 연구 문제로 남아 있다. $\square$

---

**연습문제 4.**
잣대 떠돌이 장수 문제 어림을 구체적인 보기에 써서 어림 비율이 참임을 확인하라.

??? success "연습문제 4 풀이"
    작은 보기(예컨대 꼭짓점이나 물건 5~6개)를 고른다. 어림 알고리즘을 한 걸음씩 돌린다. 알고리즘이 내놓은 것을 (작은 보기에서 막무가내로 찾은) 가장 좋은 풀이와 견준다. 비율 $ALG/OPT$(또는 $OPT/ALG$)이 밝힌 한계 안에 드는지 확인한다. 그러면 구체적인 보기에서 이론이 굳어진다. $\square$

# 플로이드-워셜 알고리즘

데이크스트라나 벨먼-포드 같은 한 샘 알고리즘은 한 샘에서 다른 모든 꼭짓점까지의 최단 경로를 찾는다. *모든* 꼭짓점 짝 사이의 최단 경로가 필요하면 꼭짓점마다 한 샘 알고리즘을 돌려도 되지만 그것이 가장 단순한 길은 아닐 수 있다. 플로이드-워셜 알고리즘은 동적 계획법으로 **모든 짝 최단 경로** 문제를 곧바로 풀며, 놀랍도록 간결한 세 겹 되풀이 짜임으로 $\Theta(V^3)$ 시간에 돈다.

## 동적 계획법 점화식

꼭짓점에 $1, 2, \dots, n$의 번호를 매기자. $d^{(k)}_{ij}$을 꼭짓점 $\{1, 2, \dots, k\}$만 가운데 꼭짓점으로 써서 $i$에서 $j$까지 가는 최단 경로의 무게라 정의한다.

**바탕 경우($k = 0$):** 가운데 꼭짓점을 쓸 수 없으므로 다음과 같다

$$
d^{(0)}_{ij} =
\begin{cases}
0 & \text{if } i = j \\
w(i, j) & \text{if } (i, j) \in E \\
\infty & \text{otherwise}
\end{cases}
$$

**되돌이 경우:** $\{1, \dots, k\}$을 가운데 꼭짓점으로 쓰는 $i$에서 $j$까지의 최단 경로는 꼭짓점 $k$을 피하거나(그러면 $\{1, \dots, k-1\}$만 쓴다) $k$을 지난다($i \leadsto k \leadsto j$으로 쪼개진다):

$$
d^{(k)}_{ij} = \min\!\left(d^{(k-1)}_{ij},\ d^{(k-1)}_{ik} + d^{(k-1)}_{kj}\right)
$$

모든 짝 $(i, j)$에 대해 $d^{(n)}_{ij}$을 셈하고 나면 그 행렬에 모든 꼭짓점 짝 사이의 최단 경로 무게가 담긴다.

## 왜 통하는가

이 점화식은 꼭짓점 $k$마다 가운데 꼭짓점 후보로 삼아 헤아린다. 최단 경로의 가장 좋은 밑짜임에 따라 부분 길 $i \leadsto k$과 $k \leadsto j$도 $\{1, \dots, k-1\}$만 가운데 꼭짓점으로 쓰는 최단 경로이다. $\min$ 연산이 더 나은 쪽을 고른다. 곧 $k$을 피하거나 $k$을 거쳐 가는 것이다.

최단 경로마다 $\{1, \dots, n\}$의 어떤 부분집합을 가운데 꼭짓점으로 쓰므로, $k$을 $1$부터 $n$까지 훑으면 모든 경우를 덮는다.

## 의사코드

```
FLOYD-WARSHALL(W):
    n = |V|
    D = W                      // D[i][j] = w(i,j) or inf
    P = predecessor matrix     // P[i][j] = i if (i,j) in E, else NIL
    for k = 1 to n:
        for i = 1 to n:
            for j = 1 to n:
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]
                    P[i][j] = P[k][j]
    return D, P
```

이 알고리즘은 거리 행렬을 제자리에서 새로 고친다. 앞선 것 행렬 $P$은 최단 경로 위 마지막 가운데 꼭짓점을 좇아 길을 되살릴 수 있게 한다.

## 복잡도

- **Time:** Three nested loops over $n$ vertices: $\Theta(n^3)$.
- **공간:** 거리 행렬과 앞선 것 행렬에 $\Theta(n^2)$.

세제곱 시간 덕분에 플로이드-워셜은 꼭짓점 수천 개까지의 빽빽한 그래프에 실전에서 쓸 만하다. 성긴 그래프에서는 존슨의 알고리즘($O(V^2 \log V + VE)$)이 더 빠를 수 있다.

## 제자리 새로 고치기가 맞음

미묘한 점 하나. 이 알고리즘은 $D^{(k-1)}$과 $D^{(k)}$ 행렬을 따로 두지 않고 $D$을 제자리에서 새로 고친다. 이는 맞다. $d^{(k)}_{ij}$을 셈할 때 값 $d^{(k)}_{ik}$과 $d^{(k)}_{kj}$이 저마다 $d^{(k-1)}_{ik}$과 $d^{(k-1)}_{kj}$과 같기 때문이다($i$에서 $k$까지나 $k$에서 $j$까지의 길에 $k$을 가운데 꼭짓점으로 더해도 $k$이 이미 끝점이므로 도움이 되지 않는다).

## 길 되살리기

앞선 것 행렬 $P$은 아무 짝 사이의 최단 경로를 되살릴 수 있게 한다. $P[i][j]$은 $i$에서 온 최단 경로 위 $j$의 앞선 꼭짓점을 담는다. $i$에서 $j$까지의 길을 되살리려면 다음과 같이 한다:

```
PRINT-PATH(P, i, j):
    if i == j:
        print i
    elif P[i][j] == NIL:
        print "no path"
    else:
        PRINT-PATH(P, i, P[i][j])
        print j
```

## 음의 고리 알아내기

플로이드-워셜을 돌린 뒤 거리 행렬의 대각선을 살펴라. 어떤 꼭짓점 $i$에 대해 $d^{(n)}_{ii} < 0$이면 그 그래프에는 $i$을 지나는 무게가 음인 고리가 있다.

## 풀이 예제

꼭짓점 4개와 다음 무게 행렬을 갖는 그래프를 생각하자:

$$
W = \begin{pmatrix}
0 & 3 & \infty & 7 \\
8 & 0 & 2 & \infty \\
5 & \infty & 0 & 1 \\
2 & \infty & \infty & 0
\end{pmatrix}
$$

**$k=1$ 뒤:** 꼭짓점 1을 가운데 꼭짓점으로.
$d_{24} = \min(\infty, d_{21} + d_{14}) = \min(\infty, 8 + 7) = 15$.
$d_{32} = \min(\infty, d_{31} + d_{12}) = \min(\infty, 5 + 3) = 8$.

**$k=2$ 뒤:** 꼭짓점 2을 가운데 꼭짓점으로.
$d_{13} = \min(\infty, d_{12} + d_{23}) = \min(\infty, 3 + 2) = 5$.

**$k=3$ 뒤:** 꼭짓점 3을 가운데 꼭짓점으로.
$d_{14} = \min(7, d_{13} + d_{34}) = \min(7, 5 + 1) = 6$.
$d_{24} = \min(15, d_{23} + d_{34}) = \min(15, 2 + 1) = 3$.

**$k=4$ 뒤:** 꼭짓점 4을 가운데 꼭짓점으로.
$d_{31} = \min(5, d_{34} + d_{41}) = \min(5, 1 + 2) = 3$.
$d_{32} = \min(8, d_{31} + d_{12}) = \min(8, 3 + 3) = 6$.

마지막 거리 행렬:

$$
D^{(4)} = \begin{pmatrix}
0 & 3 & 5 & 6 \\
5 & 0 & 2 & 3 \\
3 & 6 & 0 & 1 \\
2 & 5 & 7 & 0
\end{pmatrix}
$$

## 구현

```python
"""
플로이드-워셜 모든 짝 최단 경로 알고리즘.

모든 꼭짓점 짝 사이의 최단 경로를 O(V^3) 시간에 셈한다
중간 꼭짓점을 넓히는 동적 계획을 쓴다.
"""

from math import inf


# === 플로이드-워셜 알고리즘 ==================================================

def floyd_warshall(n: int, edges: list) -> tuple[list, list]:
    """모든 짝의 최단 경로 셈하기.

    매개변수
    ----------
    n : int
        꼭짓점의 개수(0부터 n-1까지 이름 붙임).
    edges : list of (u, v, w)
        무게 있는 방향 변.

    반환값
    -------
    dist : list of list
        dist[i][j] = i에서 j까지 최단 경로의 무게.
    pred : list of list
        pred[i][j] = 최단 i->j 경로에서 j의 앞선 꼭짓점.
    """
    # 거리 행렬과 앞선 꼭짓점 행렬의 첫값 잡기
    dist = [[inf] * n for _ in range(n)]
    pred = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w
        pred[u][v] = u

    # 주된 동적 계획 되풀이
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    return dist, pred


# === 경로 되짚기 =============================================================

def reconstruct_path(pred: list, i: int, j: int) -> list:
    """꼭짓점 i에서 j까지의 최단 경로 되짚기."""
    if pred[i][j] is None and i != j:
        return []  # 경로 없음
    path = []
    v = j
    while v != i:
        if v is None:
            return []  # 경로 없음
        path.append(v)
        v = pred[i][v]
    path.append(i)
    path.reverse()
    return path


# === 음의 순환 알아내기 ======================================================

def has_negative_cycle(dist: list) -> bool:
    """그래프에 음의 무게 순환이 있는지 살피기."""
    return any(dist[i][i] < 0 for i in range(len(dist)))


# === 보임 ====================================================================

if __name__ == "__main__":
    n = 4
    edges = [
        (0, 1, 3), (0, 3, 7),
        (1, 0, 8), (1, 2, 2),
        (2, 0, 5), (2, 3, 1),
        (3, 0, 2),
    ]

    dist, pred = floyd_warshall(n, edges)

    print("Distance matrix:")
    for row in dist:
        print([x if x != inf else "inf" for x in row])

    print(f"\nShortest path 1->3: {reconstruct_path(pred, 1, 3)}")
    print(f"Distance 1->3: {dist[1][3]}")
    print(f"Shortest path 2->1: {reconstruct_path(pred, 2, 1)}")
    print(f"Distance 2->1: {dist[2][1]}")
    print(f"Negative cycle: {has_negative_cycle(dist)}")
```

**출력:**

```
Distance matrix:
[0, 3, 5, 6]
[5, 0, 2, 3]
[3, 6, 0, 1]
[2, 5, 7, 0]

Shortest path 1->3: [1, 2, 3]
Distance 1->3: 3
Shortest path 2->1: [2, 0, 1]
Distance 2->1: 6
Negative cycle: False
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 25.2절: 플로이드-워셜 알고리즘.
- Floyd, R. W. (1962). Algorithm 97: Shortest path. *Communications of the ACM*, 5(6), 345.

## 연습문제

**연습문제 1.**
꼭짓점 3개와 변 $(0,1,4)$, $(0,2,11)$, $(1,2,2)$, $(2,0,3)$을 갖는 방향 그래프에서 플로이드-워셜 알고리즘을 따라가라. 가운데 꼭짓점 $k = 0, 1, 2$마다 뒤의 거리 행렬을 보여라.

??? success "연습문제 1 풀이"
    처음: $D^{(-1)} = \begin{pmatrix} 0 & 4 & 11 \\ \infty & 0 & 2 \\ 3 & \infty & 0 \end{pmatrix}$

    $k=0$ 뒤: 꼭짓점 0을 지나는 길을 살핀다. $D[1][2] = \min(\infty, \infty + 4) = 2$(바뀜 없음), $D[2][1] = \min(\infty, 3 + 4) = 7$. $D^{(0)} = \begin{pmatrix} 0 & 4 & 11 \\ \infty & 0 & 2 \\ 3 & 7 & 0 \end{pmatrix}$

    $k=1$ 뒤: $D[0][2] = \min(11, 4 + 2) = 6$, $D[2][2] = \min(0, 7 + 2) = 0$. $D^{(1)} = \begin{pmatrix} 0 & 4 & 6 \\ \infty & 0 & 2 \\ 3 & 7 & 0 \end{pmatrix}$

    $k=2$ 뒤: $D[0][0] = \min(0, 6+3) = 0$, $D[1][0] = \min(\infty, 2+3) = 5$, $D[1][1] = \min(0, 2+7) = 0$. $D^{(2)} = \begin{pmatrix} 0 & 4 & 6 \\ 5 & 0 & 2 \\ 3 & 7 & 0 \end{pmatrix}$ $\square$

---

**연습문제 2.**
플로이드-워셜이 모든 짝 최단 경로를 맞게 셈함을 증명하여라. 특히 $D^{(k)}[i][j]$이 $\{0, 1, \ldots, k\}$만 가운데 꼭짓점으로 쓰는 $i$에서 $j$까지의 최단 경로임을 귀납으로 보여라.

??? success "연습문제 2 풀이"
    **바탕 경우**($k = -1$): 변 $(i,j)$이 있으면 $D^{(-1)}[i][j] = w(i,j)$, $i = j$이면 $0$, 아니면 $\infty$이다. 이는 가운데 꼭짓점이 없는 최단 경로이다.

    **귀납 걸음**: $D^{(k-1)}[i][j]$이 $\{0, \ldots, k-1\}$을 가운데 꼭짓점으로 쓰는 최단 경로를 맞게 나타낸다고 놓자. $\{0, \ldots, k\}$을 가운데 꼭짓점으로 쓰는 $i$에서 $j$까지의 최단 경로는 $k$을 지나지 않거나(그러면 $D^{(k-1)}[i][j]$과 같다) $k$을 꼭 한 번 지난다(그러면 $D^{(k-1)}[i][k] + D^{(k-1)}[k][j]$이다. 최단 부분 길은 $\{0, \ldots, k-1\}$만 가운데 꼭짓점으로 쓰기 때문이다). 점화식 $D^{(k)}[i][j] = \min(D^{(k-1)}[i][j], D^{(k-1)}[i][k] + D^{(k-1)}[k][j])$이 두 경우를 모두 담는다. $\square$

---

**연습문제 3.**
플로이드-워셜이 무게가 음인 고리를 어떻게 알아내는가? 그 조건을 밝히고 왜 되는지 설명하여라.

??? success "연습문제 3 풀이"
    플로이드-워셜을 돌린 뒤 거리 행렬의 대각선을 살핀다. 어떤 꼭짓점 $i$에 대해 $D[i][i] < 0$이면 꼭짓점 $i$을 담은, 무게가 음인 고리가 있다. $D[i][i]$이 $i$에서 다시 $i$로 가는 가장 짧은 "길"(곧 고리)을 나타내므로 이렇게 된다. 이 값이 음이면 $i$을 지나는, 무게의 합이 음인 고리가 있다. 음의 고리가 없으면 모든 $i$에 대해 $D[i][i]$이 0으로 남는다(시시한 빈 길). $\square$

---

**연습문제 4.**
플로이드-워셜의 시간 복잡도를 꼭짓점마다 데이크스트라를 돌리는 것과 견주어라. 어느 쪽이 언제 나은가?

??? success "연습문제 4 풀이"
    플로이드-워셜은 변의 개수와 상관없이 늘 $O(V^3)$이다. 꼭짓점마다 (이진 힙을 쓴) 데이크스트라를 돌리면 $O(V(V + E) \log V)$이다. 빽빽한 그래프($E = \Theta(V^2)$)에서 데이크스트라는 $O(V^3 \log V)$으로 플로이드-워셜보다 나쁘다. 성긴 그래프($E = O(V)$)에서 데이크스트라는 $O(V^2 \log V)$으로 $O(V^3)$보다 훨씬 낫다. 플로이드-워셜은 구현이 더 단순하고 (데이크스트라와 달리) 음의 무게에서도 굴러가며 빽빽한 그래프에서 캐시에 더 상냥하다. 성긴 그래프에는 데이크스트라(음의 무게라면 존슨의 알고리즘을 거쳐)가 낫다. $\square$

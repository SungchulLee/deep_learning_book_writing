# kd 트리의 최근접 이웃 찾기

질의 점에 가장 가까운 점을 찾는 일은 계산 기하학, 기계 학습(k-최근접 이웃), 공간 데이터베이스에서 가장 흔한 연산 가운데 하나이다. 순진한 찾기는 질의를 점 $n$개 모두와 $O(n)$ 시간에 견준다. [kd 트리](construction.md)는 트리의 짜임으로 찾기 공간의 큰 몫을 쳐 내어 이를 평균 $O(\log n)$으로 줄인다. 핵심 통찰은 지금까지 찾은 가장 가까운 점이 쪼개기 초평면까지의 거리보다 가까우면 그 초평면 너머의 부분 트리를 통째로 건너뛸 수 있다는 것이다.

---

## 1. 알고리즘

kd 트리의 최근접 이웃 찾기는 **지금까지 가장 좋은** 점과 거리를 지키는 재귀 절차이다. 노드마다 다음과 같이 한다.

1. 질의 점 $q$에서 지금 노드의 점 $p$까지의 **거리를 셈한다**. 이 거리가 지금까지 가장 좋은 것보다 작으면 그것을 고친다.
2. **어느 자식을 먼저 들를지 가린다.** 쪼개는 축을 따라 $q$의 좌표를 $p$의 좌표와 견준다. 최근접 이웃이 거기 있을 법하므로 $q$과 같은 쪽의 자식("가까운" 자식)을 먼저 들른다.
3. **가까운 자식으로 재귀한다.**
4. **먼 자식을 들러야 하는지 살핀다.** $q$에서 쪼개기 초평면까지의 거리를 셈한다. 이 거리가 지금까지 가장 좋은 거리보다 작으면 먼 부분 트리에 더 가까운 점이 있을 수 있으므로 그리로 재귀한다. 그렇지 않으면 먼 부분 트리를 통째로 쳐 낸다.

---

## 2. 의사코드

```
NN-SEARCH(node, query, best, best_dist):
    if node is nil:
        return (best, best_dist)

    dist = distance(query, node.point)
    if dist < best_dist:
        best = node.point
        best_dist = dist

    axis = node.axis
    diff = query[axis] - node.point[axis]

    if diff <= 0:
        near, far = node.left, node.right
    else:
        near, far = node.right, node.left

    (best, best_dist) = NN-SEARCH(near, query, best, best_dist)

    if |diff| < best_dist:             # 먼 쪽에 더 가까운 점이 있을 수 있다
        (best, best_dist) = NN-SEARCH(far, query, best, best_dist)

    return (best, best_dist)
```

---

## 3. 쳐 내는 조건

쳐 내기 단계가 이 알고리즘의 효율의 알맹이이다. 노드의 쪼개기 초평면이 공간을 반공간 둘로 가른다. 질의 점 $q$에서 먼 반공간의 어떤 점까지의 최소 거리는 다음과 같다.

$$
d_{\text{hyperplane}} = |q[\text{axis}] - p[\text{axis}]|
$$

$d_{\text{hyperplane}} \ge d_{\text{best}}$이면 먼 부분 트리의 모든 점이 한 좌표를 따라 적어도 $d_{\text{hyperplane}}$만큼 떨어져 있으므로, 삼각 부등식에 따라 지금까지 가장 좋은 것보다 가까울 수 없다.

??? example "2차원에서의 최근접 이웃 찾기"
    점 $(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$과 질의 $q = (6, 3)$을 생각하자.

    (돌아가며 쪼개어 세운) kd 트리는 먼저 $x$으로, 그다음 $y$으로 쪼개는 식이다.

    1. 뿌리 $(7,2)$(축=0)에서 시작한다. $q$까지의 거리는 $\sqrt{(6-7)^2 + (3-2)^2} = \sqrt{2} \approx 1.41$이다. 가장 좋은 것은 $(7,2)$, best\_dist는 1.41이다.
    2. $q[0] = 6 < 7$이므로 왼쪽으로 먼저 간다.
    3. $(5,4)$(축=1)을 들른다. 거리는 $\sqrt{1+1} = \sqrt{2} \approx 1.41$이다. 비겼으므로 지금 것을 지킨다.
    4. 재귀를 이어 가면… 끝내 알고리즘이 $(7,2)$을 최근접 이웃으로 찾는다.

---

## 4. 구현

```python
"""kd 트리의 최근접 이웃 찾기."""

from __future__ import annotations

import math

# === 노드 정의 ===

class KDNode:
    """점과 쪼개는 축을 지닌 kd 트리 노드."""

    def __init__(self, point: list[float], axis: int):
        self.point = point
        self.axis = axis
        self.left: KDNode | None = None
        self.right: KDNode | None = None

# === 세우기 (construction.md에서) ===

def build_kdtree(points: list[list[float]], depth: int = 0) -> KDNode | None:
    """균형 잡힌 kd 트리를 세운다."""
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    node = KDNode(points[mid], axis)
    node.left = build_kdtree(points[:mid], depth + 1)
    node.right = build_kdtree(points[mid + 1:], depth + 1)
    return node

# === 최근접 이웃 찾기 ===

def nearest_neighbor(node: KDNode | None, query: list[float],
                     best: list[float] | None = None,
                     best_dist: float = math.inf
                     ) -> tuple[list[float] | None, float]:
    """kd 트리에서 *query*에 가장 가까운 점을 찾는다."""
    if node is None:
        return best, best_dist

    dist = math.sqrt(sum((q - p) ** 2 for q, p in zip(query, node.point)))
    if dist < best_dist:
        best, best_dist = node.point, dist

    axis = node.axis
    diff = query[axis] - node.point[axis]
    near = node.left if diff <= 0 else node.right
    far = node.right if diff <= 0 else node.left

    best, best_dist = nearest_neighbor(near, query, best, best_dist)

    if abs(diff) < best_dist:
        best, best_dist = nearest_neighbor(far, query, best, best_dist)

    return best, best_dist

# === 시연 ===

if __name__ == "__main__":
    pts = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    root = build_kdtree(pts)

    query = [6, 3]
    result, dist = nearest_neighbor(root, query)
    print(f"Nearest to {query}: {result} (distance={dist:.3f})")
```

**출력:**

```
Nearest to [6, 3]: [7, 2] (distance=1.414)
```

---

## 5. 복잡도

| 지표 | 평균 | 최악 |
|--------|-------------|------------|
| 시간 | $O(\log n)$ | $O(n)$ |
| 공간 | 스택 $O(\log n)$ | 스택 $O(n)$ |

평균 $O(\log n)$ 한계는 점이 어지간히 고르게 퍼져 있다고 본다. 최악의 $O(n)$은 쳐 내는 조건이 대부분의 노드에서 통하지 않을 때(이를테면 질의 점이 담긴 모든 점에서 멀고 트리가 잘 균형 잡히지 않았을 때) 일어난다.

!!! warning "차원의 저주"
    높은 차원($k \gg \log n$)에서는 초평면까지의 거리가 점까지의 거리에 견주어 작아져 쳐 내는 조건이 자주 통하지 않는다. $k \gtrsim 20$이면 kd 트리의 최근접 이웃 찾기가 $O(n)$ 쪽으로 나빠지고 (국소성 민감 해싱 같은) 어림 방법을 더 좋아한다.

---

## 6. k-최근접 이웃

하나가 아니라 $k$개의 최근접 이웃을 찾으려면 가장 좋은 점 하나를 **크기 $k$의 최대 힙**으로 바꾼다. 쳐 내는 조건은 힙에서 가장 먼 점(힙의 뿌리)까지의 거리를 쓴다. 힙에 원소가 $k$개보다 적으면 언제나 두 부분 트리를 모두 살핀다.

---

## 연습문제

**연습문제 1.**
kd 트리의 최근접 이웃 찾기의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 kd 트리의 최근접 이웃 찾기를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
kd 트리의 최근접 이웃 찾기가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.

## 정리하며

이 마당은 알고리즘、의사코드、쳐 내는 조건、구현을 차례로 짚었다.

**참고 문헌**

- Friedman, J. H., Bentley, J. L., & Finkel, R. A. (1977). An algorithm for finding best matches in logarithmic expected time. *ACM Transactions on Mathematical Software*, 3(3), 209–226.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 5. Springer.

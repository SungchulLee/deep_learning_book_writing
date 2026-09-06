# kd 트리의 범위 찾기

**범위 질의**는 "담긴 집합의 어떤 점이 주어진 영역 안에 있는가"를 묻는다. $k$차원에서 축에 나란한 직사각 영역에 대해 [kd 트리](construction.md)는 테두리 영역이 질의 직사각형과 만나지 않는 부분 트리를 쳐 내어 이 질의에 효율적으로 답한다. 2차원에서 균형 잡힌 kd 트리는 직사각 범위 안의 점 $r$개를 모두 $O(\sqrt{n} + r)$ 시간에 보고하는데, 결과가 적을 때 $O(n)$의 힘으로 밀어붙이는 훑기보다 크게 낫다.

## 질의 영역

질의는 차원마다 아래 한계와 위 한계로 정해지는, 축에 나란한 직사각형($k$차원에서는 초직사각형)이다.

$$
R = [x_1^{lo}, x_1^{hi}] \times [x_2^{lo}, x_2^{hi}] \times \cdots \times [x_k^{lo}, x_k^{hi}]
$$

모든 차원 $i = 1, \ldots, k$에서 $x_i^{lo} \le p[i] \le x_i^{hi}$이면 점 $p$을 보고한다.

## 알고리즘

범위 찾기는 kd 트리를 재귀적으로 훑으며 노드마다 세 경우를 쓴다.

1. **노드의 영역이 질의 안에 온전히 든다:** 부분 트리의 모든 점을 보고한다.
2. **노드의 영역이 질의와 만나지 않는다:** 부분 트리를 통째로 쳐 낸다.
3. **일부만 겹친다:** 지금 노드의 점이 범위에 드는지 살핀 뒤 두 자식으로 재귀한다.

노드마다 쪼개기 초평면이 질의 범위가 왼쪽 부분 트리와 만나는지, 오른쪽과 만나는지, 둘 다와 만나는지를 정한다.

```
RANGE-SEARCH(node, query_range):
    if node is nil:
        return

    if node.point is inside query_range:
        report node.point

    axis = node.axis
    if query_range.low[axis] <= node.point[axis]:
        RANGE-SEARCH(node.left, query_range)
    if query_range.high[axis] >= node.point[axis]:
        RANGE-SEARCH(node.right, query_range)
```

## 쳐 내는 조건

점 $p$과 축 $d$을 가진 노드의 쪼개기 초평면이 공간을 $p[d]$에서 가른다. 왼쪽 부분 트리는 축 $d$을 따라 좌표가 $p[d]$ 이하인 점을, 오른쪽 부분 트리는 좌표가 $p[d]$보다 큰 점을 담는다.

- $q_{low}[d] > p[d]$이면(질의 범위가 쪼갠 자리 뒤에서 시작하면) **왼쪽 부분 트리를 건너뛴다**.
- $q_{high}[d] < p[d]$이면(질의 범위가 쪼갠 자리 앞에서 끝나면) **오른쪽 부분 트리를 건너뛴다**.

두 조건이 모두 통하지 않으면(쪼개는 값이 질의 범위 안에 들면) 알고리즘이 두 부분 트리를 모두 살펴야 한다.

## 구현

```python
"""kd 트리의 범위 찾기."""

from __future__ import annotations


# === 노드 정의 ===

class KDNode:
    """점과 쪼개는 축을 지닌 kd 트리 노드."""

    def __init__(self, point: list[float], axis: int):
        self.point = point
        self.axis = axis
        self.left: KDNode | None = None
        self.right: KDNode | None = None


# === 세우기 ===

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


# === 범위 찾기 ===

def range_search(node: KDNode | None,
                 low: list[float], high: list[float]) -> list[list[float]]:
    """축에 나란한 직사각형 [low, high] 안의 모든 점을 찾는다."""
    result: list[list[float]] = []
    _range_helper(node, low, high, result)
    return result


def _range_helper(node: KDNode | None,
                  low: list[float], high: list[float],
                  result: list[list[float]]) -> None:
    """범위 찾기의 재귀 도우미."""
    if node is None:
        return

    # 지금 점이 범위 안에 있는지 살핀다
    if all(lo <= p <= hi for lo, p, hi in zip(low, node.point, high)):
        result.append(node.point)

    axis = node.axis

    # 범위와 만날 수 있는 부분 트리로 재귀한다
    if low[axis] <= node.point[axis]:
        _range_helper(node.left, low, high, result)
    if high[axis] >= node.point[axis]:
        _range_helper(node.right, low, high, result)


# === 시연 ===

if __name__ == "__main__":
    pts = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    root = build_kdtree(pts)

    low, high = [3, 2], [8, 5]
    found = range_search(root, low, high)
    print(f"Points in [{low}, {high}]: {found}")
    # 기대값: [5, 4], [7, 2] (범위 안에 다른 것이 더 있을 수 있다)
```

## 복잡도

균형 잡힌 kd 트리에서 범위 찾기의 복잡도는 차원의 수에 매인다.

| 차원 | 질의 시간 | 공간 |
|------------|-----------|-------|
| 2차원 | $O(\sqrt{n} + r)$ | $O(n)$ |
| $k$차원 | $O(n^{1 - 1/k} + r)$ | $O(n)$ |

여기서 $r$은 보고한 점의 수이다.

2차원의 $O(\sqrt{n})$ 항은 트리의 층마다 질의 범위가 (해당 축에서) 쪼개는 선을 많아야 2개 지나고 트리의 층이 $O(\log n)$개이기 때문에 생긴다. 꼼꼼히 따지면 $r = 0$일 때 들르는 노드의 수가 $O(\sqrt{n})$으로 한계 지어짐을 알 수 있다.

!!! note "아래 한계"
    2차원 범위 질의의 $O(\sqrt{n} + r)$ 한계는 kd 트리에서 빡빡하다. 노드를 $\Omega(\sqrt{n})$개 들러야 하는 질의 배치가 있다. 더 빠른 범위 질의가 필요하면 $O(n \log n)$의 공간을 대가로 $O(\log^2 n + r)$(부분 이어 넘기기를 쓰면 $O(\log n + r)$)을 이루는 **범위 트리**를 생각해 보라.

## 원형 범위 질의

직사각형이 아닌 영역(이를테면 "질의 점 $q$에서 거리 $d$ 안의 모든 점 찾기")에는 원의 테두리 직사각형을 질의 범위로 쓰고, 후보 점마다 실제 거리를 살피는 뒷거르기 단계를 더한다. kd 트리는 여전히 효율적으로 쳐 내 준다.

## 참고 문헌

- Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509–517.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 5. Springer.


## 연습문제

**연습문제 1.**
kd 트리의 범위 찾기의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 kd 트리의 범위 찾기를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
kd 트리의 범위 찾기가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.
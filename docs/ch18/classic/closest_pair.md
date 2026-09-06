# 가장 가까운 점 짝

Given $n$ points in the plane, finding the pair with the smallest Euclidean distance by brute force requires checking all $\binom{n}{2}$ pairs in $O(n^2)$ time. A divide-and-conquer approach achieves $O(n \log n)$, matching the lower bound for comparison-based algorithms. The key challenge lies in the **combine step**, where a clever geometric argument limits the number of cross-boundary pairs to examine.

## 문제 서술

Given a set $P = \{p_1, p_2, \dots, p_n\}$ of points in $\mathbb{R}^2$, find:

$$
\min_{i \ne j} d(p_i, p_j) = \min_{i \ne j} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

## 나누어 이기기 알고리즘

**1단계: 정렬.** 모든 점을 $x$-자리표로 정렬한다. $y$-자리표로 정렬한 복사본도 함께 지닌다.

**2단계: 나누기.** 가운뎃 $x$-자리표에서 $P$을 두 반쪽 $P_L$과 $P_R$으로 쪼갠다.

**Step 3: Conquer.** Recursively find the closest pair in $P_L$ (distance $\delta_L$) and in $P_R$ (distance $\delta_R$). Let $\delta = \min(\delta_L, \delta_R)$.

**Step 4: Combine.** Check whether any pair with one point in $P_L$ and the other in $P_R$ has distance less than $\delta$. This is where the algorithm's efficiency depends on a geometric insight.

## 띠 논증

Only points within distance $\delta$ of the dividing line can form a closer pair. Define the **strip**:

$$
S = \{p \in P : |p.x - x_{\text{mid}}| < \delta\}
$$

Sort the points in $S$ by $y$-coordinate. For each point $p$ in $S$, compare it only to points within $\delta$ in the $y$-direction.

!!! note "성김 보조정리"
    For any point $p$ in the strip, at most **7** other points in $S$ lie within a $\delta \times 2\delta$ rectangle centered at $p$. Therefore, the inner loop examines at most 7 candidates per point.

The proof uses a packing argument: a $\delta \times 2\delta$ rectangle can be divided into eight $(\delta/2) \times (\delta/2)$ sub-squares. Each sub-square contains at most one point (since any two points in the same half have distance at least $\delta$), so at most $8 - 1 = 7$ other points exist in the rectangle.

곧 아우르기 단계는 $O(|S|)$ 시간(띠 크기에 한 줄로 비례)이 들며 전체 되돌이 관계식은 다음과 같다:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

## 구현

```python
"""
나누어 이기기로 찾는 평면 위 가장 가까운 점 짝.

아우르기 단계에서 띠 자리가 성기다는 것을 써먹어
O(n log n) 시간을 이룬다.
"""

import math

# === 가장 가까운 짝 알고리즘 ===

def closest_pair(points: list[tuple[float, float]]) -> float:
    """가장 가까운 점 짝의 거리 찾기.

    인수:
        points: (x, y) 자리표의 목록.

    반환값:
        아무 두 점 사이의 최소 유클리드 거리.
    """
    def dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _solve(px: list, py: list) -> float:
        n = len(px)
        if n <= 3:
            # 작은 경우는 막무가내로
            best = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    best = min(best, dist(px[i], px[j]))
            return best

        mid = n // 2
        mid_x = px[mid][0]

        # y 정렬 차례를 지키며 py를 왼쪽과 오른쪽으로 쪼개기
        pyl = [p for p in py if p[0] <= mid_x]
        pyr = [p for p in py if p[0] > mid_x]

        # 가운뎃점에서 같은 값 다루기
        if len(pyl) > mid:
            excess = len(pyl) - mid
            pyr = [p for p in pyl if p[0] == mid_x][-excess:] + pyr
            pyl = [p for p in pyl if p[0] < mid_x] + \
                  [p for p in pyl if p[0] == mid_x][:-excess]

        dl = _solve(px[:mid], pyl)
        dr = _solve(px[mid:], pyr)
        delta = min(dl, dr)

        # y-자리표로 정렬한 띠 세우기
        strip = [p for p in py if abs(p[0] - mid_x) < delta]

        # 띠 안의 짝 살피기(점마다 많아야 7번 견줌)
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < delta:
                delta = min(delta, dist(strip[i], strip[j]))
                j += 1

        return delta

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return _solve(px, py)


# === 시연 ===

if __name__ == "__main__":
    points = [
        (2.0, 3.0), (12.0, 30.0), (40.0, 50.0),
        (5.0, 1.0), (12.0, 10.0), (3.0, 4.0),
    ]
    result = closest_pair(points)
    print(f"Closest pair distance: {result:.4f}")

    # 막무가내로 확인하기
    best = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.sqrt((points[i][0]-points[j][0])**2 +
                          (points[i][1]-points[j][1])**2)
            best = min(best, d)
    print(f"Brute force distance: {best:.4f}")
```

**출력:**

```
Closest pair distance: 1.4142
Brute force distance: 1.4142
```

The closest pair is $(2, 3)$ and $(3, 4)$ with distance $\sqrt{2} \approx 1.4142$. Both the divide-and-conquer and brute-force approaches find the same answer.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

The initial sort takes $O(n \log n)$. The recurrence $T(n) = 2T(n/2) + O(n)$ solves to $O(n \log n)$ by the master theorem.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 33장: Computational Geometry.
- Shamos, M. I., & Hoey, D. (1975). Closest-point problems. *IEEE Symposium on FOCS*, pp. 151--162.

## 연습문제

**연습문제 1.**
가장 가까운 점 짝의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Closest Pair of Points applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
가장 가까운 점 짝의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
가장 가까운 점 짝이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
가장 가까운 점 짝의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
